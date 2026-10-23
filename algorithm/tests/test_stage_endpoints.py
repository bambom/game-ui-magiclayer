"""阶段端点（服务端模型库 -> MAGICLAYER_{SCOPE}_{MODEL,BASE,KEY}）解析与路由。"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from algorithm.providers import openai_compat
from algorithm.providers.dispatch import dispatch
from algorithm.providers.stage import StageEndpoint, stage_endpoint


@pytest.fixture(autouse=True)
def clean_stage_env(monkeypatch):
    """隔离阶段变量与全局变量，避免本机 .env 泄进用例。"""
    for scope in ("PLANNER", "ELEMENT", "BACKGROUND"):
        for name in ("MODEL", "BASE", "KEY"):
            monkeypatch.delenv(f"MAGICLAYER_{scope}_{name}", raising=False)
    for name in ("MAGICLAYER_IMAGE_MODEL", "MAGICLAYER_PLANNER_MODEL",
                 "MAGICLAYER_OPENAI_BASE", "MAGICLAYER_API_STYLE",
                 "MAGICLAYER_IMAGE_FALLBACKS",
                 "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_stage_endpoint_reads_scope_vars_with_global_fallbacks(monkeypatch):
    monkeypatch.setenv("MAGICLAYER_ELEMENT_MODEL", "img-model-x")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_BASE", "http://gw:9000/v1/")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_KEY", "sk-element")

    endpoint = stage_endpoint("ELEMENT")
    assert endpoint == StageEndpoint(model="img-model-x",
                                     base="http://gw:9000/v1",
                                     key="sk-element")
    assert endpoint.configured

    # 未设阶段变量的阶段：模型名回落全局（图像阶段用 IMAGE_MODEL），端点为空
    monkeypatch.setenv("MAGICLAYER_IMAGE_MODEL", "global-image")
    assert stage_endpoint("BACKGROUND").model == "global-image"
    assert not stage_endpoint("BACKGROUND").configured

    # planner 回落自己的全局变量
    monkeypatch.setenv("MAGICLAYER_PLANNER_MODEL", "global-planner")
    assert stage_endpoint("PLANNER").model == "global-planner"
    assert not stage_endpoint("PLANNER").configured


def test_explicit_base_routes_to_openai_compat(monkeypatch):
    monkeypatch.setenv("MAGICLAYER_API_STYLE", "gemini")
    calls = {}

    def fake_post_chat(body, base=None, key=None):
        calls["url"] = f"{openai_compat.base_url(base)}/chat/completions"
        calls["key"] = key
        calls["model"] = body["model"]
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(openai_compat, "_post_chat", fake_post_chat)

    out = dispatch.generate_text("planner-model", "prompt", [],
                                 base="http://gw:8317/v1", key="sk-stage")
    assert out == "ok"
    # 显式 base 时无视全局 API_STYLE=gemini，按注册库条目的 OpenAI 兼容协议调用
    assert calls["url"] == "http://gw:8317/v1/chat/completions"
    assert calls["key"] == "sk-stage"
    assert calls["model"] == "planner-model"


def test_assigned_stage_goes_straight_to_images_edits(monkeypatch):
    """元素/背景修补是图生图阶段：模型库条目直走 /v1/images/edits，不碰 chat。"""
    from algorithm.providers.gemini_image import GeminiImageProvider
    from algorithm.common import decode_image

    monkeypatch.setenv("MAGICLAYER_ELEMENT_MODEL", "gpt-image-2")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_BASE", "http://gw:18080/v1")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_KEY", "sk-img")

    chat_calls = {"n": 0}

    def refuse_chat(body, base=None, key=None):
        chat_calls["n"] += 1
        return {"choices": [{"message": {"content": "must not be used"}}]}

    monkeypatch.setattr(openai_compat, "_post_chat", refuse_chat)

    posted = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"b64_json": base64.b64encode(_tiny_png()).decode()}]}

    def fake_edits_post(url, data=None, files=None, headers=None, timeout=None):
        posted["url"] = url
        posted["data"] = data
        posted["files"] = dict(files)
        return FakeResponse()

    monkeypatch.setattr(openai_compat.requests, "post", fake_edits_post)

    atlas = np.full((8, 8, 3), 200, np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    out = GeminiImageProvider().repair_elements(atlas, mask)
    assert out.shape[:2] == (8, 8)
    assert chat_calls["n"] == 0
    assert posted["url"] == "http://gw:18080/v1/images/edits"
    assert posted["data"]["model"] == "gpt-image-2"
    assert "image" in posted["files"]
    assert "mask" not in posted["files"]


def test_unassigned_stage_keeps_the_global_chat_wire(monkeypatch):
    """未分配条目（全局 .env / CLI）沿用 chat 出图线：gemini-*-image 等对话式出图模型。"""
    png = base64.b64encode(_tiny_png()).decode()

    def fake_post_chat(body, base=None, key=None):
        assert base is None
        assert body["model"] == "img-model-y"
        return {"choices": [{"message": {
            "images": [{"image_url": {"url": f"data:image/png;base64,{png}"}}]}}]}

    monkeypatch.setattr(openai_compat, "_post_chat", fake_post_chat)
    from algorithm.providers.gemini_image import GeminiImageProvider

    monkeypatch.setenv("MAGICLAYER_IMAGE_MODEL", "img-model-y")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-global")
    monkeypatch.setenv("MAGICLAYER_API_STYLE", "openai")

    edits_called = {"n": 0}

    def refuse_edits(*args, **kwargs):
        edits_called["n"] += 1
        raise AssertionError("edits wire must not be used for unassigned stages")

    import algorithm.providers.openai_compat as compat_module
    monkeypatch.setattr(compat_module, "generate_image_edits", refuse_edits)

    atlas = np.full((8, 8, 3), 128, np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    out = GeminiImageProvider().repair_elements(atlas, mask)
    assert out.shape[:2] == (8, 8)
    assert edits_called["n"] == 0


def _tiny_png() -> bytes:
    """8x8 灰 PNG（IHDR 头手写，避免依赖 cv2 编码路径）。"""
    import io
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes(8 * [128]) for _ in range(8))
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def test_image_fallback_endpoints_parses_the_server_pool(monkeypatch):
    from algorithm.providers.stage import image_fallback_endpoints

    monkeypatch.setenv("MAGICLAYER_IMAGE_FALLBACKS", """[
        {"model": "gpt-image-2", "base": "http://gw:18080/v1/", "key": "sk-a"},
        {"model": "gemini-3.1-flash-image", "base": "http://gw:8317/v1", "key": "sk-b"},
        {"model": "", "base": "http://gw:8317/v1", "key": "sk-c"},
        {"base": "http://no-key", "key": ""}
    ]""")
    pool = image_fallback_endpoints()
    assert pool == [
        StageEndpoint(model="gpt-image-2", base="http://gw:18080/v1", key="sk-a"),
        StageEndpoint(model="gemini-3.1-flash-image", base="http://gw:8317/v1", key="sk-b"),
        # 空 model 保留（调用时回落全局图像模型名）；缺 key 的条目丢弃
        StageEndpoint(model=None, base="http://gw:8317/v1", key="sk-c"),
    ]

    monkeypatch.setenv("MAGICLAYER_IMAGE_FALLBACKS", "{not json")
    assert image_fallback_endpoints() == []
    monkeypatch.delenv("MAGICLAYER_IMAGE_FALLBACKS")
    assert image_fallback_endpoints() == []


def test_cooldown_on_primary_switches_to_a_chat_line_fallback(monkeypatch):
    """首选 edits 端点长冷却时改试池内条目：gemini-*-image 走 chat 出图线。"""
    from algorithm.providers.gemini_image import GeminiImageProvider
    from algorithm.providers.openai_compat import ModelCooldownError

    monkeypatch.setenv("MAGICLAYER_ELEMENT_MODEL", "gpt-image-2")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_BASE", "http://gw:18080/v1")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_KEY", "sk-img")
    monkeypatch.setenv("MAGICLAYER_IMAGE_FALLBACKS",
                       '[{"model": "gemini-3.1-flash-image",'
                       '  "base": "http://gw:8317/v1", "key": "sk-b"}]')

    def cooling(files, data, base=None, key=None):
        raise ModelCooldownError("模型冷却中，约 1196s 后恢复")

    chat_calls = {}

    def fake_post_chat(body, base=None, key=None):
        chat_calls.update(base=base, key=key, model=body["model"])
        png = base64.b64encode(_tiny_png()).decode()
        return {"choices": [{"message": {
            "images": [{"image_url": {"url": f"data:image/png;base64,{png}"}}]}}]}

    monkeypatch.setattr(openai_compat, "_post_edits", cooling)
    monkeypatch.setattr(openai_compat, "_post_chat", fake_post_chat)

    atlas = np.full((8, 8, 3), 200, np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    out = GeminiImageProvider().repair_elements(atlas, mask)
    assert out.shape[:2] == (8, 8)
    assert chat_calls == {"base": "http://gw:8317/v1", "key": "sk-b",
                          "model": "gemini-3.1-flash-image"}


def test_cooldown_falls_through_to_an_edits_line_fallback(monkeypatch):
    """池内 gpt-image*/grok-imagine* 条目换线后仍走 edits，只是换了端点。"""
    from algorithm.providers.gemini_image import GeminiImageProvider
    from algorithm.providers.openai_compat import ModelCooldownError

    monkeypatch.setenv("MAGICLAYER_ELEMENT_MODEL", "gpt-image-2")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_BASE", "http://gw:18080/v1")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_KEY", "sk-img")
    monkeypatch.setenv("MAGICLAYER_IMAGE_FALLBACKS",
                       '[{"model": "gpt-image-2",'
                       '  "base": "http://gw:9000/v1", "key": "sk-c"}]')

    edits_calls = []

    def fake_post_edits(files, data, base=None, key=None):
        edits_calls.append((base, key, data.get("model")))
        if base == "http://gw:18080/v1":
            raise ModelCooldownError("模型冷却中，约 1196s 后恢复")
        return {"data": [{"b64_json": base64.b64encode(_tiny_png()).decode()}]}

    monkeypatch.setattr(openai_compat, "_post_edits", fake_post_edits)

    atlas = np.full((8, 8, 3), 200, np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    out = GeminiImageProvider().repair_elements(atlas, mask)
    assert out.shape[:2] == (8, 8)
    assert edits_calls == [
        ("http://gw:18080/v1", "sk-img", "gpt-image-2"),
        ("http://gw:9000/v1", "sk-c", "gpt-image-2"),
    ]


def test_non_cooldown_failure_does_not_switch_lines(monkeypatch):
    """非冷却错误（如模型被拒）不换线：配置错误应当暴露而不是被备用模型掩盖。"""
    from algorithm.providers.gemini_image import GeminiImageProvider
    from algorithm.providers.openai_compat import OpenAIError

    monkeypatch.setenv("MAGICLAYER_ELEMENT_MODEL", "gpt-image-2")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_BASE", "http://gw:18080/v1")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_KEY", "sk-img")
    monkeypatch.setenv("MAGICLAYER_IMAGE_FALLBACKS",
                       '[{"model": "gemini-3.1-flash-image",'
                       '  "base": "http://gw:8317/v1", "key": "sk-b"}]')

    def reject(files, data, base=None, key=None):
        raise OpenAIError("HTTP 400: images endpoint requires an image model")

    monkeypatch.setattr(openai_compat, "_post_edits", reject)
    monkeypatch.setattr(
        openai_compat, "_post_chat",
        lambda body, base=None, key=None: (_ for _ in ()).throw(
            AssertionError("chat wire must not be tried on non-cooldown failures")))

    atlas = np.full((8, 8, 3), 200, np.uint8)
    with pytest.raises(OpenAIError):
        GeminiImageProvider().repair_elements(atlas, np.zeros((8, 8), np.uint8))
