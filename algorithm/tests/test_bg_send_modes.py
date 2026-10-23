"""背景修补按接口选择发送模式（probe-bg-r1）与 Grok URL 回落。"""
from __future__ import annotations

import numpy as np
import pytest

from algorithm.providers.gemini_image import (
    SCENE_PROMPT, SHORT_SCENE_PROMPT, background_send_mode,
)
from algorithm.providers.stage import StageEndpoint


@pytest.fixture(autouse=True)
def _clean_bg_send_env(monkeypatch):
    monkeypatch.delenv("MAGICLAYER_BG_ATTACH_MASK", raising=False)
    monkeypatch.delenv("MAGICLAYER_BG_PROMPT", raising=False)
    for name in ("MODEL", "BASE", "KEY", "PROVIDER", "QUALITY"):
        monkeypatch.delenv(f"MAGICLAYER_BACKGROUND_{name}", raising=False)


def _edits(model: str, provider: str = "") -> StageEndpoint:
    return StageEndpoint(model=model, base="http://gw/v1", key="k",
                         provider=provider or None)


def test_send_mode_default_nomask_scene():
    mode = background_send_mode(_edits("gpt-image-2"), "gpt-image-2")
    assert mode.attach_mask is False
    assert mode.prompt == SCENE_PROMPT
    assert mode.name == "nomask-scene"


def test_send_mode_grok_edit_default_nomask_scene():
    mode = background_send_mode(_edits("grok-imagine-edit"), "grok-imagine-edit")
    assert mode.attach_mask is False
    assert mode.prompt == SCENE_PROMPT


def test_send_mode_grok_image_2_default_nomask_scene():
    mode = background_send_mode(_edits("grok-imagine-image-2.0"),
                                "grok-imagine-image-2.0")
    assert mode.attach_mask is False
    assert mode.prompt == SCENE_PROMPT


def test_send_mode_openrouter_gpt_default_nomask_scene():
    mode = background_send_mode(
        _edits("openai/gpt-image-2", provider="openrouter"),
        "openai/gpt-image-2")
    assert mode.attach_mask is False
    assert mode.prompt == SCENE_PROMPT


def test_send_mode_env_overrides(monkeypatch):
    monkeypatch.setenv("MAGICLAYER_BG_ATTACH_MASK", "1")
    monkeypatch.setenv("MAGICLAYER_BG_PROMPT", "short")
    mode = background_send_mode(_edits("gpt-image-2"), "gpt-image-2")
    assert mode.attach_mask is True
    assert mode.prompt == SHORT_SCENE_PROMPT
    assert mode.name == "mask-short"


def test_send_mode_env_mask_off(monkeypatch):
    monkeypatch.setenv("MAGICLAYER_BG_ATTACH_MASK", "0")
    mode = background_send_mode(_edits("gpt-image-2"), "gpt-image-2")
    assert mode.attach_mask is False
    assert mode.prompt == SCENE_PROMPT
    assert mode.name == "nomask-scene"


def test_repair_background_gpt_omits_mask(monkeypatch):
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod

    seen = {}

    def fake_edits(model, prompt, images, base=None, key=None, quality=None,
                   opaque_editable=False):
        seen["n"] = len(images)
        seen["prompt"] = prompt
        return np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", fake_edits)
    for var, val in {
        "MAGICLAYER_BACKGROUND_BASE": "http://gw/v1",
        "MAGICLAYER_BACKGROUND_KEY": "k",
        "MAGICLAYER_BACKGROUND_MODEL": "gpt-image-2",
    }.items():
        monkeypatch.setenv(var, val)
    gi.GeminiImageProvider().repair_background(
        np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), bool))
    assert seen["n"] == 1
    assert seen["prompt"] == SCENE_PROMPT


def test_repair_background_grok_image_omits_mask(monkeypatch):
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod

    seen = {}

    def fake_edits(model, prompt, images, base=None, key=None, quality=None,
                   opaque_editable=False):
        seen["n"] = len(images)
        seen["opaque"] = opaque_editable
        return np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", fake_edits)
    for var, val in {
        "MAGICLAYER_BACKGROUND_BASE": "http://gw/v1",
        "MAGICLAYER_BACKGROUND_KEY": "k",
        "MAGICLAYER_BACKGROUND_MODEL": "grok-imagine-image-2.0",
    }.items():
        monkeypatch.setenv(var, val)
    gi.GeminiImageProvider().repair_background(
        np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), bool))
    assert seen["n"] == 1
    assert seen["opaque"] is True


def test_repair_background_grok_edit_omits_mask_scene(monkeypatch):
    """配置 grok-imagine-edit 时自动：只送涂黑图，不另附掩膜。"""
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod

    seen = {}

    def fake_edits(model, prompt, images, base=None, key=None, quality=None,
                   opaque_editable=False):
        seen["n"] = len(images)
        seen["prompt"] = prompt
        return np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", fake_edits)
    for var, val in {
        "MAGICLAYER_BACKGROUND_BASE": "http://gw/v1",
        "MAGICLAYER_BACKGROUND_KEY": "k",
        "MAGICLAYER_BACKGROUND_MODEL": "grok-imagine-edit",
    }.items():
        monkeypatch.setenv(var, val)
    gi.GeminiImageProvider().repair_background(
        np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), bool))
    assert seen["n"] == 1
    assert seen["prompt"] == SCENE_PROMPT


def test_repair_background_openrouter_omits_mask_on_images_wire(monkeypatch):
    """配置 OpenRouter gpt-image 时自动：只送涂黑图，走 /images 线。"""
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod

    seen = {}

    def fake_or(model, prompt, images, base=None, key=None, quality=None):
        seen["n"] = len(images)
        seen["prompt"] = prompt
        seen["wire"] = "openrouter"
        seen["model"] = model
        return np.zeros((4, 4, 3), np.uint8)

    def explode(*args, **kwargs):
        raise AssertionError("OpenRouter 背景修补不得走 edits/chat 线")

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_openrouter", fake_or)
    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", explode)
    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image", explode)
    for var, val in {
        "MAGICLAYER_BACKGROUND_BASE": "https://openrouter.ai/api/v1",
        "MAGICLAYER_BACKGROUND_KEY": "sk-or-test",
        "MAGICLAYER_BACKGROUND_MODEL": "openai/gpt-image-2",
        "MAGICLAYER_BACKGROUND_PROVIDER": "openrouter",
        "MAGICLAYER_BACKGROUND_QUALITY": "low",
    }.items():
        monkeypatch.setenv(var, val)
    gi.GeminiImageProvider().repair_background(
        np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), bool))
    assert seen["wire"] == "openrouter"
    assert seen["n"] == 1
    assert seen["prompt"] == SCENE_PROMPT
    assert seen["model"] == "openai/gpt-image-2"


def test_edits_url_fallback(monkeypatch):
    """Grok 只回 data[0].url 时下载 JPEG/PNG，不再当失败。"""
    from algorithm.providers import openai_compat as oc

    tiny = np.full((6, 6, 3), (12, 34, 56), np.uint8)
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(tiny, cv2.COLOR_RGB2BGR))
    assert ok
    png = buf.tobytes()

    class PostResp:
        status_code = 200
        text = ""
        content = b"{}"

        @staticmethod
        def json():
            return {"data": [{"url": "https://imgen.example/x.jpeg",
                              "mime_type": "image/jpeg"}]}

    class GetResp:
        status_code = 200
        text = ""
        content = png

    def fake_post(url, data=None, files=None, headers=None, timeout=None):
        return PostResp()

    def fake_get(url, timeout=None, proxies=None):
        assert url.startswith("https://imgen.example/")
        return GetResp()

    monkeypatch.setattr(oc.requests, "post", fake_post)
    monkeypatch.setattr(oc.requests, "get", fake_get)
    out = oc.generate_image_edits("grok-imagine-image-2.0", "p",
                                  [np.zeros((6, 6, 3), np.uint8)],
                                  base="http://gw/v1", key="k")
    assert out.shape == (6, 6, 3)
    assert int(out[0, 0, 0]) == 12


def test_edits_still_errors_without_b64_or_url(monkeypatch):
    from algorithm.providers import openai_compat as oc
    from algorithm.providers.openai_compat import OpenAIError

    class PostResp:
        status_code = 200
        text = "{}"
        content = b"{}"

        @staticmethod
        def json():
            return {"data": [{}]}

    monkeypatch.setattr(oc.requests, "post",
                        lambda *a, **k: PostResp())
    with pytest.raises(OpenAIError, match="响应无图像"):
        oc.generate_image_edits("gpt-image-2", "p",
                                [np.zeros((4, 4, 3), np.uint8)],
                                base="http://gw/v1", key="k")


def test_scene_repair_skips_anchor_by_default():
    """MAGICLAYER_BG_ANCHOR 默认 0：provider 出图的洞内颜色不被 Telea 换掉。"""
    from algorithm.repair.background import scene_repair

    class Marker:
        name = "marker"
        degraded = False

        def repair_background(self, scene_hole_rgb, hole_mask=None):
            out = scene_hole_rgb.copy()
            out[:] = (10, 20, 30)
            return out

    work = np.full((32, 32, 3), (200, 220, 240), np.uint8)
    hard = np.zeros((32, 32), bool)
    hard[8:24, 8:24] = True
    out = scene_repair(work, hard, Marker())
    # 洞中心完全取生图（羽化 alpha=1），应接近标记色而非 Telea 纸色
    assert abs(int(out[16, 16, 0]) - 10) <= 3
