"""元素遮挡修补修复（两遍式 bbox / 镂空语义 / none 纠偏）与 run.log 行为。

修复前的缺陷：单循环内当场赋 inst.bbox 而镂空计算读子元素 bbox，规划顺序父先
子后时子 bbox 恒为 None，镂空恒空——遮挡修补从未生效。这些用例按父先/子先两种
顺序各跑一遍，验证修补写入与 alpha 并入。
"""
import base64

import numpy as np
import pytest

from algorithm import runlog
from algorithm.common import Instance, Query
from algorithm.repair.elements import build_element_layers


class MarkerProvider:
    """生图桩：Atlas 的白 mask 区一律涂成标记色，验证修补像素确实写回。

    标记色带确定性纹理（各通道 ±10 波动）：洞填再对齐（linework.relevel）只
    矫正"平色填充"（白盘缺陷形态，局部高频 RMS < 3），±10 纹理（RMS≈6）让
    桩保持"材质延续"形态，不被后处理改写。
    """

    name = "marker"
    degraded = False

    def repair_elements(self, atlas_rgb, atlas_mask):
        out = atlas_rgb.copy()
        out[atlas_mask > 0] = (10, 200, 90)
        ys, xs = np.nonzero(atlas_mask)
        out[ys, xs, 0] = (out[ys, xs, 0].astype(int)
                          + (ys * 7 + xs * 13) % 21 - 10).astype(np.uint8)
        out[ys, xs, 1] = (out[ys, xs, 1].astype(int)
                          + (ys * 5 + xs * 11) % 21 - 10).astype(np.uint8)
        out[ys, xs, 2] = (out[ys, xs, 2].astype(int)
                          + (ys * 3 + xs * 17) % 21 - 10).astype(np.uint8)
        return out

    def repair_background(self, scene_hole_rgb, hole_mask=None):
        return scene_hole_rgb.copy()


MARKER = np.array([10, 200, 90])


def _q(qid, mode="image", parent=None):
    return Query(id=qid, name=qid, kind="ui", role="foreground",
                 element_repair_mode=mode, parent_query_id=parent, z_order=0)


def _scene():
    """64x64 图：父元素占左半（红底），子元素在父内部 16:32 方块（蓝）。"""
    work = np.full((64, 64, 3), (240, 240, 240), np.uint8)
    parent_mask = np.zeros((64, 64), bool)
    parent_mask[:, :40] = True
    work[:, :40] = (200, 40, 40)
    child_mask = np.zeros((64, 64), bool)
    child_mask[16:32, 16:32] = True
    work[16:32, 16:32] = (30, 30, 220)
    return work, parent_mask, child_mask


def _by_id(results):
    return {inst.instance_id: (rgba, layer) for inst, rgba, layer in results}


@pytest.mark.parametrize("order", ["parent_first", "child_first"])
def test_occlusion_repaired_in_both_orders(order):
    work, parent_mask, child_mask = _scene()
    parent = Instance(instance_id="card", query=_q("card", "image"), hint=None,
                      mask=parent_mask)
    child = Instance(instance_id="badge", query=_q("badge", "none", parent="card"),
                     hint=None, mask=child_mask)
    instances = [parent, child] if order == "parent_first" else [child, parent]

    results, diag = build_element_layers(instances, work, MarkerProvider())
    rgba, layer = _by_id(results)["card"]
    hole = rgba[16:32, 16:32]  # 子元素遮挡区（父裁片坐标 = 全图坐标，x<40）

    # 修补像素写回镂空区 + alpha 并入（重建材质属于父层）
    assert np.linalg.norm(hole[:, :, :3].reshape(-1, 3).mean(axis=0) - MARKER) < 10
    assert (hole[:, :, 3] == 255).all()
    # 非镂空区保持父元素原色
    assert (rgba[40:60, 5:15, :3] == work[40:60, 5:15]).all()
    assert layer.repaired and not layer.degraded
    assert diag["atlas_ids"] == ["card"]


def test_surface_hole_alpha_merged():
    work, parent_mask, child_mask = _scene()
    parent = Instance(instance_id="card", query=_q("card", "surface"), hint=None,
                      mask=parent_mask)
    child = Instance(instance_id="badge", query=_q("badge", "none", parent="card"),
                     hint=None, mask=child_mask)

    results, diag = build_element_layers([parent, child], work, MarkerProvider())
    rgba, layer = _by_id(results)["card"]
    # surface 修补后镂空区 alpha 并入（原 mask 内此处本就 True，验证不回归）
    assert (rgba[16:32, 16:32, 3] == 255).all()
    assert layer.repair_mode == "surface" and layer.repaired
    assert diag["surface_ids"] == ["card"]


def test_none_with_children_coerced_to_surface():
    work, parent_mask, child_mask = _scene()
    parent = Instance(instance_id="card", query=_q("card", "none"), hint=None,
                      mask=parent_mask)
    child = Instance(instance_id="badge", query=_q("badge", "none", parent="card"),
                     hint=None, mask=child_mask)

    results, diag = build_element_layers([parent, child], work, MarkerProvider())
    rgba, layer = _by_id(results)["card"]
    assert diag["none_coerced"] == ["card"]
    assert layer.repair_mode == "surface" and layer.repaired
    assert diag["none_ids"] == ["badge"]


def test_none_without_overlap_stays_none():
    work, parent_mask, child_mask = _scene()
    far_mask = np.zeros((64, 64), bool)
    far_mask[:, 50:64] = True  # 父 bbox 外的“子元素”，实际不相交
    parent = Instance(instance_id="card", query=_q("card", "none"), hint=None,
                      mask=parent_mask)
    child = Instance(instance_id="far", query=_q("far", "none", parent="card"),
                     hint=None, mask=far_mask)

    results, diag = build_element_layers([parent, child], work, MarkerProvider())
    _, layer = _by_id(results)["card"]
    assert layer.repair_mode == "none" and not layer.repaired
    assert diag["none_coerced"] == []


# ---------------- run.log ----------------

def test_runlog_format(tmp_path):
    try:
        runlog.install(str(tmp_path), image="x.png")
        runlog.event("STAGE", "开始 ocr")
        runlog.finish({"警告": ["w1"]})
        text = (tmp_path / "run.log").read_text(encoding="utf-8")
        assert "[STAGE ] 开始 ocr" in text
        assert "[SUMMARY] 运行结束（成功）" in text
        assert "警告" in text and "w1" in text
        assert "+    0.0s]" in text  # 相对耗时
    finally:
        runlog.reset()


def test_post_chat_writes_req_line(tmp_path, monkeypatch):
    from algorithm.providers import openai_compat as oc

    class FakeResp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "hi"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr(oc.requests, "post", fake_post)
    try:
        runlog.install(str(tmp_path))
        out = oc.generate_text("m1", "hello", [], base="http://gw", key="k")
        assert out == "hi"
        text = (tmp_path / "run.log").read_text(encoding="utf-8")
        assert "[REQ   ] chat http://gw/chat/completions model=m1" in text
        assert "HTTP 200" in text and "req≈" in text
    finally:
        runlog.reset()


# ---------------- openrouter 分线 ----------------

def _patch_dispatch(monkeypatch, **replacements):
    from algorithm.providers import dispatch as dispatch_mod

    recorded = {}

    def _rec(name):
        def fn(*args, **kwargs):
            recorded[name] = (args, kwargs)
            return np.zeros((4, 4, 3), np.uint8)
        return fn

    for name in ("generate_image_openrouter", "generate_image_edits", "generate_image"):
        setattr(dispatch_mod.dispatch, name, _rec(name))
    return recorded


def test_openrouter_primary_routes_to_images_wire(monkeypatch):
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod

    recorded = {}
    calls = []

    def fake_or(model, prompt, images, base=None, key=None, quality=None):
        calls.append(("or", model, base, key, quality))
        return np.zeros((4, 4, 3), np.uint8)

    def explode(*args, **kwargs):
        raise AssertionError("openrouter 条目不得走其它线")

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_openrouter", fake_or)
    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", explode)
    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image", explode)
    for var, val in {
        "MAGICLAYER_ELEMENT_BASE": "https://openrouter.ai/api/v1",
        "MAGICLAYER_ELEMENT_KEY": "sk-or-test",
        "MAGICLAYER_ELEMENT_MODEL": "openai/gpt-image-2",
        "MAGICLAYER_ELEMENT_PROVIDER": "openrouter",
        "MAGICLAYER_ELEMENT_QUALITY": "low",
    }.items():
        monkeypatch.setenv(var, val)

    provider = gi.GeminiImageProvider()
    provider.repair_elements(np.zeros((8, 8, 3), np.uint8),
                             np.zeros((8, 8), np.uint8))
    assert calls and calls[0][0] == "or"
    assert calls[0][1] == "openai/gpt-image-2"
    assert calls[0][2] == "https://openrouter.ai/api/v1"
    assert calls[0][4] == "low"


def test_quality_propagates_to_edits_wire(monkeypatch):
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod

    calls = []

    def fake_edits(model, prompt, images, base=None, key=None, quality=None,
                   opaque_editable=False):
        calls.append((model, quality))
        return np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", fake_edits)
    for var, val in {
        "MAGICLAYER_BACKGROUND_BASE": "http://gw/v1",
        "MAGICLAYER_BACKGROUND_KEY": "k",
        "MAGICLAYER_BACKGROUND_MODEL": "gpt-image-2",
        "MAGICLAYER_BACKGROUND_QUALITY": "high",
    }.items():
        monkeypatch.setenv(var, val)

    provider = gi.GeminiImageProvider()
    provider.repair_background(np.zeros((8, 8, 3), np.uint8))
    assert calls == [("gpt-image-2", "high")]


def test_mask_polarity_propagates_to_edits_wire(monkeypatch):
    """遮罩极性：默认反转（不透明=待修补，run 9bf549dd 实测该网关按此约定执行，
    规范极性下洞区被冻结在预填色上）；显式 transparent 才走 OpenAI 规范。
    极性必须一路传到 edits 线。"""
    import cv2
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod
    from algorithm.providers.stage import stage_endpoint, image_fallback_endpoints

    seen = {}

    def fake_edits(model, prompt, images, base=None, key=None, quality=None,
                   opaque_editable=False):
        seen["opaque_editable"] = opaque_editable
        return np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", fake_edits)
    for var, val in {
        "MAGICLAYER_BACKGROUND_BASE": "http://gw/v1",
        "MAGICLAYER_BACKGROUND_KEY": "k",
        "MAGICLAYER_BACKGROUND_MODEL": "gpt-image-2",
        "MAGICLAYER_IMAGE_FALLBACKS": '[{"model":"gpt-image-2","base":"http://fb/v1",'
                                      '"key":"k2","maskPolarity":"opaque"}]',
    }.items():
        monkeypatch.setenv(var, val)

    # 默认（未设置）= 反转极性（不透明=待修补）
    monkeypatch.delenv("MAGICLAYER_BACKGROUND_MASK_POLARITY", raising=False)
    assert stage_endpoint("BACKGROUND").opaque_editable is True
    gi.GeminiImageProvider().repair_background(np.zeros((8, 8, 3), np.uint8),
                                               np.ones((8, 8), bool))
    assert seen["opaque_editable"] is True

    # 换线条目同样默认反转
    assert image_fallback_endpoints()[0].opaque_editable is True

    # 显式 transparent = OpenAI 规范极性
    monkeypatch.setenv("MAGICLAYER_BACKGROUND_MASK_POLARITY", "transparent")
    assert stage_endpoint("BACKGROUND").opaque_editable is False
    gi.GeminiImageProvider().repair_background(np.zeros((8, 8, 3), np.uint8),
                                               np.ones((8, 8), bool))
    assert seen["opaque_editable"] is False


def test_encode_rgba_mask_png_polarity():
    """默认极性=OpenAI 规范（洞透明）；反转极性=洞白+不透明。"""
    import cv2
    import numpy as np

    from algorithm.common import encode_rgba_mask_png

    mask = np.zeros((6, 6), bool)
    mask[2:4, 2:4] = True

    def decode(png):
        return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)

    spec = decode(encode_rgba_mask_png(mask))
    assert spec.shape[2] == 4
    assert (spec[2:4, 2:4, 3] == 0).all()      # 洞透明=待修补
    assert (spec[0, 0, 3] == 255).all()        # 洞外不透明=保持

    inv = decode(encode_rgba_mask_png(mask, opaque_editable=True))
    assert (inv[2:4, 2:4, 3] == 255).all()     # 洞不透明=待修补（反转网关）
    assert (inv[2:4, 2:4, :3] == 255).all()    # 且以白色标记（亮度读法兼容）
    assert (inv[0, 0, 3] == 0).all()
    assert (inv[0, 0, :3] == 0).all()


def test_scene_repair_forwards_mask_to_wire(monkeypatch):
    """背景 scene 修补：待补区涂黑，默认只送这一张图，不另附 mask。"""
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod
    from algorithm.repair.background import repair_background

    seen = {}

    def fake_edits(model, prompt, images, base=None, key=None, quality=None,
                   opaque_editable=False):
        seen["model"] = model
        seen["prompt"] = prompt
        seen["n_images"] = len(images)
        seen["image"] = images[0].copy()
        seen["mask"] = images[1].copy() if len(images) > 1 else None
        return np.zeros((8, 8, 3), np.uint8)

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", fake_edits)
    monkeypatch.delenv("MAGICLAYER_BG_ATTACH_MASK", raising=False)
    monkeypatch.delenv("MAGICLAYER_BG_PROMPT", raising=False)
    for var, val in {
        "MAGICLAYER_BACKGROUND_BASE": "http://gw/v1",
        "MAGICLAYER_BACKGROUND_KEY": "k",
        "MAGICLAYER_BACKGROUND_MODEL": "gpt-image-2",
    }.items():
        monkeypatch.setenv(var, val)

    work = np.full((32, 32, 3), (200, 220, 240), np.uint8)
    work[8:24, 8:24] = (30, 30, 220)  # 被移除元素留下的像素
    hard = np.zeros((32, 32), bool)
    hard[8:24, 8:24] = True

    repair_background(work, hard, "scene", gi.GeminiImageProvider())
    assert seen["n_images"] == 1
    assert seen["mask"] is None
    hole_pixel = seen["image"][16, 16].astype(int)
    assert abs(hole_pixel - np.array([0, 0, 0])).max() <= 0
    assert abs(int(seen["image"][0, 0, 0]) - 200) <= 3  # 洞外原背景
    assert "existing background style" in seen["prompt"]


def test_seam_band_smooth_blends_boundary_step():
    """接缝带平滑：mask 边界处的硬色阶在带内被压成渐变，远处原样。"""
    from algorithm.repair.background import seam_band_smooth

    image = np.zeros((32, 64, 3), np.uint8)
    image[:, :32] = (100, 100, 100)   # mask 内（修补区）
    image[:, 32:] = (200, 200, 200)   # mask 外（原背景）
    mask = np.zeros((32, 64), bool)
    mask[:, :32] = True

    out = seam_band_smooth(image, mask)
    # mask 内侧那列从硬阶变成渐变
    assert 110 < int(out[16, 31, 0]) < 190, int(out[16, 31, 0])
    # mask 外一个像素都不改（二期问题五a 改的契约）：洞外是真实背景，往外写
    # 就是在原图上留改写痕迹——实测洞外 1px 带差 28.7 有一份来自这里
    assert np.array_equal(out[:, 32:], image[:, 32:])
    # 远离边界的洞内像素保持原值
    assert out[16, 5, 0] == 100
    # 空 mask 原样返回
    untouched = seam_band_smooth(image, np.zeros((32, 64), bool))
    assert np.array_equal(untouched, image)


def test_model_calls_recorded_for_steps_json(tmp_path, monkeypatch):
    """生图/规划调用的结构化记录（runlog.model_call）供 steps.json 回看。"""
    from algorithm import runlog as runlog_mod
    from algorithm.providers import gemini_image as gi
    from algorithm.providers import dispatch as dispatch_mod

    def fake_edits(model, prompt, images, base=None, key=None, quality=None,
                   opaque_editable=False):
        return np.zeros((8, 8, 3), np.uint8)

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", fake_edits)
    monkeypatch.delenv("MAGICLAYER_BG_ATTACH_MASK", raising=False)
    monkeypatch.delenv("MAGICLAYER_BG_PROMPT", raising=False)
    for var, val in {
        "MAGICLAYER_BACKGROUND_BASE": "http://gw/v1",
        "MAGICLAYER_BACKGROUND_KEY": "k",
        "MAGICLAYER_BACKGROUND_MODEL": "gpt-image-2",
        "MAGICLAYER_BACKGROUND_QUALITY": "high",
    }.items():
        monkeypatch.setenv(var, val)

    runlog_mod.install(str(tmp_path))
    try:
        provider = gi.GeminiImageProvider()
        provider.repair_background(np.zeros((8, 8, 3), np.uint8),
                                    np.ones((8, 8), bool))
        records = runlog_mod.model_calls()
        assert len(records) == 1
        rec = records[0]
        assert rec["scope"] == "BACKGROUND" and rec["model"] == "gpt-image-2"
        assert rec["prompt_name"] == "SCENE_PROMPT"
        assert rec["prompt"].startswith("Fill the black missing regions")
        assert rec["wire"].startswith("POST {base}/v1/images/edits")
        assert rec["params"]["size"] == "auto"
        assert rec["params"]["quality"] == "high"
        # 默认反转极性必须在 params 里可见（面板回看时能发现洞区为什么被冻结/重绘）
        assert "mask_polarity" in rec["params"]
        assert rec["params"]["attach_mask"] is False  # 涂黑输入默认不另送 mask
        assert len(rec["images"]) == 1
        assert rec["outcome"] == "ok" and rec["duration_ms"] >= 0
    finally:
        runlog_mod.reset()


def test_openrouter_fallback_after_cooldown(tmp_path, monkeypatch):
    from algorithm.providers import dispatch as dispatch_mod
    from algorithm.providers import gemini_image as gi
    from algorithm.providers.openai_compat import ModelCooldownError

    calls = []

    def cold_edits(model, prompt, images, base=None, key=None, quality=None,
                   opaque_editable=False):
        calls.append(("edits", model))
        raise ModelCooldownError("冷却 300s")

    def warm_or(model, prompt, images, base=None, key=None, quality=None):
        calls.append(("or", model, quality))
        return np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_edits", cold_edits)
    monkeypatch.setattr(dispatch_mod.dispatch, "generate_image_openrouter", warm_or)
    monkeypatch.setenv("MAGICLAYER_ELEMENT_BASE", "http://gw/v1")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_KEY", "k")
    monkeypatch.setenv("MAGICLAYER_ELEMENT_MODEL", "gpt-image-2")
    import json

    monkeypatch.setenv("MAGICLAYER_IMAGE_FALLBACKS", json.dumps([
        {"model": "openai/gpt-image-2", "base": "https://openrouter.ai/api/v1",
         "key": "sk-or-x", "provider": "openrouter", "quality": "low"}]))

    try:
        runlog.install(str(tmp_path))
        provider = gi.GeminiImageProvider()
        provider.repair_elements(np.zeros((8, 8, 3), np.uint8),
                                 np.zeros((8, 8), np.uint8))
        assert calls[0] == ("edits", "gpt-image-2")
        assert calls[1][0] == "or"
        assert calls[1][1] == "openai/gpt-image-2"
        assert calls[1][2] == "low"
        text = (tmp_path / "run.log").read_text(encoding="utf-8")
        assert "[SWITCH]" in text and "备用图像条目" in text
    finally:
        runlog.reset()


def test_openrouter_wire_body_shape(monkeypatch):
    """openrouter 请求体：input_references 对象形态 + quality 透传 + 200 解析。"""
    from algorithm.providers import openai_compat as oc

    posted = {}

    class FakeResp:
        status_code = 200
        text = ""
        content = b"{}"

        @staticmethod
        def json():
            return {"data": [{"b64_json": base64.b64encode(_tiny_png()).decode(),
                              "media_type": "image/png"}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        posted["headers"] = headers
        return FakeResp()

    monkeypatch.setattr(oc.requests, "post", fake_post)
    img = np.full((6, 6, 3), (250, 0, 0), np.uint8)
    out = oc.generate_image_openrouter("openai/gpt-image-2", "p", [img],
                                       base="https://openrouter.ai/api/v1",
                                       key="sk-or-x", quality="low")
    assert out.shape == (4, 4, 3)  # 假响应里的 4x4 PNG
    assert posted["url"] == "https://openrouter.ai/api/v1/images"
    refs = posted["json"]["input_references"]
    assert refs[0]["type"] == "image_url" and "url" in refs[0]["image_url"]
    assert refs[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert posted["json"]["quality"] == "low"
    assert posted["headers"]["Authorization"] == "Bearer sk-or-x"


# ---------------- atlas 洞区涂黑与调试产物 ----------------

def test_atlas_prefill_hides_child_from_model():
    """洞区涂黑（文档 image25）：发给模型的 Atlas 不得再含子元素原像素或占位色。"""
    work, parent_mask, child_mask = _scene()
    parent = Instance(instance_id="card", query=_q("card", "image"), hint=None,
                      mask=parent_mask)
    child = Instance(instance_id="badge", query=_q("badge", "none", parent="card"),
                     hint=None, mask=child_mask)

    seen = {}

    class SpyProvider(MarkerProvider):
        def repair_elements(self, atlas_rgb, atlas_mask):
            seen["atlas"] = atlas_rgb.copy()
            seen["mask"] = atlas_mask.copy()
            return super().repair_elements(atlas_rgb, atlas_mask)

    build_element_layers([parent, child], work, SpyProvider())
    hole = seen["mask"] > 0
    hole_rgb = seen["atlas"][hole].reshape(-1, 3).mean(axis=0)
    assert np.linalg.norm(hole_rgb - np.array([0, 0, 0])) < 1
    assert np.linalg.norm(hole_rgb - np.array([30, 30, 220])) > 60
    assert np.linalg.norm(hole_rgb - np.array([200, 40, 40])) > 60


def test_atlas_debug_artifact_saved():
    work, parent_mask, child_mask = _scene()
    parent = Instance(instance_id="card", query=_q("card", "image"), hint=None,
                      mask=parent_mask)
    child = Instance(instance_id="badge", query=_q("badge", "none", parent="card"),
                     hint=None, mask=child_mask)

    saved = {}
    build_element_layers([parent, child], work, MarkerProvider(),
                         debug=lambda name, img: saved.update({name: img}))
    assert saved["05_atlas_input.png"].ndim == 3
    assert saved["05_atlas_mask.png"].ndim == 3
    assert saved["05_atlas.png"].ndim == 3
    assert saved["05_atlas_input.png"].shape == saved["05_atlas.png"].shape
    assert saved["05_atlas_mask.png"].shape[:2] == saved["05_atlas.png"].shape[:2]


def test_semantic_mask_image_colors_instances():
    from algorithm.pipeline import semantic_mask_image

    work, parent_mask, child_mask = _scene()
    parent = Instance(instance_id="card", query=_q("card", "image"), hint=None,
                      mask=parent_mask)
    child = Instance(instance_id="badge", query=_q("badge", "none", parent="card"),
                     hint=None, mask=child_mask)
    img = semantic_mask_image([parent, child], 64, 64)
    assert img.shape == (64, 64, 3)
    colors = {tuple(c) for c in img[parent_mask | child_mask].reshape(-1, 3)}
    assert len(colors) >= 2  # 每实例一色


def test_geometric_overlap_auto_occlusion_without_parent():
    """通用兜底：planner 未声明父子时，mask 显著几何重叠也要自动镂空底层并修补
    （线上 36d778d8：时间轴节点未挂 parent，轴线层原样留着被压住的圆徽章）。"""
    work = np.full((64, 64, 3), (240, 240, 240), np.uint8)
    line_mask = np.zeros((64, 64), bool)
    line_mask[:, 30:34] = True          # 竖线（大面积）
    work[:, 30:34] = (90, 90, 90)
    node_mask = np.zeros((64, 64), bool)
    node_mask[10:24, 22:42] = True      # 圆徽章压在竖线上（较小面积=上层）
    work[10:24, 22:42] = (30, 30, 220)

    line = Instance(instance_id="track", query=_q("track", "none"), hint=None,
                    mask=line_mask)
    node = Instance(instance_id="node#1", query=_q("node", "none"), hint=None,
                    mask=node_mask)
    results, diag = build_element_layers([line, node], work, MarkerProvider())
    rgba, layer = _by_id(results)["track"]

    # 轴线被压住处被自动镂空并修补（none 纠偏为 surface）
    assert (rgba[12:22, 30:34, :3] == MARKER).all()
    assert layer.repaired
    assert diag["auto_occluded"] == [{"instance_id": "track", "by": ["node#1"]}]
    assert "track" in diag["none_coerced"]
    # 上层徽章不受影响：没有把轴线像素反向镂空进徽章
    node_rgba, _ = _by_id(results)["node#1"]
    assert (node_rgba[:, :, :3] == work[10:24, 22:42]).all()


def test_geometric_overlap_edge_bleed_does_not_occlude():
    """SAM 边缘 1-2px 渗出（低于双门槛）不触发自动镂空。"""
    work = np.full((64, 64, 3), (240, 240, 240), np.uint8)
    a_mask = np.zeros((64, 64), bool)
    a_mask[10:30, 10:30] = True
    b_mask = np.zeros((64, 64), bool)
    b_mask[10:30, 30:31] = True         # 仅 1px 相接：20px 重叠 < 24px 门槛
    a = Instance(instance_id="a", query=_q("qa", "none"), hint=None, mask=a_mask)
    b = Instance(instance_id="b", query=_q("qb", "none"), hint=None, mask=b_mask)
    results, diag = build_element_layers([a, b], work, MarkerProvider())
    assert diag["auto_occluded"] == []
    rgba_a, _ = _by_id(results)["a"]
    assert (rgba_a[:, :, :3] == work[10:30, 10:30]).all()  # 无镂空即原像素原样


def test_child_outline_halo_swallowed_by_hole():
    """子 mask 膨胀 2px：SAM 切在描边中线时，洞外残留的描边外半圈必须被镂空吞下
    修补，否则父层沿子元素轮廓留一圈"轮廓痕迹"（线上实测 -1px 处聚集成 479px）。"""
    work, parent_mask, child_mask = _scene()
    work[14:16, 18:30] = (20, 20, 20)  # 子元素上缘外 2px 的深色描边（不在子 mask 内）
    parent = Instance(instance_id="card", query=_q("card", "image"), hint=None,
                      mask=parent_mask)
    child = Instance(instance_id="badge", query=_q("badge", "none", parent="card"),
                     hint=None, mask=child_mask)

    results, diag = build_element_layers([parent, child], work, MarkerProvider())
    rgba, _ = _by_id(results)["card"]
    # 描边深色已被修补覆盖（生成裁片整块即为结果，洞内是标记色）
    band = rgba[14:16, 18:30, :3].astype(int)
    assert not (band == 20).all(axis=2).any()
    assert (band[:, :, 1] > 100).all()  # marker 绿主导


def test_text_mask_artifacts():
    """文字 mask 产物（文档 image7 右图）：全图 removal 并集 + 字形拼图落位。"""
    from algorithm.pipeline import text_glyph_sheet, text_mask_image
    from algorithm.common import OcrRegion

    full = np.zeros((64, 64), bool)
    removal = np.zeros((64, 64), bool)
    region = OcrRegion(id="text_001", text="测", conf=0.9,
                       quad=np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32),
                       bbox=(10, 10, 20, 20))
    glyph = np.zeros((10, 10), bool)
    glyph[2:8, 2:8] = True
    region.glyph_mask = glyph
    removal[10:20, 10:20] = glyph
    region.removal_mask = removal

    merged = text_mask_image([region], 64, 64)
    assert merged.shape == (64, 64) and merged[15, 15] == 255 and merged[30, 30] == 0
    sheet = text_glyph_sheet([region], 64, 64)
    assert sheet[15, 15] == 255 and sheet[10, 10] == 0  # 拼图放在 bbox 内


def test_scene_background_masks_debug_artifact():
    """涂黑输入与硬 mask 分文件保存，不拼三联、不落软遮罩。"""
    from algorithm.repair.background import scene_repair

    class StaticProvider:
        name = "static"
        degraded = False

        def repair_background(self, scene_hole_rgb, hole_mask=None):
            return scene_hole_rgb.copy()

        def repair_elements(self, atlas_rgb, atlas_mask):
            return atlas_rgb

    work = np.full((32, 32, 3), (200, 220, 240), np.uint8)
    hard = np.zeros((32, 32), bool)
    hard[8:24, 8:24] = True
    saved = {}
    scene_repair(work, hard, StaticProvider(),
                 debug=lambda name, img: saved.update({name: img}))
    assert saved["06b_background_input.png"].shape == (32, 32, 3)
    assert saved["06b_background_mask.png"].shape == (32, 32, 3)
    assert "06b_background_masks.png" not in saved


def test_scene_repair_uses_generated_image_as_result():
    """模型出图即为最终背景，不再按洞贴回原图。"""
    from algorithm.repair.background import scene_repair

    class FillProvider:
        name = "fill"
        degraded = False

        def repair_background(self, scene_hole_rgb, hole_mask=None):
            return np.full_like(scene_hole_rgb, (0, 255, 0))

        def repair_elements(self, atlas_rgb, atlas_mask):
            return atlas_rgb

    work = np.full((32, 32, 3), (200, 0, 0), np.uint8)
    hard = np.zeros((32, 32), bool)
    hard[8:24, 8:24] = True
    out = scene_repair(work, hard, FillProvider())
    assert tuple(int(v) for v in out[16, 16]) == (0, 255, 0)
    assert tuple(int(v) for v in out[0, 0]) == (0, 255, 0)


def _tiny_png() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", np.full((4, 4, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()
