"""离线全流程冒烟：合成 UI + mock plan + local provider + 真实 SAM。

跑通 OCR 替身 -> 去字 -> 规划 -> 美术字恢复 -> SAM 分割 -> 遮挡/背景修补 ->
文字层 -> 组合输出，并校验作业目录结构（README §4）与关键行为。
"""
import json
import os

import numpy as np
import pytest

from algorithm.ocr import engine as ocr_engine
from algorithm.pipeline import Pipeline
from algorithm.tests import synth_ui


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    paths = synth_ui.fixture_paths(str(tmp_path_factory.mktemp("smoke")))
    out_dir = os.path.join(os.path.dirname(paths["image"]), "job")
    orig = ocr_engine.extract_regions
    ocr_engine.extract_regions = lambda img: synth_ui.fake_ocr_regions()
    try:
        pipe = Pipeline(planner="mock", plan_path=paths["plan"],
                        provider="local", device="cpu", out_dir=out_dir)
        summary = pipe.run(paths["image"])
    finally:
        ocr_engine.extract_regions = orig
    return paths, out_dir, summary, pipe


def test_job_layout_files(smoke_run):
    paths, out_dir, summary, pipe = smoke_run
    for name in ["status.json", "job.json", "layers.json", "input.png",
                 "background.png", "preview.png"]:
        assert os.path.exists(os.path.join(out_dir, name)), name
    for name in ["01_ocr.png", "01b_text_mask.png", "01c_text_glyphs.png",
                 "02_erased.png", "03_work.png", "03a_work_before_raster.png",
                 "04_masks.png", "05_atlas_input.png", "05_atlas_mask.png",
                 "05_atlas.png", "06_background.png",
                 "07_preview.png", "steps.json"]:
        assert os.path.exists(os.path.join(out_dir, "debug", name)), name


def test_steps_json_process_document(smoke_run):
    """debug/steps.json：分步过程文档（面板垂直回看每步输入/输出/模型输入）。"""
    paths, out_dir, summary, pipe = smoke_run
    with open(os.path.join(out_dir, "debug", "steps.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["job_state"] == "done" and doc["mode"] == "full"
    keys = [s["key"] for s in doc["steps"]]
    assert keys == ["input", "ocr", "erase", "planner", "segment",
                    "element_layers", "background", "compose"]
    assert all(s["state"] == "done" for s in doc["steps"])
    for step in doc["steps"]:
        assert "title" in step and "desc" in step and "images" in step
        assert "calls" in step and "notes" in step
        for image in step["images"]:
            # URL 指向本作业的实际产物（存在性在生成端已过滤）
            assert image["url"].startswith("/api/magiclayer/jobs/")
            assert image["role"] in ("输入", "输出", "输入+输出")
    by_key = {s["key"]: s for s in doc["steps"]}
    assert by_key["input"]["images"][0]["name"] == "input.png"
    assert any(i["name"] == "05_atlas.png" for i in by_key["element_layers"]["images"])
    assert any(i["name"] == "06_background.png" for i in by_key["background"]["images"])
    # steps.json 在 job.json artifacts 里只登记一次（幂等重写不重复登记）
    with open(os.path.join(out_dir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    steps_entries = [a for a in job["artifacts"] if a["path"] == "debug/steps.json"]
    assert len(steps_entries) == 1


def test_status_machine_done(smoke_run):
    paths, out_dir, summary, pipe = smoke_run
    with open(os.path.join(out_dir, "status.json"), encoding="utf-8") as f:
        status = json.load(f)
    assert status["state"] == "done"
    assert status["error"] is None
    for stage in ["ocr", "erase", "planner", "segment", "repair", "compose"]:
        assert stage in status["stages"], stage


def test_job_artifacts_registry(smoke_run):
    paths, out_dir, summary, pipe = smoke_run
    with open(os.path.join(out_dir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    kinds = [a["kind"] for a in job["artifacts"]]
    for kind in ["input", "plan", "contract", "background", "preview", "layer"]:
        assert kind in kinds, kind
    # 每个 layer artifact 都能对上实际文件
    for a in job["artifacts"]:
        assert os.path.exists(os.path.join(out_dir, a["path"])), a["path"]


def test_layer_counts(smoke_run):
    paths, out_dir, summary, pipe = smoke_run
    with open(os.path.join(out_dir, "layers.json"), encoding="utf-8") as f:
        contract = json.load(f)
    image_layers = [l for l in contract["layers"] if l["kind"] == "image"]
    text_layers = [l for l in contract["layers"] if l["kind"] == "text"]
    ids = {l["id"] for l in image_layers}
    # 5 个规划实例 + 管线为美术字补发的 raster 查询
    assert {"main_panel", "card_1", "card_2", "badge", "play_button",
            "raster_text_001"} <= ids
    # 普通文字与漏识文字成为文字层；美术字不重建为文字
    tids = {l["id"] for l in text_layers}
    assert "text_002" in tids and "text_c01" in tids
    assert "text_001" not in tids
    for l in image_layers:
        assert os.path.exists(os.path.join(out_dir, l["png"])), l["png"]


def test_parent_chain(smoke_run):
    paths, out_dir, summary, pipe = smoke_run
    with open(os.path.join(out_dir, "layers.json"), encoding="utf-8") as f:
        contract = json.load(f)
    by_id = {l["id"]: l for l in contract["layers"] if l["kind"] == "image"}
    assert by_id["card_1"]["parent"] == "main_panel"
    assert by_id["badge"]["parent"] == "card_1"
    assert by_id["main_panel"]["parent"] is None


def test_background_is_scene_mode(smoke_run):
    paths, out_dir, summary, pipe = smoke_run
    assert summary["background_mode"] == "scene"
    bg = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                    .open(os.path.join(out_dir, "background.png")).convert("RGB"))
    assert bg.shape[:2] == (synth_ui.H, synth_ui.W)


def test_text_style_estimated(smoke_run):
    paths, out_dir, summary, pipe = smoke_run
    with open(os.path.join(out_dir, "layers.json"), encoding="utf-8") as f:
        contract = json.load(f)
    tl = next(l for l in contract["layers"] if l["id"] == "text_002")
    st = tl["style"]
    assert st["font"] == "Arial"
    assert st["weight"] in (600, 700)
    assert 0 <= st["strokeWidth"] <= 2
    assert st["color"].startswith("#")


def test_raster_restored_before_segmentation(smoke_run):
    # 美术字恢复后进行分割：raster 图层应包含多色标题像素
    paths, out_dir, summary, pipe = smoke_run
    with open(os.path.join(out_dir, "layers.json"), encoding="utf-8") as f:
        contract = json.load(f)
    raster = next(l for l in contract["layers"] if l["id"] == "raster_text_001")
    from PIL import Image

    png = np.asarray(Image.open(os.path.join(out_dir, raster["png"])).convert("RGBA"))
    rgb = png[:, :, :3][png[:, :, 3] > 200]
    if len(rgb):
        std = rgb.astype(np.float32).std(axis=0).mean()
        assert std > 10  # 多色美术字而非单一颜色
