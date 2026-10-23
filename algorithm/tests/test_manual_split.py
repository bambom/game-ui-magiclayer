"""手动拆分扩展：ROI 拆分、单层重拆 lineage、pending 状态登记。"""
import json
import os

import numpy as np
import pytest

from algorithm.ocr import engine as ocr_engine
from algorithm.pipeline import Pipeline
from algorithm.storage import JobStore, create_pending
from algorithm.tests import synth_ui


@pytest.fixture(scope="module")
def roi_run(tmp_path_factory):
    """bbox 裁掉四周一圈：导出坐标应平移回根图坐标系（原点=(10,8)）。"""
    paths = synth_ui.fixture_paths(str(tmp_path_factory.mktemp("roi")))
    out_dir = os.path.join(os.path.dirname(paths["image"]), "job-roi")
    orig = ocr_engine.extract_regions
    ocr_engine.extract_regions = lambda img: synth_ui.fake_ocr_regions()
    try:
        pipe = Pipeline(planner="mock", plan_path=paths["plan"],
                        provider="local", device="cpu", out_dir=out_dir,
                        roi_bbox=(10, 8, synth_ui.W - 8, synth_ui.H - 6),
                        meta={"mode": "roi", "bbox": [10, 8, synth_ui.W - 8,
                                                      synth_ui.H - 6]})
        summary = pipe.run(paths["image"])
    finally:
        ocr_engine.extract_regions = orig
    return out_dir, summary, pipe


@pytest.fixture(scope="module")
def resplit_run(tmp_path_factory):
    """单层重拆：输入是图层 PNG（本测试复用合成图），原点=(30,40) 平移导出。"""
    paths = synth_ui.fixture_paths(str(tmp_path_factory.mktemp("resplit")))
    out_dir = os.path.join(os.path.dirname(paths["image"]), "job-resplit")
    orig = ocr_engine.extract_regions
    ocr_engine.extract_regions = lambda img: synth_ui.fake_ocr_regions()
    try:
        pipe = Pipeline(planner="mock", plan_path=paths["plan"],
                        provider="local", device="cpu", out_dir=out_dir,
                        origin=(30, 40),
                        meta={"mode": "resplit", "parentJob": "parent-1",
                              "parentLayer": "card_1", "origin": [30, 40]})
        summary = pipe.run(paths["image"])
    finally:
        ocr_engine.extract_regions = orig
    return out_dir, summary, pipe


def _contract(out_dir):
    with open(os.path.join(out_dir, "layers.json"), encoding="utf-8") as f:
        return json.load(f)


def test_roi_status_done(roi_run):
    out_dir, _summary, _pipe = roi_run
    with open(os.path.join(out_dir, "status.json"), encoding="utf-8") as f:
        status = json.load(f)
    assert status["state"] == "done"


def test_roi_input_is_crop(roi_run):
    out_dir, _summary, _pipe = roi_run
    img = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                     .open(os.path.join(out_dir, "input.png")).convert("RGB"))
    assert img.shape[:2] == (synth_ui.H - 14, synth_ui.W - 18)


def test_roi_contract_origin_and_meta(roi_run):
    out_dir, _summary, _pipe = roi_run
    contract = _contract(out_dir)
    assert contract["origin"] == [10, 8]
    assert contract["job"]["mode"] == "roi"
    assert contract["job"]["bbox"] == [10, 8, synth_ui.W - 8, synth_ui.H - 6]


def test_roi_layers_shifted_to_root(roi_run):
    out_dir, _summary, _pipe = roi_run
    contract = _contract(out_dir)
    layers = [l for l in contract["layers"] if l["kind"] == "image"]
    assert layers, "ROI 拆分应至少产出一个图层"
    for layer in layers:
        assert layer["x"] >= 10 and layer["y"] >= 8, layer
    for layer in [l for l in contract["layers"] if l["kind"] == "text"]:
        assert layer["x"] >= 10 and layer["y"] >= 8, layer


def test_roi_background_keeps_local_size(roi_run):
    out_dir, _summary, _pipe = roi_run
    bg = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                    .open(os.path.join(out_dir, "background.png")).convert("RGB"))
    assert bg.shape[:2] == (synth_ui.H - 14, synth_ui.W - 18)


def test_resplit_lineage_recorded(resplit_run):
    out_dir, _summary, _pipe = resplit_run
    contract = _contract(out_dir)
    assert contract["origin"] == [30, 40]
    assert contract["job"]["parentJob"] == "parent-1"
    assert contract["job"]["parentLayer"] == "card_1"
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["mode"] == "resplit"
    assert meta["parentJob"] == "parent-1"
    layers = [l for l in contract["layers"] if l["kind"] == "image"]
    for layer in layers:
        assert layer["x"] >= 30 and layer["y"] >= 40, layer


def test_create_pending_then_running(tmp_path):
    jobs_root = str(tmp_path)
    job_dir = create_pending(jobs_root, "job-pending-1",
                             meta={"mode": "full"})
    with open(os.path.join(job_dir, "status.json"), encoding="utf-8") as f:
        assert json.load(f)["state"] == "pending"
    with open(os.path.join(job_dir, "meta.json"), encoding="utf-8") as f:
        assert json.load(f)["mode"] == "full"
    # trigger 拉起管线时 JobStore 构造覆写为 running
    store = JobStore(jobs_root=jobs_root, job_id="job-pending-1")
    with open(store.status_path, encoding="utf-8") as f:
        assert json.load(f)["state"] == "running"
    # 完成时 meta.json 的 lineage 随 summary 进入 job.json
    store.artifacts.append({"kind": "input", "path": "input.png"})
    store.write_summary({"job": {"mode": "full"}})
    with open(store.job_json, encoding="utf-8") as f:
        assert json.load(f)["job"]["mode"] == "full"
