"""设置面板后端:环境变量覆盖常数、修补模式强制覆盖、阶段进度写入。

覆盖对象是 /api/magiclayer/settings -> MAGICLAYER_* env -> launch 注入链的
Python 侧:config 常数在 import 时读 env,pipeline 在像素化前应用模式覆盖,
JobStore.set_progress 供面板展示当前拆的图层。
"""
import importlib
import json
import os

import pytest

from algorithm import config
from algorithm.common import Plan, Query
from algorithm.pipeline import Pipeline
from algorithm.storage import JobStore


def _reload_config(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def restore_config():
    yield
    importlib.reload(config)  # 丢掉单测注入的 env 覆盖,恢复文档默认


def test_config_constants_read_env_overrides(monkeypatch):
    module = _reload_config(monkeypatch,
                            MAGICLAYER_OCR_MIN_CONF="0.5",
                            MAGICLAYER_SAM_RETRY_SCORE="0.6",
                            MAGICLAYER_SAM_RETRY_MAX_COMPONENTS="7",
                            MAGICLAYER_SAM_RETRY_LARGEST_RATIO="0.9",
                            MAGICLAYER_ATLAS_MAX_SIDE="2048",
                            MAGICLAYER_BG_ANCHOR="1")
    assert module.OCR_MIN_CONF == 0.5
    assert module.SAM_RETRY_SCORE == 0.6
    assert module.SAM_RETRY_MAX_COMPONENTS == 7
    assert module.SAM_RETRY_LARGEST_RATIO == 0.9
    assert module.ATLAS_MAX_SIDE == 2048
    assert module.BG_ANCHOR_ENABLE == 1


def test_config_env_overrides_fall_back_on_garbage(monkeypatch):
    module = _reload_config(monkeypatch,
                            MAGICLAYER_OCR_MIN_CONF="not-a-number",
                            MAGICLAYER_SAM_RETRY_MAX_COMPONENTS="")
    assert module.OCR_MIN_CONF == 0.35
    assert module.SAM_RETRY_MAX_COMPONENTS == 3


def _plan() -> Plan:
    return Plan(
        scene_summary="s", background_mode="scene", background_reason="",
        raster_text_ids=[], text_corrections=[], queries=[
            Query(id="q1", name="card", kind="button", role="foreground",
                  element_repair_mode="image", parent_query_id=None, z_order=1),
            Query(id="q2", name="title", kind="logo", role="foreground",
                  element_repair_mode="none", parent_query_id="q1", z_order=2),
        ])


def test_repair_overrides_force_both_modes(monkeypatch):
    monkeypatch.setenv("MAGICLAYER_ELEMENT_REPAIR", "surface")
    monkeypatch.setenv("MAGICLAYER_BACKGROUND_REPAIR", "none")
    plan = _plan()
    Pipeline()._apply_repair_overrides(plan)
    assert plan.background_mode == "none"
    assert all(q.element_repair_mode == "surface" for q in plan.queries)


def test_repair_overrides_ignore_auto_and_invalid(monkeypatch):
    monkeypatch.setenv("MAGICLAYER_ELEMENT_REPAIR", "auto")
    monkeypatch.setenv("MAGICLAYER_BACKGROUND_REPAIR", "explode")
    plan = _plan()
    Pipeline()._apply_repair_overrides(plan)
    assert plan.background_mode == "scene"          # 规划结果原样保留
    assert [q.element_repair_mode for q in plan.queries] == ["image", "none"]


def test_set_progress_writes_status_and_stages_clear_it(tmp_path):
    store = JobStore(out_dir=str(tmp_path / "job"))

    def read():
        with open(store.status_path, encoding="utf-8") as f:
            return json.load(f)

    store.begin_stage("segment")
    assert read()["progress"] == {"stage": "segment"}

    store.set_progress("segment", "q1（card）", 1, 4)
    status = read()
    assert status["stage"] == "segment"
    assert status["progress"] == {"stage": "segment", "current": "q1（card）",
                                  "index": 1, "total": 4}

    store.set_progress("repair", "背景修补（scene）")
    assert read()["progress"] == {"stage": "repair", "current": "背景修补（scene）"}

    store.end_stage("repair")
    assert "progress" not in read()  # 阶段结束清掉,面板不再显示过期层名
    store.complete()
    assert "progress" not in read()
    assert read()["state"] == "done"
