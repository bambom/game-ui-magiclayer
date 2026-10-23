"""mediatransformation 语义引擎：确定性变换、笔刷遮罩栅格化、作业生命周期。

生图/SAM 通道由 Java 侧服务测试与真实上游覆盖，这里聚焦纯本地可验证的行为：
ADJUSTMENT_V3 调色数学、UPSCALE 几何、模糊遮罩、笔刷坐标 wire 解码、
mask 作业的代理产物与失败词表、status.json 状态机。
"""
import base64
import json
import struct

import cv2
import numpy as np
import pytest

from algorithm import transform
from algorithm.transform import Frame, TransformFailure, rasterize_strokes


def _f32(values):
    return base64.b64encode(np.asarray(values, np.float32).tobytes()).decode()


def _frame(width=64, height=48, seed=7):
    rng = np.random.default_rng(seed)
    return Frame(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


# ---------------------------------------------------------------- 调整数学

def test_adjustment_brightness_and_contrast_direction():
    frame = Frame(np.full((8, 8, 3), 128, np.uint8))
    brighter = transform.op_adjustment(frame, {"globalParams": {"brightness": 1.0}})
    assert int(brighter.rgb[0, 0, 0]) > 128
    darker = transform.op_adjustment(frame, {"globalParams": {"brightness": -1.0}})
    assert int(darker.rgb[0, 0, 0]) < 128
    contrast = transform.op_adjustment(Frame(np.zeros((8, 8, 3), np.uint8)),
                                       {"globalParams": {"contrast": 1.0}})
    assert int(contrast.rgb[0, 0, 0]) == 0  # 黑端不被负向对比拉开


def test_adjustment_saturation_gray_pull_and_temperature_channel():
    frame = _frame()
    saturated = transform.op_adjustment(frame, {"globalParams": {"saturation": -1.0}})
    channels = saturated.rgb[0, 0]
    assert int(abs(int(channels[0]) - int(channels[1]))) <= 1  # 完全去饱和 -> 灰
    warm = transform.op_adjustment(frame, {"globalParams": {"temperature": 1.0}})
    diff = warm.rgb[:, :, 0].astype(int) - frame.rgb[:, :, 0].astype(int)
    assert diff.max() >= 0 and warm.rgb[0, 0, 0] >= frame.rgb[0, 0, 0]


def test_adjustment_alpha_carried_through():
    frame = Frame(np.full((8, 8, 3), 100, np.uint8), np.full((8, 8), 200, np.uint8))
    out = transform.op_adjustment(frame, {"globalParams": {"brightness": 0.5}})
    assert out.alpha is not None and out.alpha[0, 0] == 200


def test_adjustment_out_of_range_params_clamped():
    frame = _frame()
    out = transform.op_adjustment(frame, {"globalParams": {"brightness": 99, "vignette": -5}})
    assert out.rgb.dtype == np.uint8 and out.rgb.min() >= 0


# ---------------------------------------------------------------- 几何类

def test_upscale_doubles_and_keeps_alpha():
    frame = Frame(np.zeros((10, 20, 3), np.uint8), np.full((10, 20), 255, np.uint8))
    out = transform.op_upscale(frame, {}, 1600)
    assert (out.height, out.width) == (20, 40)
    assert out.alpha is not None and out.alpha.shape == (20, 40)


def test_gaussian_blur_blends_only_stroked_region():
    frame = Frame(np.zeros((40, 40, 3), np.uint8))
    frame.rgb[10:30, 10:30] = 255  # 中心方块
    strokes = [{"A?": "A", "A": {"A": _f32([0.5]), "B": _f32([0.5])}, "B": 0.1, "C": "ffffff"}]
    out = transform.op_gaussian_blur(
        frame, {"intensityMask": {"maskType": "MASK_SELECTION", "maskId": "m1"}},
        {"m1": {"mask": {"A": strokes}}})
    center = out.rgb[20, 20].astype(int)
    corner = out.rgb[2, 2]
    assert center.mean() < 255 and corner[0] == 0  # 方块边缘被模糊、角落未动


def test_rasterize_strokes_polyline_and_radius():
    strokes = [{"A": {"A": _f32([0.1, 0.3, 0.5]), "B": _f32([0.5, 0.5, 0.5])}, "B": 0.2}]
    mask = rasterize_strokes(strokes, 100, 100)
    assert mask[50, 10] > 0 and mask[50, 50] > 0
    assert mask[5, 5] == 0


# ---------------------------------------------------------------- 失败词表

def test_background_removal_without_subject_raises_official_error(monkeypatch):
    monkeypatch.setattr(transform, "_subject_mask", lambda frame, points=None: None)
    with pytest.raises(TransformFailure) as failure:
        transform.op_background_removal(_frame())
    assert failure.value.step == "BACKGROUND_REMOVAL"
    assert failure.value.error_type == "NO_FOREGROUND_FOUND"


def test_mask_jobs_face_fails_and_depth_proxies(monkeypatch):
    monkeypatch.setattr(transform, "_subject_mask", lambda frame, points=None: None)
    with pytest.raises(TransformFailure) as failure:
        transform.produce_mask("FACE_SEGMENTATION_MASK", _frame())
    assert failure.value.error_type == "NO_FACES_FOUND"
    with pytest.raises(TransformFailure) as seg_failure:
        transform.produce_mask("SEGMENTATION_MASK", _frame())
    assert seg_failure.value.error_type == "NO_SEGMENTS_FOUND"
    depth = transform.produce_mask("DEPTHMAP_MASK", _frame())
    assert depth.shape == (48, 64) and depth[24, 32] >= 250


def test_unknown_transformation_type_rejected():
    with pytest.raises(TransformFailure) as failure:
        transform.run_stack(_frame(), [{"type": "QUANTUM_FLIP"}],
                            transform.JobContext(max_edge=1600, masks={}))
    assert failure.value.error_type == "BAD_TRANSFORMATION"


def test_prompt_editing_requires_text():
    with pytest.raises(TransformFailure) as failure:
        transform.op_prompt_editing(_frame(), {"prompt": {"text": " "}})
    assert failure.value.error_type == "BAD_TRANSFORMATION"


# ---------------------------------------------------------------- 作业生命周期

def _job(tmp_path, request_body, source):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "request.json").write_text(json.dumps(request_body), encoding="utf-8")
    ok, buf = cv2.imencode(".png", cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
    (job_dir / "source.png").write_bytes(buf.tobytes())
    return str(job_dir)


def test_run_job_writes_output_and_status(tmp_path):
    body = {"type": "TRANSFORMATION_REQUEST", "maxLongestEdge": 1600,
            "transformations": [{"type": "ADJUSTMENT_V3", "globalParams": {"brightness": 0.2}}]}
    transform.run_job(_job(tmp_path, body, np.full((16, 16, 3), 60, np.uint8)))
    status = json.loads((tmp_path / "job" / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done" and status["kind"] == "image"
    assert (tmp_path / "job" / "output.png").exists()
    assert (tmp_path / "job" / "run.log").exists() is False  # run.log 由 main() 安装


def test_run_job_mask_kind_records_mask_type(tmp_path, monkeypatch):
    monkeypatch.setattr(transform, "_subject_mask",
                        lambda frame, points=None: np.ones((10, 10), bool))
    body = {"type": "SEGMENTATION_MASK", "maxLongestEdge": 1600, "transformations": []}
    transform.run_job(_job(tmp_path, body, np.zeros((10, 10, 3), np.uint8)))
    status = json.loads((tmp_path / "job" / "status.json").read_text(encoding="utf-8"))
    assert status == {**status, "state": "done", "kind": "mask",
                      "mask_type": "SEGMENTATION_RESPONSE"}
    assert (tmp_path / "job" / "mask.png").exists()


def test_main_failure_is_zero_exit_with_failed_status(tmp_path):
    body = {"type": "TRANSFORMATION_REQUEST", "maxLongestEdge": 1600,
            "transformations": [{"type": "PROMPT_EDITING", "prompt": {"text": "hi"}}]}
    job = _job(tmp_path, body, np.zeros((8, 8, 3), np.uint8))
    import algorithm.providers.gemini_image as gemini_image

    original = gemini_image.GeminiImageProvider._generate_repair
    gemini_image.GeminiImageProvider._generate_repair = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no upstream"))
    try:
        code = transform.main(["--job-dir", job])
    finally:
        gemini_image.GeminiImageProvider._generate_repair = original
    assert code == 0
    status = json.loads((tmp_path / "job" / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["failed_step"] == "PROMPT_EDITING"
    assert status["error_type"] == "PROVIDER_FAILED"
    assert (tmp_path / "job" / "run.log").exists()


def test_inpainting_falls_back_to_telea_without_upstream(tmp_path):
    source = np.zeros((20, 20, 3), np.uint8)
    source[5:15, 5:15] = 200
    body = {"type": "TRANSFORMATION_REQUEST", "maxLongestEdge": 1600,
            "transformations": [{"type": "INPAINTING",
                                 "mask": {"maskType": "MASK_SELECTION", "maskId": "u1"}}]}
    job_dir = _job(tmp_path, body, source)
    (tmp_path / "job" / "mask-u1.json").write_text(json.dumps(
        {"mask": {"A": [{"A": {"A": _f32([0.5]), "B": _f32([0.5])}, "B": 0.3}]}}),
        encoding="utf-8")
    import algorithm.providers.gemini_image as gemini_image

    original = gemini_image.GeminiImageProvider._generate_repair
    gemini_image.GeminiImageProvider._generate_repair = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no upstream"))
    try:
        transform.run_job(job_dir)
    finally:
        gemini_image.GeminiImageProvider._generate_repair = original
    assert (tmp_path / "job" / "output.png").exists()
