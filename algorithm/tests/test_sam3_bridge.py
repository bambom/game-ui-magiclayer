"""SAM3 子进程桥：用 stub worker 单测协议与适配层（不依赖 CUDA/3.45G 权重）。"""
import base64
import io
import json
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

from algorithm.common import Hint
from algorithm.segment import sam
from algorithm.segment.sam import Sam3Segmenter, get_sam3_worker

# 纯协议 stub：ready -> set_image 记尺寸 -> predict 回 bbox 填充 mask。
# 跑在与测试相同的解释器下（主 .venv 有 PIL/cv2/numpy）。
_STUB = r"""
import base64, io, json, sys
import numpy as np, cv2
from PIL import Image
size = None
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req["op"] == "set_image":
        img = Image.open(io.BytesIO(base64.b64decode(req["png_b64"])))
        size = img.size
        resp = {"id": req.get("id"), "ok": True, "size": list(size)}
    elif req["op"] == "predict":
        x0, y0, x1, y1 = req.get("box") or [0, 0, *size]
        m = np.zeros((size[1], size[0]), np.uint8)
        m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 255
        ok, buf = cv2.imencode(".png", m)
        resp = {"id": req.get("id"), "ok": True,
                "masks": [base64.b64encode(buf.tobytes()).decode()],
                "scores": [0.9]}
    else:
        resp = {"id": req.get("id"), "ok": False, "error": "bad op"}
    print(json.dumps(resp), flush=True)
"""

_ERROR_STUB = r"""
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    print(json.dumps({"id": req.get("id"), "ok": False,
                      "error": "boom-" + req.get("op", "?")}), flush=True)
"""


@pytest.fixture(autouse=True)
def _reset_worker():
    sam._SAM3_WORKER = None
    yield
    worker = sam._SAM3_WORKER
    if worker is not None:
        worker.close()
    sam._SAM3_WORKER = None


def _install_stub(monkeypatch, tmp_path, script):
    stub = tmp_path / "stub_worker.py"
    stub.write_text(script, encoding="utf-8")
    monkeypatch.setenv("MAGICLAYER_SAM3_PYTHON", sys.executable)
    monkeypatch.setenv("MAGICLAYER_SAM3_WORKER", str(stub))
    # 权重存在性检查：拿 stub 文件本身顶替即可
    monkeypatch.setenv("MAGICLAYER_SAM3_WEIGHTS", str(stub))
    return stub


def _hint():
    return Hint(bbox=(10, 12, 58, 44),
                positive_points=[(30, 25), (40, 30)],
                negative_points=[(5, 5)],
                bbox_norm=[], positive_points_norm=[], negative_points_norm=[])


def test_stub_roundtrip_returns_bbox_mask(monkeypatch, tmp_path):
    _install_stub(monkeypatch, tmp_path, _STUB)
    image = np.full((60, 80, 3), (120, 140, 90), np.uint8)
    segmenter = Sam3Segmenter(image)
    mask, detail = segmenter.segment_hint(_hint(), 80, 60)
    assert mask is not None and mask.any()
    assert detail["candidates"] == 1 and detail["sam_score"] == pytest.approx(0.9)
    # stub 回的是 [x0,y0,x1-1,y1-1] 填充矩形：mask 外接框应贴近提示框
    ys, xs = np.where(mask)
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (10, 12, 57, 43)
    # 正点在 mask 内、负点不在
    assert mask[25, 30] and mask[30, 40] and not mask[5, 5]


def test_stub_predict_error_propagates(monkeypatch, tmp_path):
    _install_stub(monkeypatch, tmp_path, _ERROR_STUB)
    image = np.zeros((40, 50, 3), np.uint8)
    with pytest.raises(RuntimeError, match="boom-set_image"):
        Sam3Segmenter(image)  # set_image is also ok:false -> construction immediately raises


def test_missing_python_hint(monkeypatch, tmp_path):
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB, encoding="utf-8")
    monkeypatch.setenv("MAGICLAYER_SAM3_PYTHON", str(tmp_path / "nope.exe"))
    monkeypatch.setenv("MAGICLAYER_SAM3_WORKER", str(stub))
    monkeypatch.setenv("MAGICLAYER_SAM3_WEIGHTS", str(stub))
    with pytest.raises(FileNotFoundError, match=r"\.venv-sam3"):
        get_sam3_worker()


def test_create_impl_routes_sam3(monkeypatch, tmp_path):
    _install_stub(monkeypatch, tmp_path, _STUB)
    monkeypatch.setenv("MAGICLAYER_SAM_MODEL", "sam3")
    assert sam.sam_model() == "sam3"
    image = np.zeros((40, 50, 3), np.uint8)
    impl = sam._create_impl(image, "cpu")
    assert isinstance(impl, Sam3Segmenter)
    impl.predictor._worker.close()
