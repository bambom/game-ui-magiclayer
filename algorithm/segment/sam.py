"""SAM1 ViT-B 提示分割（文档 §2.2）：整图编码一次，逐实例三候选。

后端可配（偏离文档，需显式选择）：MAGICLAYER_SAM_MODEL 默认 `vit_b`（文档）；
`sam2.1-hiera-tiny|small|base_plus|large` 走 SAM2ImagePredictor（官方包 SAM-2 +
本地权重 algorithm/models/*.pt，2026-09-03 环境已装好，零网络依赖）；
`sam3` 走 .venv-sam3 独立环境子进程桥（torch 2.10+cu128，RTX 4090；
主管线 .venv 的 CPU torch 不升级——红线）。
三后端共用三候选评分与 mask 清理。
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import subprocess
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .. import config, runlog
from ..common import BBox, Hint
from .cleanup import clean_mask
from .scoring import pick_best

_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
_PREDICTOR = None
_SAM2_PREDICTORS: dict = {}
_SAM3_WORKER = None
_LOAD_LOCK = threading.Lock()

# SAM2.1 模型名 -> (本地权重文件, hydra 配置)。权重一次性下到 algorithm/models/
# （运行时零网络）：经 VPN 代理 curl -L
# https://huggingface.co/facebook/sam2.1-hiera-<tier>/resolve/main/<file>.pt
_SAM2_MODELS = {
    "sam2.1-hiera-tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "sam2.1-hiera-small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "sam2.1-hiera-base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "sam2.1-hiera-large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def checkpoint_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models", _CHECKPOINT_NAME)


def _models_dir() -> str:
    return os.path.dirname(checkpoint_path())


def sam_model() -> str:
    """MAGICLAYER_SAM_MODEL：vit_b（默认，文档）、SAM2.1 模型名或 sam3。"""
    return os.environ.get("MAGICLAYER_SAM_MODEL", "vit_b").strip()


def warmup(device: str = "cpu") -> None:
    """按当前配置预热所选后端（管线规划阶段后台调用）。"""
    model = sam_model()
    if model == "vit_b":
        get_predictor(device)
        return
    if model == "sam3":
        get_sam3_worker()
        return
    get_sam2_predictor(model, device)


def get_predictor(device: str = "cpu"):
    """懒加载 SAM ViT-B；权重在 algorithm/models/ 下。

    加锁：管线在规划阶段起后台线程预热权重，与主线程的首次获取可能并发。
    """
    global _PREDICTOR
    if _PREDICTOR is None:
        with _LOAD_LOCK:
            if _PREDICTOR is None:
                from segment_anything import SamPredictor, sam_model_registry

                ckpt = checkpoint_path()
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(
                        f"缺少 SAM 权重 {ckpt}；请下载 "
                        f"https://dl.fbaipublicfiles.com/segment_anything/{_CHECKPOINT_NAME}")
                sam = sam_model_registry["vit_b"](checkpoint=ckpt)
                sam.to(device=device)
                _PREDICTOR = SamPredictor(sam)
    return _PREDICTOR


def get_sam2_predictor(model: str, device: str = "cpu"):
    """懒加载并缓存 SAM2.1 predictor（本地权重 build_sam2，零网络依赖）。

    逐图工作只在 set_image——predictor 本身按 (模型,设备) 缓存复用，与 vit_b 的
    _PREDICTOR 同一模式（同一进程内一次只跑一个作业，set_image 的共享状态可接受）。
    """
    key = (model, device)
    pred = _SAM2_PREDICTORS.get(key)
    if pred is None:
        with _LOAD_LOCK:
            pred = _SAM2_PREDICTORS.get(key)
            if pred is None:
                try:
                    from sam2.build_sam import build_sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                except ImportError as e:
                    raise ImportError(_sam2_missing_hint()) from e
                spec = _SAM2_MODELS.get(model)
                if spec is None:
                    raise ValueError(f"未知 SAM2 模型 {model}；"
                                     f"可选：{sorted(_SAM2_MODELS)}")
                ckpt_name, cfg_name = spec
                ckpt = os.path.join(_models_dir(), ckpt_name)
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(
                        f"缺少 SAM2.1 权重 {ckpt}；请下载 "
                        f"https://huggingface.co/facebook/sam2.1-hiera-"
                        f"{model.split('-')[-1]}/resolve/main/{ckpt_name} "
                        f"到该路径（经 VPN 代理 curl -L）")
                sam2_model = build_sam2(cfg_name, ckpt_path=ckpt, device=device)
                pred = SAM2ImagePredictor(sam2_model)
                _SAM2_PREDICTORS[key] = pred
    return pred


def _sam2_missing_hint() -> str:
    return ("MAGICLAYER_SAM_MODEL 选择了 SAM2 但运行环境缺少依赖/权重："
            "请 `pip install git+https://github.com/facebookresearch/sam2.git`"
            "（官方包 SAM-2，2026-09-03 已装入 .venv）并确认本地权重 "
            "algorithm/models/ 存在（{' / '.join(f for f, _ in _SAM2_MODELS.values())}）；"
            "未配置时保持默认 vit_b 即为文档算法。")


# ---------------- SAM3 子进程桥（.venv-sam3，2026-09-04） ----------------

def _sam3_python() -> str:
    """SAM3 独立环境解释器：env 可覆盖，默认 magiclayer/.venv-sam3。"""
    override = os.environ.get("MAGICLAYER_SAM3_PYTHON", "").strip()
    if override:
        return override
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, ".venv-sam3", "Scripts", "python.exe")


def _sam3_weights() -> str:
    override = os.environ.get("MAGICLAYER_SAM3_WEIGHTS", "").strip()
    if override:
        return override
    return os.path.join(_models_dir(), "sam3.pt")


def _sam3_worker_script() -> str:
    override = os.environ.get("MAGICLAYER_SAM3_WORKER", "").strip()
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_worker.py")


def _sam3_missing_hint(python: str, weights: str) -> str:
    return ("MAGICLAYER_SAM_MODEL 选择了 sam3 但环境不完整：需要独立环境 "
            f"{python}（Python 3.12 + torch cu128 + 官方 third_party/sam3，"
            f"已有 .venv-sam3）与权重 {weights}（algorithm/models/sam3.pt）；"
            "SAM3 推理需 CUDA。未配置时保持默认 vit_b 即为文档算法。")


class _Sam3Worker:
    """SAM3 行式 JSON 子进程：set_image 一次、多次 predict，超时可配。

    stderr 默认丢弃（sam3 内部日志很密，管道无人读会撑爆缓冲）；
    MAGICLAYER_SAM3_DEBUG=1 时继承到控制台便于排查。
    """

    def __init__(self):
        python = _sam3_python()
        weights = _sam3_weights()
        if not os.path.isfile(python) or not os.path.isfile(weights):
            raise FileNotFoundError(_sam3_missing_hint(python, weights))
        stderr = None if os.environ.get("MAGICLAYER_SAM3_DEBUG") else subprocess.DEVNULL
        self._proc = subprocess.Popen(
            [python, _sam3_worker_script(), "--weights", weights],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, encoding="utf-8", bufsize=1)
        first = self._readline(config.SAM3_TIMEOUT)
        ready = json.loads(first) if first else {}
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"SAM3 worker 未就绪：{first or '无输出（build 失败？开 MAGICLAYER_SAM3_DEBUG=1 看 stderr）'}"
                               + (f" / {ready.get('fatal', '')}" if ready.get("fatal") else ""))

    def _readline(self, timeout: float) -> Optional[str]:
        import threading as _t

        received: List[str] = []

        def _read():
            line = self._proc.stdout.readline()
            if line:
                received.append(line)

        reader = _t.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout)
        return received[0] if received else None

    def _call(self, request: dict) -> dict:
        if self._proc.poll() is not None:
            raise RuntimeError("SAM3 worker 已退出（build/推理崩溃？开 MAGICLAYER_SAM3_DEBUG=1 复现）")
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        line = self._readline(config.SAM3_TIMEOUT)
        if not line:
            raise RuntimeError(f"SAM3 worker {request.get('op')} 超时（>{config.SAM3_TIMEOUT}s）")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"SAM3 worker {request.get('op')} 失败：{response.get('error')}")
        return response

    def set_image(self, image_rgb: np.ndarray) -> None:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("set_image PNG 编码失败")
        self._call({"id": 1, "op": "set_image",
                    "png_b64": base64.b64encode(buf.tobytes()).decode("ascii")})

    def predict(self, coords: Optional[np.ndarray], labels: Optional[np.ndarray],
                box: np.ndarray):
        response = self._call({
            "id": 2, "op": "predict",
            "points": (coords.astype(int).tolist() if coords is not None else []),
            "labels": (labels.astype(int).tolist() if labels is not None else []),
            "box": ([int(v) for v in box] if box is not None else [])})
        masks = []
        for encoded in response["masks"]:
            raw = np.frombuffer(base64.b64decode(encoded), np.uint8)
            decoded = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            masks.append(decoded > 127)
        return masks, np.asarray(response["scores"], np.float32)

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
            except OSError:
                pass
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()


def get_sam3_worker() -> _Sam3Worker:
    """懒加载并缓存 SAM3 worker（模型 build ~7s + 3.45G 权重，作业内复用）。

    与 predictor 缓存同一模式：同一进程一次只跑一个作业，set_image 的
    共享状态可接受；进程退出时 atexit 兜底终止子进程。
    """
    global _SAM3_WORKER
    if _SAM3_WORKER is None:
        with _LOAD_LOCK:
            if _SAM3_WORKER is None:
                _SAM3_WORKER = _Sam3Worker()
                atexit.register(_SAM3_WORKER.close)
    return _SAM3_WORKER


def _grid_points(box: BBox) -> List[Tuple[int, int]]:
    """bbox 中心 + 四分点（碎 mask 重试用）。"""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    qx, qy = (x1 - x0) // 4, (y1 - y0) // 4
    return [(cx, cy), (x0 + qx, y0 + qy), (x1 - qx, y0 + qy),
            (x0 + qx, y1 - qy), (x1 - qx, y1 - qy)]


def _fragmented(mask: np.ndarray) -> tuple:
    """(是否碎裂, 连通域数, 最大组件占比)：8 邻域连通域统计。"""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    count = n - 1
    if count <= 1:
        return False, count, 1.0
    sizes = np.bincount(labels.ravel())[1:]
    largest = float(sizes.max() / max(sizes.sum(), 1))
    frag = bool(count > config.SAM_RETRY_MAX_COMPONENTS
                and largest < config.SAM_RETRY_LARGEST_RATIO)
    return frag, count, largest


class Sam1Segmenter:
    """SAM1 ViT-B 后端（文档 §2.2）。"""

    def __init__(self, image_rgb: np.ndarray, device: str = "cpu"):
        self.predictor = get_predictor(device)
        self.predictor.set_image(image_rgb)

    def segment_hint(self, hint: Hint, width: int, height: int):
        return _predict(self.predictor, hint)


class Sam2Segmenter:
    """SAM2.1 后端（可配升级，非文档算法）。"""

    def __init__(self, image_rgb: np.ndarray, device: str = "cpu",
                 model: str = "sam2.1-hiera-small"):
        self.predictor = get_sam2_predictor(model, device)
        self.predictor.set_image(image_rgb)

    def segment_hint(self, hint: Hint, width: int, height: int):
        return _predict(self.predictor, hint)


class _Sam3PredictorAdapter:
    """把 worker 的两段式调用适配成 SAM1 predictor 的 predict 接口（复用 _predict）。"""

    def __init__(self, worker: _Sam3Worker):
        self._worker = worker

    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=True):
        masks, scores = self._worker.predict(point_coords, point_labels, box)
        return masks, scores, None


class Sam3Segmenter:
    """SAM3 后端（.venv-sam3 子进程桥，非文档算法；CUDA/bfloat16）。"""

    def __init__(self, image_rgb: np.ndarray, device: str = "cpu"):
        worker = get_sam3_worker()
        worker.set_image(image_rgb)
        self.predictor = _Sam3PredictorAdapter(worker)

    def segment_hint(self, hint: Hint, width: int, height: int):
        return _predict(self.predictor, hint)


def _predict(predictor, hint: Hint):
    box = hint.bbox
    box_xyxy = np.array([box[0], box[1], box[2] - 1, box[3] - 1], np.float32)
    point_coords, point_labels = [], []
    for p in hint.positive_points:
        point_coords.append(p)
        point_labels.append(1)
    for p in hint.negative_points:
        point_coords.append(p)
        point_labels.append(0)
    coords = np.array(point_coords, np.float32) if point_coords else None
    labels = np.array(point_labels, np.int32) if point_labels else None

    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        box=box_xyxy,
        multimask_output=True,
    )
    best_mask, detail = pick_best(
        [m.astype(bool) for m in masks], scores.tolist(), box,
        hint.positive_points, hint.negative_points)
    cleaned = clean_mask(best_mask, hint.positive_points)
    detail["candidates"] = len(masks)
    detail["cleaned_removed_ratio"] = float(
        1.0 - cleaned.sum() / max(best_mask.sum(), 1))
    return cleaned, detail


def _needs_retry(mask: np.ndarray, detail: dict) -> tuple:
    """(是否重试, 原因)：SAM 分过低或连通域碎裂。"""
    frag, count, largest = _fragmented(mask)
    if detail.get("sam_score", 1.0) < config.SAM_RETRY_SCORE:
        return True, f"sam_score={detail.get('sam_score', 0):.3f} 低于 {config.SAM_RETRY_SCORE}"
    if frag:
        return True, f"连通域 {count} 个且最大占比 {largest:.2f} 碎裂"
    return False, ""


def _create_impl(image_rgb: np.ndarray, device: str):
    model = sam_model()
    if model == "vit_b":
        return Sam1Segmenter(image_rgb, device)
    if model == "sam3":
        return Sam3Segmenter(image_rgb)
    return Sam2Segmenter(image_rgb, device, model)


class Segmenter:
    """后端无关门面：整图编码一次，逐提示三候选评分 + 碎 mask 网格补点重试。"""

    def __init__(self, image_rgb: np.ndarray, device: str = "cpu"):
        self._impl = _create_impl(image_rgb, device)

    def segment_hint(self, hint: Hint, width: int, height: int) -> Tuple[Optional[np.ndarray], dict]:
        mask, detail = self._impl.segment_hint(hint, width, height)
        if mask is None or not mask.any():
            return mask, detail
        retry, why = _needs_retry(mask, detail)
        if not retry:
            return mask, detail
        runlog.event("LAYER", f"分割质量重试：{why}，网格补点重跑")
        retry_hint = Hint(bbox=hint.bbox,
                          positive_points=_grid_points(hint.bbox),
                          negative_points=hint.negative_points,
                          bbox_norm=hint.bbox_norm,
                          positive_points_norm=[],
                          negative_points_norm=hint.negative_points_norm)
        mask2, detail2 = self._impl.segment_hint(retry_hint, width, height)
        if mask2 is not None and mask2.any() and detail2["total"] >= detail["total"]:
            mask, detail = mask2, detail2
        detail["retried"] = True
        return mask, detail
