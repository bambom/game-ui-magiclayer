"""公共数据结构与几何/图像工具。"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1（右/下开区间）


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_of_mask(mask: np.ndarray) -> Optional[BBox]:
    """mask 外接框；空 mask 返回 None。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def quad_to_mask(quad: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """OCR 四边形（4x2, float）转局部 allowed mask。"""
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(quad).astype(np.int32)], 1)
    return m.astype(bool)


def iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / float(area_a + area_b - inter)


def norm_bbox_to_pixels(bbox_norm: Sequence[float], width: int, height: int) -> BBox:
    x0, y0, x1, y1 = bbox_norm
    px0 = int(round(clamp(float(x0), 0.0, 1.0) * width))
    py0 = int(round(clamp(float(y0), 0.0, 1.0) * height))
    px1 = int(round(clamp(float(x1), 0.0, 1.0) * width))
    py1 = int(round(clamp(float(y1), 0.0, 1.0) * height))
    px1 = max(px1, px0 + 1)
    py1 = max(py1, py0 + 1)
    return min(px0, width - 1), min(py0, height - 1), min(px1, width), min(py1, height)


def norm_points_to_pixels(points: Sequence[Sequence[float]], width: int, height: int) -> List[Point]:
    out = []
    for p in points or []:
        x = int(round(clamp(float(p[0]), 0.0, 1.0) * (width - 1)))
        y = int(round(clamp(float(p[1]), 0.0, 1.0) * (height - 1)))
        out.append((x, y))
    return out


def encode_png(image_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def encode_rgba_mask_png(mask: np.ndarray, opaque_editable: bool = False) -> bytes:
    """二值遮罩 -> /v1/images/edits 用的 RGBA PNG。

    默认按 OpenAI edits 规范：待修补区域（白/True）alpha=0 透明。实测部分网关
    （2026-09-02 run 9bf549dd，gpt-image-2 @ localhost:18080）按相反约定执行——
    重绘不透明区、冻结透明区，洞区原样返回预填色。``opaque_editable`` 供这类
    条目翻转极性：待修补区以白+不透明标记（兼容按亮度读遮罩的网关）。
    """
    hole = mask.astype(bool) if mask.dtype == bool else mask > 127
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)
    if opaque_editable:
        rgba[..., :3] = np.where(hole[:, :, None], 255, 0).astype(np.uint8)
        rgba[..., 3] = np.where(hole, 255, 0).astype(np.uint8)
    else:
        rgba[..., 3] = np.where(hole, 0, 255)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """任意 PNG/JPG 字节 -> RGB。"""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError("image decode failed")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


@dataclass
class OcrRegion:
    """一条 OCR 记录（文档 §1.1）。id 形如 text_001。"""
    id: str
    text: str
    conf: float
    quad: np.ndarray            # 4x2 float，图像坐标
    bbox: BBox                  # quad 外接框
    is_raster: bool = False     # LLM 判定的美术字
    is_correction: bool = False # 来自 text_corrections 的漏识文字
    removal_mask: Optional[np.ndarray] = None  # 全图坐标的硬 removal mask
    mask_mode: str = ""         # estimated_glyphs | coarse
    glyph_mask: Optional[np.ndarray] = None  # 裁片坐标的字形二值 mask（样式估计用）
    local_bg: Optional[Tuple[float, float, float]] = None  # 局部底色 RGB（样式估计用）
    style: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]


@dataclass
class Hint:
    """一个几何实例提示（规划 query.geometry_hints 的像素化形态）。"""
    bbox: BBox
    positive_points: List[Point]
    negative_points: List[Point]
    bbox_norm: List[float]
    positive_points_norm: List[List[float]]
    negative_points_norm: List[List[float]]


@dataclass
class Query:
    """LLM 规划出的素材查询（文档 §2.1.1）。"""
    id: str
    name: str
    kind: str
    role: str
    element_repair_mode: str   # image | surface | none
    parent_query_id: Optional[str]
    z_order: int
    hints: List[Hint] = field(default_factory=list)


@dataclass
class Instance:
    """query x hint 展开后的一个实例（SAM 与图层构建的单位）。"""
    instance_id: str            # 单提示用 query.id，多提示加序号
    query: Query
    hint: Hint
    mask: Optional[np.ndarray] = None  # 全图 bool
    bbox: Optional[BBox] = None


@dataclass
class ImageLayer:
    """导出的图片素材层。"""
    instance_id: str
    name: str
    kind: str
    role: str
    x: int
    y: int
    w: int
    h: int
    z: int
    parent: Optional[str]      # 父 instance_id
    png_rel: str               # 相对输出目录的 PNG 路径
    repair_mode: str
    repaired: bool = False
    degraded: bool = False


@dataclass
class TextLayer:
    """导出的可编辑文字层。"""
    region_id: str
    text: str
    x: float
    y: float
    w: float
    h: float
    conf: float
    style: dict


@dataclass
class Plan:
    """校验后的规划结果。"""
    scene_summary: str
    background_mode: str        # scene | surface | none
    background_reason: str
    raster_text_ids: List[str]
    text_corrections: List[dict]
    queries: List[Query]
    raw: dict = field(default_factory=dict)


def dump_json(path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not serializable: {type(o)}")
