"""mask 修补与连通域筛选（文档 §2.4）+ 边界整形（二期问题三，文档外补齐）。"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import config

_K3 = np.ones((3, 3), np.uint8)
_SHAPE_MIN_SIDE = 16          # 外轮廓最小边长：更小的形状谈不上"边界锯齿"
_SHAPE_MIN_HOLE_PX = 64       # 内域（框环开口）小于此面积不参与吸附


def close_mask(mask: np.ndarray) -> np.ndarray:
    """3x3 全 1 核闭运算（先膨胀后腐蚀），连接 1-2px 断裂并填小针孔。"""
    kernel = np.ones((config.CLOSE_KERNEL, config.CLOSE_KERNEL), np.uint8)
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return out.astype(bool)


def filter_components(mask: np.ndarray,
                      positive_points: Sequence[Tuple[int, int]]) -> np.ndarray:
    """8 邻域连通域筛选：
    1) 含正点的组件必保留；
    2) 若无组件命中正点，保留最大组件；
    3) 面积 >= 最大组件 8% 的组件保留；
    4) 其余碎屑删除。
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max()) if len(areas) else 0

    labels_with_points = set()
    for x, y in positive_points:
        h, w = labels.shape
        if 0 <= x < w and 0 <= y < h:
            lab = int(labels[y, x])
            if lab > 0:
                labels_with_points.add(lab)

    keep = np.zeros(num, bool)
    if labels_with_points:
        for lab in labels_with_points:
            keep[lab] = True
    else:
        max_lab = int(np.argmax(areas)) + 1
        keep[max_lab] = True
    for lab in range(1, num):
        if areas[lab - 1] >= max_area * config.COMPONENT_KEEP_RATIO:
            keep[lab] = True

    out = keep[labels]
    return out


def clean_mask(mask: np.ndarray,
               positive_points: Sequence[Tuple[int, int]]) -> np.ndarray:
    return filter_components(close_mask(mask), positive_points)


# ---------------- 边界整形（2026-09-03 二期问题三b） ----------------

def _iou_mask(a: np.ndarray, b: np.ndarray) -> float:
    union = float((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def _rounded_rect(shape, box, radius: int) -> np.ndarray:
    """轴对齐圆角矩形光栅（radius<=0 即直角矩形）。"""
    x0, y0, x1, y1 = box
    out = np.zeros(shape, np.uint8)
    w, h = x1 - x0, y1 - y0
    r = int(min(radius, min(w, h) // 2))
    if r <= 0:
        out[y0:y1, x0:x1] = 1
        return out.astype(bool)
    cv2.rectangle(out, (x0 + r, y0), (x1 - 1 - r, y1 - 1), 1, -1)
    cv2.rectangle(out, (x0, y0 + r), (x1 - 1, y1 - 1 - r), 1, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - 1 - r, y0 + r),
                   (x0 + r, y1 - 1 - r), (x1 - 1 - r, y1 - 1 - r)):
        cv2.circle(out, (cx, cy), r, 1, -1)
    return out.astype(bool)


def _contours_ccomp(mask: np.ndarray):
    """(外轮廓列表, 内域轮廓列表)：RETR_CCOMP 的 parent==-1 即外轮廓。"""
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                             cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = found[-2], found[-1]
    if hierarchy is None:
        return [], []
    outer, holes = [], []
    for c, node in zip(contours, hierarchy[0]):
        (outer if node[3] == -1 else holes).append(c)
    return outer, holes


def _contour_region(contour: np.ndarray):
    """(bbox, 轮廓填充后的局部光栅)。与 _rounded_rect 同一光栅口径。"""
    x, y, w, h = cv2.boundingRect(contour)
    tmp = np.zeros((h, w), np.uint8)
    cv2.drawContours(tmp, [contour - (x, y)], -1, 1, -1)
    return (x, y, w, h), tmp.astype(bool)


def _fit_radius(region: np.ndarray) -> int:
    """在轮廓局部光栅上按 IoU 拟合圆角半径（先梯子后 1px 精修）。

    不按 面积 = w·h − (4−π)·r² 反解：边界锯齿只会抠掉像素，面积亏空会被整笔
    算成圆角——实测 224x140 的直角卡框被反解出 r=32，重建反而更差。IoU 拟合
    对这种零均值以下的噪声稳健，代价是 ~25 次 bbox 级光栅（毫秒内）。
    """
    h, w = region.shape
    rmax = min(h, w) // 2
    step = max(rmax // 8, 1)
    best, best_iou = 0, -1.0
    for r in list(range(0, rmax + 1, step)) + [rmax]:
        v = _iou_mask(region, _rounded_rect((h, w), (0, 0, w, h), r))
        if v > best_iou:
            best, best_iou = r, v
    for r in range(max(best - step, 0), min(best + step, rmax) + 1):
        v = _iou_mask(region, _rounded_rect((h, w), (0, 0, w, h), r))
        if v > best_iou:
            best, best_iou = r, v
    return best


def _snap_rounded_rect(mask: np.ndarray) -> Optional[np.ndarray]:
    """圆角矩形吸附候选：逐轮廓拟合圆角半径后按 bbox 重光栅。

    不需要角点拟合或边角度判据。是否真是矩形由调用方用重建 IoU 自证——不是
    矩形的形状拟合出的候选与原 mask 差很多，自动落空到轮廓平滑档；圆角矩形
    族本身涵盖直角矩形（r=0）与圆/胶囊（r=min边/2），圆形元素也走这一档。
    """
    outer, holes = _contours_ccomp(mask)
    if not outer:
        return None
    cand = np.zeros(mask.shape, bool)
    for c in outer:
        (x, y, w, h), region = _contour_region(c)
        if min(w, h) < _SHAPE_MIN_SIDE:
            return None
        cand |= _rounded_rect(mask.shape, (x, y, x + w, y + h),
                              _fit_radius(region))
    for c in holes:                     # 框环开口：同款吸附后扣掉
        (x, y, w, h), region = _contour_region(c)
        if int(region.sum()) < _SHAPE_MIN_HOLE_PX:
            continue
        cand &= ~_rounded_rect(mask.shape, (x, y, x + w, y + h),
                               _fit_radius(region))
    return cand if cand.any() else None


def _smooth_contours(mask: np.ndarray) -> Optional[np.ndarray]:
    """轮廓高斯重光栅候选：紧凑形状才做，多刺轮廓（植被/火焰）原样不动。

    compactness = 周长²/(4π·面积)（圆=1、方≈1.27），超上限说明轮廓本身就该
    有大量细节，平滑会削掉真内容。平滑沿轮廓序列环形卷积，只动边界抖动。
    """
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    outer, holes = _contours_ccomp(closed.astype(bool))
    if not outer:
        return None
    sigma = config.SHAPE_SMOOTH_SIGMA
    if sigma <= 0:
        return None
    cand = np.zeros(mask.shape, np.uint8)
    for group, value in ((outer, 1), (holes, 0)):
        for c in group:
            pts = c.reshape(-1, 2)
            if len(pts) < 12:
                if value:
                    cv2.fillPoly(cand, [pts.astype(np.int32)], 1)
                continue
            per = float(cv2.arcLength(c, True))
            area = float(cv2.contourArea(c))
            if area <= 0:
                continue
            if value and per * per > (config.SHAPE_MAX_COMPACTNESS
                                      * 4.0 * math.pi * area):
                return None            # 多刺外轮廓：整个 mask 不整形
            cv2.fillPoly(cand, [_smooth_ring(pts, sigma)], value)
    out = cand.astype(bool)
    return out if out.any() else None


def _smooth_ring(pts: np.ndarray, sigma: float) -> np.ndarray:
    """闭合点序列的环形高斯平滑（首尾相接，不在起点处留折角）。"""
    n = len(pts)
    radius = max(int(round(3 * sigma)), 1)
    kernel = cv2.getGaussianKernel(2 * radius + 1, sigma).ravel()
    wrapped = np.concatenate([pts[-radius:], pts, pts[:radius]]).astype(np.float64)
    smooth = np.stack([np.convolve(wrapped[:, i], kernel, mode="valid")
                       for i in (0, 1)], axis=1)
    return np.round(smooth[:n]).astype(np.int32)


def _within_shift(mask: np.ndarray, cand: np.ndarray, max_shift: int) -> bool:
    """整形只许在原边界 max_shift 内挪动：越界即形状被改坏（误判保护）。"""
    xor = mask ^ cand
    if not xor.any():
        return True
    if max_shift <= 0:
        return False
    boundary = mask ^ cv2.erode(mask.astype(np.uint8), _K3).astype(bool)
    near = cv2.dilate(boundary.astype(np.uint8), _K3,
                      iterations=max_shift).astype(bool)
    return not (xor & ~near).any()


def refine_boundary(mask: np.ndarray) -> Tuple[np.ndarray, dict]:
    """SAM mask 边界确定性整形，返回 (mask, info)。

    先试圆角矩形吸附（框/卡/tab 这类矩形元素直接把四边拉直、圆角吸附），
    重建 IoU 不够则退到轮廓高斯重光栅；两档都要通过位移与 IoU 守卫，
    否则原样返回。整形是纯几何操作，不看 query 声明的 kind——"是不是矩形"
    由重建自证，比按 kind 分派可靠。
    """
    info = {"applied": False, "mode": "", "iou": 1.0, "changed": 0}
    if config.SHAPE_DISABLE or mask is None or not mask.any():
        return mask, info
    cand = _snap_rounded_rect(mask)
    mode = "rect"
    if cand is None or _iou_mask(mask, cand) < config.SHAPE_SNAP_MIN_IOU:
        cand, mode = _smooth_contours(mask), "smooth"
    if cand is None:
        return mask, info
    iou = _iou_mask(mask, cand)
    changed = int((mask ^ cand).sum())
    if changed == 0 or iou < config.SHAPE_MIN_IOU \
            or not _within_shift(mask, cand, config.SHAPE_MAX_SHIFT_PX):
        return mask, info
    info.update(applied=True, mode=mode, iou=iou, changed=changed)
    return cand, info
