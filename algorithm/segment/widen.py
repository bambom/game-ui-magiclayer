"""环形 mask 向外扩展（框太细，2026-09-03 用户裁定，文档未定）。

SAM 对框环的 mask 边界落在描边中线，导出框层缺外半圈笔画——与 elements.py
镂空膨胀补偿的是同一偏差，但那里只修「父层留下的洞」，框自己导出的 mask
此前从未补偿。本模块做确定性外扩，两条路：

- 自身有内开口（SAM 直接切出的环）：开口尺寸不动（先膨胀再按原开口抠回），
  外边界外推；
- 实心父框（开口要等镂空才出现）：给 children（子实例 mask）时按其并集
  预演镂空量环带，只推外边界、不动内部——开口仍是 elements.py 镂空的事。

是否扩、扩多少由算法配置决定（off / fixed / adaptive），几何自证，不看
planner 声明的 kind（父子关系与镂空同一口径，只当结构提示用）。
adaptive 按环带宽度推外扩量：SAM 切在中线时环带 ≈ 笔画内半圈，
外半圈 ≈ 一个环带宽，ratio=1.0 即补回。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import config

_K3 = np.ones((3, 3), np.uint8)
_MIN_SIDE = 16            # 环外接 bbox 最小边：与 cleanup._SHAPE_MIN_SIDE 同族
_MIN_HOLE_PX = 64         # 开口面积阈值：与 cleanup._SHAPE_MIN_HOLE_PX 同族


def widen_hollow_mask(mask: np.ndarray,
                      children: Optional[Sequence[np.ndarray]] = None
                      ) -> Tuple[np.ndarray, dict]:
    """环形/父框 mask 外扩，返回 (mask, info)。非环形（无开口、无子占位）原样返回。"""
    info: dict = {"applied": False, "method": "", "px": 0, "band": 0.0}
    if config.MASK_WIDEN_METHOD == "off" or mask is None or not mask.any():
        return mask, info
    u8 = mask.astype(np.uint8)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if min(x1 - x0, y1 - y0) < _MIN_SIDE:
        return mask, info
    # 内开口 = 外轮廓填充 − mask。自身无开口但带子实例时，按子 mask 并集
    # 预演镂空（环带 = mask 去掉子占位），开口本身留给 elements.py 切
    found = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(u8)
    cv2.drawContours(filled, found[-2], -1, 1, -1)
    holes = filled.astype(bool) & ~mask
    ring = mask
    solid_parent = False
    if int(holes.sum()) < _MIN_HOLE_PX and children:
        inner = np.zeros_like(mask)
        for child in children:
            inner |= child.astype(bool)
        holes = inner & mask
        ring = mask & ~inner
        solid_parent = True
    if int(holes.sum()) < _MIN_HOLE_PX:
        return mask, info

    if config.MASK_WIDEN_METHOD == "fixed":
        px = max(0, int(config.MASK_WIDEN_PX))
    else:
        # 环带宽度估计：距离变换中轴区域（≥峰值一半）的 75 分位 × 2。峰值会被
        # 矩形角部抬高（L 拐角中轴离外沿更远），纯中位数又被偶数宽压低，长直边
        # 主导的 75 分位两头都稳（合成环实测误差 ≤5%）。SAM 切在描边中线时
        # 环带 ≈ 笔画内半圈，被切掉的外半圈 ≈ 一个环带宽
        dt = cv2.distanceTransform(ring.astype(np.uint8), cv2.DIST_L2,
                                   cv2.DIST_MASK_5)
        medial = dt[dt >= dt.max() * 0.5]
        if medial.size == 0:
            return mask, info
        band = float(np.percentile(medial, 75) * 2.0)
        px = int(math.ceil(config.MASK_WIDEN_RATIO * band))
        px = max(1, min(px, int(config.MASK_WIDEN_MAX_PX)))
        info["band"] = band
    if px <= 0:
        return mask, info
    grown = cv2.dilate(u8, _K3, iterations=px).astype(bool)
    # 自身开口的环：膨胀会吃开口，按原开口抠回（内边界不动）。实心父框不抠
    # ——子区域镂空是 elements.py 的职责（那边带自己的膨胀量）
    out = grown & ~holes if not solid_parent else grown
    info.update(applied=True, method=config.MASK_WIDEN_METHOD, px=px)
    return out, info
