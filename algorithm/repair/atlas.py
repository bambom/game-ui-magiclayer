"""Atlas 元素生图修补（文档 §3.1 + image24/25 示意）：原尺寸零缩放拼进正方形
画布 -> 一次修补 -> 按记录的位置/尺寸整块切回。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .. import config, runlog


@dataclass
class AtlasItem:
    item_id: str
    crop_rgb: np.ndarray          # 元素裁片（RGB）
    hole_mask: np.ndarray         # 裁片内的镂空（bool）


@dataclass
class _Placement:
    item: AtlasItem
    x: int
    y: int
    w: int                        # 画布上的贴放宽 = 裁片原始宽（零缩放）
    h: int


def _shelf_pack(sizes: List[Tuple[int, int]], gap: int, max_side: int) -> Tuple[int, int, List[Tuple[int, int]]]:
    """货架式装箱：按高降序逐行摆放，返回画布宽高与每个 item 的左上角。"""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    positions = [None] * len(sizes)
    x = y = shelf_h = 0
    canvas_w = 0
    for i in order:
        w, h = sizes[i]
        if x > 0 and x + w > max_side:
            x = 0
            y += shelf_h + gap
            shelf_h = 0
        positions[i] = (x, y)
        x += w + gap
        shelf_h = max(shelf_h, h)
        canvas_w = max(canvas_w, x - gap)
    return canvas_w, y + shelf_h, positions


def _square_side(sizes: List[Tuple[int, int]], gap: int) -> int:
    """正方形边长：不小于 ATLAS_MIN_SIDE，逐档放大直到原尺寸装箱放得下。

    不缩放元素（2026-09-03 用户裁定：元素大小不能变），画布只能长大；
    起始值取面积开方与最宽 item 的较大者，避免一个超宽条带把货架挤爆。
    """
    total = sum((w + gap) * (h + gap) for w, h in sizes)
    side = max(config.ATLAS_MIN_SIDE,
               int(np.ceil(np.sqrt(total))),
               max(w for w, _ in sizes))
    while True:
        cw, ch, _ = _shelf_pack(sizes, gap, side)
        if cw <= side and ch <= side:
            return side
        side = int(side * 1.25) + 1


def build_atlas(items: List[AtlasItem]) -> Tuple[np.ndarray, np.ndarray, List[_Placement]]:
    """把待修补元素按原始尺寸零缩放拼进正方形 Atlas 与同尺寸二值 mask（白=待补）。

    2026-09-03 用户裁定：元素大小不能变——此前按 ATLAS_MAX_SIDE 等比缩小、
    切回再 LANCZOS 放大，是输出"被拉伸/大小改变"的一层来源。画布取 1:1
    正方形且最小 ATLAS_MIN_SIDE(1024)：网关出图宽高比归一，方进方出形变最小。
    位置与尺寸由 placements 携带并落 debug/05_atlas_placements.json，
    slice_back 按此整块取图即得原始尺寸的元素 PNG。
    """
    gap = config.ATLAS_GAP
    sizes = [(it.crop_rgb.shape[1], it.crop_rgb.shape[0]) for it in items]
    if not sizes:
        raise ValueError("Atlas 为空：没有需要 image 修补的元素")
    side = _square_side(sizes, gap)

    atlas = np.full((side, side, 3), config.ATLAS_GAP_COLOR, np.uint8)
    mask = np.zeros((side, side), np.uint8)
    placements: List[_Placement] = []
    _cw, _ch, packed = _shelf_pack(sizes, gap, side)
    for item, (px, py) in zip(items, packed):
        h, w = item.crop_rgb.shape[:2]
        crop = item.crop_rgb
        hole = item.hole_mask
        if hole.any():
            # 文档 image25：待补区涂黑 + 第二张 mask。不用中位色——占位色会被当成
            # 独立画框；不用 Telea——大洞放射渐变会被当成结构；也不留原子像素。
            t0 = time.time()
            crop = crop.copy()
            crop[hole] = (0, 0, 0)
            runlog.event("LAYER", f"Atlas 洞区涂黑 {item.item_id} "
                                  f"{(time.time() - t0) * 1000:.0f}ms")
        atlas[py:py + h, px:px + w] = crop
        mask[py:py + h, px:px + w] = hole.astype(np.uint8) * 255
        placements.append(_Placement(item=item, x=px, y=py, w=w, h=h))
    if side > config.ATLAS_MAX_SIDE:
        runlog.event("WARN", f"Atlas 画布 {side}x{side} 超过 "
                             f"ATLAS_MAX_SIDE={config.ATLAS_MAX_SIDE}"
                             f"（原尺寸零缩放裁定，不再缩小；网关出图≈1.5MP，"
                             f"切回分辨率可能低于原始）")
    return atlas, mask, placements


def placements_json(canvas_w: int, canvas_h: int,
                    placements: List[_Placement]) -> str:
    """位置/大小记录（debug/05_atlas_placements.json）：切回与外部复核共用。"""
    return json.dumps({
        "canvas": [int(canvas_w), int(canvas_h)],
        "items": [{"id": p.item.item_id, "x": p.x, "y": p.y,
                   "w": p.w, "h": p.h} for p in placements],
    }, ensure_ascii=False, indent=2)


def slice_back(repaired_atlas: np.ndarray, _mask: np.ndarray,
               placements: List[_Placement]) -> List[Tuple[str, np.ndarray]]:
    """切回各元素：按 placements 记录的位置/尺寸整块取图，即该元素 PNG。

    拼放零缩放，切出天然是原始尺寸；防御性 resize 仅在生成图几何被外部改写
    （网关出图不保真被 _fit_size 拉回）时兜底。不按洞贴回原裁片、不接缝融合
    （2026-09-03 用户裁定，同背景直采生成图）。
    """
    out: List[Tuple[str, np.ndarray]] = []
    for p in placements:
        region = repaired_atlas[p.y:p.y + p.h, p.x:p.x + p.w]
        if region.shape[:2] != (p.h, p.w):
            region = cv2.resize(region, (p.w, p.h), interpolation=cv2.INTER_LANCZOS4)
        else:
            region = region.copy()
        out.append((p.item.item_id, region))
    return out
