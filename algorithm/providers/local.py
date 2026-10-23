"""离线降级 provider：不走生图，用传统算法兜底。

文档 §3.2 说明 surface 路径与文字清除修补相似（取 mask 周围像素填补），
未给出独立参数；这里按 Telea radius=5（与 §1.2.1 一致）实现，标记 degraded。
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .base import ImageRepairProvider


def _telea(rgb: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class LocalProvider(ImageRepairProvider):
    name = "local"
    degraded = True

    def repair_elements(self, atlas_rgb: np.ndarray, atlas_mask: np.ndarray) -> np.ndarray:
        # Atlas 可能很大，按连通外接框分块 Telea，避免全图修补耗时失控
        out = atlas_rgb.copy()
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            atlas_mask.astype(np.uint8), connectivity=8)
        margin = 12
        h, w = out.shape[:2]
        for lab in range(1, num):
            x, y, bw, bh = (stats[lab, cv2.CC_STAT_LEFT], stats[lab, cv2.CC_STAT_TOP],
                            stats[lab, cv2.CC_STAT_WIDTH], stats[lab, cv2.CC_STAT_HEIGHT])
            x0, y0 = max(x - margin, 0), max(y - margin, 0)
            x1, y1 = min(x + bw + margin, w), min(y + bh + margin, h)
            sub_mask = (labels[y0:y1, x0:x1] == lab).astype(np.uint8) * 255
            sub = out[y0:y1, x0:x1]
            sub_out = _telea(sub, sub_mask, config.INPAINT_RADIUS)
            sub[sub_mask > 0] = sub_out[sub_mask > 0]
        return out

    def repair_background(self, scene_hole_rgb: np.ndarray,
                          hole_mask: np.ndarray = None) -> np.ndarray:
        """边界颜色/低频延续：降采样 Telea 后放大，再做高斯低频化。"""
        h, w = scene_hole_rgb.shape[:2]
        if hole_mask is not None:
            hole = hole_mask.astype(bool)
        else:
            hole = scene_hole_rgb[:, :, 0] == 0
            if not hole.any():
                return scene_hole_rgb.copy()
            # 用暗度判定黑孔：三通道均值 < 8 视为诊断孔
            hole = scene_hole_rgb.mean(axis=2) < 8
        scale = config.SURFACE_BG_DOWNSCALE
        small = cv2.resize(scene_hole_rgb, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(hole.astype(np.uint8) * 255, (w // scale, h // scale),
                                interpolation=cv2.INTER_NEAREST)
        filled = _telea(small, small_mask, max(config.SURFACE_BG_INPAINT_RADIUS // scale, 3))
        up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
        low = cv2.GaussianBlur(up, (0, 0), config.BG_SOFT_FEATHER_SIGMA * 2)
        out = scene_hole_rgb.copy()
        m = cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        out[m] = low[m]
        return out
