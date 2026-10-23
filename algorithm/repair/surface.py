"""element_repair_mode=surface：从缺口周围像素修补近似均匀材质（文档 §3.2）。

文档未给出独立采样参数；按「与文字清除修补相似」取 Telea radius=5。
"""
from __future__ import annotations

import numpy as np

from .. import config


def surface_fill(crop_rgb: np.ndarray, hole_mask: np.ndarray,
                 margin: int = 16) -> np.ndarray:
    """对裁片内的 hole_mask 镂空做 Telea 修补，返回修补后的裁片副本。"""
    if not hole_mask.any():
        return crop_rgb.copy()
    h, w = crop_rgb.shape[:2]
    x0 = max(int(np.nonzero(hole_mask.any(axis=0))[0].min()) - margin, 0)
    x1 = min(int(np.nonzero(hole_mask.any(axis=0))[0].max()) + 1 + margin, w)
    y0 = max(int(np.nonzero(hole_mask.any(axis=1))[0].min()) - margin, 0)
    y1 = min(int(np.nonzero(hole_mask.any(axis=1))[0].max()) + 1 + margin, h)
    sub = crop_rgb[y0:y1, x0:x1]
    sub_mask = hole_mask[y0:y1, x0:x1].astype(np.uint8) * 255
    import cv2

    filled = cv2.inpaint(cv2.cvtColor(sub, cv2.COLOR_RGB2BGR), sub_mask,
                         config.INPAINT_RADIUS, cv2.INPAINT_TELEA)
    out = crop_rgb.copy()
    out[y0:y1, x0:x1][sub_mask > 0] = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)[sub_mask > 0]
    return out
