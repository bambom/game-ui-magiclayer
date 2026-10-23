"""初次全 OCR 去字（文档 §1.2）与美术字恢复（§1.4）。"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .. import config
from ..common import OcrRegion, clamp, quad_to_mask
from ..ocr.engine import is_cjk_char


def _otsu_threshold(values: np.ndarray) -> float:
    """在给定像素值集合（0..255）上计算 Otsu 阈值。"""
    hist = np.bincount(values, minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    level = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = total - w0
    sum_all = (hist * level).sum()
    sum0 = np.cumsum(hist * level)
    sum1 = sum_all - sum0
    valid = (w0 > 0) & (w1 > 0)
    mu0 = np.where(valid, sum0 / np.maximum(w0, 1e-9), 0.0)
    mu1 = np.where(valid, sum1 / np.maximum(w1, 1e-9), 0.0)
    var_between = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, 0.0)
    return float(np.argmax(var_between))


def _dilate_radius(text: str, box_h: int) -> int:
    if len(text.strip()) == 1 and is_cjk_char(text.strip()):
        factor, lo, hi = config.ERASE_DILATE_CJK
    else:
        factor, lo, hi = config.ERASE_DILATE_OTHER
    return int(clamp(round(factor * box_h), lo, hi))


def _border_band(allowed: np.ndarray, band: int) -> np.ndarray:
    """裁片四周 band 像素边带 ∩ allowed。"""
    h, w = allowed.shape
    m = np.zeros((h, w), bool)
    m[:band, :] = True
    m[h - band:, :] = True
    m[:, :band] = True
    m[:, w - band:] = True
    return m & allowed


def erase_region(image_rgb: np.ndarray, region: OcrRegion) -> None:
    """按文档 §1.2.1 处理单个 OCR 区域，原地填写 region 的 removal/glyph/local_bg。

    removal_mask 为全图坐标 bool；修补像素由调用方统一执行。
    """
    x0, y0, x1, y1 = region.bbox
    crop = image_rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    allowed = quad_to_mask(region.quad - np.array([x0, y0], np.float32), (ch, cw))
    allowed_px = int(allowed.sum())
    if allowed_px == 0:
        region.removal_mask = np.zeros(image_rgb.shape[:2], bool)
        region.mask_mode = "empty"
        region.glyph_mask = np.zeros((ch, cw), bool)
        return

    # 2. 估计局部底色：边带 RGB 中位数
    band = int(clamp(round(min(ch, cw) / config.ERASE_BAND_DIV),
                     config.ERASE_BAND_MIN, config.ERASE_BAND_MAX))
    band = min(band, max(min(ch, cw) // 2, 1))
    band_mask = _border_band(allowed, band)
    src = band_mask if band_mask.any() else allowed
    bg = np.median(crop[src], axis=0)  # RGB

    # 3-4. 颜色距离 + Otsu 分割字形
    diff = crop.astype(np.float32) - bg.astype(np.float32)
    dist = np.linalg.norm(diff, axis=2)
    response = np.clip(dist * config.ERASE_DIST_SCALE, 0, 255).astype(np.uint8)
    t = _otsu_threshold(response[allowed])
    thresh = max(config.ERASE_OTSU_FLOOR, config.ERASE_OTSU_FACTOR * t)
    glyph = allowed & (response >= thresh)

    # 5. 连接笔画：3x3 闭运算
    kernel = np.ones((3, 3), np.uint8)
    glyph = cv2.morphologyEx(glyph.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # 6. 质量约束
    area_ratio = float(glyph.sum()) / float(allowed_px)
    mask_mode = "estimated_glyphs"
    if area_ratio < config.ERASE_AREA_MIN or area_ratio > config.ERASE_AREA_MAX:
        glyph = allowed.copy()
        mask_mode = "coarse"

    region.glyph_mask = glyph
    region.mask_mode = mask_mode
    region.local_bg = (float(bg[0]), float(bg[1]), float(bg[2]))

    # 7. 椭圆膨胀覆盖抗锯齿/描边/阴影（形态学不得越出四边形）
    r = _dilate_radius(region.text, ch)
    k = 2 * r + 1
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(glyph.astype(np.uint8), ell).astype(bool) & allowed

    removal = np.zeros(image_rgb.shape[:2], bool)
    removal[y0:y1, x0:x1] = dilated
    region.removal_mask = removal


def inpaint_mask_region(image_rgb: np.ndarray, removal: np.ndarray,
                        margin: int = 16) -> np.ndarray:
    """对 removal mask 区域做 Telea 修补，返回修补后的整图副本。

    在裁片 + margin 上下文内修补以采样周边像素（文档 §1.2.1 第 8 步）。
    """
    ys, xs = np.nonzero(removal)
    if len(xs) == 0:
        return image_rgb.copy()
    h, w = image_rgb.shape[:2]
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + 1 + margin, w)
    y1 = min(int(ys.max()) + 1 + margin, h)
    crop = image_rgb[y0:y1, x0:x1]
    mask = removal[y0:y1, x0:x1].astype(np.uint8) * 255
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, config.INPAINT_RADIUS, cv2.INPAINT_TELEA)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    result = image_rgb.copy()
    region_slice = result[y0:y1, x0:x1]
    region_slice[mask > 0] = out_rgb[mask > 0]
    return result


def erase_all(image_rgb: np.ndarray, regions: List[OcrRegion]) -> np.ndarray:
    """初次全 OCR 去字（文档 §1.2）：逐区域估计 + Telea。"""
    work = image_rgb
    for region in regions:
        erase_region(work, region)
        if region.removal_mask is not None and region.removal_mask.any():
            work = inpaint_mask_region(work, region.removal_mask)
    return work


def restore_raster(work: np.ndarray, original: np.ndarray,
                   raster_regions: List[OcrRegion]) -> np.ndarray:
    """美术字分支（文档 §1.4 第 4 步）：删除对应文字 removal mask，恢复原图栅格字形。"""
    out = work
    for region in raster_regions:
        if region.removal_mask is None:
            continue
        out = out.copy()
        out[region.removal_mask] = original[region.removal_mask]
    return out
