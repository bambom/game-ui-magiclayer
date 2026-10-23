"""去字逐区域算法（文档 §1.2.1）的阈值与行为。"""
import numpy as np

from algorithm.common import OcrRegion
from algorithm.texterase.erase import (_otsu_threshold, erase_region,
                                       inpaint_mask_region)


def make_region(w, h):
    quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return OcrRegion(id="text_001", text="金币", conf=0.98, quad=quad,
                     bbox=(0, 0, w, h))


def flat_bg(w, h, color=(60, 127, 139)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    return img


def test_otsu_bimodal():
    # 三值分布：类间方差最大切在 {10,100} | {200} 之间
    vals = np.array([10] * 400 + [100] * 100 + [200] * 400)
    t = _otsu_threshold(vals)
    assert 50 <= t < 200


def test_glyph_estimated_and_removed():
    # 深底 + 中央白色字形（占比约 25%，落在 1.5%~68% 之间 -> estimated_glyphs）
    img = flat_bg(120, 60)
    img[15:45, 30:90] = (240, 244, 241)
    r = make_region(120, 60)
    erase_region(img, r)
    assert r.mask_mode == "estimated_glyphs"
    assert r.removal_mask is not None and r.removal_mask.any()
    out = inpaint_mask_region(img, r.removal_mask)
    # 字形区域被修补回底色附近
    filled = out[20:40, 40:80].reshape(-1, 3).mean(axis=0)
    assert np.linalg.norm(filled - np.array([60, 127, 139])) < 25


def test_tiny_glyph_falls_back_to_coarse():
    # 字形占比 < 1.5% -> mask_mode=coarse，mask 取完整四边形
    img = flat_bg(120, 60)
    img[29:31, 59:61] = (240, 244, 241)
    r = make_region(120, 60)
    erase_region(img, r)
    assert r.mask_mode == "coarse"
    assert r.removal_mask.sum() == 120 * 60


def test_solid_glyph_falls_back_to_coarse():
    # 整块反色（占比 100% > 68%）-> coarse
    img = flat_bg(120, 60)
    img[:] = (240, 244, 241)
    r = make_region(120, 60)
    erase_region(img, r)
    assert r.mask_mode == "coarse"


def test_dilation_stays_within_quad():
    # 形态学不得越出四边形：quad 内缩于 bbox 时，removal mask 在 quad 外无像素
    img = flat_bg(120, 60)
    img[15:45, 30:90] = (240, 244, 241)
    quad = np.array([[10, 10], [110, 10], [110, 50], [10, 50]], np.float32)
    r = OcrRegion(id="text_001", text="金币", conf=0.98, quad=quad, bbox=(0, 0, 120, 60))
    erase_region(img, r)
    outside = np.ones((60, 120), bool)
    outside[10:50, 10:110] = False
    assert not (r.removal_mask & outside).any()
