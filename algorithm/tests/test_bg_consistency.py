"""背景一致性融合残差的确定性修复（2026-09-03 二期问题五，val-postproc-3）。

5a 接缝带只写 mask 内侧 + 阶跃超阈自适应加宽；
5b 被压过平的洞（k_grain 低）从干净环带镜像平铺颗粒补回 RMS；
5c 环带统计全局:局部混合 + 洞内均值向环带均值 ≤3/通道 微平移。
"""
import cv2
import numpy as np

from algorithm import config
from algorithm.repair.background import (anchor_scene_fill, seam_band_smooth,
                                         seam_step)

PAPER = np.array([208, 212, 218], dtype=float)


def _paper(h, w, sigma=6.0, seed=4):
    """带真实颗粒的纸底（细颗粒 RMS 约 6）。"""
    rng = np.random.default_rng(seed)
    img = PAPER + rng.normal(0, sigma, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------- 5a 接缝带 ----------------

def test_seam_smooth_never_writes_outside_mask():
    """洞外一个像素都不许改：洞外真实背景上不留改写痕迹。"""
    img = _paper(120, 120)
    mask = np.zeros((120, 120), bool)
    mask[40:80, 40:80] = True
    img[mask] = (150, 160, 170)          # 洞内是另一种颜色，制造大阶跃
    out = seam_band_smooth(img, mask)
    assert np.array_equal(out[~mask], img[~mask])


def _step_case(delta=40, size=160):
    img = _paper(size, size)
    mask = np.zeros((size, size), bool)
    mask[size // 3:size * 2 // 3, size // 3:size * 2 // 3] = True
    img[mask] = np.clip(PAPER + delta, 0, 255).astype(np.uint8)
    return img, mask


def test_seam_smooth_widens_until_step_drops():
    """固定 2px 吃不掉的阶跃，自适应加宽后明显变小。

    只写内侧时模糊有数学上限：洞外不动，对称核最多把内侧拉到内外中点，
    所以阶跃最好也就减到约 Δ/2——清零要靠梯度域（见下一个用例）。
    """
    img, mask = _step_case()
    before = seam_step(img, mask)
    after = seam_step(seam_band_smooth(img, mask), mask)
    assert after < 0.7 * before, (before, after)
    assert after >= 0.35 * before                 # Δ/2 上限，别写出假期望


def test_poisson_harmonize_removes_the_step_when_enabled(monkeypatch):
    """开了梯度域收尾且给了参照图：阶跃被清到阈值以内，洞外仍不动。"""
    monkeypatch.setattr(config, "BG_SEAM_POISSON", 1)
    img, mask = _step_case()
    reference = _paper(160, 160)                  # 修补前的真实背景
    out = seam_band_smooth(img, mask, reference=reference)
    assert seam_step(out, mask) < seam_step(seam_band_smooth(img, mask), mask)
    assert np.array_equal(out[~mask], img[~mask])


def test_poisson_skipped_without_reference(monkeypatch):
    """开了开关但调用方没给参照图：退回加宽结果，不假装做了。"""
    monkeypatch.setattr(config, "BG_SEAM_POISSON", 1)
    img, mask = _step_case()
    assert np.array_equal(seam_band_smooth(img, mask),
                          seam_band_smooth(img, mask, reference=None))


def test_seam_smooth_leaves_matched_boundary_alone():
    """本来就没阶跃时停在起始带宽，不加宽（少改一个像素是一个）。"""
    from algorithm.repair.background import _blend_band

    img = _paper(120, 120)
    mask = np.zeros((120, 120), bool)
    mask[40:80, 40:80] = True
    assert seam_step(img, mask) <= config.BG_SEAM_RESID_THR
    out = seam_band_smooth(img, mask)
    assert np.array_equal(out, _blend_band(img, mask, config.BG_SEAM_BAND_ITERS))


def test_seam_smooth_zero_iters_is_noop(monkeypatch):
    monkeypatch.setattr(config, "BG_SEAM_BAND_ITERS", 0)
    img = _paper(80, 80)
    mask = np.zeros((80, 80), bool)
    mask[30:50, 30:50] = True
    assert seam_band_smooth(img, mask) is img


# ---------------- 5b 纹理注入 ----------------

def _hp_rms(img, mask, sigma):
    """总高频 RMS（= 注入的口径，也是证据里"边缘密度"的家族）。"""
    f = img.astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (0, 0), sigma)
    return float(np.sqrt((hp[mask] ** 2).mean()))


def test_grain_injection_lifts_over_flat_hole():
    """洞里是纯平生图（k_grain→0）：注入后洞内颗粒回到环带水平。"""
    work = _paper(240, 240, sigma=8.0)
    hole = np.zeros((240, 240), bool)
    hole[80:160, 80:160] = True
    fill = work.copy()
    fill[hole] = PAPER.astype(np.uint8)   # 洞内零颗粒
    out = anchor_scene_fill(work, fill, hole)
    ann = (cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=48).astype(bool)
           & ~cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=12).astype(bool) & ~hole)
    sigma = config.BG_ANCHOR_SIGMA
    got, want = _hp_rms(out, hole, sigma), _hp_rms(work, ann, sigma)
    assert got > config.BG_GRAIN_INJECT_BELOW * want, (got, want)


def test_grain_injection_disabled_leaves_hole_flat(monkeypatch):
    monkeypatch.setattr(config, "BG_GRAIN_INJECT_BELOW", 0.0)
    work = _paper(240, 240, sigma=8.0)
    hole = np.zeros((240, 240), bool)
    hole[80:160, 80:160] = True
    fill = work.copy()
    fill[hole] = PAPER.astype(np.uint8)
    out = anchor_scene_fill(work, fill, hole)
    assert _hp_rms(out, hole, config.BG_ANCHOR_SIGMA) < 1.0


def test_reflect_tile_is_deterministic_and_seamless():
    from algorithm.repair.background import _reflect_tile

    patch = np.arange(16, dtype=np.float32).reshape(4, 4)
    a = _reflect_tile(patch, 10, 10)
    b = _reflect_tile(patch, 10, 10)
    assert np.array_equal(a, b)          # 无随机源
    assert a.shape == (10, 10)
    # 块接缝处镜像相接：第 3/4 列（块边界）取值相同
    assert a[0, 3] == a[0, 4]


# ---------------- 5c 洞间一致性 ----------------

def test_mean_shift_is_capped(monkeypatch):
    """洞内均值只向环带均值削峰，不硬拉（逐通道 ≤ 上限）。"""
    monkeypatch.setattr(config, "BG_GRAIN_INJECT_BELOW", 0.0)
    work = _paper(240, 240)
    hole = np.zeros((240, 240), bool)
    hole[80:160, 80:160] = True
    fill = work.copy()
    fill[hole] = np.clip(PAPER + 60, 0, 255).astype(np.uint8)  # 洞内整体偏亮
    out = anchor_scene_fill(work, fill, hole)
    no_shift = _anchor_without_shift(work, fill, hole, monkeypatch)
    delta = np.abs(out[hole].astype(float).mean(axis=0)
                   - no_shift[hole].astype(float).mean(axis=0))
    assert delta.max() <= config.BG_HOLE_MEAN_SHIFT_MAX + 1.0


def _anchor_without_shift(work, fill, hole, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(config, "BG_HOLE_MEAN_SHIFT_MAX", 0.0)
        return anchor_scene_fill(work, fill, hole)


def test_global_mix_reduces_spread_between_holes():
    """两个洞的局部环带颗粒差异很大时，全局混合把 k_grain 拉近。"""
    from algorithm.repair.background import anchor_scene_fill as anchor

    rng = np.random.default_rng(2)
    work = np.zeros((260, 460, 3), np.float32) + PAPER
    work[:, :230] += rng.normal(0, 3.0, (260, 230, 3))    # 左半细腻
    work[:, 230:] += rng.normal(0, 12.0, (260, 230, 3))   # 右半粗糙
    work = np.clip(work, 0, 255).astype(np.uint8)
    hole = np.zeros((260, 460), bool)
    hole[100:180, 60:160] = True
    hole[100:180, 300:400] = True
    fill = work.copy()
    fill[hole] = np.clip(PAPER + rng.normal(0, 20.0, (int(hole.sum()), 3)),
                         0, 255).astype(np.uint8)
    out = anchor(work, fill, hole)
    num, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    sigma = config.BG_ANCHOR_SIGMA
    rms = [_hp_rms(out, labels == lab, sigma) for lab in range(1, num)]
    assert len(rms) == 2
    # 混合后两洞颗粒量级接近（纯局部估计会拉开到 3 倍以上）
    assert max(rms) < 3.0 * max(min(rms), 1e-6)
