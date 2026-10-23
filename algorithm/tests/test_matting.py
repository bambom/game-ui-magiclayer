"""半透明区域的确定性连续 alpha（2026-09-03 二期问题四）。

导出层 alpha 一直是 SAM 二值 mask，雾化/玻璃/光晕边缘被硬切。观测
I = α·F + (1−α)·B，B 由边界低频延续估、F 由邻域前景均值估，α 是闭式解。
这里用"已知真值 α"的合成图验证解的精度，并验证三条兜底：
|F−B| 不足回退二值、硬边元素不走 matting、开关关掉即原样二值。
"""
import cv2
import numpy as np

from algorithm import config
from algorithm.common import Hint, Instance, Query
from algorithm.repair.elements import _rgba, _wants_matting
from algorithm.repair.matting import (edge_softness, feather_binary, is_soft_edge,
                                      solve_alpha, trimap_width)

SIZE = 200
CENTER = (100, 100)
RADIUS = 60
RAMP = 4.0          # 真值 α 在 r±RAMP 内线性过渡（宽 8px，落在未知带里）

BG = np.array([200, 202, 198], np.float32)
FG = np.array([60, 80, 240], np.float32)


def _alpha_disc(fg=FG, bg=BG):
    """返回 (合成图, 真值 α, 二值 mask)：平坦背景上的 α 渐变圆。"""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    dist = np.sqrt((xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2)
    a = np.clip((RADIUS + RAMP - dist) / (2 * RAMP), 0.0, 1.0)
    img = a[..., None] * fg + (1.0 - a[..., None]) * bg
    return (np.clip(np.round(img), 0, 255).astype(np.uint8), a, a >= 0.5)


def _q(kind="decoration", role="decoration"):
    return Query(id="veg", name="vegetation", kind=kind, role=role,
                 element_repair_mode="none", parent_query_id=None, z_order=1)


def _inst(kind="decoration", role="decoration"):
    hint = Hint(bbox=(0, 0, SIZE, SIZE), positive_points=[CENTER],
                negative_points=[], bbox_norm=[0, 0, 1, 1],
                positive_points_norm=[[0.5, 0.5]], negative_points_norm=[])
    return Instance(instance_id="veg", query=_q(kind, role), hint=hint)


# ---------------- 闭式解精度 ----------------

def test_solve_alpha_recovers_known_ground_truth():
    """已知真值 α 的渐变圆：解出的 α 与真值 MAE < 0.05。"""
    img, truth, mask = _alpha_disc()
    alpha, _, info = solve_alpha(img, mask)
    assert info["applied"] and info["solved_px"] > 0
    mae = float(np.abs(alpha.astype(np.float32) / 255.0 - truth).mean())
    assert mae < 0.05, (mae, info)


def test_solve_alpha_is_continuous_not_bimodal():
    """未知带里必须出现中间值——直方图不是 0/255 双峰。"""
    img, _, mask = _alpha_disc()
    alpha, _, _ = solve_alpha(img, mask)
    mid = int(((alpha > 12) & (alpha < 243)).sum())
    assert mid > 200, mid          # 圆周约 380px × 未知带宽


def test_solve_alpha_beats_binary_on_the_ramp():
    """在过渡带上，连续 α 的误差显著小于二值 mask。"""
    img, truth, mask = _alpha_disc()
    alpha, _, _ = solve_alpha(img, mask)
    band = (truth > 0.05) & (truth < 0.95)
    got = float(np.abs(alpha[band].astype(np.float32) / 255.0 - truth[band]).mean())
    binary = float(np.abs(mask[band].astype(np.float32) - truth[band]).mean())
    assert got < 0.5 * binary, (got, binary)


def test_foreground_rgb_drops_the_background_mix():
    """未知带 RGB 换成 F 估计：不把背景色混进素材（移动图层不露馅）。"""
    img, truth, mask = _alpha_disc()
    _, rgb, _ = solve_alpha(img, mask)
    band = (truth > 0.2) & (truth < 0.8)
    before = float(np.abs(img[band].astype(np.float32) - FG).mean())
    after = float(np.abs(rgb[band].astype(np.float32) - FG).mean())
    assert after < 0.3 * before, (before, after)
    # 确定区的像素一个都不许动
    core = truth >= 1.0
    assert np.array_equal(rgb[core], img[core])


# ---------------- 兜底 ----------------

def test_low_contrast_falls_back_to_binary():
    """|F−B| 低于门槛（前景与背景几乎同色）：数值上不可解，回退二值。"""
    img, _, mask = _alpha_disc(fg=BG + 2.0)
    alpha, rgb, info = solve_alpha(img, mask)
    assert not info["applied"] and info["solved_px"] == 0
    assert np.array_equal(alpha, mask.astype(np.uint8) * 255)
    assert rgb is img


def test_disable_switch_returns_plain_binary(monkeypatch):
    monkeypatch.setattr(config, "MATTING_DISABLE", 1)
    img, _, mask = _alpha_disc()
    alpha, rgb, info = solve_alpha(img, mask)
    assert not info["applied"]
    assert np.array_equal(alpha, mask.astype(np.uint8) * 255) and rgb is img


def test_empty_mask_is_safe():
    img, _, _ = _alpha_disc()
    mask = np.zeros((SIZE, SIZE), bool)
    alpha, rgb, info = solve_alpha(img, mask)
    assert not info["applied"] and not alpha.any() and rgb is img


def test_trimap_width_clamps_to_config():
    small = np.zeros((40, 40), bool)
    small[18:22, 18:22] = True                  # 4px 元素 -> 撞下限
    assert trimap_width(small) == config.MAT_TRIMAP_MIN
    big = np.ones((900, 900), bool)             # 0.04x900=36 -> 撞上限
    assert trimap_width(big) == config.MAT_TRIMAP_MAX
    mid = np.zeros((400, 400), bool)
    mid[100:300, 100:300] = True                # 0.04x200=8
    assert trimap_width(mid) == 8


# ---------------- 软/硬边判据 ----------------

def test_soft_edge_detected_on_the_gradient_disc():
    img, _, mask = _alpha_disc()
    assert is_soft_edge(img, mask)


def test_edge_softness_measures_the_transition_width():
    """等效宽度必须随真实过渡宽度单调上升（梯度中位数做不到这件事）。

    锁死判据标尺：硬阶跃与 1px 斜坡同为 2.0，4px 斜坡 4.0，8px 7.9。
    """
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    dist = np.sqrt((xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2)
    got = []
    for full in (1.0, 2.0, 4.0, 8.0):
        a = np.clip((RADIUS + full / 2 - dist) / full, 0.0, 1.0)
        img = np.clip(np.round(a[..., None] * FG + (1.0 - a[..., None]) * BG),
                      0, 255).astype(np.uint8)
        got.append(edge_softness(img, a >= 0.5))
    assert got == sorted(got), got
    assert got[0] < config.MAT_SOFT_EDGE_WIDTH <= got[2]
    assert abs(got[3] - 8.0) < 1.5, got


def test_edge_softness_zero_without_contrast():
    """前景与背景几乎同色：α 本就不可解，直接判硬边（别做无意义的解算）。"""
    img, _, mask = _alpha_disc(fg=BG + 2.0)
    assert edge_softness(img, mask) == 0.0
    assert not is_soft_edge(img, mask)


def test_hard_edge_rejected():
    """高对比直边（卡框/按钮）不该走 matting——只会把干净边啃出毛边。"""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:] = BG.astype(np.uint8)
    mask = np.zeros((SIZE, SIZE), bool)
    mask[50:150, 50:150] = True
    img[mask] = FG.astype(np.uint8)
    assert not is_soft_edge(img, mask)
    assert not _wants_matting(_inst(kind="frame", role="panel"), img, mask)


def test_decoration_always_gets_matting():
    """kind/role=decoration 走连续 alpha（植被/雾气一律软边）。"""
    img, _, mask = _alpha_disc()
    assert _wants_matting(_inst(), img, mask)


# ---------------- 羽化与装配 ----------------

def test_feather_touches_only_the_boundary_ring():
    mask = np.zeros((80, 80), bool)
    mask[20:60, 20:60] = True
    out = feather_binary(mask)
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    assert out[mask & ~ring].min() == 255            # 内部实心
    assert out[~mask & ~ring].max() == 0             # 外部全透
    assert 0 < out[ring].max() < 255                 # 边界确实有中间值


def test_feather_sigma_zero_is_plain_binary(monkeypatch):
    monkeypatch.setattr(config, "EDGE_FEATHER_SIGMA", 0.0)
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    assert set(np.unique(feather_binary(mask))) <= {0, 255}


def test_rgba_assembly_uses_continuous_alpha_for_decoration():
    img, truth, mask = _alpha_disc()
    rgba = _rgba(_inst(), img, mask)
    assert rgba.shape == (SIZE, SIZE, 4)
    mae = float(np.abs(rgba[..., 3].astype(np.float32) / 255.0 - truth).mean())
    assert mae < 0.05, mae


def test_rgba_keeps_repaired_holes_opaque():
    """镂空区是我们重建出来的材质，一律不透明——不能被 matting 解成半透。"""
    img, _, mask = _alpha_disc()
    holes = np.zeros((SIZE, SIZE), bool)
    holes[92:108, 92:108] = True                  # 洞在元素中心
    rgba = _rgba(_inst(), img, mask, holes)
    assert rgba[..., 3][holes].min() == 255


def test_rgba_hard_edge_layer_stays_binary_ish():
    """硬边层走羽化路径：半透明只出现在边界那一圈，内部实心、外部全透。"""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:] = BG.astype(np.uint8)
    mask = np.zeros((SIZE, SIZE), bool)
    mask[50:150, 50:150] = True
    img[mask] = FG.astype(np.uint8)
    rgba = _rgba(_inst(kind="frame", role="panel"), img, mask)
    a = rgba[..., 3]
    assert a[60:140, 60:140].min() == 255
    assert a[:40, :40].max() == 0
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    assert not ((a > 12) & (a < 243) & ~ring).any()   # 半透明不越出边界环
