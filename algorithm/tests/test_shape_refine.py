"""mask 边界确定性整形（2026-09-03 二期问题三b）。

框/卡/tab 这类矩形元素吸附成圆角矩形把锯齿拉直;非矩形退轮廓平滑;
多刺轮廓（植被）与会改坏形状的候选一律放弃整形。
"""
import cv2
import numpy as np
import pytest

from algorithm.segment.cleanup import refine_boundary
from algorithm.tests.test_halo_and_profile import _disc


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / float((a | b).sum())


def _jitter_x(mask: np.ndarray) -> int:
    """逐行端点跳动次数（scripts/measure_edge_jitter.py 同款指标）。"""
    counts, prev = 0, None
    for line in mask:
        xs = np.nonzero(line)[0]
        cur = (int(xs[0]), int(xs[-1])) if len(xs) else None
        if prev is not None and cur is not None:
            counts += int(cur[0] != prev[0]) + int(cur[1] != prev[1])
        prev = cur
    return counts


def _jagged_rect(h=200, w=300, box=(40, 30, 260, 170), amp=2, seed=7):
    """锯齿矩形：四边逐行/逐列随机抖 ±amp px（SAM 边缘的合成形态）。"""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), bool)
    for y in range(y0, y1):
        m[y, x0 + rng.integers(-amp, amp + 1):x1 + rng.integers(-amp, amp + 1)] = True
    for x in range(x0, x1):
        col = m[:, x]
        col[:y0 + rng.integers(-amp, amp + 1)] = False
        col[y1 + rng.integers(-amp, amp + 1):] = False
    return m


def test_rect_snap_straightens_jagged_card_frame():
    mask = _jagged_rect()
    before = _jitter_x(mask)
    out, info = refine_boundary(mask)
    assert info["applied"] and info["mode"] == "rect", info
    assert _jitter_x(out) < before / 8, (before, _jitter_x(out))
    assert info["iou"] >= 0.95


def test_rect_snap_keeps_ring_frame_opening():
    """环形框：外轮廓与内域开口各自吸附，开口不被填掉。"""
    mask = np.zeros((200, 200), bool)
    mask[20:180, 20:180] = True
    mask[60:140, 60:140] = False
    out, info = refine_boundary(mask)
    assert info["mode"] == "rect"
    assert not out[100, 100]              # 开口保持镂空
    assert out[30, 100] and out[100, 30]  # 框环保持实心


def test_rect_snap_recovers_rounded_corners():
    """圆角卡片吸附后仍是圆角（按面积反解半径），不被拉成直角。"""
    from algorithm.segment.cleanup import _rounded_rect

    ref = _rounded_rect((160, 160), (30, 30, 130, 130), 20)
    rng = np.random.default_rng(3)
    noisy = ref.copy()
    edge = ref ^ cv2.erode(ref.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    flip = edge & (rng.random(ref.shape) < 0.5)
    noisy[flip] = False
    out, info = refine_boundary(noisy)
    assert info["applied"]
    assert not out[31, 31]                # 左上角外仍是圆角空白
    assert out[80, 32] and out[32, 80]    # 四边中点实心


def test_disc_snaps_within_rounded_rect_family():
    """圆盘 = 圆角矩形 r=min边/2：走吸附档，抖动被拉回正圆。"""
    disc = _disc(200, (100, 100), 70)
    rng = np.random.default_rng(5)
    edge = disc ^ cv2.erode(disc.astype(np.uint8),
                            np.ones((3, 3), np.uint8)).astype(bool)
    noisy = disc.copy()
    noisy[edge & (rng.random(disc.shape) < 0.5)] = False
    out, info = refine_boundary(noisy)
    assert info["applied"], info
    # 曲线边界的端点跳动有固有下限（正圆每行端点都在变），这里看形状保真：
    # 抖动被拉回一个干净正圆（跳动指标只对直边形状可读，见量测脚本注记）
    assert _iou(out, disc) > 0.97
    assert abs(int(out.sum()) - int(disc.sum())) < 0.05 * disc.sum()


def test_smooth_path_for_shape_outside_rect_family():
    """十字形（工具图标）不在圆角矩形族里：吸附落空，退轮廓平滑。"""
    cross = np.zeros((200, 200), bool)
    cross[80:120, 30:170] = True
    cross[30:170, 80:120] = True
    rng = np.random.default_rng(9)
    edge = cross ^ cv2.erode(cross.astype(np.uint8),
                             np.ones((3, 3), np.uint8)).astype(bool)
    noisy = cross.copy()
    noisy[edge & (rng.random(cross.shape) < 0.5)] = False
    out, info = refine_boundary(noisy)
    assert info["applied"] and info["mode"] == "smooth", info
    assert not out[40, 40]                # 四个凹角没被吸附填平


def test_spiky_contour_is_left_alone():
    """多刺轮廓（植被剪影）：compactness 超限，不整形。"""
    m = np.zeros((200, 200), bool)
    m[120:170, 20:180] = True
    for x in range(20, 180, 6):           # 一排尖刺树冠
        cv2.fillPoly(m.view(np.uint8), [np.array(
            [[x, 120], [x + 3, 60], [x + 6, 120]], np.int32)], 1)
    spiky = m | (m.view(np.uint8).astype(bool))
    out, info = refine_boundary(spiky)
    assert not info["applied"]
    assert np.array_equal(out, spiky)


def test_guard_reverts_when_candidate_moves_boundary_too_far():
    """L 形（tab 带缺角）：吸附候选会把缺角填平，位移守卫拦下。"""
    mask = np.zeros((200, 200), bool)
    mask[40:160, 40:160] = True
    mask[40:100, 100:160] = False         # 缺掉右上一大块
    out, info = refine_boundary(mask)
    assert not info["applied"] or info["mode"] == "smooth"
    assert not out[50, 150]               # 缺角没被填平


def test_disable_switch(monkeypatch):
    from algorithm import config

    monkeypatch.setattr(config, "SHAPE_DISABLE", 1)
    mask = _jagged_rect()
    out, info = refine_boundary(mask)
    assert not info["applied"]
    assert out is mask
