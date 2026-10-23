"""洞缘弧斑残留的确定性修复（2026-09-03 二期问题一，val-postproc-3 遗留）。

1a 紧凑团块镂空膨胀自适应（徽章光晕按尺寸外扩，细长结构仍 2px）；
1b 洞外环带内被关住的偏离连通域并入洞（真实穿洞结构必须留下）；
1c 续接剖面章改两侧逐行插值（消洞上下口的粗细/深浅跳变）。
"""
import cv2
import numpy as np

from algorithm.repair.elements import _dilate_iters
from algorithm.repair.linework import postprocess_repair

PAPER = np.array([208, 212, 218], dtype=float)
LINE = np.array([60, 62, 70], dtype=float)


def _disc(size, center, radius, thickness=-1) -> np.ndarray:
    """bool 圆盘/圆环（cv2 绘图不吃 bool，画在 uint8 上再转）。"""
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, center, radius, 1, thickness)
    return m.astype(bool)


# ---------------- 1a 自适应膨胀 ----------------

def test_dilate_iters_scales_with_compact_badge_size():
    """圆徽章按 min 边 12% 外扩：80px 徽章 -> 10 被上限截到 8。"""
    assert _dilate_iters(_disc(120, (60, 60), 40)) == 8   # round(.12*80)=10 -> 上限
    assert _dilate_iters(_disc(60, (30, 30), 15)) == 4    # round(.12*30)=4


def test_dilate_iters_keeps_2px_for_thin_and_sparse():
    """细长（轴线）与稀疏（环框）形状不外扩：会吃掉大片父材质。"""
    axis = np.zeros((200, 40), bool)
    axis[:, 18:22] = True
    assert _dilate_iters(axis) == 2
    ring = np.zeros((120, 120), bool)
    ring[10:110, 10:110] = True
    ring[30:90, 30:90] = False                # bbox 填充率 0.64 但…
    ring[20:100, 20:100] = False              # …打薄到 0.36：稀疏
    assert _dilate_iters(ring) == 2


# ---------------- 1b 残晕吸收 ----------------

def _disc_hole_with_arc():
    """纸底裁片：中央圆盘洞 + 洞缘左右各一道弧斑（光晕残留）。"""
    crop = np.full((120, 120, 3), PAPER.astype(np.uint8), np.uint8)
    hole = _disc(120, (60, 60), 24)
    halo = _disc(120, (60, 60), 27, 3) & ~hole   # 洞外 3px 环：贴洞缘、被环带关住
    crop[halo] = (150, 150, 158)              # 比纸暗的光晕
    crop[hole] = (235, 235, 238)              # 模型填的亮盘
    return crop, hole, halo


def test_absorb_halo_clears_arc_around_hole():
    crop, hole, halo = _disc_hole_with_arc()
    out, info = postprocess_repair(crop.copy(), hole)
    assert info["halo_absorbed"] >= int(halo.sum() * 0.8)
    # 弧斑像素被按外侧材质补平：不再偏离纸底
    assert np.abs(out[halo].astype(float).mean(axis=0) - PAPER).max() < 12.0
    assert info["hole"].sum() > hole.sum()    # 有效洞含吸收带


def test_absorb_halo_keeps_line_crossing_the_hole():
    """穿洞的轴线本体越出环带，必须留下（弧斑与线的分辨判据）。"""
    crop = np.full((160, 60, 3), PAPER.astype(np.uint8), np.uint8)
    crop[:, 28:32] = LINE.astype(np.uint8)
    hole = np.pad(_disc(40, (20, 20), 18), ((60, 60), (10, 10)))
    crop[hole] = (235, 235, 238)
    out, info = postprocess_repair(crop.copy(), hole)
    assert info["halo_absorbed"] == 0
    # 洞外的线段仍在（吸收没吞掉结构）
    assert np.abs(out[8, 28:32].astype(float) - LINE).max() < 12.0


def test_absorb_halo_skips_when_neighbourhood_is_heterogeneous():
    """洞紧邻的合法异质材质（框环金边）向外延伸出环带，不当残晕。"""
    crop = np.full((120, 120, 3), PAPER.astype(np.uint8), np.uint8)
    hole = np.zeros((120, 120), bool)
    hole[40:80, 40:80] = True
    crop[hole] = (235, 235, 238)
    crop[30:90, 30:40] = (190, 150, 60)       # 左侧一整条金边贴着洞
    crop[30:90, 80:90] = (190, 150, 60)       # 右侧同款：环带过半是异质材质
    out, info = postprocess_repair(crop.copy(), hole)
    assert info["halo_absorbed"] == 0


def test_absorb_halo_disabled_by_zero_ring(monkeypatch):
    from algorithm import config

    monkeypatch.setattr(config, "ATLAS_HALO_RING_PX", 0)
    crop, hole, _halo = _disc_hole_with_arc()
    _out, info = postprocess_repair(crop.copy(), hole)
    assert info["halo_absorbed"] == 0


# ---------------- 1c 剖面渐变 ----------------

def test_reconnect_profile_interpolates_between_two_sides():
    """上深下浅的渐变线：续接段在洞上下口与原生线段无跳变。"""
    crop = np.full((120, 44, 3), PAPER.astype(np.uint8), np.uint8)
    for y in range(120):                      # 线色沿 y 从深到浅线性渐变
        shade = LINE + (PAPER - LINE) * 0.55 * (y / 119.0)
        crop[y, 19:21] = np.round(shade).astype(np.uint8)
        crop[y, 24:26] = np.round(shade).astype(np.uint8)
    hole = np.zeros((120, 44), bool)
    hole[40:80, 2:42] = True
    crop[hole] = (235, 235, 238)
    out, info = postprocess_repair(crop.copy(), hole)
    assert info["reconnected"] == 1
    # 洞上口：续接首行与洞外最后一行的线心色差；洞下口同理。合并中位剖面时
    # 两端各偏离渐变中值约 (PAPER-LINE)*0.55*0.33 ≈ 25，插值后应显著更小
    top_gap = float(np.abs(out[40, 19:21].astype(float)
                           - out[39, 19:21].astype(float)).max())
    bot_gap = float(np.abs(out[79, 19:21].astype(float)
                           - out[80, 19:21].astype(float)).max())
    assert top_gap < 10.0, top_gap
    assert bot_gap < 10.0, bot_gap
    # 续接段整体仍是渐变（洞内上段比下段深）
    upper = out[44:52, 19:21].astype(float).mean()
    lower = out[68:76, 19:21].astype(float).mean()
    assert lower > upper + 8.0
