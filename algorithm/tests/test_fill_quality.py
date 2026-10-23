"""洞区修补质量三遗留问题的确定性修复（2026-09-03，run 4968ff94）。

1) 背景低频锚定合成：颜色场取边界延续、纹理细节取生图（宽松 img2img 网关
   在洞内自行构图出暖褐卡板，提示词拦不住）；
2) Atlas 洞填再对齐 + 细线穿洞续接（时间轴节点洞被补成白盘、双线全断）；
3) 内容子层 mask 确定性剥离父框环（SAM 把整卡含框圈进子 mask）。
"""
import cv2
import numpy as np
import pytest

from algorithm.common import Instance, Query
from algorithm.config import BG_ANCHOR_SIGMA
from algorithm.repair.background import anchor_scene_fill
from algorithm.repair.elements import _clip_child_mask, build_element_layers
from algorithm.repair.linework import postprocess_repair

PAPER = np.array([208, 212, 218], dtype=float)   # 冷调纸
LINE = np.array([60, 62, 70], dtype=float)       # 深色双线
WARM = np.array([205, 170, 130], dtype=float)    # 暖褐卡板（模型自行构图）


def _hp_rms(img: np.ndarray, mask: np.ndarray) -> float:
    f = img.astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (0, 0), BG_ANCHOR_SIGMA)
    return float(np.sqrt((hp[mask] ** 2).mean()))


# ---------------- 1) 背景低频锚定 ----------------

def test_anchor_takes_boundary_color_field_not_model_palette():
    """暖褐卡板填充被锚定回周边冷调纸雾的颜色场。"""
    h, w = 160, 200
    work = np.zeros((h, w, 3), np.uint8)
    work[:] = PAPER.astype(np.uint8)
    hole = np.zeros((h, w), bool)
    hole[40:120, 60:140] = True
    fill = np.full((h, w, 3), 0, np.uint8)
    fill[:] = WARM.astype(np.uint8)
    out = anchor_scene_fill(work, fill, hole)
    hole_mean = out[hole].reshape(-1, 3).mean(axis=0)
    assert np.linalg.norm(hole_mean - PAPER) < 12.0, hole_mean
    # 洞外保持生图像素原样（融合只发生在洞内）
    assert np.array_equal(out[~hole], fill[~hole])


def test_anchor_keeps_model_texture_on_textured_surroundings():
    """周边是山浪级纹理时 k≈1，生图纹理细节保留。"""
    rng = np.random.default_rng(11)
    h, w = 160, 200
    noise = rng.integers(-25, 26, (h, w, 3)).astype(np.int16)
    base = np.full((h, w, 3), 190.0) + noise
    work = np.clip(base, 0, 255).astype(np.uint8)
    hole = np.zeros((h, w), bool)
    hole[40:120, 60:140] = True
    fill = np.full((h, w, 3), 190, np.uint8)
    fill[hole] = np.clip(190.0 + rng.integers(-25, 26, (int(hole.sum()), 3)),
                         0, 255).astype(np.uint8)
    out = anchor_scene_fill(work, fill, hole)
    assert _hp_rms(out, hole) >= 0.6 * _hp_rms(fill, hole)


def test_anchor_small_hole_takes_lowfreq_base():
    """小洞直接取低频延续：接近周边色，与生图调色板无关。"""
    work = np.full((40, 40, 3), PAPER.astype(np.uint8), np.uint8)
    hole = np.zeros((40, 40), bool)
    hole[16:24, 16:24] = True
    fill = np.full((40, 40, 3), WARM.astype(np.uint8), np.uint8)
    out = anchor_scene_fill(work, fill, hole)
    assert np.linalg.norm(out[hole].reshape(-1, 3).mean(axis=0) - PAPER) < 10.0


# ---------------- 2) Atlas 洞填再对齐 + 细线续接 ----------------

def _axis_strip_with_disc():
    """44 宽轴条裁片：纸底 + 深色双线，中段 40 行的洞被补成白盘。"""
    crop = np.full((120, 44, 3), PAPER.astype(np.uint8), np.uint8)
    crop[:, 19:21] = LINE.astype(np.uint8)
    crop[:, 24:26] = LINE.astype(np.uint8)
    hole = np.zeros((120, 44), bool)
    hole[40:80, 2:42] = True
    crop[hole] = (235, 235, 238)
    return crop, hole


def test_relevel_shifts_white_disc_back_to_paper():
    crop, hole = _axis_strip_with_disc()
    out, info = postprocess_repair(crop.copy(), hole)
    assert info["releveled"] == 1
    paper_zone = out[44:76, 6:16].astype(float)
    assert np.linalg.norm(paper_zone.reshape(-1, 3).mean(axis=0) - PAPER) < 10.0


def test_reconnect_restores_double_line_through_hole():
    crop, hole = _axis_strip_with_disc()
    out, info = postprocess_repair(crop.copy(), hole)
    assert info["reconnected"] == 1
    for y in range(44, 76):
        row = out[y, 12:34].astype(int)
        close = np.abs(row - LINE.astype(int)).sum(axis=1) < 120
        assert close.any(), f"y={y} 无续接线像素"
    # 洞底 = 父材质延续（纸），不是模型填的亮盘（宽松 img2img 常把被删的节点
    # 徽章画回来——亮盘边缘贴纸色，relevel 环带统计测不出，整域 Telea 覆盖）
    disc_zone = out[50:70, 4:14].astype(float)
    assert np.linalg.norm(disc_zone.reshape(-1, 3).mean(axis=0) - PAPER) < 12.0
    # 洞及其 1px 环之外保持原样
    keep = ~cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert np.array_equal(out[keep], crop[keep])


def test_reconnect_skips_when_no_line_nearby():
    crop = np.full((120, 44, 3), PAPER.astype(np.uint8), np.uint8)
    hole = np.zeros((120, 44), bool)
    hole[40:80, 2:42] = True
    crop[hole] = (235, 235, 238)
    out, info = postprocess_repair(crop, hole)
    assert info["reconnected"] == 0


def test_reconnect_skips_misaligned_sides():
    crop = np.full((120, 60, 3), PAPER.astype(np.uint8), np.uint8)
    crop[:40, 12:14] = LINE.astype(np.uint8)    # 上侧线偏左
    crop[80:, 44:46] = LINE.astype(np.uint8)    # 下侧线偏右：非同一条线
    hole = np.zeros((120, 60), bool)
    hole[40:80, 2:58] = True
    crop[hole] = (235, 235, 238)
    out, info = postprocess_repair(crop, hole)
    assert info["reconnected"] == 0


def test_reconnect_horizontal_line():
    """横向细线（轨道）同样续接：经转置分支。"""
    crop = np.full((44, 120, 3), PAPER.astype(np.uint8), np.uint8)
    crop[19:21, :] = LINE.astype(np.uint8)
    crop[24:26, :] = LINE.astype(np.uint8)
    hole = np.zeros((44, 120), bool)
    hole[2:42, 40:80] = True
    crop[hole] = (235, 235, 238)
    out, info = postprocess_repair(crop, hole)
    assert info["reconnected"] == 1
    for x in range(44, 76):
        col = out[12:34, x].astype(int)
        close = np.abs(col - LINE.astype(int)).sum(axis=1) < 120
        assert close.any(), f"x={x} 无续接线像素"


# ---------------- 3) 内容子层剥离父框 ----------------

def _ring_parent():
    """120x120 方环框（环厚 20）。"""
    m = np.zeros((120, 120), bool)
    m[10:110, 10:110] = True
    m[30:90, 30:90] = False
    return m


def test_clip_ring_parent_strips_ring_keeps_interior():
    parent = _ring_parent()
    child = np.zeros((120, 120), bool)
    child[10:110, 10:110] = True  # 子盖满整卡含框环（线上失败形态）
    out = _clip_child_mask(child, parent, (10, 10, 110, 110))
    assert out is not None
    assert not (out & parent).any()      # 框环像素全部剥离
    assert out[40:80, 40:80].all()       # 内域画作保留


def test_clip_plate_parent_strips_outer_band():
    parent = np.zeros((100, 160), bool)
    parent[10:90, 10:150] = True         # 整板框
    child = parent.copy()
    out = _clip_child_mask(child, parent, (10, 10, 150, 90))
    assert out is not None
    strip = int(np.clip(round(0.06 * 80), 4, 16))
    expected = cv2.erode(parent.astype(np.uint8),
                         np.ones((2 * strip + 1,) * 2, np.uint8)).astype(bool)
    assert np.array_equal(out, expected)


def test_clip_guard_reverts_when_child_lives_on_ring():
    """子元素本来就骑在框环上（环上徽章）：裁后所剩无几，回退不动。"""
    parent = _ring_parent()
    child = np.zeros((120, 120), bool)
    child[8:28, 50:70] = True            # 全在顶环带上
    assert _clip_child_mask(child, parent, (10, 10, 110, 110)) is None


def test_clip_skips_non_contained_child():
    parent = _ring_parent()
    child = np.zeros((120, 120), bool)
    child[100:119, 40:80] = True         # 骑出父框下缘：横跨元素本该带框外像素
    assert _clip_child_mask(child, parent, (10, 10, 110, 110)) is None


def test_clip_skips_small_or_sparse_parent():
    small = np.zeros((40, 40), bool)
    small[4:36, 4:36] = True             # 边长 < 48：不看做框
    child = small.copy()
    assert _clip_child_mask(child, small, (4, 4, 36, 36)) is None
    sparse = np.zeros((120, 120), bool)
    sparse[:, 55:65] = True              # 细长条（轴线）：既非环也非整板
    child2 = np.zeros((120, 120), bool)
    child2[40:80, 52:68] = True
    assert _clip_child_mask(child2, sparse, (0, 55, 120, 65)) is None


class _MarkerProvider:
    name = "marker"
    degraded = False

    def repair_elements(self, atlas_rgb, atlas_mask):
        return atlas_rgb.copy()

    def repair_background(self, scene_hole_rgb, hole_mask=None):
        return scene_hole_rgb.copy()


def _q(qid, mode="image", parent=None):
    return Query(id=qid, name=qid, kind="card", role="container",
                 element_repair_mode=mode, parent_query_id=parent, z_order=1)


def test_build_element_layers_applies_child_clip():
    """端到端：环框 + 盖满整卡的子层 → 子 mask 被裁剪并登记 diag。"""
    parent = _ring_parent()
    child = np.zeros((120, 120), bool)
    child[10:110, 10:110] = True
    work = np.full((120, 120, 3), (240, 240, 240), np.uint8)
    frame = Instance(instance_id="frame#1", query=_q("frame", "image"),
                     hint=None, mask=parent)
    illo = Instance(instance_id="illo#1", query=_q("illo", "none", parent="frame"),
                    hint=None, mask=child)
    results, diag = build_element_layers([frame, illo], work, _MarkerProvider())
    assert diag["child_clipped"] == ["illo#1"]
    rgba = next(r for i, r, _l in results if i.instance_id == "illo#1")
    layer = next(lay for _i, _r, lay in results if lay.instance_id == "illo#1")
    # 裁剪后的子层 bbox 缩进内域；框环像素不在子层
    assert (layer.w, layer.h) == (58, 58)
    full_alpha = np.zeros((120, 120), np.uint8)
    full_alpha[layer.y:layer.y + layer.h, layer.x:layer.x + layer.w] = rgba[..., 3]
    assert not (full_alpha.astype(bool) & parent).any()
    assert (full_alpha[40:80, 40:80] == 255).all()  # 内域画作保留
