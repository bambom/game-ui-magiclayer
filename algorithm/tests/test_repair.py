"""Atlas 拼接/切回与背景修补。"""
import json

import numpy as np

from algorithm.repair.atlas import (AtlasItem, build_atlas, placements_json,
                                    slice_back)
from algorithm.repair.background import build_hidden_mask, repair_background
from algorithm.providers.local import LocalProvider


def test_atlas_square_min_side_and_zero_scaling():
    """2026-09-03 裁定：画布正方形且 >=1024；元素原尺寸零缩放；洞涂黑。"""
    crop = np.full((40, 60, 3), (200, 100, 50), np.uint8)
    hole = np.zeros((40, 60), bool)
    hole[10:30, 20:40] = True
    items = [AtlasItem(item_id="a", crop_rgb=crop, hole_mask=hole)]
    atlas, mask, placements = build_atlas(items)
    h, w = atlas.shape[:2]
    assert h == w and w >= 1024
    p = placements[0]
    assert (p.w, p.h) == (60, 40)
    # 贴放内容零缩放：非洞像素原样、洞区涂黑
    region = atlas[p.y:p.y + p.h, p.x:p.x + p.w]
    assert (region[~hole] == crop[~hole]).all()
    assert (region[hole] == 0).all()
    assert mask[p.y:p.y + p.h, p.x:p.x + p.w][hole].min() == 255


def test_atlas_square_grows_for_oversize_items():
    """元素原尺寸放不下时画布长大而不是缩放元素（含超宽条带）。"""
    tall = np.full((900, 300, 3), 30, np.uint8)
    wide = np.full((60, 1500, 3), 60, np.uint8)
    items = [AtlasItem("tall", tall, np.zeros((900, 300), bool)),
             AtlasItem("wide", wide, np.zeros((60, 1500), bool))]
    atlas, _mask, placements = build_atlas(items)
    h, w = atlas.shape[:2]
    assert h == w and w >= 1500
    by_id = {p.item.item_id: p for p in placements}
    assert (by_id["tall"].w, by_id["tall"].h) == (300, 900)
    assert (by_id["wide"].w, by_id["wide"].h) == (1500, 60)
    # 切回天然原始尺寸
    out = dict(slice_back(atlas, _mask, placements))
    assert out["tall"].shape == (900, 300, 3)
    assert out["wide"].shape == (60, 1500, 3)


def test_placements_json_records_geometry():
    crop = np.full((40, 60, 3), 90, np.uint8)
    items = [AtlasItem(item_id="a", crop_rgb=crop,
                       hole_mask=np.zeros((40, 60), bool))]
    atlas, _mask, placements = build_atlas(items)
    data = json.loads(placements_json(atlas.shape[1], atlas.shape[0], placements))
    assert data["canvas"] == [atlas.shape[1], atlas.shape[0]]
    assert data["items"] == [{"id": "a", "x": placements[0].x,
                              "y": placements[0].y, "w": 60, "h": 40}]


def test_atlas_roundtrip_local_provider():
    crop = np.full((40, 60, 3), (200, 100, 50), np.uint8)
    hole = np.zeros((40, 60), bool)
    hole[10:30, 20:40] = True
    crop[hole] = (0, 0, 0)  # 镂空
    items = [AtlasItem(item_id="a", crop_rgb=crop, hole_mask=hole)]
    atlas, mask, placements = build_atlas(items)
    assert mask.shape == atlas.shape[:2]
    provider = LocalProvider()
    repaired = provider.repair_elements(atlas, mask)
    out = dict(slice_back(repaired, mask, placements))["a"]
    # 镂空被修补为接近元素底色
    filled = out[15:25, 25:35].reshape(-1, 3).mean(axis=0)
    assert np.linalg.norm(filled - np.array([200, 100, 50])) < 30
    # LocalProvider 只写洞像素，洞外与生成图一致（即原裁片）
    assert (out[~hole] == crop[~hole]).all()


def test_slice_back_uses_generated_crop_as_result():
    """模型出图切片即为元素，不再按洞贴回原裁片。"""
    crop = np.full((40, 60, 3), (200, 0, 0), np.uint8)
    hole = np.zeros((40, 60), bool)
    hole[10:30, 20:40] = True
    items = [AtlasItem(item_id="a", crop_rgb=crop, hole_mask=hole)]
    atlas, mask, placements = build_atlas(items)
    repaired = np.full_like(atlas, (0, 255, 0))
    out = dict(slice_back(repaired, mask, placements))["a"]
    assert out.shape == crop.shape
    assert (out == (0, 255, 0)).all()


def test_atlas_multi_item_no_overlap():
    a = np.full((20, 30, 3), 10, np.uint8)
    b = np.full((40, 25, 3), 20, np.uint8)
    items = [AtlasItem("a", a, np.zeros((20, 30), bool)),
             AtlasItem("b", b, np.zeros((40, 25), bool))]
    atlas, mask, placements = build_atlas(items)
    assert len(placements) == 2
    (pa, pb) = placements[0], placements[1]
    ax0, ay0 = pa.x, pa.y
    bx0, by0 = pb.x, pb.y
    overlap_x = max(ax0, bx0) < min(ax0 + 30, bx0 + 25)
    overlap_y = max(ay0, by0) < min(ay0 + 20, by0 + 40)
    assert not (overlap_x and overlap_y)


def test_hidden_mask_union():
    import numpy as np

    m1 = np.zeros((10, 10), bool); m1[0:5] = True
    m2 = np.zeros((10, 10), bool); m2[5:] = True
    u = build_hidden_mask([m1, m2])
    assert u.all()


def test_background_none_keeps():
    img = np.full((20, 20, 3), 99, np.uint8)
    hole = np.zeros((20, 20), bool)
    out, degraded = repair_background(img, hole, "none", LocalProvider())
    assert (out == img).all()


def test_background_surface_fills_hole():
    img = np.full((60, 60, 3), 99, np.uint8)
    img[20:40, 20:40] = (0, 0, 0)  # 隐藏区
    hole = np.zeros((60, 60), bool)
    hole[20:40, 20:40] = True
    out, degraded = repair_background(img, hole, "surface", LocalProvider())
    assert out[30, 30].mean() > 30  # 不再是黑孔
