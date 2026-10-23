"""mask 清理（文档 §2.4）：闭运算 + 8 邻域连通域筛选。"""
import numpy as np

from algorithm.segment.cleanup import close_mask, filter_components


def test_close_fills_pinhole():
    m = np.ones((9, 9), bool)
    m[4, 4] = False  # 1px 针孔
    out = close_mask(m)
    assert out[4, 4]


def test_close_does_not_expand_to_bbox():
    m = np.zeros((9, 9), bool)
    m[2:7, 2:7] = True
    out = close_mask(m)
    assert not out[0, :].any() and not out[:, 0].any()


def test_keep_components_with_positive_points():
    m = np.zeros((30, 30), bool)
    m[0:5, 0:5] = True      # 小组件 A，含正点
    m[10:25, 10:25] = True  # 最大组件 B
    out = filter_components(m, [(2, 2)])
    assert out[2, 2] and out[15, 15]  # A 因正点保留，B 因最大保留


def test_keep_components_above_8pct():
    m = np.zeros((40, 40), bool)
    m[0:3, 0:3] = True       # 9px 小碎屑，无正点，< 8%
    m[10:30, 10:30] = True   # 最大 400px
    m[35:39, 0:4] = True     # 16px = 4% of 400 -> 删除
    m[0:10, 20:28] = True    # 80px = 20% of 400 -> 保留
    out = filter_components(m, [])
    assert not out[1, 1]
    assert out[20, 20]
    assert not out[36, 1]
    assert out[3, 24]


def test_no_point_component_keeps_max_only():
    m = np.zeros((30, 30), bool)
    m[0:4, 0:4] = True       # 16px
    m[10:25, 10:25] = True   # 225px 最大
    out = filter_components(m, [])
    assert not out[1, 1]
    assert out[15, 15]
