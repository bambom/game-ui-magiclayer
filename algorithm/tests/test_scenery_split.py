"""装饰大横带的确定性纵向分割（2026-09-03 二期问题二）。

planner 把左山群与右树丛合成一条全宽横带、且裁掉向下延伸时的几何兜底：
按列投影谷位切簇 + 逐簇下扩重跑 SAM 收回延伸（重跑不合格保留原簇）。
"""
import numpy as np

from algorithm.common import Hint, Instance, Query, bbox_of_mask
from algorithm.segment.split import split_wide_decorations

W, H = 800, 600


def _q(kind="decoration", role="decoration"):
    return Query(id="scenery", name="scenery cluster", kind=kind, role=role,
                 element_repair_mode="none", parent_query_id=None, z_order=1)


def _hint():
    return Hint(bbox=(40, 40, 760, 180), positive_points=[(400, 100)],
                negative_points=[], bbox_norm=[0.05, 0.067, 0.95, 0.3],
                positive_points_norm=[[0.5, 0.167]], negative_points_norm=[])


def _band(gap=(360, 400)):
    """全宽横带：左右两簇 + 中间一段雾气间隙（无像素）。"""
    m = np.zeros((H, W), bool)
    m[60:170, 50:gap[0]] = True
    m[60:170, gap[1]:750] = True
    return m


def _inst(mask, kind="decoration", role="decoration"):
    inst = Instance(instance_id="scenery", query=_q(kind, role),
                    hint=_hint(), mask=mask)
    inst.bbox = bbox_of_mask(mask)
    return inst


class _NullSeg:
    """不改变结果的分割器：重跑返回空，逐簇 mask 保持原样。"""

    calls = 0

    def segment_hint(self, hint, width, height):
        type(self).calls += 1
        return np.zeros((height, width), bool), {}


class _ExtendSeg:
    """重跑把簇向下延伸一截（树冠/山脚被横带裁掉的部分）。"""

    def __init__(self, extend=90):
        self.extend = extend
        self.hints = []

    def segment_hint(self, hint, width, height):
        self.hints.append(hint)
        x0, y0, x1, y1 = hint.bbox
        m = np.zeros((height, width), bool)
        m[y0:min(y0 + 110 + self.extend, y1), x0 + 4:x1 - 4] = True
        return m, {}


class _GreedySeg:
    """重跑吞进大片背景：应被面积上限拒绝。"""

    def segment_hint(self, hint, width, height):
        m = np.zeros((height, width), bool)
        m[:, :] = True
        return m, {}


def test_splits_wide_band_at_mist_gap():
    out = split_wide_decorations([_inst(_band())], _NullSeg(), W, H)
    assert [i.instance_id for i in out] == ["scenery#s1", "scenery#s2"]
    b1, b2 = out[0].bbox, out[1].bbox
    assert b1[2] <= 380 and b2[0] >= 380      # 各自 bbox 不含雾气间隙
    assert out[0].query is out[1].query        # 同一 query，只是多实例


def test_regrow_recovers_downward_extent():
    seg = _ExtendSeg()
    out = split_wide_decorations([_inst(_band())], seg, W, H)
    assert len(out) == 2
    for inst in out:
        assert inst.bbox[3] > 170              # 下缘越过原横带的 170
    # 下扩提示确实把 bbox 下缘外推了簇高的 1 倍（110 -> +110）
    assert seg.hints[0].bbox[3] >= 170 + 100


def test_regrow_rejects_background_grab():
    out = split_wide_decorations([_inst(_band())], _GreedySeg(), W, H)
    assert len(out) == 2
    for inst in out:
        assert int(inst.mask.sum()) < 60000    # 保留原簇，没吞下整图
        assert inst.bbox[3] <= 170


def test_no_split_without_valley():
    """连续横带（无间隙）：找不到谷位，原实例不动。"""
    solid = np.zeros((H, W), bool)
    solid[60:170, 50:750] = True
    inst = _inst(solid)
    out = split_wide_decorations([inst], _NullSeg(), W, H)
    assert out == [inst]


def test_narrow_gap_is_not_a_cluster_boundary():
    """树冠间 4px 透光缺口不算簇边界（谷宽下限）。"""
    m = np.zeros((H, W), bool)
    m[60:170, 50:400] = True
    m[60:170, 404:750] = True
    inst = _inst(m)
    assert split_wide_decorations([inst], _NullSeg(), W, H) == [inst]


def test_skips_non_decoration_and_non_band():
    """非装饰类、或不够扁的装饰，都不参与分割。"""
    card = _inst(_band(), kind="card", role="surface")
    assert split_wide_decorations([card], _NullSeg(), W, H) == [card]
    tall = np.zeros((H, W), bool)
    tall[60:400, 50:400] = True
    tall[60:400, 440:750] = True               # 有谷位但宽高比不足
    blocky = _inst(tall)
    assert split_wide_decorations([blocky], _NullSeg(), W, H) == [blocky]


def test_disable_switch(monkeypatch):
    from algorithm import config

    monkeypatch.setattr(config, "SPLIT_DISABLE", 1)
    inst = _inst(_band())
    assert split_wide_decorations([inst], _NullSeg(), W, H) == [inst]
