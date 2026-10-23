"""装饰大横带的确定性纵向分割（2026-09-03 二期问题二，文档外补齐）。

planner 规则 v2 已写"每个独立簇一条查询（左山群与右树丛是两条）"，但模型不守：
run 4968ff94 的 scenery_relic_cluster 只有 1 条查询，bbox_norm
[0.052,0.088,0.893,0.165] 是一条全宽横带，且下缘把向下延伸的树冠/山脚裁掉。
按项目分工原则，规划这种概率行为不能用来保正确性——几何问题几何解：

1. 按列像素投影找簇间谷位（雾气间隙）把横带切成多簇；
2. 逐簇把 bbox 下缘外扩后重跑一次 SAM，把被横带裁掉的向下延伸收回来
   （新 mask 必须仍盖住原簇且没吞进背景，否则保留原簇——SAM 已切出的
   部分不许丢）；
3. 原实例换成 `<query_id>#s1..sK` 多实例，后续 bbox/镂空/导出全走既有代码。
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .. import config, runlog
from ..common import Hint, Instance, bbox_of_mask

_K3 = np.ones((3, 3), np.uint8)


def _is_wide_band(inst: Instance, width: int) -> bool:
    """该实例是否是"装饰大横带"：装饰类 + 占宽够 + 够扁。"""
    if inst.query.kind != "decoration" and inst.query.role != "decoration":
        return False
    box = bbox_of_mask(inst.mask)
    if box is None:
        return False
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return (bw >= config.SPLIT_MIN_WIDTH_RATIO * width
            and bw >= config.SPLIT_MIN_ASPECT * max(bh, 1))


def _valley_cuts(mask: np.ndarray, box) -> List[int]:
    """按列投影找谷位切点（全图 x 坐标）。无谷返回空表。

    投影先做一维高斯平滑（sigma 取带宽的 1/200 量级，抹掉单列噪声但保住
    十几像素宽的雾气间隙），谷 = 连续的"投影低于正列中位数 ratio 倍"区间，
    且宽度够（窄缺口是树冠之间的透光，不是簇边界）。切点取谷中点。
    """
    x0, _y0, x1, _y1 = box
    proj = mask[:, x0:x1].sum(axis=0).astype(np.float32)
    sigma = max((x1 - x0) / 200.0, 1.0)
    ksize = int(sigma * 6) | 1
    proj = cv2.GaussianBlur(proj.reshape(1, -1), (ksize, 1), sigma).ravel()
    positive = proj[proj > 0]
    if positive.size == 0:
        return []
    thr = config.SPLIT_VALLEY_RATIO * float(np.median(positive))
    low = proj <= thr
    cuts: List[int] = []
    start = None
    for i, is_low in enumerate(low):
        if is_low and start is None:
            start = i
        elif not is_low and start is not None:
            if i - start >= config.SPLIT_VALLEY_MIN_W and start > 0:
                cuts.append(x0 + (start + i) // 2)
            start = None
    # 尾部低谷贴着 bbox 右缘，不是簇间间隙（bbox 本就切在内容上），忽略
    return cuts


def _clusters_from_cuts(mask: np.ndarray, box, cuts: List[int]) -> List[np.ndarray]:
    """按切点把 mask 切成多簇（各簇保留自身连通域，丢弃碎片）。"""
    x0, _y0, x1, _y1 = box
    bounds = [x0] + cuts + [x1]
    out: List[np.ndarray] = []
    for left, right in zip(bounds, bounds[1:]):
        piece = np.zeros_like(mask)
        piece[:, left:right] = mask[:, left:right]
        if int(piece.sum()) >= config.SPLIT_MIN_CLUSTER_PX:
            out.append(piece)
    return out


def _extend_hint(cluster: np.ndarray, base: Hint, width: int, height: int
                 ) -> Optional[Hint]:
    """簇的下扩提示：bbox 下缘外扩簇高的 EXTEND_DOWN 倍，左右各 4px。

    正点取簇质心与"最高列"的中点：质心可能落在两片树冠之间的空隙上，最高
    列（该簇像素最多的列）的中点必在实体上。
    """
    box = bbox_of_mask(cluster)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    ch = y1 - y0
    ex0 = max(x0 - 4, 0)
    ex1 = min(x1 + 4, width)
    ey1 = min(y1 + int(round(config.SPLIT_EXTEND_DOWN * ch)), height)
    ys, xs = np.nonzero(cluster)
    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))
    points = []
    if cluster[cy, cx]:
        points.append((cx, cy))
    col = x0 + int(np.argmax(cluster[:, x0:x1].sum(axis=0)))
    col_ys = np.nonzero(cluster[:, col])[0]
    if len(col_ys):
        points.append((col, int(col_ys[len(col_ys) // 2])))
    if not points:
        return None
    return Hint(bbox=(ex0, y0, ex1, ey1),
                positive_points=points,
                negative_points=list(base.negative_points),
                bbox_norm=[ex0 / width, y0 / height, ex1 / width, ey1 / height],
                positive_points_norm=[[x / max(width - 1, 1), y / max(height - 1, 1)]
                                      for x, y in points],
                negative_points_norm=list(base.negative_points_norm))


def _regrow(cluster: np.ndarray, hint: Hint, segmenter, width: int,
            height: int, label: str) -> np.ndarray:
    """下扩重跑一次 SAM 收回向下延伸；不合格则原样返回簇 mask。"""
    try:
        grown, _detail = segmenter.segment_hint(hint, width, height)
    except Exception as e:                      # 重跑失败不该拖垮分割
        runlog.event("WARN", f"{label} 下扩重跑失败，保留原簇：{e}")
        return cluster
    if grown is None or not grown.any():
        return cluster
    area = float(cluster.sum())
    cover = float((grown & cluster).sum()) / max(area, 1.0)
    growth = float(grown.sum()) / max(area, 1.0)
    if cover < config.SPLIT_KEEP_MIN_COVER:
        runlog.event("LAYER", f"{label} 下扩重跑覆盖率 {cover:.2f} 不足，保留原簇")
        return cluster
    if growth > config.SPLIT_KEEP_MAX_GROWTH:
        runlog.event("LAYER", f"{label} 下扩重跑面积涨 {growth:.2f}x 疑似吞背景，"
                              f"保留原簇")
        return cluster
    # 并上原簇：重跑可能在某处收窄，SAM 已切出的部分不许丢
    out = grown | cluster
    runlog.event("LAYER", f"{label} 下扩重跑收回延伸：面积 {int(area)}->"
                          f"{int(out.sum())} bbox={bbox_of_mask(out)}")
    return out


def split_wide_decorations(instances: List[Instance], segmenter, width: int,
                           height: int) -> List[Instance]:
    """把装饰大横带实例替换成逐簇实例。无横带或切不出多簇时原表返回。"""
    if config.SPLIT_DISABLE:
        return instances
    out: List[Instance] = []
    for inst in instances:
        if inst.mask is None or not inst.mask.any() or not _is_wide_band(inst, width):
            out.append(inst)
            continue
        box = bbox_of_mask(inst.mask)
        cuts = _valley_cuts(inst.mask, box)
        clusters = _clusters_from_cuts(inst.mask, box, cuts) if cuts else []
        if len(clusters) < 2:
            runlog.event("LAYER", f"{inst.instance_id}（{inst.query.name}）"
                                  f"横带未找到簇间谷位（谷 {len(cuts)} 处），不分割")
            out.append(inst)
            continue
        runlog.event("LAYER", f"{inst.instance_id}（{inst.query.name}）横带分割 "
                              f"{len(clusters)} 簇，切点 x={cuts}")
        for i, cluster in enumerate(clusters, 1):
            label = f"{inst.query.id}#s{i}"
            hint = _extend_hint(cluster, inst.hint, width, height)
            mask = cluster if hint is None else _regrow(
                cluster, hint, segmenter, width, height, label)
            piece = Instance(instance_id=label, query=inst.query,
                             hint=hint or inst.hint, mask=mask)
            piece.bbox = bbox_of_mask(mask)
            runlog.event("LAYER", f"{label} 簇 bbox={piece.bbox} "
                                  f"面积={int(mask.sum())}")
            out.append(piece)
    return out
