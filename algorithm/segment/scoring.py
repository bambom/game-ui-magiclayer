"""三候选评分（文档 §2.3）：

总分 = SAM模型分 + 0.22 x 正点命中率 - 0.38 x 负点命中率
     + 0.10 x 框内包含度 + 0.06 x 外接框IoU

文档示例验证：0.9833+0.22+0.09956+0.05868 = 1.3615 ✓
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .. import config
from ..common import BBox, bbox_of_mask, iou

POS_WEIGHT = config.SCORE_POS_WEIGHT
NEG_WEIGHT = config.SCORE_NEG_WEIGHT
CONTAIN_WEIGHT = config.SCORE_CONTAIN_WEIGHT
IOU_WEIGHT = config.SCORE_IOU_WEIGHT


def positive_hit(mask: np.ndarray, points: Sequence[Tuple[int, int]]) -> float:
    if not points:
        return 1.0
    h, w = mask.shape
    hits = sum(1 for x, y in points
               if 0 <= x < w and 0 <= y < h and mask[y, x])
    return hits / float(len(points))


def containment_ratio(mask: np.ndarray, box: BBox) -> float:
    """候选 mask 位于提示框内的比例。"""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    x0, y0, x1, y1 = box
    inside = mask[y0:y1, x0:x1].sum()
    return float(inside) / float(total)


def total_score(sam_score: float, pos_hit: float, neg_hit: float,
                containment: float, box_iou: float) -> float:
    return (sam_score
            + POS_WEIGHT * pos_hit - NEG_WEIGHT * neg_hit
            + CONTAIN_WEIGHT * containment + IOU_WEIGHT * box_iou)


def score_candidate(mask: np.ndarray, sam_score: float, box: BBox,
                    positive_points: Sequence[Tuple[int, int]],
                    negative_points: Sequence[Tuple[int, int]]) -> dict:
    pos = positive_hit(mask, positive_points)
    neg = positive_hit(mask, negative_points)
    contain = containment_ratio(mask, box)
    m_bbox = bbox_of_mask(mask) or (0, 0, 0, 0)
    box_iou = iou(m_bbox, box)
    return {
        "total": float(total_score(sam_score, pos, neg, contain, box_iou)),
        "sam_score": float(sam_score),
        "positive_hit": float(pos),
        "negative_hit": float(neg),
        "containment": float(contain),
        "bbox_iou": float(box_iou),
    }


def pick_best(masks: List[np.ndarray], scores: Sequence[float], box: BBox,
              positive_points, negative_points) -> Tuple[np.ndarray, dict]:
    """返回总分最高的候选 mask 及其评分明细。"""
    assert len(masks) == len(scores) >= 1
    details = [
        score_candidate(m, s, box, positive_points, negative_points)
        for m, s in zip(masks, scores)
    ]
    best = max(range(len(masks)), key=lambda i: details[i]["total"])
    details[best]["chosen"] = True
    return masks[best], details[best]
