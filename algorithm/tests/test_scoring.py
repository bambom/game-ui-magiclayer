"""三候选评分公式（文档 §2.3 示例表逐值验证）。"""
from algorithm.segment.scoring import (containment_ratio, positive_hit,
                                       score_candidate, total_score)


def test_doc_example_scores():
    # 文档 §2.3 表格：候选 1/2/3 的输入与总分
    assert abs(total_score(0.9833, 1.0, 0.0, 0.9956, 0.9780) - 1.3615) < 5e-5
    assert abs(total_score(0.9565, 1.0, 0.0, 0.9958, 0.9735) - 1.3345) < 5e-5
    assert abs(total_score(0.9050, 1.0, 0.0, 0.9966, 0.9798) - 1.2835) < 5e-4


def test_positive_hit_and_negatives():
    import numpy as np

    mask = np.zeros((10, 10), bool)
    mask[2:8, 2:8] = True
    assert positive_hit(mask, [(3, 3), (5, 5)]) == 1.0
    assert positive_hit(mask, [(0, 0), (5, 5)]) == 0.5
    assert positive_hit(mask, []) == 1.0
    assert positive_hit(mask, [(9, 9)]) == 0.0  # 越界点不算命中


def test_containment_and_iou():
    import numpy as np

    mask = np.zeros((10, 10), bool)
    mask[0:5, 0:10] = True  # 50 像素，25 在框内
    box = (0, 0, 5, 5)
    assert containment_ratio(mask, box) == 0.5


def test_score_candidate_prefers_higher_sam_score():
    import numpy as np

    mask = np.zeros((10, 10), bool)
    mask[1:9, 1:9] = True
    box = (1, 1, 9, 9)
    pts = [(4, 4)]
    a = score_candidate(mask, 0.98, box, pts, [])
    b = score_candidate(mask, 0.90, box, pts, [])
    assert a["total"] > b["total"]
    assert a["positive_hit"] == 1.0
    assert a["containment"] == 1.0
    assert a["bbox_iou"] == 1.0
