"""OCR 过滤规则（文档 §1.1）。"""
import numpy as np

from algorithm.ocr.engine import filter_regions, is_single_latin
from algorithm.common import OcrRegion


def region(text, conf, w, h):
    quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return OcrRegion(id="", text=text, conf=conf, quad=quad, bbox=(0, 0, w, h))


def test_min_confidence():
    kept = filter_regions([region("金币", 0.34, 40, 20), region("金币", 0.35, 40, 20)])
    assert [k.conf for k in kept] == [0.35]


def test_single_latin_icon_rejection():
    kept = filter_regions([region("A", 0.6, 12, 14), region("A", 0.85, 12, 14),
                           region("AB", 0.5, 20, 14)])
    assert len(kept) == 2
    assert kept[0].conf == 0.85 and kept[1].text == "AB"


def test_vertical_decor_ratio():
    # 低置信度且高宽比 >= 2.4 -> 剔除；高置信度竖排保留
    kept = filter_regions([region("丨", 0.5, 10, 26), region("丨", 0.9, 10, 26)])
    assert len(kept) == 1 and kept[0].conf == 0.9


def test_is_single_latin():
    assert is_single_latin("a")
    assert not is_single_latin("ab")
    assert not is_single_latin("金")
