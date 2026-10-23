"""RapidOCR 封装与过滤规则（文档 §1.1）。

- 最低置信度 0.35；
- 单个拉丁字母 conf < 0.85 视为图标误识别剔除；
- 低置信度且高宽比 >= 2.4 的竖排装饰候选剔除（低置信度分界文档未定，见 config）。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .. import config
from ..common import BBox, OcrRegion

_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    return _ENGINE


def is_single_latin(text: str) -> bool:
    t = text.strip()
    return len(t) == 1 and t.isascii() and t.isalpha()


def is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
    )


def contains_cjk(text: str) -> bool:
    return any(is_cjk_char(c) for c in text)


def filter_regions(cands: List[OcrRegion]) -> List[OcrRegion]:
    kept = []
    for r in cands:
        if r.conf < config.OCR_MIN_CONF:
            continue
        if is_single_latin(r.text) and r.conf < config.SINGLE_LATIN_MIN_CONF:
            continue
        h = max(r.bbox[3] - r.bbox[1], 1)
        w = max(r.bbox[2] - r.bbox[0], 1)
        if r.conf < config.VERTICAL_DECOR_MAX_CONF and (h / w) >= config.VERTICAL_DECOR_MIN_ASPECT:
            continue
        kept.append(r)
    return kept


def extract_regions(image_rgb: np.ndarray) -> List[OcrRegion]:
    """运行 OCR 并应用过滤，返回带稳定 ID 的区域列表。"""
    result = _get_engine()(image_rgb)
    cands: List[OcrRegion] = []
    if result:
        items = result[0] if isinstance(result, tuple) else result
        for item in items or []:
            quad = np.asarray(item[0], np.float32)
            text = str(item[1])
            conf = float(item[2])
            x0, y0 = quad.min(axis=0)
            x1, y1 = quad.max(axis=0)
            bbox: BBox = (int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)))
            cands.append(OcrRegion(id="", text=text, conf=conf, quad=quad, bbox=bbox))
    kept = filter_regions(cands)
    for i, r in enumerate(kept, 1):
        r.id = f"text_{i:03d}"
    return kept
