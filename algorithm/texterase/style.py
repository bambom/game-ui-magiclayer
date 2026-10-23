"""文字样式估计与渲染（文档 §1.3）。

阴影参数：文档未给出从框高 h 推导的通用公式（只有一个示例值），
按文档 §7 的约束不从示例臆造，输出中不含阴影字段。
"""
from __future__ import annotations

import platform
from typing import Optional, Tuple

import numpy as np

from .. import config
from ..common import OcrRegion, clamp
from ..ocr.engine import contains_cjk, is_cjk_char

_FONT_FILES = {
    ("Windows", config.FONT_CJK): "msyh.ttc",
    ("Windows", config.FONT_LATIN): "arial.ttf",
    ("Darwin", config.FONT_CJK): "/System/Library/Fonts/PingFang.ttc",
    ("Darwin", config.FONT_LATIN): "/System/Library/Fonts/Helvetica.ttc",
}


def _lum(rgb: Tuple[float, float, float]) -> float:
    """Rec.709 亮度。"""
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hex(rgb: Tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c)) for c in rgb))


def _load_font(size: int, family: str):
    from PIL import ImageFont

    key = (platform.system(), family)
    name = _FONT_FILES.get(key)
    if name is None:  # Linux 等其他平台的可用字体映射
        name = "msyh.ttc" if family == config.FONT_CJK else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _measure(text: str, font, stroke_width: int) -> Tuple[int, int]:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return r - l, b - t


def estimate_foreground(crop_rgb: np.ndarray, glyph_mask: np.ndarray,
                        local_bg: Tuple[float, float, float]) -> str:
    """前景色：字形像素按 32 量化到桶，桶得分 = 像素数 x (24 + 与底色距离)。"""
    pixels = crop_rgb[glyph_mask]
    if len(pixels) == 0:
        return "#000000"
    q = (pixels // config.STYLE_QUANT_STEP).astype(np.int32) * config.STYLE_QUANT_STEP + 16
    q = np.clip(q, 0, 255)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    bg = np.asarray(local_bg, np.float32)
    best_key, best_score = None, -1.0
    for key, count in zip(keys, counts):
        dist = float(np.linalg.norm(key.astype(np.float32) - bg))
        score = float(count) * (config.STYLE_BUCKET_BASE_SCORE + dist)
        if score > best_score:
            best_score, best_key = score, key
    return _hex(tuple(best_key))


def estimate_style(image_rgb: np.ndarray, region: OcrRegion) -> dict:
    """按文档 §1.3 估计颜色/字体/字号/字重/描边。"""
    x0, y0, x1, y1 = region.bbox
    w, h = x1 - x0, y1 - y0
    crop = image_rgb[y0:y1, x0:x1]
    glyph = region.glyph_mask if region.glyph_mask is not None else np.zeros(crop.shape[:2], bool)
    local_bg = region.local_bg if region.local_bg is not None else (0.0, 0.0, 0.0)

    fg_hex = estimate_foreground(crop, glyph, local_bg)

    family = config.FONT_CJK if contains_cjk(region.text) else config.FONT_LATIN

    text = region.text.strip()
    single_cjk = len(text) == 1 and is_cjk_char(text)
    size = int(round((config.STYLE_SIZE_CJK if single_cjk else config.STYLE_SIZE_OTHER) * h))
    size = max(size, config.STYLE_SIZE_MIN)

    coverage = float(glyph.sum()) / max(float(w * h), 1.0)
    weight = config.STYLE_WEIGHT_BOLD if coverage >= config.STYLE_WEIGHT_COVERAGE else config.STYLE_WEIGHT_NORMAL

    stroke_w = round(config.STYLE_STROKE_WIDE * h if coverage > config.STYLE_STROKE_COVERAGE
                     else config.STYLE_STROKE_NARROW * h)
    stroke_w = int(clamp(stroke_w, 0, config.STYLE_STROKE_MAX))

    stroke_hex = (config.STYLE_STROKE_DARK if _lum(_hex_to_rgb(fg_hex)) >= _lum(local_bg)
                  else config.STYLE_STROKE_LIGHT)

    # 字号逐像素收缩到完整放入 OCR 框（含描边测量）
    fit_to_box = False
    while size > config.STYLE_SIZE_MIN:
        fw, fh = _measure(text, _load_font(size, family), stroke_w)
        if fw <= w and fh <= h:
            break
        size -= 1
        fit_to_box = True

    return {
        "font": family,
        "size": size,
        "weight": weight,
        "color": fg_hex,
        "stroke": stroke_hex,
        "strokeWidth": stroke_w,
        "shadow": None,  # 文档未定义通用公式，不从示例臆造
        "coverage": round(coverage, 4),
        "fit_to_box": fit_to_box,
        "mask_mode": region.mask_mode,
    }


def _hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def render_text_png(text: str, style: dict) -> Optional[np.ndarray]:
    """按估计样式渲染 RGBA 预览图（调试/组合预览用）。"""
    from PIL import Image, ImageDraw

    family, size = style["font"], style["size"]
    stroke_w = int(style.get("strokeWidth") or 0)
    font = _load_font(size, family)
    tw, th = _measure(text, font, stroke_w)
    pad = 2
    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fg = tuple(int(style["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    stroke = tuple(int(style["stroke"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    draw.text((pad, pad), text, font=font, fill=fg, stroke_width=stroke_w, stroke_fill=stroke)
    return np.asarray(img)
