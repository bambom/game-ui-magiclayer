"""文字样式估计（文档 §1.3）。"""
import numpy as np

from algorithm.common import OcrRegion
from algorithm.texterase.erase import erase_region
from algorithm.texterase.style import _lum, estimate_foreground, estimate_style


def make_region(w, h, text="金币：100"):
    quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return OcrRegion(id="text_001", text=text, conf=0.98, quad=quad, bbox=(0, 0, w, h))


def text_image(w, h, fg=(239, 240, 240), bg=(70, 131, 130)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bg
    # 粗笔画字形占 coverage > 17%
    img[8:40, 10:14] = fg
    img[8:40, 20:24] = fg
    img[8:40, 30:34] = fg
    img[8:40, 40:44] = fg
    return img


def test_foreground_picks_glyph_bucket():
    img = text_image(120, 48)
    r = make_region(120, 48)
    erase_region(img, r)
    fg = estimate_foreground(img, r.glyph_mask, r.local_bg)
    rgb = tuple(int(fg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    # 亮色字形被选中（每个通道均应显著亮于底色）
    assert all(c > 150 for c in rgb)


def test_weight_bold_on_high_coverage():
    img = text_image(120, 48)
    r = make_region(120, 48)
    erase_region(img, r)
    style = estimate_style(img, r)
    coverage = style["coverage"]
    assert style["weight"] == (700 if coverage >= 0.17 else 600)


def test_stroke_width_clamped():
    img = text_image(400, 120)  # 高 h -> 0.045h 超上限
    r = make_region(400, 120)
    erase_region(img, r)
    style = estimate_style(img, r)
    assert 0 <= style["strokeWidth"] <= 2


def test_stroke_dark_for_bright_fg():
    img = text_image(120, 48)
    r = make_region(120, 48)
    erase_region(img, r)
    style = estimate_style(img, r)
    fg = style["color"]
    rgb = tuple(int(fg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    if _lum(rgb) >= _lum(r.local_bg):
        assert style["stroke"] == "#1e2322"
    else:
        assert style["stroke"] == "#f0f4f1"


def test_font_mapping():
    img = text_image(120, 48)
    r = make_region(120, 48, text="金币")
    erase_region(img, r)
    assert estimate_style(img, r)["font"] == "Microsoft YaHei"
    r2 = make_region(120, 48, text="100")
    erase_region(img, r2)
    assert estimate_style(img, r2)["font"] == "Arial"


def test_size_fit_shrinks():
    # 长文本塞进小框 -> fit_to_box 生效
    img = text_image(60, 20)
    r = make_region(60, 20, text="特别长的文字内容示例")
    erase_region(img, r)
    style = estimate_style(img, r)
    assert style["fit_to_box"] is True
    assert style["size"] >= 8
