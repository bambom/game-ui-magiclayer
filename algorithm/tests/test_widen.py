"""环形 mask 外扩（框太细）：只扩有开口的环、开口尺寸不变、三种算法路径。"""
import importlib

import numpy as np
import pytest

from algorithm import config
from algorithm.segment.widen import widen_hollow_mask


def _reload_config(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def restore_config():
    yield
    importlib.reload(config)  # 丢掉单测注入的 env 覆盖，恢复默认


def _ring(h=140, w=180, x0=20, y0=20, x1=160, y1=120, band=6):
    """矩形环：外框 [x0,y0,x1,y1)，四边条宽 band。"""
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    m[y0 + band:y1 - band, x0 + band:x1 - band] = False
    return m


def _holes_of(mask):
    """外轮廓填充 − mask，即开口区域（用于断言开口不被吃掉）。"""
    import cv2
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, found[-2], -1, 1, -1)
    return filled.astype(bool) & ~mask


def test_off_returns_mask_unchanged(monkeypatch):
    _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="off")
    m = _ring()
    out, info = widen_hollow_mask(m)
    assert out is m and not info["applied"]


def test_fixed_widens_outward_only(monkeypatch):
    _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="fixed",
                   MAGICLAYER_MASK_WIDEN_PX="3")
    m = _ring(band=6)
    out, info = widen_hollow_mask(m)
    assert info["applied"] and info["method"] == "fixed" and info["px"] == 3
    # 外边界向外推 3px，开口原样保留
    assert _holes_of(out).sum() == _holes_of(m).sum()
    ys, xs = np.where(out)
    assert (xs.min(), ys.min()) == (20 - 3, 20 - 3)
    assert (xs.max(), ys.max()) == (159 + 3, 119 + 3)
    # 外扩后 mask ⊇ 原 mask（只增不减）
    assert (out & m).sum() == m.sum()


def test_adaptive_derives_px_from_band(monkeypatch):
    _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="adaptive",
                   MAGICLAYER_MASK_WIDEN_RATIO="1.0",
                   MAGICLAYER_MASK_WIDEN_MAX_PX="12")
    m = _ring(band=6)
    out, info = widen_hollow_mask(m)
    # 环带 6px → ratio 1.0 → 外扩 6px（未触上限）
    assert info["applied"] and info["band"] == pytest.approx(6.0)
    assert info["px"] == 6 and info["method"] == "adaptive"
    assert _holes_of(out).sum() == _holes_of(m).sum()
    ys, xs = np.where(out)
    assert (xs.min(), ys.min()) == (14, 14) and (xs.max(), ys.max()) == (165, 125)


def test_adaptive_clamped_by_max_px(monkeypatch):
    _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="adaptive",
                   MAGICLAYER_MASK_WIDEN_RATIO="1.0",
                   MAGICLAYER_MASK_WIDEN_MAX_PX="4")
    m = _ring(band=40)  # 环带 40px，比例外扩本应 40px
    out, info = widen_hollow_mask(m)
    assert info["px"] == 4
    assert _holes_of(out).sum() == _holes_of(m).sum()


def test_solid_mask_without_hole_untouched(monkeypatch):
    _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="adaptive")
    m = np.zeros((140, 180), bool)
    m[20:120, 20:160] = True  # 实心矩形，无开口
    out, info = widen_hollow_mask(m)
    assert not info["applied"] and (out == m).all()


def test_solid_parent_widens_by_children_preview(monkeypatch):
    """实心父框：子 mask 预演镂空量环带，只推外边界；子区域镂空归 elements.py。"""
    _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="adaptive",
                   MAGICLAYER_MASK_WIDEN_RATIO="1.0",
                   MAGICLAYER_MASK_WIDEN_MAX_PX="12")
    m = np.zeros((140, 180), bool)
    m[20:120, 20:160] = True  # 实心框（无自身开口，开口靠后续镂空）
    child = np.zeros((140, 180), bool)
    child[26:114, 26:154] = True  # 子实例占满开口 → 预演环带 6px
    out, info = widen_hollow_mask(m, [child])
    assert info["applied"] and info["method"] == "adaptive" and info["px"] == 6
    ys, xs = np.where(out)
    assert (xs.min(), ys.min()) == (14, 14) and (xs.max(), ys.max()) == (165, 125)
    # 外扩只推外边界：mask ⊇ 原实心框，子区域不在此抠洞
    assert (out & m).sum() == m.sum()
    assert out[26:114, 26:154].all()


def test_small_ring_untouched(monkeypatch):
    _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="fixed",
                   MAGICLAYER_MASK_WIDEN_PX="3")
    m = _ring(h=60, w=80, x0=30, y0=25, x1=50, y1=40, band=3)  # bbox 20x15 < 16 最小边
    out, info = widen_hollow_mask(m)
    assert not info["applied"] and (out == m).all()


def test_tiny_hole_ring_untouched(monkeypatch):
    _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="adaptive")
    m = np.zeros((140, 180), bool)
    m[20:120, 20:160] = True
    m[60:63, 80:83] = False  # 9px 针孔 < 64 阈值
    out, info = widen_hollow_mask(m)
    assert not info["applied"] and (out == m).all()


def test_pipeline_settings_env_registers(monkeypatch):
    """面板 env 链：MAGICLAYER_MASK_WIDEN_* 能覆盖四个常数。"""
    module = _reload_config(monkeypatch,
                            MAGICLAYER_MASK_WIDEN_METHOD="fixed",
                            MAGICLAYER_MASK_WIDEN_PX="5",
                            MAGICLAYER_MASK_WIDEN_RATIO="0.5",
                            MAGICLAYER_MASK_WIDEN_MAX_PX="8")
    assert module.MASK_WIDEN_METHOD == "fixed"
    assert module.MASK_WIDEN_PX == 5
    assert module.MASK_WIDEN_RATIO == 0.5
    assert module.MASK_WIDEN_MAX_PX == 8


def test_bad_method_falls_back_to_default(monkeypatch):
    module = _reload_config(monkeypatch, MAGICLAYER_MASK_WIDEN_METHOD="bogus")
    assert module.MASK_WIDEN_METHOD == "adaptive"
