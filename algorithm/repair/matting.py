"""半透明区域的确定性 alpha 求解（2026-09-03 二期问题四，文档外补齐）。

导出层 alpha 一直是 SAM 二值 mask（0/255）：植被雾化边缘、玻璃条、光晕被硬
切，叠回修补背景时边缘生硬。SAM 系（含 SAM2）本质都不产连续 alpha，靠模型
出透明图又要赌网关是否透传 background=transparent——但这问题根本不用赌：

观测 I = α·F + (1−α)·B。B（元素背后的背景）可由边界低频延续估出，F（前景色）
对雾/玻璃/光晕这类近均匀前景可由邻域前景均值估出，于是 α 是最小二乘闭式解

    α = ((I−B)·(F−B)) / |F−B|²        逐像素三通道点积，clamp 到 [0,1]

只在 trimap 的未知带内解（确定前景/背景直接给 255/0），|F−B| 太小的像素数值
上不可解，回退二值 mask。毫秒级，无外部依赖，符合"几何/统计问题确定性解"
的分工原则。
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .background import _lowfreq_continuation

_K3 = np.ones((3, 3), np.uint8)


def trimap_width(mask: np.ndarray) -> int:
    """未知带宽度 = clamp(RATIO·min(bbox 边), MIN, MAX)。"""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return config.MAT_TRIMAP_MIN
    side = min(int(xs.max()) - int(xs.min()) + 1,
               int(ys.max()) - int(ys.min()) + 1)
    return int(np.clip(round(config.MAT_TRIMAP_RATIO * side),
                       config.MAT_TRIMAP_MIN, config.MAT_TRIMAP_MAX))


def edge_softness(crop_rgb: np.ndarray, mask: np.ndarray) -> float:
    """边界过渡的等效宽度（px）：跨界总色差 / 峰值斜率。

    梯度幅值本身不能判软硬——它是「对比度 / 过渡宽度」，同一个 8px 雾边在
    高对比前景上梯度比低对比硬边还大。必须除掉对比度。线性斜坡的 Sobel-3
    响应 = 8·斜率，故 等效宽度 = 8·C / 峰值梯度。

    峰值取带内 p90 而非中位数：中位数在带宽固定时随过渡宽度非单调（实测
    1/2/4/8px 斜坡依次 99/151/188/109，硬边 77——完全不可用），p90 才是
    「最陡处斜率」。实测本式给出 1px 斜坡 2.0、2px 2.35、4px 4.0、8px 7.9、
    硬阶跃 2.0，单调可分。

    C 在带外采样（带内像素本身处于混色中）：前景取 erode(w) 的灰度中位数，
    背景取 dilate(2w) − dilate(w) 环带的中位数。C 不足 MAT_MIN_CONTRAST 时
    α 数值上本就不可解，直接判硬边。
    """
    m8 = mask.astype(np.uint8)
    band = (cv2.dilate(m8, _K3, iterations=2).astype(bool)
            & ~cv2.erode(m8, _K3, iterations=2).astype(bool))
    if not band.any():
        return 0.0
    w = trimap_width(mask)
    fg = cv2.erode(m8, _K3, iterations=w).astype(bool)
    near = cv2.dilate(m8, _K3, iterations=2 * w).astype(bool)
    bg = near & ~cv2.dilate(m8, _K3, iterations=w).astype(bool)
    if not fg.any() or not bg.any():
        return 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    contrast = abs(float(np.median(gray[fg])) - float(np.median(gray[bg])))
    if contrast < config.MAT_MIN_CONTRAST:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    peak = float(np.percentile(np.sqrt(gx * gx + gy * gy)[band], 90))
    return 8.0 * contrast / max(peak, 1e-6)


def is_soft_edge(crop_rgb: np.ndarray, mask: np.ndarray) -> bool:
    """过渡宽度达门槛即"软边"（雾化/玻璃/光晕）。

    硬边元素（卡框、按钮）过渡只有 1-2px——对它们求 α 只会把干净的直边啃出
    毛边，交给 1px 羽化就够。
    """
    return edge_softness(crop_rgb, mask) >= config.MAT_SOFT_EDGE_WIDTH


def _local_mean(crop: np.ndarray, where: np.ndarray, ksize: int) -> np.ndarray:
    """where 内像素的局部均值场（盒滤波加权除以计数，空邻域回落全局均值）。

    近均匀前景（雾/玻璃/光晕）的 F 用邻域均值估计就够，比 inpaint 快一个量级
    且不引入放射状伪结构。
    """
    k = (max(ksize, 3) | 1,) * 2
    sel = where.astype(np.float32)
    acc = cv2.blur(crop.astype(np.float32) * sel[..., None], k)
    cnt = cv2.blur(sel, k)[..., None]
    fallback = (crop[where].astype(np.float32).mean(axis=0) if where.any()
                else np.zeros(3, np.float32))
    out = np.where(cnt > 1e-4, acc / np.maximum(cnt, 1e-4), fallback)
    return out.astype(np.float32)


def solve_alpha(crop_rgb: np.ndarray, mask: np.ndarray) -> tuple:
    """返回 (alpha_u8, fg_rgb, info)。

    alpha：确定前景 255、确定背景 0、未知带内为闭式解；不可解处回退二值。
    fg_rgb：未知带内用 F 估计替换（去掉背景混色），其余保持原像素——否则
    半透明边缘会把背景色一起带进素材，移动图层后露馅。
    """
    info = {"applied": False, "unknown_px": 0, "solved_px": 0, "width": 0}
    if config.MATTING_DISABLE or mask is None or not mask.any():
        return (mask.astype(np.uint8) * 255 if mask is not None else None,
                crop_rgb, info)
    width = trimap_width(mask)
    m8 = mask.astype(np.uint8)
    fg = cv2.erode(m8, _K3, iterations=width).astype(bool)
    bg = ~cv2.dilate(m8, _K3, iterations=width).astype(bool)
    unknown = ~fg & ~bg
    info["width"] = width
    info["unknown_px"] = int(unknown.sum())
    if not unknown.any() or not fg.any():
        return mask.astype(np.uint8) * 255, crop_rgb, info

    obs = crop_rgb.astype(np.float32)
    # B：元素背后的背景——边界低频延续（把整个元素连未知带一起当洞挖掉采样，
    # 否则半透明边缘的混色会被当成背景）
    bg_field = _lowfreq_continuation(
        crop_rgb, cv2.dilate(m8, _K3, iterations=width).astype(bool)
    ).astype(np.float32)
    # F：邻域前景均值（窗口取未知带宽的 4 倍，覆盖到确定前景侧）
    fg_field = _local_mean(crop_rgb, fg, 4 * width)

    diff = fg_field - bg_field
    den = (diff * diff).sum(axis=2)
    num = ((obs - bg_field) * diff).sum(axis=2)
    solved = np.clip(num / np.maximum(den, 1e-6), 0.0, 1.0)
    solvable = unknown & (np.sqrt(den) >= config.MAT_MIN_CONTRAST)
    info["solved_px"] = int(solvable.sum())
    if not solvable.any():
        return mask.astype(np.uint8) * 255, crop_rgb, info

    alpha = mask.astype(np.float32)          # 不可解的未知像素回退二值
    alpha[solvable] = solved[solvable]
    # 未知带内中值滤波去噪（3x3，只回写未知带，别动确定区）
    smooth = cv2.medianBlur((alpha * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    alpha[solvable] = smooth[solvable]
    alpha[fg] = 1.0
    alpha[bg] = 0.0

    out_rgb = crop_rgb.copy()
    out_rgb[solvable] = np.clip(np.round(fg_field[solvable]), 0, 255).astype(np.uint8)
    info["applied"] = True
    return np.clip(np.round(alpha * 255), 0, 255).astype(np.uint8), out_rgb, info


def feather_binary(mask: np.ndarray) -> np.ndarray:
    """二值 alpha 的 1px 羽化（未走 matting 的层用）：只动边界那一圈。

    治标但便宜：0/255 硬边界在导出 PNG 上就是锯齿，1px 渐变能明显缓解视感，
    又不像 matting 那样需要背景估计。
    """
    binary = mask.astype(np.uint8) * 255
    sigma = float(config.EDGE_FEATHER_SIGMA)
    if sigma <= 0 or not mask.any():
        return binary
    m8 = mask.astype(np.uint8)
    ring = (cv2.dilate(m8, _K3).astype(bool) & ~cv2.erode(m8, _K3).astype(bool))
    blur = cv2.GaussianBlur(binary, (0, 0), sigma)
    out = binary.copy()
    out[ring] = blur[ring]
    return out
