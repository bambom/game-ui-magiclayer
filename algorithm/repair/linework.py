"""Atlas 洞区修补的确定性后处理（2026-09-03，run 4968ff94 遗留问题二）。

生图模型在洞内的两个系统性毛病由毫秒级几何操作兜底，不靠提示词概率：
1. 洞填色场不接边界——时间轴节点洞被补成比纸更亮的白盘。按洞缘内外的
   中位色差把填充的低频场平移回边界统计（relevel），纹理按周边/洞内高频
   能量比缩放（背景低频锚定的元素版，元素保留自身渐变所以用"平移"而非
   整体替换颜色场）。
2. 细长结构（时间轴双线、进度轨道）穿洞被拦断。洞两侧都探测到同向细线
   且线心对齐时，按两侧横剖面把线续接过洞（reconnect）；只有一侧有线可能
   是结构端点，不续接。

二期（2026-09-03，val-postproc-3 遗留问题一）再补两件，都发生在洞缘：
3. 洞外还贴着被抠元素的光晕/外环残留（弧斑），子 mask 膨胀没吃到——被洞外
   环带关住的偏离连通域并入洞（_absorb_halo）；
4. 续接线在洞上下口与原生线段有粗细/深浅跳变——剖面章从"两侧合并中位"改成
   两侧剖面逐行线性插值，把渐变接回去。
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config

_K3 = np.ones((3, 3), np.uint8)

_RELEVEL_BAND_PX = 2        # 洞缘内外统计带宽度
_LINE_BAND_PX = 12          # 续接：洞外采样带半宽
_LINE_PROFILE_HALF = 10     # 续接：横剖面半宽
_LINE_ALIGN_TOL_PX = 8      # 两侧线心允许错位
_LINE_MIN_ROW_RATIO = 0.6   # 侧带内出现线的行占比下限
_LINE_AREA_MIN = 0.01       # 带内"线像素"占比合理区间（低于=无线，高于=非细线）
_LINE_AREA_MAX = 0.45
_LINE_DEV_FLOOR = 10.0      # 线像素亮度偏离下限
_LINE_SKEW_RATIO = 0.25     # PCA 主轴斜度上限（超过视为斜线，不续接）
_PROFILE_EDGE_MARGIN = 12.0  # 剖面里偏离纸底多少算"有结构"
_RELEVEL_MAX_OFFSET = 40.0   # 洞缘外-内中位色差的逐通道上限；更大视为"合法异质
                             # 材质相邻"（桩标记色、框环金边、描边），不是白盘缺陷
_HALO_MAX_RING_RATIO = 0.9   # 单块残晕占洞外环带面积的上限（徽章光晕径向对称，
                             # 本就会占满环带，只留个兜底上限）


def postprocess_repair(crop_rgb: np.ndarray, hole_mask: np.ndarray) -> tuple:
    """入口：残晕吸收 -> relevel -> 续接。返回 (crop, info)。

    info["hole"] 是吸收残晕后的有效洞——调用方拿它做接缝平滑，否则吸收带的
    新边界会留一条硬边。无洞或自门槛不触发时原样返回。
    """
    info = {"releveled": 0, "reconnected": 0, "halo_absorbed": 0,
            "hole": hole_mask}
    if hole_mask is None or not hole_mask.any():
        return crop_rgb, info
    out, hole = _absorb_halo(crop_rgb, hole_mask, info)
    info["hole"] = hole
    out = _relevel_hole_fill(out, hole, info)
    out = _reconnect_lines(out, hole, info)
    return out, info


def _absorb_halo(crop: np.ndarray, hole_mask: np.ndarray,
                 info: dict) -> tuple:
    """洞邻域残晕并入洞并按外侧材质补平（2026-09-03 二期问题一b）。

    圆徽章的光晕/外环没被子 mask 自适应膨胀吃干净时（光晕不规则、或非徽章类
    残留），会贴着洞缘留下弧斑（val-postproc-3 时间轴节点洞左右两侧实测）。
    判据是几何的：洞外环带内偏离纸底的连通域，凡**被环带关住**（触不到环带
    外沿）就是残晕；触到外沿的会继续往外延伸，那是真实结构（穿洞的轴线本体、
    相邻框线），必须留下。弧斑本身沿洞缘切向拉长，光看伸长率与线分不开，
    "是否越出环带"才分得开。

    吸收带按 洞∪吸收带 之外的材质 Telea 补平（只写吸收带）：洞内的生图内容
    留给 relevel/续接处置，吸收带取干净外侧延续，不把亮盘颜色拉出来。
    """
    from .surface import surface_fill

    ring_px = int(config.ATLAS_HALO_RING_PX)
    if ring_px <= 0:
        return crop, hole_mask
    hole_u8 = hole_mask.astype(np.uint8)
    grown = cv2.dilate(hole_u8, _K3, iterations=ring_px).astype(bool)
    ring = grown & ~hole_mask
    clean = (cv2.dilate(hole_u8, _K3, iterations=ring_px + 4).astype(bool)
             & ~grown)
    if not ring.any() or not clean.any():
        return crop, hole_mask
    lum = crop.astype(np.float32).sum(axis=2) / 3.0
    med = float(np.median(lum[clean]))
    mad = float(np.median(np.abs(lum[clean] - med))) + 1e-6
    dev = ring & (np.abs(lum - med) > max(_LINE_DEV_FLOOR, 2.5 * mad))
    if not dev.any():
        return crop, hole_mask
    rim = grown & ~cv2.dilate(hole_u8, _K3,
                              iterations=max(ring_px - 1, 0)).astype(bool)
    touch = cv2.dilate(hole_u8, _K3).astype(bool)
    num, labels, stats, _cent = cv2.connectedComponentsWithStats(
        dev.astype(np.uint8), connectivity=8)
    absorbed = np.zeros_like(hole_mask)
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < config.ATLAS_HALO_MIN_PX:
            continue
        if area > _HALO_MAX_RING_RATIO * float(ring.sum()):
            continue        # 占掉环带过半：邻域本就是异质材质，不是残晕
        comp = labels == lab
        if not (comp & touch).any():
            continue        # 不贴洞缘：与洞无关的独立元素
        if (comp & rim).any():
            continue        # 越出环带：真实结构继续往外延伸
        absorbed |= comp
    if not absorbed.any():
        return crop, hole_mask
    out = crop.copy()
    filled = surface_fill(crop, hole_mask | absorbed)
    out[absorbed] = filled[absorbed]
    info["halo_absorbed"] = int(absorbed.sum())
    return out, hole_mask | absorbed


def _relevel_hole_fill(crop: np.ndarray, hole_mask: np.ndarray, info: dict) -> np.ndarray:
    """逐洞连通域把填充的低频场平移回边界统计：new = low + offset + (crop − low)。

    offset = 洞缘外中位色 − 洞缘内中位色（低频场整体平移；元素自身渐变保留在
    low + residual 里，与背景"整体换颜色场"的低频锚定不同）。|offset| 逐通道
    超过 _RELEVEL_MAX_OFFSET 视为"合法异质材质相邻"（生图桩标记色、框环金边、
    描边——洞填本来就接它们，不该被拉向边界），不碰；白盘缺陷（run 4968ff94
    时间轴节点洞，白 235 vs 纸 213）偏移 ~27 在管辖内。填充本就接边界时
    offset≈0，逐像素差 ≤1 自门槛不写回。
    """
    crop_f = crop.astype(np.float32)
    num, labels = cv2.connectedComponents(hole_mask.astype(np.uint8), connectivity=8)
    sigma = config.ATLAS_RELEVEL_SIGMA
    low = cv2.GaussianBlur(crop_f, (0, 0), sigma)
    residual = crop_f - low
    touched = False
    for lab in range(1, num):
        comp = labels == lab
        comp_u8 = comp.astype(np.uint8)
        in_band = comp & ~cv2.erode(comp_u8, _K3,
                                    iterations=_RELEVEL_BAND_PX).astype(bool)
        out_band = (cv2.dilate(comp_u8, _K3, iterations=_RELEVEL_BAND_PX + 1).astype(bool)
                    & ~cv2.dilate(comp_u8, _K3).astype(bool))
        if not in_band.any() or not out_band.any():
            continue
        offset = (np.median(crop_f[out_band], axis=0)
                  - np.median(crop_f[in_band], axis=0))
        if float(np.abs(offset).max()) > _RELEVEL_MAX_OFFSET:
            continue
        new = np.clip(np.round(low + offset + residual), 0, 255)
        if np.abs(new[comp] - crop_f[comp]).max() <= 1.0:
            continue
        crop_f[comp] = new[comp]
        info["releveled"] += 1
        touched = True
    return np.round(crop_f).astype(np.uint8) if touched else crop


def _reconnect_lines(crop: np.ndarray, hole_mask: np.ndarray, info: dict) -> np.ndarray:
    """细线双向续接：洞两侧同向对齐的细线按横剖面穿过洞重画（只写洞及其 1px 环）。"""
    num, labels = cv2.connectedComponents(hole_mask.astype(np.uint8), connectivity=8)
    out = crop.copy()
    for lab in range(1, num):
        comp = labels == lab
        ys, xs = np.nonzero(comp)
        comp_box = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
        lum = out.astype(np.float32).sum(axis=2) / 3.0
        stamped = _reconnect_one(out, lum, comp, comp_box, vertical=True) \
            or _reconnect_one(out, lum, comp, comp_box, vertical=False)
        if stamped:
            info["reconnected"] += 1
    return out


def _reconnect_one(out: np.ndarray, lum: np.ndarray, comp: np.ndarray,
                   comp_box, vertical: bool) -> bool:
    """单个洞连通域的一个方向续接。成功写回 out 并返回 True。"""
    if not vertical:
        # 连续拷贝：Telea/cv2 不接受转置视图；成功路径末尾整体写回
        work = np.ascontiguousarray(np.transpose(out, (1, 0, 2)))
        lum_t = np.ascontiguousarray(lum.T)
        comp_t = np.ascontiguousarray(comp.T)
        y0, x0, y1, x1 = comp_box
        stamped = _reconnect_vertical(work, lum_t, comp_t, (x0, y0, x1, y1))
        if stamped:
            out[...] = np.transpose(work, (1, 0, 2))
        return stamped
    return _reconnect_vertical(out, lum, comp, comp_box)


def _reconnect_vertical(out: np.ndarray, lum: np.ndarray, comp: np.ndarray,
                        comp_box) -> bool:
    """竖直细线续接（out 的形状已被调用方转到该方向）。"""
    h, w = comp.shape
    y0, x0, y1, x1 = comp_box
    band = cv2.dilate(comp.astype(np.uint8), _K3,
                      iterations=_LINE_BAND_PX).astype(bool) & ~comp
    if not band.any():
        return False
    band_vals = lum[band]
    med = float(np.median(band_vals))
    mad = float(np.median(np.abs(band_vals - med))) + 1e-6
    thr = max(_LINE_DEV_FLOOR, 2.5 * mad)
    line_px = band & (np.abs(lum - med) > thr)
    frac = float(line_px.sum()) / max(float(band.sum()), 1.0)
    if not (_LINE_AREA_MIN <= frac <= _LINE_AREA_MAX):
        return False
    pys, pxs = np.nonzero(line_px)
    vx, vy = float(pxs.var()) + 1e-6, float(pys.var()) + 1e-6
    if vy >= vx:
        if vx > _LINE_SKEW_RATIO * vy:
            return False  # 太斜，不做方向外推
    else:
        return False  # 主轴非竖直（横向由调用方转置后重试）
    # 两侧各有线：逐行线心
    sides = {}
    lx0, lx1 = max(x0 - 6, 0), min(x1 + 6, w)
    for name, rows in (("top", range(max(y0 - _LINE_BAND_PX, 0), y0)),
                       ("bottom", range(y1, min(y1 + _LINE_BAND_PX, h)))):
        cents = []
        for y in rows:
            xs_line = np.nonzero(line_px[y, lx0:lx1])[0]
            if len(xs_line):
                cents.append(float(xs_line.mean()) + lx0)
        sides[name] = cents
    need = _LINE_MIN_ROW_RATIO * _LINE_BAND_PX
    if len(sides["top"]) < need or len(sides["bottom"]) < need:
        return False  # 只有一侧有线：可能是结构端点，不续接
    xc_top = float(np.median(sides["top"]))
    xc_bot = float(np.median(sides["bottom"]))
    if abs(xc_top - xc_bot) > _LINE_ALIGN_TOL_PX:
        return False
    # 横剖面：两侧有线行以各自线心对齐取逐偏移中位色。两侧分开存——线本身
    # 常带渐变（描边深浅、粗细变化），合并成单一中位会在洞上下口留跳变，
    # 盖章时按行进度在两侧剖面间线性插值才接得上（二期问题一c）。
    acc: dict = {"top": {}, "bottom": {}}
    for name, rows in (("top", sides["top"]), ("bottom", sides["bottom"])):
        rows_range = range(max(y0 - _LINE_BAND_PX, 0), y0) if name == "top" \
            else range(y1, min(y1 + _LINE_BAND_PX, h))
        xc_side = xc_top if name == "top" else xc_bot
        for y, xc in zip(rows_range, rows):
            if abs(xc - xc_side) > 3:
                continue
            xi = int(round(xc))
            for d in range(-_LINE_PROFILE_HALF, _LINE_PROFILE_HALF + 1):
                x = xi + d
                if 0 <= x < w:
                    acc[name].setdefault(d, []).append(out[y, x])
    offsets = sorted(set(acc["top"]) | set(acc["bottom"]))
    if len(offsets) < 3:
        return False
    # 某侧某偏移无样本（裁片边界）时借另一侧同偏移，两侧都无的偏移已不在 offsets
    med_top = {d: np.median(np.asarray(v), axis=0) for d, v in acc["top"].items()}
    med_bot = {d: np.median(np.asarray(v), axis=0) for d, v in acc["bottom"].items()}
    prof_top = {d: med_top.get(d, med_bot.get(d)) for d in offsets}
    prof_bot = {d: med_bot.get(d, med_top.get(d)) for d in offsets}
    far = [p[d] for p in (prof_top, prof_bot) for d in offsets
           if abs(d) >= _LINE_PROFILE_HALF]
    if not far:  # 裁片太窄、剖面边缘偏移越界：取最远两端当纸底参考
        dext = max(abs(d) for d in offsets)
        far = [p[d] for p in (prof_top, prof_bot) for d in (dext, -dext)
               if d in p]
    edge = np.median(np.vstack(far), axis=0)
    # 任一侧偏离纸底即保留该偏移：单侧的渐变淡端不该被另一侧裁掉
    keep = [d for d in offsets
            if max(float(np.abs(prof_top[d] - edge).max()),
                   float(np.abs(prof_bot[d] - edge).max()))
            > _PROFILE_EDGE_MARGIN] or [0]
    dmin, dmax = min(keep), max(keep)
    # 洞底先补成父材质延续（Telea，洞尺寸级小洞无星暴风险）：细线穿过的洞
    # 真相就是"父材质 + 线"，模型填的任何东西（宽松 img2img 常把被删的节点
    # 徽章原样画回来——亮盘边缘还贴着纸色，relevel 的环带统计都测不出偏移）
    # 都不作数，整域覆盖再盖章。
    from .surface import surface_fill

    filled = surface_fill(out, comp)
    out[comp] = filled[comp]
    ring1 = cv2.dilate(comp.astype(np.uint8), _K3).astype(bool)
    span = max(y1 - y0, 1)
    for y in range(y0, y1):
        t = (y - y0 + 0.5) / span
        xi = int(round(xc_top + (xc_bot - xc_top) * t))
        for d in range(dmin, dmax + 1):
            x = xi + d
            if d not in prof_top or not (0 <= x < w):
                continue
            if comp[y, x] or ring1[y, x]:
                color = prof_top[d] * (1.0 - t) + prof_bot[d] * t
                out[y, x] = np.round(color).astype(np.uint8)
    return True
