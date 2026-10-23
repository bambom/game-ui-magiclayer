"""背景修补（文档 §3.4）：scene（生图）/ surface（边界延续）/ none。

scene 线（文档 image26）：待补区涂黑 + 第二张 mask 一起送给模型，避免占位
色矩形被当成独立画框；膨胀吞残边；融合后接缝带平滑。
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .. import config
from .. import runlog
from ..providers.base import ImageRepairProvider

_K3 = np.ones((3, 3), np.uint8)

# 低频锚定：周边纹理采样环带宽度（3x3 迭代）与"小洞直接取低频延续"的面积下限
_BG_ANCHOR_RING_PX = 4
_BG_ANCHOR_MIN_PX = 64
# 干净环形采样带（3x3 迭代）：内界避开洞缘残边/混色，外界取周边真实背景；
# 排除一切洞（邻洞平涂预填不得污染统计）
_BG_ANCHOR_ANNULUS_IN = 12
_BG_ANCHOR_ANNULUS_OUT = 48
# 构图检测（2026-09-03 重放实验）：填充低频场与边界延续场的中位偏离——
# 诚实延续基线 2~12，本网关对卡片形洞的填充 27~95（自行构图的暖褐卡板，
# 两次独立重放一致）。结构带（框线/装饰纹等 >2px 的成构图内容）按它指数
# 衰减：构图填充连纹理结构都不可信（换色后仍以幽灵框残留）；细颗粒带
# （纸纤维级 <2px 无构图能力）只按干净环带的真实颗粒配比，保住纸感。
_BG_ANCHOR_AGREE_GRACE = 12.0
_BG_ANCHOR_AGREE_TAU = 10.0
_BG_ANCHOR_K_FLOOR = 0.05
_BG_ANCHOR_BAND_SIGMA = 2.0


def build_hidden_mask(instance_masks) -> np.ndarray:
    """UI 隐藏区硬 mask = 全部实例 mask 的并集。"""
    out = None
    for m in instance_masks:
        out = m.copy() if out is None else (out | m)
    return out


def _prefill_holes(work_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """待补区涂黑（文档 image25/26）：黑=缺失内容，配合第二张 mask 送给模型。

    不用中位色平涂——占位色矩形会被当成独立画框，洞里自己构图（job 9bd4d1c0
    幽灵框）。不用 Telea——大洞放射渐变会被当成结构。发给模型的图里不能再认出
    被抠走的 UI。
    """
    out = work_rgb.copy()
    if mask.any():
        out[mask.astype(bool)] = (0, 0, 0)
    return out


def _lowfreq_continuation(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """洞区低频边界延续场：降采样 Telea -> 放大 -> 低通。只保证洞内取值有意义。

    供低频锚定合成用（surface_repair 是"低频场+软 alpha 回贴"，这里要的是
    未回贴的原始场）。在预填图上做降采样，洞边界的采样像素不会被洞内原
    像素污染。
    """
    h, w = rgb.shape[:2]
    scale = max(config.SURFACE_BG_DOWNSCALE, 1)
    sw, sh = max(w // scale, 1), max(h // scale, 1)
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), (sw, sh),
                            interpolation=cv2.INTER_NEAREST)
    small_mask = cv2.morphologyEx(small_mask, cv2.MORPH_CLOSE, _K3)
    filled = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), small_mask,
                         max(config.SURFACE_BG_INPAINT_RADIUS // scale, 3),
                         cv2.INPAINT_TELEA)
    up = cv2.resize(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(up, (0, 0), config.BG_ANCHOR_SIGMA)


def anchor_scene_fill(diag_rgb: np.ndarray, fill_rgb: np.ndarray,
                      hole_mask: np.ndarray) -> np.ndarray:
    """洞区低频锚定合成（2026-09-03）：final = 边界低频延续 + 分带配比的生图细节。

    宽松 img2img 网关（run 4968ff94，gpt-image-2）在洞内自行构图——周边是冷调
    纸雾，洞里画出暖褐"卡板"延续，提示词拦不住。按项目分工原则做确定性后
    处理：颜色场整体取边界延续（_lowfreq_continuation），生图细节分两带配比
    （逐连通域）：
    - 细颗粒带（<2px，纸纤维级，无构图能力）：k_grain = 干净环带真实颗粒RMS /
      洞内颗粒RMS（≤1）——压到周边纸感水平，保住质地；
    - 结构带（>2px 的成构图内容：框线/装饰纹/笔触）：k_struct = min(k_grain,
      k_agree)，k_agree = exp(−(偏离−宽限)/τ)，偏离 = 填充低频场与边界延续
      场的中位差（重放实测：诚实延续 2~12，构图卡板 27~95）——模型在构图时
      结构不可信：换色后框线仍以"幽灵框"残留（val-postproc-2 用户实测反馈），
      唯有压掉；诚实延续时 k_agree≈1 全保留（含真实山浪笔触）。
    小洞（<64px）直接取低频延续。毫秒级。

    二期（问题五b/c）再补两件一致性收尾：环带颗粒统计按全局:局部混合消洞间
    方差、被压得过平的洞（k_grain 很低）从干净环带镜像平铺颗粒补回 RMS、
    洞内均值向环带均值做 ≤3/通道 的微平移。
    """
    fill = fill_rgb.astype(np.float32)
    if not hole_mask.any():
        return fill_rgb
    base = _lowfreq_continuation(diag_rgb, hole_mask).astype(np.float32)
    sigma = config.BG_ANCHOR_SIGMA
    fill_low = cv2.GaussianBlur(fill, (0, 0), sigma)
    residual = fill - fill_low
    band_blur = cv2.GaussianBlur(residual, (0, 0), _BG_ANCHOR_BAND_SIGMA)
    grain_band = residual - band_blur     # 细颗粒（纸纤维级）
    struct_band = band_blur               # 结构（框线/装饰纹/笔触）
    src = diag_rgb.astype(np.float32)
    hp_src = src - cv2.GaussianBlur(src, (0, 0), sigma)
    hp_band = cv2.GaussianBlur(hp_src, (0, 0), _BG_ANCHOR_BAND_SIGMA)
    hp_grain = hp_src - hp_band
    num, labels, _stats, _cent = cv2.connectedComponentsWithStats(
        hole_mask.astype(np.uint8), connectivity=8)
    # 全局环带颗粒（二期问题五c）：逐洞独立估计 k_grain 会引入洞间方差
    # （实测密度比散布 0.65~1.60），与局部统计按 BG_ANN_GLOBAL_MIX 混合
    holes_u8 = hole_mask.astype(np.uint8)
    annulus_all = (cv2.dilate(holes_u8, _K3,
                              iterations=_BG_ANCHOR_ANNULUS_OUT).astype(bool)
                   & ~cv2.dilate(holes_u8, _K3,
                                 iterations=_BG_ANCHOR_ANNULUS_IN).astype(bool)
                   & ~hole_mask)
    if annulus_all.any():
        ann_grain_global = float(np.sqrt((hp_grain[annulus_all] ** 2).mean()))
        ann_hp_global = float(np.sqrt((hp_src[annulus_all] ** 2).mean()))
    else:
        ann_grain_global = ann_hp_global = 0.0
    mix_cfg = float(np.clip(config.BG_ANN_GLOBAL_MIX, 0.0, 1.0))
    out = fill.copy()
    for lab in range(1, num):
        comp = labels == lab
        if int(comp.sum()) < _BG_ANCHOR_MIN_PX:
            out[comp] = base[comp]
            continue
        # 干净环形采样带：避开洞缘残边，排除一切洞的预填
        comp_u8 = comp.astype(np.uint8)
        annulus = (cv2.dilate(comp_u8, _K3,
                              iterations=_BG_ANCHOR_ANNULUS_OUT).astype(bool)
                   & ~cv2.dilate(comp_u8, _K3,
                                 iterations=_BG_ANCHOR_ANNULUS_IN).astype(bool)
                   & ~hole_mask)
        if annulus.any():
            mix = mix_cfg if ann_grain_global > 0 else 0.0
            ann_local = float(np.sqrt((hp_grain[annulus] ** 2).mean()))
            ann_grain = (1.0 - mix) * ann_local + mix * ann_grain_global
            ann_hp = ((1.0 - mix) * float(np.sqrt((hp_src[annulus] ** 2).mean()))
                      + mix * ann_hp_global)
        else:
            ann_local, ann_grain, ann_hp = (ann_grain_global, ann_grain_global,
                                            ann_hp_global)
        hole_grain = float(np.sqrt((grain_band[comp] ** 2).mean()))
        k_grain = 1.0 if hole_grain <= 1e-6 else min(1.0, ann_grain / hole_grain)
        dev = float(np.median(np.abs(fill_low[comp] - base[comp])).mean())
        k_agree = float(np.exp(-max(0.0, dev - _BG_ANCHOR_AGREE_GRACE)
                               / _BG_ANCHOR_AGREE_TAU))
        k_struct = max(min(k_grain, k_agree), _BG_ANCHOR_K_FLOOR)
        composed_hp = (grain_band[comp] * k_grain + struct_band[comp] * k_struct)
        out[comp] = base[comp] + composed_hp
        # 洞内合成后的总高频 RMS（低频锚定的 base 本身无高频，故即为 composed_hp）
        now_hp = float(np.sqrt((composed_hp ** 2).mean()))
        gain = _inject_grain(out, comp, hp_src, annulus, ann_hp, now_hp)
        shift = _shift_to_annulus(out, comp, diag_rgb, annulus)
        runlog.event("BG", f"低频锚定 连通域{lab} 面积={int(comp.sum())} "
                           f"环颗粒={ann_grain:.1f}（局部 {ann_local:.1f}/全局 "
                           f"{ann_grain_global:.1f}） 洞颗粒={hole_grain:.1f} "
                           f"k_grain={k_grain:.2f} 偏离={dev:.0f} "
                           f"k_agree={k_agree:.2f} k_struct={k_struct:.2f} "
                           f"洞高频={now_hp:.1f}/环{ann_hp:.1f} 注入={gain:.2f} "
                           f"均值平移={shift}")
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _reflect_tile(patch: np.ndarray, h: int, w: int) -> np.ndarray:
    """镜像平铺到 hxw：奇数块翻转，块接缝处不产生突变。

    用镜像而非随机拼贴：没有随机源，同输入同输出，重放可复现（确定性原则）。
    """
    ph, pw = patch.shape[:2]
    rows = []
    for j in range(-(-h // ph)):
        cols = []
        for i in range(-(-w // pw)):
            blk = patch[:, ::-1] if i % 2 else patch
            cols.append(blk[::-1] if j % 2 else blk)
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)[:h, :w]


def _clean_window(annulus: np.ndarray, side: int):
    """环带内一个完整 side x side 干净窗口的中心（取最靠左上的，确定性）。

    腐蚀后仍为真的像素，恰是"以它为中心的 side x side 窗口整块落在环带内"。
    """
    if side < 4:
        return None
    ero = cv2.erode(annulus.astype(np.uint8), np.ones((side, side), np.uint8))
    ys, xs = np.nonzero(ero)
    if not len(xs):
        return None
    i = int(np.lexsort((xs, ys))[0])
    return int(xs[i]), int(ys[i])


def _inject_grain(out: np.ndarray, comp: np.ndarray, hp_src: np.ndarray,
                  annulus: np.ndarray, ann_hp: float, now: float) -> float:
    """洞内过平时从干净环带镜像平铺真实纹理，补到环带高频水平（二期问题五b）。

    口径是总高频（= 证据里的"边缘密度"家族）：洞内高频低于环带的
    BG_GRAIN_INJECT_BELOW 倍才注入。被压掉的主要是结构带（k_struct），拿真纸
    颗粒补回密度而不是把不可信的构图结构放回来——注入的是零均值高频场，
    低频锚定的颜色场不动。返回实际增益（0=没注入）。
    """
    below = float(config.BG_GRAIN_INJECT_BELOW)
    if below <= 0 or ann_hp <= 0 or not annulus.any() or now >= below * ann_hp:
        return 0.0
    deficit = ann_hp * ann_hp - now * now
    if deficit <= 0:
        return 0.0
    ys, xs = np.nonzero(comp)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    side = min(32, _BG_ANCHOR_ANNULUS_OUT - _BG_ANCHOR_ANNULUS_IN)
    center = _clean_window(annulus, side)
    if center is None:
        runlog.event("BG", f"环带放不下 {side}x{side} 干净窗口，跳过纹理注入")
        return 0.0
    cx, cy = center
    half = side // 2
    patch = hp_src[cy - half:cy - half + side, cx - half:cx - half + side]
    if patch.shape[:2] != (side, side):     # 采样窗贴到图边被截断
        return 0.0
    patch_rms = float(np.sqrt((patch ** 2).mean()))
    if patch_rms <= 1e-6:
        return 0.0
    gain = float(np.sqrt(deficit)) / patch_rms
    tile = _reflect_tile(patch, y1 - y0, x1 - x0) * gain
    local = comp[y0:y1, x0:x1]
    out[y0:y1, x0:x1][local] += tile[local]
    return gain


def _shift_to_annulus(out: np.ndarray, comp: np.ndarray, diag_rgb: np.ndarray,
                      annulus: np.ndarray) -> list:
    """洞内均值向环带均值做上限受控的微平移（二期问题五c）。

    实测洞内均值与各自环带均值差 8~32：大头是合法材质差（洞里本来就压着
    不同东西），只削峰——逐通道 clamp 到 ±BG_HOLE_MEAN_SHIFT_MAX。
    """
    cap = float(config.BG_HOLE_MEAN_SHIFT_MAX)
    if cap <= 0 or not annulus.any():
        return [0.0, 0.0, 0.0]
    delta = (np.median(diag_rgb[annulus].astype(np.float32), axis=0)
             - np.median(out[comp], axis=0))
    shift = np.clip(delta, -cap, cap)
    out[comp] += shift
    return [round(float(v), 1) for v in shift]


def seam_step(image: np.ndarray, mask: np.ndarray) -> float:
    """接缝可见度：洞缘内 1px 环与洞缘外 1px 环的中位色差（逐通道取最大）。

    自适应加宽的判据，也是 scripts/measure_bg_consistency.py 量的同一个数。
    """
    if image.ndim != 3 or not mask.any():
        return 0.0
    m8 = mask.astype(np.uint8)
    inner = mask & ~cv2.erode(m8, _K3).astype(bool)
    outer = cv2.dilate(m8, _K3).astype(bool) & ~mask
    if not inner.any() or not outer.any():
        return 0.0
    return float(np.abs(np.median(image[inner].astype(np.float32), axis=0)
                        - np.median(image[outer].astype(np.float32),
                                    axis=0)).max())


def _blend_band(image: np.ndarray, mask: np.ndarray, iters: int) -> np.ndarray:
    """把 mask 内侧 iters 宽的环带与高斯模糊版按软 alpha 混合。

    只写 mask 以内：洞外像素不在我们的管辖权里（隐藏区 mask 已按
    BG_REPAIR_DILATE_ITERS 膨胀过，该吞的残边都吞进来了），往外写反而在
    真实背景上留下"改写痕迹"——实测洞外 1px 带差 28.7 有一份就来自这里。
    模糊核照旧从洞外真实像素采样，连续性不受影响。

    模糊 sigma 随带宽一起长：跨边界的混合量由 sigma 决定而非带宽，只加宽
    带、不加大 sigma 的话洞缘那一圈的阶跃根本不动（自适应就白做了）。
    base 宽度下 sigma 仍是 BG_SEAM_BLUR_SIGMA，行为与一期一致。
    """
    band = mask & ~cv2.erode(mask.astype(np.uint8), _K3,
                             iterations=iters).astype(bool)
    if not band.any():
        return image
    alpha = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.0)[..., None]
    alpha = np.clip(alpha * 1.6, 0, 1)
    alpha[~mask] = 0.0                    # 高斯尾巴渗到洞外，掐掉
    sigma = max(config.BG_SEAM_BLUR_SIGMA, 0.6 * iters)
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    out = image.astype(np.float32) * (1 - alpha) + blur.astype(np.float32) * alpha
    # 高斯尾巴 alpha~1e-11 会让整数值截断成 199——先 round 再落 uint8
    return np.round(out).astype(np.uint8)


def _poisson_harmonize(filled: np.ndarray, reference: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    """梯度域调和（可选收尾）：洞内保留纹理梯度，洞缘取真实背景的值。

    模糊有个数学上限——洞外我们不写，对称核最多把内侧拉到内外中点，阶跃只能
    减半。泊松（seamlessClone: src=合成图, dst=真实背景, mask=洞）解的是"梯度
    像 src、边界值像 dst"，阶跃从根上消掉，且洞外一个像素不动（dst 原样）。

    非凸/贴边 mask 是 seamlessClone 的已知崩溃点，故限定在 bbox 内缩一圈并
    整体 try：失败即保留加宽结果（那已经可用，这一步只是锦上添花）。
    """
    if reference is None or reference.shape != filled.shape:
        runlog.event("BG", "梯度域调和缺参照图（调用方未传），保留加宽结果")
        return filled
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return filled
    h, w = filled.shape[:2]
    x0, x1 = max(int(xs.min()) - 2, 1), min(int(xs.max()) + 3, w - 1)
    y0, y1 = max(int(ys.min()) - 2, 1), min(int(ys.max()) + 3, h - 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return filled
    sub = mask[y0:y1, x0:x1]
    if sub[0].any() or sub[-1].any() or sub[:, 0].any() or sub[:, -1].any():
        runlog.event("BG", "洞贴着裁剪边界，跳过梯度域调和（seamlessClone 崩溃点）")
        return filled
    try:
        out = filled.copy()
        out[y0:y1, x0:x1] = cv2.seamlessClone(
            np.ascontiguousarray(filled[y0:y1, x0:x1]),
            np.ascontiguousarray(reference[y0:y1, x0:x1]),
            np.ascontiguousarray(sub.astype(np.uint8) * 255),
            ((x1 - x0) // 2, (y1 - y0) // 2), cv2.NORMAL_CLONE)
        runlog.event("BG", "接缝带梯度域调和完成")
        return out
    except cv2.error as e:
        runlog.event("WARN", f"接缝带梯度域调和失败，保留加宽结果：{e}")
        return filled


def seam_band_smooth(image: np.ndarray, mask: np.ndarray,
                     reference: np.ndarray = None,
                     iters: int = None) -> np.ndarray:
    """融合接缝带平滑（自适应加宽）：mask 内侧环带与模糊版按软 alpha 混合。

    羽化粘贴产生的接缝与色阶突变发生在我们自己的合成代码里，模型与提示词
    都无法触达；纯本地毫秒级操作，背景融合与元素切片回填共用。

    固定 2px 带吃不掉 val-postproc-3 的 ≤3px 过渡带（1px 带差 28.7），所以
    平滑后量接缝阶跃，超阈就加宽重做（每轮都从原图重算，避免反复模糊糊掉
    纹理），到 BG_SEAM_MAX_ITERS 为止。注意模糊的上限是"阶跃减半"（洞外不
    写，对称核最多把内侧拉到中点）——真要清零得靠梯度域，故 reference（修补
    前的真实图）给到时、且开了 BG_SEAM_POISSON，才做最后那一步。
    """
    if image.ndim != 3 or not mask.any():
        return image
    base = int(config.BG_SEAM_BAND_ITERS if iters is None else iters)
    if base <= 0:
        return image
    # 逐洞独立决定带宽：加宽是按洞的尺寸买单的（见 _afford_width），一张图上
    # 大洞与徽章级小洞并存时不能用同一个宽度
    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = image
    for lab in range(1, num):
        out = _smooth_one_hole(out, labels == lab, base, reference)
    return out


def _afford_width(comp: np.ndarray, base: int) -> int:
    """该洞能承受的接缝带宽度上限。

    带每加宽 1px 都在啃洞内内容——窄洞啃不起：16x16 的徽章洞若按大洞那样
    加到 4px，整个洞会被模糊糊掉（重建材质的均值都被拉走）。取洞内切半径
    （距离变换峰值）的 1/4：徽章洞正好还是 base 的 2px（保持一期行为），
    大洞才放开到 BG_SEAM_MAX_ITERS。
    """
    dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
    afford = int(float(dist.max()) // 4)
    return max(base, min(int(config.BG_SEAM_MAX_ITERS), afford))


def _smooth_one_hole(image: np.ndarray, comp: np.ndarray, base: int,
                     reference: np.ndarray) -> np.ndarray:
    """单洞的接缝带平滑：从 base 起加宽到阶跃达标或到该洞的宽度上限。"""
    limit = _afford_width(comp, base)
    out, step = image, 0.0
    for width in range(base, limit + 1):
        out = _blend_band(image, comp, width)
        step = seam_step(out, comp)
        if step <= config.BG_SEAM_RESID_THR:
            if width > base:
                runlog.event("BG", f"接缝带加宽到 {width}px 后阶跃 {step:.1f}")
            return out
    runlog.event("BG", f"接缝带加满 {limit}px（洞宽上限）仍有阶跃 {step:.1f}"
                       f"（阈值 {config.BG_SEAM_RESID_THR}）")
    if not config.BG_SEAM_POISSON:
        return out
    return _poisson_harmonize(out, reference, comp)


def scene_repair(work_rgb: np.ndarray, hard_mask: np.ndarray,
                 provider: ImageRepairProvider, debug=None) -> np.ndarray:
    """scene 路径：待补区涂黑后只送这一张图，模型出图即为最终背景。

    涂黑图只当输入；debug 分文件保存涂黑输入和硬 mask，不拼图、不和原图融合。
    若只送了洞区裁剪，生成图整块替换该裁剪（裁剪外未送给模型，保持原图像素）。
    """
    h, w = work_rgb.shape[:2]
    diag = _prefill_holes(work_rgb, hard_mask)

    cx0, cy0, cx1, cy1 = _hole_context_box(hard_mask, w, h)
    runlog.event("BG", f"scene 生图裁剪上下文框 ({cx0},{cy0})-({cx1},{cy1}) "
                       f"（全图 {w}x{h}，隐藏区 {int(hard_mask.sum())}px）")
    t0 = time.time()
    filled_crop = provider.repair_background(diag[cy0:cy1, cx0:cx1],
                                             hard_mask[cy0:cy1, cx0:cx1])
    runlog.event("BG", f"scene 生图修补 {(time.time() - t0) * 1000:.0f}ms "
                       f"provider={provider.name}")

    if config.BG_ANCHOR_ENABLE:
        filled_crop = anchor_scene_fill(diag[cy0:cy1, cx0:cx1], filled_crop,
                                        hard_mask[cy0:cy1, cx0:cx1])
    else:
        runlog.event("BG", "跳过低频锚定（MAGICLAYER_BG_ANCHOR=0）")

    if debug is not None:
        debug("06b_background_input.png", diag)
        debug("06b_background_mask.png",
              cv2.cvtColor(hard_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB))
    if (cx0, cy0, cx1, cy1) == (0, 0, w, h):
        return filled_crop
    out = work_rgb.copy()
    out[cy0:cy1, cx0:cx1] = filled_crop
    return out


def _hole_context_box(hard_mask: np.ndarray, w: int, h: int):
    """洞区外扩 35%（至少 64px 上下文），长边不足 512 时以洞为中心扩到 512（边界裁断）。"""
    ys, xs = np.nonzero(hard_mask)
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mx = max(int((x1 - x0) * 0.35), 64)
    my = max(int((y1 - y0) * 0.35), 64)
    cx0, cy0 = max(x0 - mx, 0), max(y0 - my, 0)
    cx1, cy1 = min(x1 + mx, w), min(y1 + my, h)
    if max(cy1 - cy0, cx1 - cx0) < 512:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        tx, ty = min(512, w), min(512, h)
        cx0 = max(min(cx - tx // 2, w - tx), 0)
        cy0 = max(min(cy - ty // 2, h - ty), 0)
        cx1, cy1 = cx0 + tx, cy0 + ty
    return cx0, cy0, cx1, cy1


def surface_repair(work_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    """surface 路径：只延续边界颜色/低频成分（降采样 Telea + 低频化）。"""
    h, w = work_rgb.shape[:2]
    if not hard_mask.any():
        return work_rgb.copy()
    scale = config.SURFACE_BG_DOWNSCALE
    small = cv2.resize(work_rgb, (max(w // scale, 1), max(h // scale, 1)),
                       interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(hard_mask.astype(np.uint8) * 255,
                            (max(w // scale, 1), max(h // scale, 1)),
                            interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), small_mask,
                         max(config.SURFACE_BG_INPAINT_RADIUS // scale, 3),
                         cv2.INPAINT_TELEA)
    up = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    up = cv2.resize(up, (w, h), interpolation=cv2.INTER_CUBIC)
    low = cv2.GaussianBlur(up, (0, 0), config.BG_SOFT_FEATHER_SIGMA * 2)
    alpha = cv2.GaussianBlur(hard_mask.astype(np.float32), (0, 0), 1.0)[..., None]
    out = low.astype(np.float32) * alpha + work_rgb.astype(np.float32) * (1 - alpha)
    return out.astype(np.uint8)


def none_repair(work_rgb: np.ndarray) -> np.ndarray:
    """none：不合成隐藏像素，保留当前清理背景。"""
    return work_rgb.copy()


def repair_background(work_rgb: np.ndarray, hard_mask: np.ndarray,
                      mode: str, provider: ImageRepairProvider,
                      debug=None) -> tuple:
    """返回 (背景图, 是否降级)。scene 生图失败时退回 surface 延续。"""
    t0 = time.time()
    out, degraded = _repair_background(work_rgb, hard_mask, mode, provider, debug)
    runlog.event("BG", f"模式={mode} 总耗时 {(time.time() - t0) * 1000:.0f}ms "
                       f"降级={degraded} provider={provider.name}")
    return out, degraded


def _repair_background(work_rgb: np.ndarray, hard_mask: np.ndarray,
                       mode: str, provider: ImageRepairProvider,
                       debug=None) -> tuple:
    if mode == "none":
        return none_repair(work_rgb), provider.degraded
    # 隐藏区膨胀：把并集 mask 外没扣干净的残边（抗锯齿半像素/描边外半圈/投影）
    # 吞进重绘管辖权——edits 线 mask 外像素原样返回，提示词改不了它们。
    # 并集外本来就是背景，膨胀只多吃背景侧几像素，语义安全。
    mask = cv2.dilate(hard_mask.astype(np.uint8), _K3,
                      iterations=config.BG_REPAIR_DILATE_ITERS).astype(bool)
    if mode == "scene":
        if not provider.degraded:
            try:
                return scene_repair(work_rgb, mask, provider, debug), False
            except Exception as e:  # 生图不可用（配额/网络）-> 传统延续兜底
                import warnings as _w

                _w.warn(f"scene 生图修补失败，退回 surface 兜底: {e}")
                runlog.event("WARN", f"scene 生图修补失败，退回 surface 兜底: {e}")
        return surface_repair(work_rgb, mask), True
    if mode == "surface":
        return surface_repair(work_rgb, mask), provider.degraded
    return none_repair(work_rgb), provider.degraded
