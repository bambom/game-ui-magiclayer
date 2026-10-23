"""元素图层构建与遮挡修补（文档 §3）：顶层直取，底层扣子元素后按材质修补。"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import config, runlog
from ..common import ImageLayer, Instance, bbox_of_mask
from ..providers.base import ImageRepairProvider
from .atlas import AtlasItem, build_atlas, placements_json, slice_back
from .matting import feather_binary, is_soft_edge, solve_alpha
from .surface import surface_fill


_K3 = np.ones((3, 3), np.uint8)


def _children_map(instances: List[Instance]) -> Dict[str, List[str]]:
    """按 query id 建父子表。"""
    by_query: Dict[str, List[Instance]] = {}
    for inst in instances:
        by_query.setdefault(inst.query.id, []).append(inst)
    out: Dict[str, List[str]] = {}
    for inst in instances:
        pid = inst.query.parent_query_id
        if pid:
            out.setdefault(pid, []).append(inst.query.id)
    return out


def _descendant_query_ids(instances: List[Instance]) -> Dict[str, set]:
    """每个 query 的全部传递后代 query id。"""
    children = _children_map(instances)
    memo: Dict[str, set] = {}

    def visit(qid: str) -> set:
        if qid in memo:
            return memo[qid]
        memo[qid] = set()  # 环保护
        acc = set()
        for c in children.get(qid, []):
            acc.add(c)
            acc |= visit(c)
        memo[qid] = acc
        return acc

    for inst in instances:
        visit(inst.query.id)
    return memo


# 无父子声明的实例间几何重叠自动镂空的下限：重叠像素数与占较小 mask 面积的比例
# 双门槛。低于门槛视为 SAM 边缘渗出（1-2px），不做镂空。
_AUTO_OCCLUDE_MIN_PX = 24
_AUTO_OCCLUDE_MIN_RATIO = 0.05

# 内容子层剥离父框（2026-09-03）的判定常数：父框最小边长、包围内域占父 bbox
# 面积比例（≥ 视为环形框）、父 mask 占 bbox 覆盖率（≥ 视为整板框）、整板框的
# 环带宽度系数、子 mask 沾框带比例下限、误判保护（裁后内域像素绝对下限与
# 相对比例——盖满整卡的子层裁后剩的正是内域，合法情形只占原面积三成上下，
# 所以比例门槛必须远低于"环上小元素"判反的用法，后者靠绝对下限拦截）。
_FRAME_MIN_SIDE = 48
_RING_INTERIOR_MIN_RATIO = 0.08
_PLATE_MIN_COVERAGE = 0.85
_RING_STRIP_RATIO = 0.06
_BAND_OVERLAP_MIN_RATIO = 0.12
_CLIP_MIN_INNER_PX = 576   # 24x24：裁后内域不足此面积视为误判
_CLIP_KEEP_MIN_RATIO = 0.05


def _enclosed_interior(mask: np.ndarray) -> np.ndarray:
    """闭合 mask 的包围内域：从外域 flood fill，不可达部分即内域（局部坐标）。

    先 pad 一圈假外域：框环常紧贴父 bbox，四角可能全是框像素，不 pad 就没有
    外域种子可淹。
    """
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = ~mask
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)
    return padded[1:-1, 1:-1].astype(bool)


def _clip_child_mask(child_mask: np.ndarray, parent_mask: np.ndarray,
                     parent_bbox) -> Optional[np.ndarray]:
    """单对父子：把子 mask 里混进的父框像素几何剥离。

    环形父（包围内域显著）裁到 内域 − 框环带；整板父裁到 腐蚀环带后的内板
    （环带宽度 = clamp(6%·min边, 4, 16)，装饰环总在板外缘）。子 mask 不
    containment 于父框（横跨/压角元素本就该带框外像素）、沾框带不足、父框
    形状不明或裁后所剩无几时一律不动（误判保护）。
    """
    px0, py0, px1, py1 = parent_bbox
    pw, ph = px1 - px0, py1 - py0
    if min(pw, ph) < _FRAME_MIN_SIDE:
        return None
    cb = bbox_of_mask(child_mask)
    if cb[0] < px0 - 4 or cb[1] < py0 - 4 or cb[2] > px1 + 4 or cb[3] > py1 + 4:
        return None
    sub = parent_mask[py0:py1, px0:px1]
    bbox_area = float(pw * ph)
    parent_u8 = parent_mask.astype(np.uint8)
    interior = _enclosed_interior(sub)
    if int(interior.sum()) >= _RING_INTERIOR_MIN_RATIO * bbox_area:
        inner = np.zeros_like(parent_mask)
        inner[py0:py1, px0:px1] = interior
        inner &= ~cv2.dilate(parent_u8, _K3).astype(bool)
        band = parent_mask
    else:
        if parent_mask.sum() < _PLATE_MIN_COVERAGE * bbox_area:
            return None
        strip = int(np.clip(round(_RING_STRIP_RATIO * min(pw, ph)), 4, 16))
        kernel = np.ones((2 * strip + 1,) * 2, np.uint8)
        inner = cv2.erode(parent_u8, kernel).astype(bool)
        band = parent_mask & ~inner  # 恰为将被剥掉的环带
    if not inner.any():
        return None
    overlap = float((child_mask & band).sum())
    if overlap < _BAND_OVERLAP_MIN_RATIO * float(child_mask.sum()):
        return None
    clipped = child_mask & inner
    if int(clipped.sum()) < _CLIP_MIN_INNER_PX \
            or float(clipped.sum()) < _CLIP_KEEP_MIN_RATIO * float(child_mask.sum()):
        return None
    return clipped


def _clip_children_to_frames(valid: List[Instance]) -> List[str]:
    """内容子层确定性剥离父框（2026-09-03，run 4968ff94 遗留问题三a）。

    planner 规则要求子 bbox 在框环以内，但 SAM 实际常把整卡（含装饰框环）
    圈进子 mask——五个 event_illustrations 实例全部盖满整卡，移动插画会把
    框边带走。树关系已在，几何可算：对每个带父实例、且父框是"环形/整板框"
    的子实例裁掉框带；裁后内域不足 _CLIP_MIN_INNER_PX 视为误判回退。
    返回被裁剪的实例 id 列表。
    """
    by_query: Dict[str, List[Instance]] = {}
    for inst in valid:
        by_query.setdefault(inst.query.id, []).append(inst)
    changed: List[str] = []
    for inst in valid:
        pid = inst.query.parent_query_id
        parents = by_query.get(pid) if pid else None
        if not parents:
            continue
        cb = bbox_of_mask(inst.mask)
        best = None
        best_overlap = 0
        for p in parents:
            if p is inst:
                continue
            pb = bbox_of_mask(p.mask)
            ix0, iy0 = max(cb[0], pb[0]), max(cb[1], pb[1])
            ix1, iy1 = min(cb[2], pb[2]), min(cb[3], pb[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            ov = int((inst.mask[iy0:iy1, ix0:ix1]
                      & p.mask[iy0:iy1, ix0:ix1]).sum())
            if ov > best_overlap:
                best, best_overlap = p, ov
        if best is None:
            continue
        clipped = _clip_child_mask(inst.mask, best.mask, bbox_of_mask(best.mask))
        if clipped is None:
            continue
        before = int(inst.mask.sum())
        runlog.event("LAYER", f"{inst.instance_id} 子 mask 剥离父框 "
                              f"{best.instance_id}：面积 {before}->"
                              f"{int(clipped.sum())}")
        inst.mask = clipped
        changed.append(inst.instance_id)
    return changed


# 自适应镂空膨胀的「紧凑团块」判据（2026-09-03 二期问题一a）：长宽比与 bbox
# 填充率。细长（轴线/丝带/进度轨）与稀疏形状外扩会吃掉大片父材质，保持 2px；
# 徽章/图标/圆章这类紧凑团块才按尺寸外扩去吞自己的光晕。
_HALO_MAX_ASPECT = 2.0
_HALO_MIN_FILL = 0.45


def _dilate_iters(mask: np.ndarray) -> int:
    """该子 mask 参与镂空时的 3x3 膨胀迭代数。

    固定 2px 只够吃描边外半圈与抗锯齿（一期结论）。圆徽章另有一圈光晕/外环，
    半径按徽章尺寸走（val-postproc-3 实测 ≈10~15%），落在 2px 之外就留在父层
    成弧斑（时间轴节点洞两侧）。紧凑团块按 min 边比例外扩，上限见 config。
    """
    box = bbox_of_mask(mask)
    if box is None:
        return 2
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if max(bw, bh) > _HALO_MAX_ASPECT * max(min(bw, bh), 1):
        return 2
    if float(mask.sum()) < _HALO_MIN_FILL * float(bw * bh):
        return 2
    return int(np.clip(round(config.HALO_DILATE_RATIO * min(bw, bh)),
                       2, max(config.HALO_DILATE_MAX, 2)))


def _geometric_overlays(valid: List[Instance]) -> Dict[str, List[Instance]]:
    """按 mask 几何重叠推定无父子声明实例间的上下关系（通用兜底）。

    planner 漏声明 parent 时（沿线节点压轴线、徽章压面板角、丝带压卡片……任何
    "一个已提取元素压在另一个上面"的形态），底层实例同样要在被压住处镂空修补，
    不能只依赖提示词概率。判定：bbox 相交 + mask 重叠显著；树关系对（祖先/后代）
    交给既有父子镂空，跳过。上下关系：z_order 高者在上；同 z 时较小 mask 在上
    （节点/徽章压线/面板）；面积也相同按规划顺序后者在上。
    """
    descendants = _descendant_query_ids(valid)
    areas = {inst.instance_id: int(inst.mask.sum()) for inst in valid}
    out: Dict[str, List[Instance]] = {}
    for i, a in enumerate(valid):
        ax0, ay0, ax1, ay1 = a.bbox
        for b in valid[i + 1:]:
            if b.query.id in descendants.get(a.query.id, ()) \
                    or a.query.id in descendants.get(b.query.id, ()):
                continue
            bx0, by0, bx1, by1 = b.bbox
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            overlap = int((a.mask[iy0:iy1, ix0:ix1]
                           & b.mask[iy0:iy1, ix0:ix1]).sum())
            if overlap < _AUTO_OCCLUDE_MIN_PX \
                    or overlap < _AUTO_OCCLUDE_MIN_RATIO * min(areas[a.instance_id],
                                                               areas[b.instance_id]):
                continue
            za, zb = a.query.z_order or 0, b.query.z_order or 0
            if za != zb:
                overlay, under = (a, b) if za > zb else (b, a)
            elif areas[a.instance_id] != areas[b.instance_id]:
                overlay, under = ((b, a) if areas[a.instance_id] < areas[b.instance_id]
                                  else (a, b))
            else:
                overlay, under = b, a
            out.setdefault(under.instance_id, []).append(overlay)
    return out


def build_element_layers(instances: List[Instance], work_rgb: np.ndarray,
                         provider: ImageRepairProvider, debug=None, progress=None
                         ) -> Tuple[List[Tuple[Instance, np.ndarray, ImageLayer]], dict]:
    """返回 [(instance, rgba, ImageLayer)] 与诊断信息。

    - alpha = 实例 mask（遮挡材质补齐后作为完整素材导出，文档 §0.1）；
    - 镂空 = 传递后代 mask 的并集 ∩ 裁片（不与父 mask 再相交：修补区完整并入
      材质，image/surface 导出 alpha |= 镂空，保证重建像素可见）；另含无父子
      声明但 mask 显著几何重叠的上层实例（_geometric_overlays 通用兜底）；
    - image 模式批量进 Atlas 生图；surface 模式 Telea；none 保持镂空，
      但带子元素遮挡的 none 自动纠偏为 surface（不修补会导出空洞素材）。
    - debug 可选回调 (文件名, RGB 图)：Atlas 修补时分文件落涂黑输入、硬 mask、生图输出。

    bbox 计算分两遍：先给全部有效实例赋 bbox，再算镂空——规划顺序是父先子后，
    单循环里父处理时子 bbox 还没赋值（None），镂空恒为空，遮挡修补会全部失效。
    """
    descendants = _descendant_query_ids(instances)
    by_query: Dict[str, List[Instance]] = {}
    for inst in instances:
        by_query.setdefault(inst.query.id, []).append(inst)

    results: List[Tuple[Instance, np.ndarray, ImageLayer]] = []
    atlas_items: List[Tuple[Instance, np.ndarray, np.ndarray, Tuple[int, int]]] = []
    diag = {"atlas_ids": [], "surface_ids": [], "none_ids": [], "empty_masks": [],
            "none_coerced": [], "auto_occluded": []}

    # 第一遍：全部有效实例的 bbox（第二遍的镂空计算要读子元素 bbox）
    valid: List[Instance] = []
    for inst in instances:
        if inst.mask is None or not inst.mask.any():
            diag["empty_masks"].append(inst.instance_id)
            continue
        valid.append(inst)
    # 子层剥离父框须在 bbox 之前：裁剪改变子 mask 与其 bbox，也顺带让父层的
    # 镂空只剩"框环以内"的内画区（框环像素留在父层，不再被错误挖掉修补）
    diag["child_clipped"] = _clip_children_to_frames(valid)
    for inst in valid:
        inst.bbox = bbox_of_mask(inst.mask)
    auto_overlays = _geometric_overlays(valid)

    # 第二遍：裁片 + 镂空 + 按材质修补
    # 子 mask 先膨胀再参与镂空：SAM mask 只切到描边中线，外半圈描边/抗锯齿
    # 若留在父层会形成一圈"轮廓痕迹"（线上实测残留在洞外 -1px 处聚集成 479px）。
    # 膨胀量按形状自适应（_dilate_iters）：紧凑团块要连自己的光晕一起吞。
    _k3 = np.ones((3, 3), np.uint8)
    dilated = {}
    for inst in valid:
        iters = _dilate_iters(inst.mask)
        dilated[inst.instance_id] = cv2.dilate(
            inst.mask.astype(np.uint8), _k3, iterations=iters).astype(bool)
        if iters > 2:
            runlog.event("LAYER", f"{inst.instance_id}（{inst.query.name}）"
                                  f"紧凑团块镂空膨胀 {iters}px（光晕外扩）")
    for i, inst in enumerate(valid):
        if progress is not None:
            progress(f"{inst.instance_id}（{inst.query.name}）", i + 1, len(valid))
        x0, y0, x1, y1 = inst.bbox
        crop = work_rgb[y0:y1, x0:x1]
        alpha = inst.mask[y0:y1, x0:x1]

        holes = np.zeros_like(alpha)
        for dq in descendants.get(inst.query.id, ()):
            for child in by_query.get(dq, ()):
                if child.mask is None or child.bbox is None:
                    continue
                cx0, cy0, cx1, cy1 = child.bbox
                if cx1 + 2 <= x0 or cx0 - 2 >= x1 or cy1 + 2 <= y0 or cy0 - 2 >= y1:
                    continue
                holes |= dilated[child.instance_id][y0:y1, x0:x1]
        overlays = auto_overlays.get(inst.instance_id)
        if overlays:
            for overlay in overlays:
                holes |= dilated[overlay.instance_id][y0:y1, x0:x1]
            diag["auto_occluded"].append(
                {"instance_id": inst.instance_id,
                 "by": [o.instance_id for o in overlays]})

        mode = inst.query.element_repair_mode
        hole_px = int(holes.sum())
        if mode == "none" and hole_px:
            # none 只对独立素材合理；被后代/上层压住的区域不修补会导出空洞
            mode = "surface"
            diag["none_coerced"].append(inst.instance_id)
            runlog.event("LAYER", f"{inst.instance_id}（{inst.query.name}）"
                                  f"mode none 被遮挡 {hole_px}px，纠偏为 surface")
        runlog.event("LAYER", f"{inst.instance_id}（{inst.query.name}）mode={mode} "
                              f"bbox={inst.bbox} 镂空={hole_px}px")

        if mode == "image":
            atlas_items.append((inst, crop, holes, (x0, y0)))
            diag["atlas_ids"].append(inst.instance_id)
            results.append((inst, _rgba(inst, crop, alpha),
                            _make_layer(inst, inst.bbox, mode)))
        elif mode == "surface":
            filled = surface_fill(crop, holes)
            diag["surface_ids"].append(inst.instance_id)
            results.append((inst, _rgba(inst, filled, alpha, holes),
                            _make_layer(inst, inst.bbox, mode)))
        else:
            diag["none_ids"].append(inst.instance_id)
            results.append((inst, _rgba(inst, crop, alpha),
                            _make_layer(inst, inst.bbox, mode)))

    # Atlas 生图修补（一次调用）；生图不可用时退回传统修补并标记降级
    if atlas_items:
        items = [AtlasItem(item_id=inst.instance_id, crop_rgb=crop, hole_mask=holes)
                 for inst, crop, holes, _ in atlas_items]
        atlas_rgb, atlas_mask, placements = build_atlas(items)
        runlog.event("LAYER", f"Atlas 拼接 {len(items)} 个元素 "
                              f"画布 {atlas_rgb.shape[1]}x{atlas_rgb.shape[0]}"
                              f"（正方形、元素原尺寸零缩放）")
        if debug is not None:
            debug("05_atlas_placements.json",
                  placements_json(atlas_rgb.shape[1], atlas_rgb.shape[0],
                                  placements))
        used_provider = provider
        t0 = time.time()
        if progress is not None:
            progress(f"Atlas 生图修补（{len(items)} 个元素）")
        try:
            repaired = used_provider.repair_elements(atlas_rgb, atlas_mask)
        except Exception as e:
            from ..providers.local import LocalProvider

            diag["atlas_fallback"] = diag.get("atlas_fallback", []) + [str(e)]
            runlog.event("WARN", f"Atlas 生图修补失败退回传统修补: {e}")
            used_provider = LocalProvider()
            repaired = used_provider.repair_elements(atlas_rgb, atlas_mask)
        runlog.event("LAYER", f"Atlas 生图修补完成 {(time.time() - t0) * 1000:.0f}ms "
                              f"provider={used_provider.name} "
                              f"degraded={used_provider.degraded}")
        if used_provider.degraded:
            diag["atlas_degraded"] = True
        if debug is not None:
            debug("05_atlas_input.png", atlas_rgb)
            debug("05_atlas_mask.png",
                  cv2.cvtColor(atlas_mask, cv2.COLOR_GRAY2RGB))
            debug("05_atlas.png", repaired)
        for item_id, filled_crop in slice_back(repaired, atlas_mask, placements):
            inst, _, holes, (x0, y0) = next(
                t for t in atlas_items if t[0].instance_id == item_id)
            alpha = inst.mask[inst.bbox[1]:inst.bbox[3], inst.bbox[0]:inst.bbox[2]]
            rgba = _rgba(inst, filled_crop, alpha, holes)
            idx = next(i for i, r in enumerate(results) if r[0].instance_id == item_id)
            results[idx] = (inst, rgba, results[idx][2])
            results[idx][2].repaired = True
            results[idx][2].degraded = used_provider.degraded
    for inst, _, layer in results:
        if layer.repair_mode == "surface":
            layer.repaired = True  # surface 即文档 §3.2 传统路径，非降级
    return results, diag


def _rgba(inst: Instance, crop: np.ndarray, alpha: np.ndarray,
          holes: Optional[np.ndarray] = None) -> np.ndarray:
    """导出 RGBA 的唯一装配点（二期问题四把连续 alpha 落在这里）。

    alpha 语义不变：实例 mask ∪ 镂空（修补出的材质完整并入素材，文档 §0.1），
    变的是边界那一圈的取值——
    - 软边元素（植被雾化/玻璃/光晕，或 kind=decoration）走 matting 解出连续
      alpha，并把未知带的 RGB 换成前景估计（不把背景混色带进素材）；
    - 其余元素退 1px 羽化，缓解 0/255 硬边的锯齿视感。
    镂空区一律不透明：那是我们重建出来的材质，不参与半透明。
    """
    solid = alpha if holes is None else (alpha | holes)
    if _wants_matting(inst, crop, alpha):
        a, rgb, info = solve_alpha(crop, solid)
        if info["applied"]:
            if holes is not None:
                a = np.maximum(a, holes.astype(np.uint8) * 255)
            runlog.event("LAYER", f"{inst.instance_id}（{inst.query.name}）"
                                  f"连续 alpha：未知带 {info['unknown_px']}px "
                                  f"解出 {info['solved_px']}px 带宽 {info['width']}")
            return np.dstack([rgb, a])
    return np.dstack([crop, feather_binary(solid)])


def _wants_matting(inst: Instance, crop: np.ndarray, alpha: np.ndarray) -> bool:
    """该层是否走连续 alpha：装饰类，或边界确实是软的。"""
    if config.MATTING_DISABLE or not alpha.any():
        return False
    if inst.query.kind == "decoration" or inst.query.role == "decoration":
        return True
    return is_soft_edge(crop, alpha)


def _make_layer(inst: Instance, bbox, repair_mode: Optional[str] = None) -> ImageLayer:
    x0, y0, x1, y1 = bbox
    return ImageLayer(
        instance_id=inst.instance_id,
        name=inst.query.name,
        kind=inst.query.kind,
        role=inst.query.role,
        x=x0, y=y0, w=x1 - x0, h=y1 - y0,
        z=inst.query.z_order,
        parent=None,  # 导出时由 pipeline 按 query 父链映射
        png_rel="",
        repair_mode=repair_mode if repair_mode is not None
        else inst.query.element_repair_mode,
    )
