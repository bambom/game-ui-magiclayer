"""LLM 返回 JSON 的 schema 校验与规范化（文档 §2.1.1）。

策略：能收敛的收敛（大小写、缺省名、未知父级置空），不可收敛的丢弃该
query/hint 并记录 warning；全部丢弃才视为失败。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..common import Hint, Instance, Plan, Query

KINDS = {"panel", "card", "button", "icon", "illustration", "badge", "decoration", "logo"}
ROLES = {"container", "surface", "button", "foreground", "decoration"}
REPAIR_MODES = {"image", "surface", "none"}
BG_MODES = {"scene", "surface", "none"}


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN 拒绝


def _parse_bbox_norm(v) -> Optional[List[float]]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    vals = [_num(x) for x in v]
    if any(x is None for x in vals):
        return None
    x0, y0, x1, y1 = vals
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return [x0, y0, x1, y1]


def _parse_points(v, lo: int, hi: int) -> Optional[List[List[float]]]:
    if v is None:
        return [] if lo == 0 else None
    if not isinstance(v, list) or not (lo <= len(v) <= hi):
        return None
    out = []
    for p in v:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        x, y = _num(p[0]), _num(p[1])
        if x is None or y is None:
            return None
        out.append([x, y])
    return out


def parse_plan(raw: dict, ocr_ids: List[str]) -> Tuple[Plan, List[str]]:
    warnings: List[str] = []
    known_ids = set(ocr_ids)

    bg = raw.get("background_repair") or {}
    bg_mode = str(bg.get("mode", "")).lower()
    if bg_mode not in BG_MODES:
        warnings.append(f"background_repair.mode 非法: {bg_mode!r}，按 none 处理")
        bg_mode = "none"

    raster_ids = []
    for rid in raw.get("raster_text_ids") or []:
        if rid in known_ids:
            raster_ids.append(rid)
        else:
            warnings.append(f"raster_text_ids 未知 id 被丢弃: {rid!r}")

    corrections = []
    for c in raw.get("text_corrections") or []:
        if not isinstance(c, dict):
            continue
        bbox = _parse_bbox_norm(c.get("bbox_norm"))
        text = str(c.get("text") or "").strip()
        if not text or bbox is None:
            warnings.append(f"text_corrections 条目被丢弃: {c!r}")
            continue
        corrections.append({"text": text, "bbox_norm": bbox,
                            "confidence": _num(c.get("confidence")) or 0.5})

    queries: List[Query] = []
    raw_queries = raw.get("queries") or []
    id_set = set()
    for q in raw_queries:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        if not qid or qid in id_set:
            warnings.append(f"query id 缺失或重复，条目被丢弃: {q.get('id')!r}")
            continue
        kind = str(q.get("kind") or "").lower()
        role = str(q.get("role") or "").lower()
        repair = str(q.get("element_repair_mode") or "").lower()
        if kind not in KINDS or role not in ROLES or repair not in REPAIR_MODES:
            warnings.append(f"query {qid} 枚举字段非法被丢弃 (kind={kind}, role={role}, repair={repair})")
            continue
        try:
            z = int(q.get("z_order", 0))
        except (TypeError, ValueError):
            warnings.append(f"query {qid} z_order 非法，按 0 处理")
            z = 0
        hints: List[Hint] = []
        for h in q.get("geometry_hints") or []:
            if not isinstance(h, dict):
                continue
            bbox = _parse_bbox_norm(h.get("bbox_norm"))
            pos = _parse_points(h.get("positive_points_norm"), 1, 3)
            neg = _parse_points(h.get("negative_points_norm"), 0, 2)
            if bbox is None or pos is None or neg is None:
                warnings.append(f"query {qid} 的 hint 被丢弃: {h!r}")
                continue
            hints.append(Hint(bbox_norm=bbox, positive_points_norm=pos,
                              negative_points_norm=neg, bbox=(0, 0, 0, 0),
                              positive_points=[], negative_points=[]))
        if not hints:
            warnings.append(f"query {qid} 无有效 geometry_hints，被丢弃")
            continue
        parent = q.get("parent_query_id")
        parent = str(parent).strip() if parent else None
        id_set.add(qid)
        queries.append(Query(
            id=qid, name=str(q.get("name") or qid), kind=kind, role=role,
            element_repair_mode=repair, parent_query_id=parent or None,
            z_order=z, hints=hints))

    # 父级引用修正 + 环检测
    by_id = {q.id: q for q in queries}
    for q in queries:
        if q.parent_query_id is not None and q.parent_query_id not in by_id:
            warnings.append(f"query {q.id} 的 parent_query_id {q.parent_query_id!r} 不存在，置空")
            q.parent_query_id = None
    for q in queries:
        seen, node = {q.id}, q
        while node.parent_query_id is not None:
            if node.parent_query_id in seen:
                warnings.append(f"query {q.id} 父链成环，已断开")
                node.parent_query_id = None
                break
            seen.add(node.parent_query_id)
            node = by_id[node.parent_query_id]

    plan = Plan(
        scene_summary=str(raw.get("scene_summary") or ""),
        background_mode=bg_mode,
        background_reason=str(bg.get("reason") or ""),
        raster_text_ids=raster_ids,
        text_corrections=corrections,
        queries=queries,
        raw=raw,
    )
    return plan, warnings


def expand_instances(plan: Plan) -> List[Instance]:
    """query x hint 展开为实例；单 hint 实例沿用 query.id。"""
    out: List[Instance] = []
    for q in plan.queries:
        if len(q.hints) == 1:
            out.append(Instance(instance_id=q.id, query=q, hint=q.hints[0]))
        else:
            for i, h in enumerate(q.hints, 1):
                out.append(Instance(instance_id=f"{q.id}#{i}", query=q, hint=h))
    return out
