"""为真实样例图生成离线联调 plan.json（mock planner 用）。

仅用于管线联调与回归；真实运行由 Gemini 规划。
用法：python scripts/make_sample_plan.py <image> <out.json>
"""
from __future__ import annotations

import json
import sys


def grid_cards():
    """M-MpArcANJw.png（840x1872）的 5x5 卡牌网格近似框。"""
    cols = [55, 210, 362, 513, 665]
    rows = [245, 545, 838, 1130, 1420]
    cw, ch = 120, 272
    hints = []
    for r, y in enumerate(rows, 1):
        for c, x in enumerate(cols, 1):
            x1, y1 = min(x + cw, 840), min(y + ch, 1872)
            hints.append({
                "bbox_norm": [x / 840, y / 1872, x1 / 840, y1 / 1872],
                "positive_points_norm": [[(x + cw / 2) / 840, (y + ch / 2) / 1872]],
                "negative_points_norm": [],
            })
    return hints


def plan_for_m_mp_arcanjw() -> dict:
    def box(x0, y0, x1, y1):
        return {"bbox_norm": [x0 / 840, y0 / 1872, x1 / 840, y1 / 1872],
                "positive_points_norm": [[(x0 + x1) / 2 / 840, (y0 + y1) / 2 / 1872]],
                "negative_points_norm": []}

    def q(qid, kind, role, repair, parent, z, *hints):
        return {"id": qid, "name": qid.replace("_", " "), "kind": kind,
                "role": role, "element_repair_mode": repair,
                "parent_query_id": parent, "z_order": z,
                "geometry_hints": list(hints)}

    return {
        "scene_summary": "card collection screen: white panel with 25 card slots on sky background",
        "background_repair": {"mode": "scene",
                              "reason": "mountain sky visible at edges behind the panel"},
        "text_corrections": [],
        "raster_text_ids": ["text_001", "text_002"],
        "queries": [
            q("main_panel", "panel", "container", "image", None, 1,
              box(20, 160, 820, 1730)),
            q("crest_icon", "logo", "foreground", "none", None, 2,
              box(25, 25, 195, 195)),
            q("res_bar_left", "panel", "container", "surface", None, 2,
              box(425, 55, 650, 145)),
            q("res_bar_right", "panel", "container", "surface", None, 2,
              box(645, 55, 805, 145)),
            q("title_frame", "decoration", "decoration", "surface", None, 2,
              box(245, 165, 595, 235)),
            q("nav_bar", "panel", "container", "surface", "main_panel", 4,
              box(40, 1650, 800, 1725)),
            q("footer_bar", "panel", "container", "surface", None, 1,
              box(0, 1735, 840, 1872)),
            q("card_slot", "card", "container", "image", "main_panel", 3,
              *grid_cards()),
        ],
    }


if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv) > 2 else "plan.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(plan_for_m_mp_arcanjw(), f, ensure_ascii=False, indent=2)
    print(f"已生成 {out}: 8 query / {len(grid_cards()) + 7} 实例提示")
