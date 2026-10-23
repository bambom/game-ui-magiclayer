"""LLM 图层规划 Prompt（文档 §2.1.3 英文原文逐字）与 OCR 区域列表序列化。"""
from __future__ import annotations

import json
from typing import List

from ..common import OcrRegion

PLANNER_PROMPT = """Analyze a game UI and return a compact executable layer plan.

Two images are provided in this order: (1) the original image, (2) an OCR-cleaned working image.

OCR regions are listed with stable ids below. First classify their rendering mode:
- ordinary readable UI copy (labels, values, hints, button captions) remains editable text;
- a logo, wordmark or decorative display lettering whose irregular shape, multicolor fill, thick custom outline, embedded mascot/icon or hand-drawn geometry is part of its identity must remain a raster image asset. Put only those OCR ids in raster_text_ids.

Do not classify text as raster merely because it is bold, outlined or placed on a button. A normal font that can be acceptably recreated as text must stay editable. This is a rendering decision, not an OCR-confidence decision.

The cleaned image initially has all OCR glyphs removed. For each raster_text_id, also emit exactly one non-text query (prefer kind=logo, role=foreground) using its location in the original image. The pipeline will restore that artwork before segmentation. Never emit ordinary text as a query.

Find all visible non-text assets that should be independently editable:
- large panels, boards, frames and dialogs whose exposed material/boundary is visible;
- every repeated card, slot, tab or row surface as a separate geometry hint;
- complete action/navigation buttons as one object, including any integrated icon shape;
- independent icons, item artwork, portraits, silhouettes and framed illustrations;
- badges, ribbons, locks, counters and grouped status markers with their own visible shape.

Game-UI decomposition rules:
- A card or panel whose inner content is separable must be split: one query for the frame/plate itself (covering the full frame with its border) and one child query per separable non-text content asset (illustration, portrait, badge, mini-map, inner plate), each child's parent_query_id pointing at the frame. A child's bbox and mask must stay strictly inside the frame's inner opening and never include the ornamental frame ring or border; the pipeline strips any frame pixels it finds in a child mask anyway. Split even when the content fully fills the frame: the ornamental frame ring is its own layer and the inner artwork is another. Ordinary text inside a card is never a query; it stays with OCR. Choose the frame's element_repair_mode from its material: "surface" for a near flat-colour plate (the removed content is then filled with the plate colour), "image" for a textured or gradient plate.
- A long structural line (timeline axis, progress track, connector line) is its own decoration query, and repeated node icons, orbs or markers sitting on it form a separate query with one geometry hint per instance; never merge them into the line's mask.
- General overlay rule: whenever one extracted element visually sits on top of or crosses another extracted element (a node on a line, a badge on a panel corner, a ribbon across a card, a lock on a slot), set the upper element's parent_query_id to the lower one, so the lower layer is repaired where it was covered.
- A prominent free-standing scenery cluster inside the background (a mountain group, tree cluster, rock formation or large ornament motif) that a user would edit separately is a decoration query with tight hints; it is not part of the full-screen background. Emit one query per distinct cluster: a left mountain group and a right tree cluster are two queries, not one strip covering both. Only diffuse all-over texture (paper, gradients, mist) stays with the background.

Worked example for the two rules above (abbreviated, coordinates are illustrative). An image with a mountain group on the left and a torii-with-trees cluster on the right, both sitting in the upper background, yields two decoration queries, never one strip:
{"id": "mountain_group_left", "kind": "decoration", "role": "decoration", "element_repair_mode": "none", "parent_query_id": null, "z_order": 1, "geometry_hints": [{"bbox_norm": [0.06, 0.09, 0.37, 0.31], "positive_points_norm": [[0.19, 0.19]], "negative_points_norm": []}]}
{"id": "torii_tree_cluster_right", "kind": "decoration", "role": "decoration", "element_repair_mode": "none", "parent_query_id": null, "z_order": 1, "geometry_hints": [{"bbox_norm": [0.56, 0.08, 0.91, 0.34], "positive_points_norm": [[0.72, 0.21]], "negative_points_norm": []}]}
Each bbox hugs one cluster and includes its full downward extent (tree crowns, mountain feet, trailing mist-free silhouette); it must not stop at a horizontal line that cuts the cluster in half, and a single bbox must never span the empty gap between the two clusters.

Do not emit the full-screen background, ordinary text, shadows, gaps, hairline separators that carry no iconography, speculative objects, duplicate composite descriptions, or people/props inside one framed illustration. A parent surface and its foreground child must be separate only when both are real editable assets. Repeated instances share one query but each instance needs its own hint.

Every hint is passed directly to segmentation:
- bbox_norm=[x0,y0,x1,y1] tightly encloses exactly one instance, normalized to 0..1;
- positive_points_norm has 1-3 points on solid target pixels, not edges, holes or background;
- negative_points_norm is empty unless a real competing object is near/inside the box; when used, put 1-2 points on that competitor, never on an overlaid child of an underlying surface;
- inspect each repeated instance separately. Do not mechanically copy relative point positions.
- z_order is back-to-front. Parent surfaces must have a lower z_order than their children.
- parent_query_id is required for every element visually carried by another exported element. Use the nearest direct material parent, not merely the largest outer panel: artwork, silhouette, badge or label inside a card belongs to that card; the card belongs to its panel. Leave it null only for true root elements. This parent chain is exported and also controls occlusion repair.

For every query choose one element_repair_mode:
- "image" for a textured, patterned, illustrated, translucent or gradient-bearing parent material whose child elements must later be removed and reconstructed by the image model;
- "surface" only for a nearly uniform flat-colour material that can be reconstructed from measured neighbouring pixels;
- "none" for a topmost foreground object or an object with no meaningful material behind children.

This is a material decision. A panel/card/frame with artwork, glow or visible texture is "image".

Choose one background_repair.mode for the pixels hidden by the extracted UI:
- "scene": the hidden area is a real textured/illustrated scene and should be reconstructed from the visible scene boundary. The repair image will contain black holes for missing content; do not treat the black as scene content.
- "surface": the hidden area is visually close to a uniform or translucent colour/material. Use measured boundary colour/low-frequency continuation only; do not invent scenery or texture.
- "none": do not synthesize hidden pixels; preserve the current cleaned background.

Select from the image evidence, not from the number of detected layers.

Return JSON only, with no prose and no fields outside this schema. The original image is provided only for the text audit; the text-cleaned image is the source for non-text layer planning.

{
  "scene_summary": "short description",
  "background_repair": {"mode": "scene|surface|none", "reason": "short evidence-based reason"},
  "text_corrections": [{"text": "visible text missed by OCR", "bbox_norm": [0.1,0.1,0.2,0.2], "confidence": 0.8}],
  "raster_text_ids": ["text_001"],
  "queries": [{
    "id": "snake_case_id",
    "name": "short user-facing name",
    "kind": "panel|card|button|icon|illustration|badge|decoration|logo",
    "role": "container|surface|button|foreground|decoration",
    "element_repair_mode": "image|surface|none",
    "parent_query_id": null,
    "z_order": 0,
    "geometry_hints": [{"bbox_norm": [0.1,0.1,0.2,0.2], "positive_points_norm": [[0.15,0.15]], "negative_points_norm": []}]
  }]
}"""


def format_ocr_regions(regions: List[OcrRegion]) -> str:
    """把带稳定 ID 的 OCR 区域列表序列化进 prompt（格式文档未定，取紧凑 JSON 行）。"""
    lines = []
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        lines.append(json.dumps({
            "id": r.id,
            "text": r.text,
            "confidence": round(r.conf, 4),
            "bbox": [x0, y0, x1, y1],
        }, ensure_ascii=False))
    return "\n".join(lines)
