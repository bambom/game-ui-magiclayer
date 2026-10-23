"""图层组合（文档 §4）：背景 + 图片层（父子/ z 序）+ 文字层 + 复原图。"""
from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np

from ..common import ImageLayer, TextLayer


def order_layers(layers: List[Tuple[ImageLayer, np.ndarray]]) -> List[Tuple[ImageLayer, np.ndarray]]:
    """先根后子（父链拓扑）、同层按 z 从后到前。"""
    by_id = {layer.instance_id: (layer, rgba) for layer, rgba in layers}
    children: dict = {}
    for layer, _ in layers:
        children.setdefault(layer.parent, []).append(layer.instance_id)

    ordered_ids: List[str] = []

    def visit(lid: str) -> None:
        if lid in ordered_ids:
            return
        ordered_ids.append(lid)

    roots = [lid for lid, (l, _) in by_id.items() if l.parent not in by_id]
    stack = sorted(roots, key=lambda i: by_id[i][0].z)
    queue = list(stack)
    while queue:
        lid = queue.pop(0)
        visit(lid)
        kids = sorted(children.get(lid, []), key=lambda i: by_id[i][0].z)
        for k in kids:
            if k not in ordered_ids:
                queue.append(k)
    for lid in by_id:
        visit(lid)
    return [by_id[lid] for lid in ordered_ids]


def compose_preview(background: np.ndarray,
                    layers: List[Tuple[ImageLayer, np.ndarray]],
                    text_layers: List[TextLayer],
                    text_pngs: dict) -> np.ndarray:
    """按坐标与层级合成复原图（最后渲染文字层）。"""
    canvas = background.copy().astype(np.uint8)
    for layer, rgba in order_layers(layers):
        if rgba is None:
            continue
        _paste(canvas, rgba, layer.x, layer.y)
    for tl in text_layers:
        png = text_pngs.get(tl.region_id)
        if png is None:
            continue
        _paste(canvas, png, int(round(tl.x)), int(round(tl.y)))
    return canvas


def _paste(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    h, w = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    if x1 <= x0 or y1 <= y0:
        return
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (sub[:, :, 3:4].astype(np.float32) / 255.0)
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (sub[:, :, :3].astype(np.float32) * a + region * (1 - a)).astype(np.uint8)


def save_outputs(out_dir: str, background: np.ndarray,
                 layers: List[Tuple[ImageLayer, np.ndarray]],
                 text_layers: List[TextLayer], text_pngs: dict,
                 preview: np.ndarray) -> None:
    os.makedirs(os.path.join(out_dir, "layers"), exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "background.png"),
                cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "preview.png"),
                cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    for layer, rgba in layers:
        if rgba is None:
            continue
        layer.png_rel = f"layers/{_safe(layer.instance_id)}.png"
        cv2.imwrite(os.path.join(out_dir, layer.png_rel),
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    for tl in text_layers:
        png = text_pngs.get(tl.region_id)
        if png is None:
            continue
        os.makedirs(os.path.join(out_dir, "layers_text"), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "layers_text", f"{_safe(tl.region_id)}.png"),
                    cv2.cvtColor(png, cv2.COLOR_RGBA2BGRA))


def _safe(name: str) -> str:
    return name.replace("#", "_").replace("/", "_")
