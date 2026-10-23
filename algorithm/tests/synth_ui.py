"""合成游戏 UI 图 + 配套 mock plan：离线冒烟测试夹具。

布局（480x800）：
- 天空渐变场景背景（scene 修补目标）
- main_panel (30,150,450,700) 纹理面板，z1，image 修补，根
- card_1 (50,230,240,420) 渐变卡，z2，image，父 main_panel
- card_2 (260,230,430,420) 渐变卡，z2，image，父 main_panel
- badge 圆形徽章 @ (215,245) r18，z3，none，父 card_1
- play_button (150,730,330,780)，z1，none，根
- 美术字标题（OCR text_001 -> raster）
- 普通文字标签（OCR text_002 -> 可编辑文字层）
- text_corrections 一条漏识文字
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw

W, H = 480, 800
GOLD = (240, 200, 80)


def _gradient(size, top, bottom):
    w, h = size
    t = np.linspace(0, 1, h)[:, None]
    arr = (np.array(top)[None, None, :] * (1 - t[..., None])
           + np.array(bottom)[None, None, :] * t[..., None])
    return np.tile(arr, (1, w, 1)).astype(np.uint8)


def build_image() -> np.ndarray:
    img = _gradient((W, H), (110, 160, 220), (235, 242, 250)).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    # 云朵
    for cx, cy, r in [(80, 90, 40), (140, 110, 30), (380, 60, 45)]:
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=(250, 250, 252))
    arr = np.asarray(pil).copy()
    # main_panel：带噪点纹理的面板
    noise = np.random.RandomState(7).normal(0, 6, (700 - 150, 450 - 30, 1))
    base = np.array([180, 120, 90], np.float32)  # 暖木色
    arr[150:700, 30:450] = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    # 两张渐变卡
    arr[230:420, 50:240] = _gradient((190, 190), (90, 190, 110), (40, 120, 70))
    arr[230:420, 260:430] = _gradient((170, 190), (150, 90, 200), (80, 40, 130))
    # badge 圆形
    cv2_circle(arr, (215, 245), 18, GOLD)
    # play_button
    arr[730:780, 150:330] = np.array([200, 60, 60], np.uint8)[None, None, :]
    arr[734:776, 154:326] = np.array([230, 90, 90], np.uint8)[None, None, :]
    # 漏识文字区（普通面板色，无字形；二次文字处理会把它擦除）
    return arr


def cv2_circle(arr, center, radius, color):
    cy, cx = center[1], center[0]
    yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    m = (xx - center[0]) ** 2 + (yy - cy) ** 2 <= radius ** 2
    arr[m] = color


def draw_texts(arr: np.ndarray) -> np.ndarray:
    """画美术字标题与普通文字标签。"""
    from PIL import ImageFont

    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    try:
        big = ImageFont.truetype("arial.ttf", 64)
        small = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        big = small = None
    # 标题：多色粗字（模拟美术字）
    x = 60
    for i, ch in enumerate("MAGIC"):
        color = [(220, 60, 60), (240, 160, 40), (60, 170, 90),
                 (60, 110, 220), (150, 70, 200)][i]
        draw.text((x, 60), ch, font=big, fill=color, stroke_width=3,
                  stroke_fill=(40, 40, 40))
        x += 66
    draw.text((60, 160), "Score: 1234", font=small, fill=(40, 60, 90))
    return np.asarray(pil)


def mock_plan() -> dict:
    return {
        "scene_summary": "synthetic game ui with panel, cards and button",
        "background_repair": {"mode": "scene", "reason": "sky and clouds visible behind panel"},
        "text_corrections": [
            {"text": "Bonus", "bbox_norm": [0.55, 0.75, 0.78, 0.79], "confidence": 0.8}
        ],
        "raster_text_ids": ["text_001"],
        "queries": [
            {"id": "main_panel", "name": "Main panel", "kind": "panel",
             "role": "container", "element_repair_mode": "image",
             "parent_query_id": None, "z_order": 1,
             "geometry_hints": [{"bbox_norm": [0.0625, 0.1875, 0.9375, 0.875],
                                 "positive_points_norm": [[0.125, 0.5625]],
                                 "negative_points_norm": []}]},
            {"id": "card_1", "name": "Card 1", "kind": "card",
             "role": "container", "element_repair_mode": "image",
             "parent_query_id": "main_panel", "z_order": 2,
             "geometry_hints": [{"bbox_norm": [0.104, 0.2875, 0.5, 0.525],
                                 "positive_points_norm": [[0.25, 0.4]],
                                 "negative_points_norm": []}]},
            {"id": "card_2", "name": "Card 2", "kind": "card",
             "role": "container", "element_repair_mode": "image",
             "parent_query_id": "main_panel", "z_order": 2,
             "geometry_hints": [{"bbox_norm": [0.5417, 0.2875, 0.8958, 0.525],
                                 "positive_points_norm": [[0.7083, 0.4]],
                                 "negative_points_norm": []}]},
            {"id": "badge", "name": "Collected badge", "kind": "badge",
             "role": "foreground", "element_repair_mode": "none",
             "parent_query_id": "card_1", "z_order": 3,
             "geometry_hints": [{"bbox_norm": [0.4104, 0.2838, 0.4854, 0.3288],
                                 "positive_points_norm": [[0.4479, 0.3063]],
                                 "negative_points_norm": []}]},
            {"id": "play_button", "name": "Play button", "kind": "button",
             "role": "button", "element_repair_mode": "none",
             "parent_query_id": None, "z_order": 1,
             "geometry_hints": [{"bbox_norm": [0.3125, 0.9125, 0.6875, 0.975],
                                 "positive_points_norm": [[0.5, 0.944]],
                                 "negative_points_norm": []}]},
        ],
    }


def fixture_paths(tmp_dir: str) -> dict:
    os.makedirs(tmp_dir, exist_ok=True)
    img_path = os.path.join(tmp_dir, "synth_ui.png")
    plan_path = os.path.join(tmp_dir, "plan.json")
    Image.fromarray(draw_texts(build_image())).save(img_path)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(mock_plan(), f, ensure_ascii=False, indent=2)
    return {"image": img_path, "plan": plan_path}


def fake_ocr_regions():
    """真实 RapidOCR 的确定性替身（冒烟测试用）。"""
    from algorithm.common import OcrRegion

    title = np.array([[60, 60], [420, 60], [420, 140], [60, 140]], np.float32)
    label = np.array([[60, 160], [300, 160], [300, 195], [60, 195]], np.float32)
    return [
        OcrRegion(id="text_001", text="MAGIC", conf=0.9, quad=title,
                  bbox=(60, 60, 420, 140)),
        OcrRegion(id="text_002", text="Score: 1234", conf=0.95, quad=label,
                  bbox=(60, 160, 300, 195)),
    ]
