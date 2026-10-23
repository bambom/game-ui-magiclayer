"""离线联调用 mock planner：从 JSON 文件读取规划结果。

仅供管线联调与回归测试；真实运行必须用 gemini planner。
"""
from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np

from ..common import OcrRegion, Plan
from .schema import parse_plan


def load_plan(path: str, regions: List[OcrRegion]) -> Tuple[Plan, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_plan(raw, [r.id for r in regions])
