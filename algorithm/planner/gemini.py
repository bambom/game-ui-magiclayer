"""Gemini 3.7 Flash 图层规划调用（文档 §2.1），经 dispatch 兼容双协议。"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from .. import config
from .. import runlog
from ..common import OcrRegion, Plan
from ..providers.dispatch import dispatch
from ..providers.gemini import extract_json
from .prompt import PLANNER_PROMPT, format_ocr_regions
from .schema import parse_plan


class PlannerError(RuntimeError):
    pass


def run_planner(original_rgb: np.ndarray, cleaned_rgb: np.ndarray,
                regions: List[OcrRegion]) -> tuple[Plan, List[str]]:
    """原图 + 去字图 + OCR 稳定 ID + Prompt -> 校验后的 Plan。"""
    from ..providers.stage import stage_endpoint

    endpoint = stage_endpoint("PLANNER")
    model = endpoint.model or config.PLANNER_MODEL_DEFAULT
    prompt = PLANNER_PROMPT + "\n\nOCR regions:\n" + format_ocr_regions(regions)
    record = {
        "scope": "PLANNER", "model": model,
        "base": endpoint.base or "（全局默认网关）",
        "wire": "POST {base}/chat/completions（文本输出，图片以 base64 内联）",
        "params": {}, "prompt_name": "PLANNER_PROMPT", "prompt": prompt,
        "images": ["原图（input.png）", "去字图（02_erased.png）"],
    }
    t0 = time.time()
    try:
        text = dispatch.generate_text(model, prompt,
                                      [original_rgb, cleaned_rgb],
                                      base=endpoint.base, key=endpoint.key)
    except Exception as e:
        record.update(duration_ms=round((time.time() - t0) * 1000),
                      outcome="失败: " + str(e)[:160])
        runlog.model_call(**record)
        raise
    record.update(duration_ms=round((time.time() - t0) * 1000),
                  outcome="ok（返回 JSON 图层规划）")
    runlog.model_call(**record)
    raw = extract_json(text)
    return parse_plan(raw, [r.id for r in regions])
