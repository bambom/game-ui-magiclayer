"""Gemini REST 调用（规划与生图共用）。

环境变量：
- GEMINI_API_KEY（或 MAGICLAYER_GEMINI_KEY）
- MAGICLAYER_GEMINI_BASE：端点基址，默认官方 generativelanguage.googleapis.com
- MAGICLAYER_PLANNER_MODEL（默认 gemini-3.5-flash，文档指定 3.7 但网关需另配）
- MAGICLAYER_IMAGE_MODEL（默认 gemini-3.1-flash-image，未配置时自动探测）
"""
from __future__ import annotations

import base64
import json as _json
import os
import time
from typing import List, Optional

import requests

from .. import config
from .. import runlog
from ..common import decode_image, encode_png


def _resp_bytes(resp) -> int:
    """响应字节数（fake 响应对象缺 content 时回落 text，再缺则 0）。"""
    try:
        return len(resp.content)
    except AttributeError:
        return len(getattr(resp, "text", "") or "")


class GeminiError(RuntimeError):
    pass


def api_key(override: Optional[str] = None) -> str:
    if override:
        return override
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("MAGICLAYER_GEMINI_KEY") or ""
    if not key:
        raise GeminiError("缺少 GEMINI_API_KEY 环境变量")
    return key


def base_url(override: Optional[str] = None) -> str:
    if override:
        return override.rstrip("/")
    return os.environ.get("MAGICLAYER_GEMINI_BASE", config.GEMINI_BASE).rstrip("/")


def _headers(key: Optional[str] = None) -> dict:
    token = api_key(key)
    # 官方用 x-goog-api-key；本地网关多用 Bearer，两个都带以兼容
    return {"Content-Type": "application/json",
            "x-goog-api-key": token,
            "Authorization": f"Bearer {token}"}


def _post(url: str, body: dict, key: Optional[str] = None) -> dict:
    req_bytes = len(_json.dumps(body)) if runlog.enabled() else 0
    last_err: Optional[str] = None
    for attempt in range(config.GEMINI_RETRIES):
        t0 = time.time()
        try:
            resp = requests.post(url, json=body, headers=_headers(key),
                                 timeout=config.GEMINI_TIMEOUT)
        except requests.RequestException as e:
            runlog.event("REQ", f"gemini {url} attempt={attempt + 1} 网络错误 "
                                f"{time.time() - t0:.1f}s req≈{req_bytes}B: {e}")
            last_err = f"网络错误: {e}"
        else:
            runlog.event("REQ", f"gemini {url} attempt={attempt + 1} -> "
                                f"HTTP {resp.status_code} {time.time() - t0:.1f}s "
                                f"req≈{req_bytes}B resp={_resp_bytes(resp)}B")
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code not in (429, 500, 502, 503):
                raise GeminiError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        time.sleep(2 ** attempt)
    raise GeminiError(f"重试 {config.GEMINI_RETRIES} 次后仍失败：{last_err}")


def _part_image(image_rgb) -> dict:
    data = base64.b64encode(encode_png(image_rgb)).decode()
    return {"inlineData": {"mimeType": "image/png", "data": data}}


def _extract_text(resp: dict) -> str:
    parts = (resp.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text:
        raise GeminiError(f"响应无文本: {str(resp)[:500]}")
    return text


def extract_json(text: str) -> dict:
    """剥掉代码围栏/前后杂文后解析 JSON。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise GeminiError(f"响应不是 JSON: {text[:300]}")
    import json

    return json.loads(t[start:end + 1])


def list_models() -> List[dict]:
    url = f"{base_url()}/v1beta/models?pageSize=200"
    resp = requests.get(url, headers=_headers(), timeout=60)
    if resp.status_code != 200:
        raise GeminiError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("models", [])


def resolve_image_model() -> str:
    """环境变量 > 配置默认 > 自动探测（取版本最新的图像输出模型）。"""
    configured = os.environ.get("MAGICLAYER_IMAGE_MODEL", "") or config.IMAGE_MODEL_DEFAULT
    if configured:
        return configured
    models = list_models()
    cands = []
    for m in models:
        name = m.get("name", "").removeprefix("models/")
        methods = m.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        if "image" in name and "imagen" not in name and "embedding" not in name:
            cands.append(name)
    if not cands:
        raise GeminiError(f"未找到支持图像输出的模型，可用: {[m.get('name') for m in models][:40]}")
    return sorted(cands)[-1]


def generate_text(model: str, prompt: str, images: List,
                  base: Optional[str] = None, key: Optional[str] = None) -> str:
    """多图 + 文本 -> 文本（planner 用）。"""
    parts = [_part_image(im) for im in images]
    parts.append({"text": prompt})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    url = f"{base_url(base)}/v1beta/models/{model}:generateContent"
    return _extract_text(_post(url, body, key=key))


def generate_image(model: str, prompt: str, images: List,
                   base: Optional[str] = None, key: Optional[str] = None) -> object:
    """多图 + 文本 -> 图像（修补用）。返回 RGB ndarray。"""
    parts = [_part_image(im) for im in images]
    parts.append({"text": prompt})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    url = f"{base_url(base)}/v1beta/models/{model}:generateContent"
    resp = _post(url, body, key=key)
    for p in (resp.get("candidates") or [{}])[0].get("content", {}).get("parts") or []:
        inline = p.get("inlineData") or p.get("inline_data")
        if inline and inline.get("data"):
            return decode_image(base64.b64decode(inline["data"]))
    raise GeminiError(f"响应无图像: {str(resp)[:500]}")
