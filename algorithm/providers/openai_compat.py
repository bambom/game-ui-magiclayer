"""OpenAI 兼容协议（/v1/chat/completions）实现。

用于本地网关（antigravity 等）；与 Gemini 原生协议经 dispatch 切换。
生图响应形状（实测）：choices[0].message.images[].image_url.url = data URI；
个别实现把图放在 content 的 markdown data URI 里，一并兼容。
OpenRouter 的图像走专用 POST {base}/images（input_references 图生图）。
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
from typing import List, Optional

import requests

from .. import config
from .. import runlog
from ..common import decode_image, encode_png, encode_rgba_mask_png


def _resp_bytes(resp) -> int:
    """响应字节数（fake 响应对象缺 content 时回落 text，再缺则 0）。"""
    try:
        return len(resp.content)
    except AttributeError:
        return len(getattr(resp, "text", "") or "")


class OpenAIError(RuntimeError):
    pass


class ModelCooldownError(OpenAIError):
    """上游全部凭据处于长冷却（配额用尽）：重试无意义，可换线或降级。"""


def api_key(override: Optional[str] = None) -> str:
    if override:
        return override
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        raise OpenAIError("缺少 GEMINI_API_KEY/OPENAI_API_KEY 环境变量")
    return key


def base_url(override: Optional[str] = None) -> str:
    if override:
        return override.rstrip("/")
    return os.environ.get("MAGICLAYER_OPENAI_BASE", config.OPENAI_BASE).rstrip("/")


def _post_chat(body: dict, base: Optional[str] = None,
               key: Optional[str] = None) -> dict:
    url = f"{base_url(base)}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key(key)}"}
    req_bytes = len(json.dumps(body)) if runlog.enabled() else 0
    last_err: Optional[str] = None
    for attempt in range(config.GEMINI_RETRIES):
        t0 = time.time()
        try:
            resp = requests.post(url, json=body, headers=headers,
                                 timeout=config.GEMINI_TIMEOUT)
        except requests.RequestException as e:
            runlog.event("REQ", f"chat {url} model={body.get('model')} "
                                f"attempt={attempt + 1} 网络错误 {time.time() - t0:.1f}s "
                                f"req≈{req_bytes}B: {e}")
            last_err = f"网络错误: {e}"
        else:
            runlog.event("REQ", f"chat {url} model={body.get('model')} "
                                f"attempt={attempt + 1} -> HTTP {resp.status_code} "
                                f"{time.time() - t0:.1f}s req≈{req_bytes}B "
                                f"resp={_resp_bytes(resp)}B")
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                # 长冷却（上游配额用尽）重试无意义，快速失败交给上层降级
                try:
                    detail = resp.json()
                    reset = int(detail.get("error", {}).get("reset_seconds", 0) or 0)
                except ValueError:
                    reset = 0
                if reset > 120:
                    raise ModelCooldownError(f"模型冷却中，约 {reset}s 后恢复: {resp.text[:200]}")
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise OpenAIError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        time.sleep(2 ** attempt)
    raise OpenAIError(f"重试 {config.GEMINI_RETRIES} 次后仍失败：{last_err}")


def _image_part(image_rgb) -> dict:
    data = base64.b64encode(encode_png(image_rgb)).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"}}


def _message_content(msg: dict) -> object:
    c = msg.get("content")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def generate_text(model: str, prompt: str, images: List,
                  base: Optional[str] = None, key: Optional[str] = None) -> str:
    parts: List[dict] = [{"type": "text", "text": prompt}]
    parts += [_image_part(im) for im in images]
    resp = _post_chat({"model": model, "messages": [{"role": "user", "content": parts}]},
                      base=base, key=key)
    try:
        text = _message_content(resp["choices"][0]["message"])
    except (KeyError, IndexError) as e:
        raise OpenAIError(f"响应缺少 choices: {str(resp)[:400]}") from e
    if not text:
        raise OpenAIError(f"响应无文本: {str(resp)[:400]}")
    return text


_DATA_URI = re.compile(r"data:image/[a-zA-Z]+;base64,([A-Za-z0-9+/=\s]+)")


def generate_image(model: str, prompt: str, images: List,
                   base: Optional[str] = None, key: Optional[str] = None):
    parts: List[dict] = [{"type": "text", "text": prompt}]
    parts += [_image_part(im) for im in images]
    resp = _post_chat({"model": model, "messages": [{"role": "user", "content": parts}]},
                      base=base, key=key)
    msg = resp["choices"][0]["message"]

    for im in msg.get("images") or []:
        url = (im.get("image_url") or {}).get("url", "")
        m = _DATA_URI.search(url or "")
        if m:
            return _decode_b64(m.group(1))

    text = _message_content(msg)
    if text:
        m = _DATA_URI.search(text)
        if m:
            return _decode_b64(m.group(1))
    raise OpenAIError(f"响应无图像: {str(resp)[:400]}")


def _decode_b64(b64: str):
    try:
        return decode_image(base64.b64decode(re.sub(r"\s", "", b64)))
    except (binascii.Error, ValueError) as e:
        raise OpenAIError(f"图像 base64 解码失败: {e}") from e


def _post_edits(files: list, data: dict, base: Optional[str] = None,
                key: Optional[str] = None) -> dict:
    url = f"{base_url(base)}/images/edits"
    headers = {"Authorization": f"Bearer {api_key(key)}"}
    req_bytes = (sum(len(payload) for _, (_, payload, _) in files) + len(str(data))) \
        if runlog.enabled() else 0
    last_err: Optional[str] = None
    for attempt in range(config.GEMINI_RETRIES):
        t0 = time.time()
        try:
            resp = requests.post(url, data=data, files=files, headers=headers,
                                 timeout=config.GEMINI_TIMEOUT)
        except requests.RequestException as e:
            runlog.event("REQ", f"edits {url} model={data.get('model')} "
                                f"attempt={attempt + 1} 网络错误 {time.time() - t0:.1f}s "
                                f"req≈{req_bytes}B: {e}")
            last_err = f"网络错误: {e}"
        else:
            runlog.event("REQ", f"edits {url} model={data.get('model')} "
                                f"attempt={attempt + 1} -> HTTP {resp.status_code} "
                                f"{time.time() - t0:.1f}s req≈{req_bytes}B "
                                f"resp={_resp_bytes(resp)}B")
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                # 长冷却（上游配额用尽）重试无意义，快速失败交给上层降级
                try:
                    detail = resp.json()
                    reset = int(detail.get("error", {}).get("reset_seconds", 0) or 0)
                except ValueError:
                    reset = 0
                if reset > 120:
                    raise ModelCooldownError(f"模型冷却中，约 {reset}s 后恢复: {resp.text[:200]}")
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise OpenAIError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        time.sleep(2 ** attempt)
    raise OpenAIError(f"重试 {config.GEMINI_RETRIES} 次后仍失败：{last_err}")


def generate_image_edits(model: str, prompt: str, images: List,
                         base: Optional[str] = None, key: Optional[str] = None,
                         quality: Optional[str] = None,
                         opaque_editable: bool = False):
    """图进图出：multipart POST {base}/images/edits。

    images[0]=原图（RGB ndarray）；images[1] 可选二值遮罩（白=待修补，按 edits 线
    约定编码；``opaque_editable`` 供遮罩极性反转的网关——不透明=待修补）。
    无遮罩时整图交由 prompt 描述修补（背景 scene 场景）。输出尺寸不保真，
    调用方负责缩放回输入几何。quality（high/medium/low）是 gpt-image 系的保真档位。
    """
    files = [("image", ("image.png", encode_png(images[0]), "image/png"))]
    if len(images) > 1:
        files.append(("mask", ("mask.png",
                               encode_rgba_mask_png(images[1],
                                                    opaque_editable=opaque_editable),
                               "image/png")))
    data = {"model": model, "prompt": prompt, "size": "auto"}
    if quality:
        data["quality"] = quality
    resp = _post_edits(files, data, base=base, key=key)
    item = (resp.get("data") or [{}])[0] or {}
    b64 = item.get("b64_json", "") or ""
    if b64:
        return _decode_b64(b64)
    # Grok edits（probe-bg-r1）：常回 data[0].url JPEG 而无 b64_json。
    url = (item.get("url") or "").strip()
    if url.startswith("http://") or url.startswith("https://"):
        runlog.event("REQ", f"edits 无 b64_json，改下 URL {url[:90]}")
        return _download_url_image(url)
    raise OpenAIError(f"响应无图像: {str(resp)[:400]}")


def _download_url_image(url: str):
    """下载 edits 线返回的图像 URL（Grok 网关）。超时单独封顶 60s。

    本地 edits POST 走 18080，结果 JPEG 在 imgen.x.ai；直连超时则改试
    ``MAGICLAYER_IMAGE_URL_PROXY``（与 OpenRouter 同一 VPN 口）。
    """
    timeout = min(60, int(config.GEMINI_TIMEOUT) or 60)
    proxy = os.environ.get("MAGICLAYER_IMAGE_URL_PROXY", "").strip()
    attempts = []
    if proxy:
        attempts.append({"http": proxy, "https": proxy})
    attempts.append(None)
    last_err: Optional[str] = None
    for proxies in attempts:
        via = (proxies or {}).get("https") or "direct"
        try:
            resp = requests.get(url, timeout=timeout, proxies=proxies)
        except requests.RequestException as e:
            last_err = f"网络错误 via {via}: {e}"
            runlog.event("REQ", f"edits-url GET {url[:90]} via {via} {last_err}")
            continue
        runlog.event("REQ", f"edits-url GET {url[:90]} via {via} "
                            f"-> HTTP {resp.status_code} resp={_resp_bytes(resp)}B")
        if resp.status_code == 200:
            return decode_image(resp.content)
        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
    raise OpenAIError(f"图像 URL 下载失败：{last_err}")


def generate_image_openrouter(model: str, prompt: str, images: List,
                              base: Optional[str] = None, key: Optional[str] = None,
                              quality: Optional[str] = None):
    """OpenRouter 专用图像线：POST {base}/images，input_references 图生图。

    实测（2026-08-31）：请求体 {"model","prompt","input_references":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,.."}}]}；遮罩
    （白=待修补）作为第二张参考图随 ATLAS_PROMPT 的文字约定一起送；响应
    data[0].b64_json。不发送 size——按模型原生尺寸出图，调用方 _fit_size
    缩回输入几何；quality 仅模型声明支持时发送（gpt-image 系），否则 400。
    """
    refs = [{"type": "image_url",
             "image_url": {"url": "data:image/png;base64,"
                                  + base64.b64encode(encode_png(im)).decode()}}
            for im in images]
    body = {"model": model, "prompt": prompt, "input_references": refs}
    if quality:
        body["quality"] = quality
    url = f"{base_url(base)}/images"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key(key)}"}
    req_bytes = len(json.dumps(body)) if runlog.enabled() else 0
    last_err: Optional[str] = None
    for attempt in range(config.GEMINI_RETRIES):
        t0 = time.time()
        try:
            resp = requests.post(url, json=body, headers=headers,
                                 timeout=config.GEMINI_TIMEOUT)
        except requests.RequestException as e:
            runlog.event("REQ", f"openrouter-images {url} model={model} "
                                f"attempt={attempt + 1} 网络错误 "
                                f"{time.time() - t0:.1f}s req≈{req_bytes}B: {e}")
            last_err = f"网络错误: {e}"
        else:
            runlog.event("REQ", f"openrouter-images {url} model={model} "
                                f"attempt={attempt + 1} -> HTTP {resp.status_code} "
                                f"{time.time() - t0:.1f}s req≈{req_bytes}B "
                                f"resp={_resp_bytes(resp)}B")
            if resp.status_code == 200:
                b64 = ((resp.json().get("data") or [{}])[0]).get("b64_json", "")
                if not b64:
                    raise OpenAIError(f"响应无图像: {resp.text[:400]}")
                return _decode_b64(b64)
            if resp.status_code == 429:
                try:
                    detail = resp.json()
                    reset = int(detail.get("error", {}).get("reset_seconds", 0) or 0)
                except ValueError:
                    reset = 0
                if reset > 120:
                    raise ModelCooldownError(f"模型冷却中，约 {reset}s 后恢复: {resp.text[:200]}")
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise OpenAIError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        time.sleep(2 ** attempt)
    raise OpenAIError(f"重试 {config.GEMINI_RETRIES} 次后仍失败：{last_err}")


def list_models() -> List[str]:
    resp = requests.get(f"{base_url()}/models",
                        headers={"Authorization": f"Bearer {api_key()}"},
                        timeout=30)
    if resp.status_code != 200:
        raise OpenAIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return [m["id"] for m in resp.json().get("data", [])]


def resolve_image_model() -> str:
    configured = os.environ.get("MAGICLAYER_IMAGE_MODEL", "") or config.IMAGE_MODEL_DEFAULT
    if configured:
        return configured
    for name in list_models():
        if "image" in name and "embedding" not in name:
            return name
    raise OpenAIError("网关无图像输出模型")
