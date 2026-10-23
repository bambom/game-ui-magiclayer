"""Gemini 生图修补 provider。默认只送一张涂黑图，不另附 mask。

各 API 仍走各自网关线（edits / OpenRouter /images / chat），但载荷都是
「原图上有黑洞」这一张。MAGICLAYER_BG_ATTACH_MASK=1 才加第二张 mask。

经 dispatch 支持 Gemini 原生与 OpenAI 兼容两种协议。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import config
from .. import runlog
from .base import ImageRepairProvider

ATLAS_PROMPT = """Fill the black missing regions by continuing the surrounding material and style of this same element.
The black is a hole, not content to keep. Do not add text, icons, or new objects. Keep everything else unchanged.
Output one complete image at the same size."""

SCENE_PROMPT = """Fill the black missing regions by continuing this image's existing background style, texture, and scenery.
The black is a hole, not scenery to keep. Do not add UI, text, logos, or new objects. Keep everything else unchanged.
Output one complete image at the same size."""

# probe-bg-r1 合格-2/3：无掩膜时的中文口语（网页对话路径）。有掩膜的 short 格用户未裁定。
SHORT_SCENE_PROMPT = (
    "移除图上所有 UI 元素、文字和图标，把背景按周围内容自然补全，不要添加任何新东西"
)

# 网关 /v1/images/edits 只收 gpt-image-*/grok-imagine-*；其余图像模型走 chat 出图线
_EDITS_LINE_MODELS = re.compile(r"gpt-image|grok-imagine", re.IGNORECASE)


@dataclass(frozen=True)
class BackgroundSendMode:
    """背景修补按接口选择的发送模式（文档 §3.4 probe-bg-r1）。

    attach_mask: 是否把硬洞图（mask_crop 同类，白=待补）作为第二张图送出。
    prompt: SCENE_PROMPT 或 SHORT_SCENE_PROMPT。
    name: run.log 短名。
    """
    attach_mask: bool
    prompt: str
    name: str


def background_send_mode(endpoint, model: Optional[str]) -> BackgroundSendMode:
    """背景发送模式。调用时读环境变量，不在 import 冻结。

    默认（MAGICLAYER_BG_ATTACH_MASK=auto / MAGICLAYER_BG_PROMPT=auto）：
    待补区已涂黑，只送这一张图（涂黑即唯一待补标记，不再另附 mask）。
    ``MAGICLAYER_BG_ATTACH_MASK=1`` 才加一张硬 mask。
    ``endpoint``/``model`` 保留给调用方（冷却换线仍按新端点重选，便于覆盖）。
    覆盖：``MAGICLAYER_BG_ATTACH_MASK=0|1``，``MAGICLAYER_BG_PROMPT=scene|short``。
    元素 Atlas 同样只送涂黑图，不另附 mask。
    """
    _ = (endpoint, model)
    attach = False
    prompt = SCENE_PROMPT

    attach_ov = os.environ.get("MAGICLAYER_BG_ATTACH_MASK", "auto").strip().lower()
    if attach_ov in ("0", "false", "no", "nomask"):
        attach = False
    elif attach_ov in ("1", "true", "yes", "mask"):
        attach = True

    prompt_ov = os.environ.get("MAGICLAYER_BG_PROMPT", "auto").strip().lower()
    if prompt_ov == "scene":
        prompt = SCENE_PROMPT
    elif prompt_ov == "short":
        prompt = SHORT_SCENE_PROMPT
    name = ("mask" if attach else "nomask") + (
        "-short" if prompt is SHORT_SCENE_PROMPT else "-scene")
    return BackgroundSendMode(attach_mask=attach, prompt=prompt, name=name)


def _images_for_mode(mode: BackgroundSendMode, scene_rgb, hole_mask) -> list:
    if mode.attach_mask and hole_mask is not None:
        return [scene_rgb, hole_mask]
    return [scene_rgb]


class GeminiImageProvider(ImageRepairProvider):
    name = "gemini"
    degraded = False

    def __init__(self):
        self._model = None
        self._element_endpoint = None
        self._background_endpoint = None

    @property
    def model(self) -> str:
        if self._model is None:
            from .dispatch import dispatch

            self._model = dispatch.resolve_image_model()
        return self._model

    def _endpoint(self, scope: str):
        """该阶段的模型端点（服务端模型库分配；未分配时为空，回落全局配置）。"""
        from .stage import stage_endpoint

        if scope == "ELEMENT" and self._element_endpoint is None:
            self._element_endpoint = stage_endpoint("ELEMENT")
        if scope == "BACKGROUND" and self._background_endpoint is None:
            self._background_endpoint = stage_endpoint("BACKGROUND")
        return self._element_endpoint if scope == "ELEMENT" else self._background_endpoint

    def _generate_repair(self, scope: str, prompt: str, images: list,
                         bg_scene=None, bg_mask=None):
        """元素/背景修补是图生图阶段：模型库条目按 provider 分线——openrouter 走
        其专用 /images（input_references），其余条目直走 /v1/images/edits（OpenAI 线
        的图生图端点，只收 gpt-image-*/grok-imagine-* 等真图生图模型）；未分配条目时
        沿用全局配置的 chat 出图 / Gemini 原生线（gemini-*-image 等对话式出图模型）。
        首选端点处于长冷却（配额用尽）时依次改试服务端下发的其余图像条目，
        全部冷却才向上抛出，交给传统修补降级。

        每次调用经 runlog.model_call 记录结构化信息（wire/参数/prompt 全文/时延），
        由 pipeline 汇总进 debug/steps.json 供面板按步骤回看模型输入。
        ``bg_scene``/``bg_mask`` 仅背景阶段传入，供冷却换线时按新模型重选发送模式。"""
        from .dispatch import dispatch
        from .openai_compat import ModelCooldownError
        from .stage import image_fallback_endpoints

        endpoint = self._endpoint(scope)
        model = endpoint.model or dispatch.resolve_image_model()
        wire, params = _wire_info(endpoint)
        params = dict(params)
        params["attach_mask"] = len(images) > 1
        record = {
            "scope": scope, "model": model,
            "base": endpoint.base or "（全局默认网关）",
            "wire": wire, "params": params,
            "prompt_name": _prompt_name(prompt), "prompt": prompt,
            "images": _image_descs(scope, images),
        }
        t0 = time.time()
        try:
            out, switched = self._dispatch_repair(
                dispatch, endpoint, model, prompt, images, scope,
                bg_scene=bg_scene, bg_mask=bg_mask)
        except Exception as e:
            record.update(duration_ms=round((time.time() - t0) * 1000),
                          outcome="失败: " + str(e)[:160])
            runlog.model_call(**record)
            raise
        record.update(duration_ms=round((time.time() - t0) * 1000), outcome="ok")
        if switched:
            record["switched_to"] = switched
        runlog.model_call(**record)
        return out

    def _dispatch_repair(self, dispatch, endpoint, model: str, prompt: str,
                         images: list, scope: str, bg_scene=None, bg_mask=None):
        """按端点分线发起调用；长冷却时依次改试备用条目。

        返回 (输出, 切换到的备用模型 or None)；全部冷却时向上抛 ModelCooldownError。
        背景换线按备用端点重新 ``background_send_mode``（默认仍是涂黑+送 mask；
        环境变量覆盖会跟着生效）。
        """
        from .openai_compat import ModelCooldownError
        from .stage import image_fallback_endpoints

        try:
            if endpoint.configured:
                if _is_openrouter(endpoint):
                    return dispatch.generate_image_openrouter(
                        model, prompt, images, base=endpoint.base,
                        key=endpoint.key, quality=endpoint.quality), None
                return dispatch.generate_image_edits(model, prompt, images,
                                                     base=endpoint.base,
                                                     key=endpoint.key,
                                                     quality=endpoint.quality,
                                                     opaque_editable=endpoint.opaque_editable), None
            return dispatch.generate_image(model, prompt, images,
                                           base=endpoint.base,
                                           key=endpoint.key), None
        except ModelCooldownError:
            primary = (endpoint.base, endpoint.model, endpoint.key)
            for fallback in image_fallback_endpoints():
                if (fallback.base, fallback.model, fallback.key) == primary:
                    continue
                fallback_model = fallback.model or dispatch.resolve_image_model()
                fb_prompt, fb_images = prompt, images
                if scope == "BACKGROUND" and bg_scene is not None:
                    mode = background_send_mode(fallback, fallback_model)
                    fb_prompt = mode.prompt
                    fb_images = _images_for_mode(mode, bg_scene, bg_mask)
                    runlog.event("BG", f"冷却换线发送模式 {mode.name}")
                try:
                    out = _fallback_wire(dispatch, fallback, fallback_model,
                                         fb_prompt, fb_images)
                    note = (f"{scope} 首选模型冷却，已切换备用图像条目 "
                            f"{fallback_model} @ {fallback.base}")
                    print(f"[repair] {note}", flush=True)
                    runlog.event("SWITCH", note)
                    return out, f"{fallback_model} @ {fallback.base}"
                except ModelCooldownError:
                    continue
            raise

    def repair_elements(self, atlas_rgb: np.ndarray, atlas_mask: np.ndarray) -> np.ndarray:
        # 待补区已涂黑，只送 Atlas 图，不另附 mask。
        _ = atlas_mask
        out = self._generate_repair("ELEMENT", ATLAS_PROMPT, [atlas_rgb])
        return _fit_size(out, atlas_rgb)

    def repair_background(self, scene_hole_rgb: np.ndarray,
                          hole_mask: np.ndarray = None) -> np.ndarray:
        # 默认只送涂黑图；MAGICLAYER_BG_ATTACH_MASK=1 才另附一张 mask。
        endpoint = self._endpoint("BACKGROUND")
        from .dispatch import dispatch
        model = endpoint.model or dispatch.resolve_image_model()
        mode = background_send_mode(endpoint, model)
        images = _images_for_mode(mode, scene_hole_rgb, hole_mask)
        runlog.event("BG", f"发送模式 {mode.name} attach_mask={mode.attach_mask} "
                           f"n_images={len(images)} model={model}")
        out = self._generate_repair("BACKGROUND", mode.prompt, images,
                                    bg_scene=scene_hole_rgb, bg_mask=hole_mask)
        return _fit_size(out, scene_hole_rgb)


def generate_edit(scope: str, prompt: str, images: list) -> np.ndarray:
    """任意提示词的图生图变换（mediatransformation 语义引擎复用修补通道）。

    images 语义与 /v1/images/edits 一致：images[0]=原图，images[1] 可选白=待编辑
    的二值遮罩；端点选择（openrouter/edits/chat 三线）与冷却换线全部沿用
    ``GeminiImageProvider._generate_repair``。
    """
    return GeminiImageProvider()._generate_repair(scope, prompt, images)


def _is_openrouter(endpoint) -> bool:
    return (getattr(endpoint, "provider", "") or "").strip().lower() == "openrouter"


def _prompt_name(prompt: str) -> str:
    if prompt == ATLAS_PROMPT:
        return "ATLAS_PROMPT"
    if prompt == SCENE_PROMPT:
        return "SCENE_PROMPT"
    if prompt == SHORT_SCENE_PROMPT:
        return "SHORT_SCENE_PROMPT"
    return "（自定义提示词）"


def _wire_info(endpoint):
    """该端点实际使用的线与随请求参数（steps.json 过程可视化用）。"""
    if not endpoint.configured:
        return "POST {base}/chat/completions（对话式出图，图片以 base64 内联）", {}
    if _is_openrouter(endpoint):
        params = {}
        if endpoint.quality:
            params["quality"] = endpoint.quality
        return ("POST {base}/images（OpenRouter 图生图：input_references="
                "[原图, 遮罩]，遮罩语义靠提示词约定）", params)
    params = {"size": "auto"}
    if endpoint.quality:
        params["quality"] = endpoint.quality
    if getattr(endpoint, "opaque_editable", False):
        params["mask_polarity"] = "opaque（默认反转：白区不透明=待修补）"
        return ("POST {base}/v1/images/edits（multipart：image=原图，"
                "mask=遮罩且白区保持不透明=待修补——默认反转极性）", params)
    params["mask_polarity"] = "transparent（规范：白区转 alpha=0 透明=待修补）"
    return ("POST {base}/v1/images/edits（multipart：image=原图，"
            "mask=遮罩且白区转 alpha=0 透明=待修补）", params)


def _image_descs(scope: str, images: list):
    """随调用送出的每张图的语义说明（数量即实际送出张数）。"""
    if scope == "ELEMENT":
        descs = ["Atlas 拼接图（待补区已涂黑）",
                 "Atlas 二值遮罩（白=待修补）"]
    else:
        descs = ["背景图（待补区已涂黑）",
                 "背景遮罩（白=待重绘）"]
    return descs[:len(images)]


def _fallback_wire(dispatch, endpoint, model: str, prompt: str, images: list):
    """冷却池条目分线：openrouter 走专用 /images；gpt-image*/grok-imagine* 走
    /images/edits；其余（gemini-*-image 等对话式出图模型）走 chat 出图线。"""
    if _is_openrouter(endpoint):
        return dispatch.generate_image_openrouter(model, prompt, images,
                                                  base=endpoint.base,
                                                  key=endpoint.key,
                                                  quality=endpoint.quality)
    if _EDITS_LINE_MODELS.search(model or ""):
        return dispatch.generate_image_edits(model, prompt, images,
                                             base=endpoint.base, key=endpoint.key,
                                             quality=endpoint.quality,
                                             opaque_editable=getattr(
                                                 endpoint, "opaque_editable", False))
    return dispatch.generate_image(model, prompt, images,
                                   base=endpoint.base, key=endpoint.key)


def _fit_size(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """生图输出尺寸与输入不完全一致时缩放回输入尺寸。"""
    if out.shape[:2] == ref.shape[:2]:
        return out
    import cv2

    return cv2.resize(out, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)


def create_provider(kind: str) -> ImageRepairProvider:
    if kind == "gemini":
        return GeminiImageProvider()
    if kind == "local":
        from .local import LocalProvider

        return LocalProvider()
    if kind == "null":
        return NullProvider()
    raise ValueError(f"未知 image repair provider: {kind}")
