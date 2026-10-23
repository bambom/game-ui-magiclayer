"""协议分发：按 MAGICLAYER_API_STYLE（auto|gemini|openai）路由到对应实现。

auto 规则：MAGICLAYER_OPENAI_BASE 或 GEMINI_BASE 路径含 /v1（非 v1beta）视为
openai 兼容；否则 gemini 原生。

显式传入 base（阶段端点，来自服务端模型库条目）时固定走 openai 兼容——注册库
条目按构造全部是 OpenAI 兼容端点，与全局 API_STYLE 无关。
"""
from __future__ import annotations

import os
from typing import List, Optional

from .. import config
from . import gemini as gemini_api
from . import openai_compat as openai_api


class _Dispatch:
    @property
    def style(self) -> str:
        s = os.environ.get("MAGICLAYER_API_STYLE", "auto").lower()
        if s in ("gemini", "openai"):
            return s
        return "openai" if os.environ.get("MAGICLAYER_OPENAI_BASE") else "gemini"

    @property
    def api(self):
        return openai_api if self.style == "openai" else gemini_api

    @staticmethod
    def _api_for(base: Optional[str]):
        return openai_api if base else None

    def generate_text(self, model: str, prompt: str, images: List,
                      base: Optional[str] = None, key: Optional[str] = None) -> str:
        api = self._api_for(base) or self.api
        return api.generate_text(model, prompt, images, base=base, key=key)

    def generate_image(self, model: str, prompt: str, images: List,
                       base: Optional[str] = None, key: Optional[str] = None):
        api = self._api_for(base) or self.api
        return api.generate_image(model, prompt, images, base=base, key=key)

    def generate_image_edits(self, model: str, prompt: str, images: List,
                             base: Optional[str] = None, key: Optional[str] = None,
                             quality: Optional[str] = None,
                             opaque_editable: bool = False):
        """图生图专用的 /images/edits 线，仅存在于 OpenAI 兼容网关。

        ``opaque_editable``：遮罩极性反转（不透明=待修补），由条目的
        maskPolarity 设置驱动（规范默认透明=待修补）。
        """
        return openai_api.generate_image_edits(model, prompt, images, base=base,
                                               key=key, quality=quality,
                                               opaque_editable=opaque_editable)

    def generate_image_openrouter(self, model: str, prompt: str, images: List,
                                  base: Optional[str] = None, key: Optional[str] = None,
                                  quality: Optional[str] = None):
        """OpenRouter 专用图像线（POST {base}/images + input_references）。"""
        return openai_api.generate_image_openrouter(model, prompt, images, base=base,
                                                    key=key, quality=quality)

    def resolve_image_model(self) -> str:
        configured = os.environ.get("MAGICLAYER_IMAGE_MODEL", "")
        if configured:
            return configured
        return self.api.resolve_image_model()


dispatch = _Dispatch()
