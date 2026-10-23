"""按阶段（stage）解析模型端点。

服务端模型库（Java MagicAiModelRegistry：多个模型条目 + 各阶段分配）在启动拆图
任务时把每个阶段的条目展开为 ``MAGICLAYER_{SCOPE}_{MODEL,BASE,KEY,PROVIDER,QUALITY}``
环境变量（SCOPE ∈ PLANNER / ELEMENT / BACKGROUND，条目按构造均为 OpenAI 兼容端点；
provider=openrouter 的条目走其专用 /images 线）。
本模块按阶段解析；各值都可缺失，缺失时调用方回落到既有全局配置
（magiclayer/.env、MAGICLAYER_API_STYLE 自动判定），CLI 与离线用法不变。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class StageEndpoint:
    model: Optional[str]
    base: Optional[str]  # 显式 base => 注册库条目，按 OpenAI 兼容协议调用
    key: Optional[str]
    provider: Optional[str] = None  # openrouter 等需要专用 wire 的提供方
    quality: Optional[str] = None   # gpt-image 系的保真档位（high/medium/low）
    # 遮罩极性："transparent"=OpenAI 规范（透明=待修补）；None=反转（不透明=待修补，
    # 默认——2026-09-02 run 9bf549dd 实测本地 gpt-image-2 网关按此约定执行，规范极性下
    # 洞区被冻结在预填色上原样返回）。显式 transparent 才走规范。
    mask_polarity: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.base and self.key)

    @property
    def opaque_editable(self) -> bool:
        return self.mask_polarity != "transparent"


def _var(scope: str, name: str) -> str:
    return os.environ.get(f"MAGICLAYER_{scope}_{name}", "").strip()


def _polarity(raw: str) -> Optional[str]:
    return "transparent" if raw.strip().lower() == "transparent" else None


def stage_endpoint(scope: str) -> StageEndpoint:
    """该阶段的模型端点：阶段变量优先，模型名回落全局（planner/image 各自的既有变量）。"""
    model = _var(scope, "MODEL") or None
    if model is None and scope == "PLANNER":
        model = os.environ.get("MAGICLAYER_PLANNER_MODEL", "").strip() or None
    elif model is None:
        model = os.environ.get("MAGICLAYER_IMAGE_MODEL", "").strip() or None
    base = _var(scope, "BASE").rstrip("/") or None
    return StageEndpoint(model=model, base=base, key=_var(scope, "KEY") or None,
                         provider=_var(scope, "PROVIDER") or None,
                         quality=_var(scope, "QUALITY") or None,
                         mask_polarity=_polarity(_var(scope, "MASK_POLARITY")))


def image_fallback_endpoints() -> List[StageEndpoint]:
    """冷却换线池：服务端模型库导出的全部图像条目（JSON 数组
    ``[{"model":..,"base":..,"key":..,"provider":..,"quality":..}, ..]``）。
    缺 base/key 的条目忽略；解析失败按空池处理（换线不可用，维持既有单端点行为）。"""
    raw = os.environ.get("MAGICLAYER_IMAGE_FALLBACKS", "").strip()
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        return []
    pool: List[StageEndpoint] = []
    if not isinstance(items, list):
        return pool
    for item in items:
        if not isinstance(item, dict):
            continue
        base = str(item.get("base", "")).rstrip("/")
        key = str(item.get("key", "") or "")
        if not base or not key:
            continue
        pool.append(StageEndpoint(model=str(item.get("model", "") or "") or None,
                                  base=base, key=key,
                                  provider=str(item.get("provider", "") or "") or None,
                                  quality=str(item.get("quality", "") or "") or None,
                                  mask_polarity=_polarity(
                                      str(item.get("maskPolarity", "") or ""))))
    return pool
