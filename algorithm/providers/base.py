"""图像修补 provider 抽象。

两条修补路径（文档 §3）：
- elements：Atlas 元素修补（两张输入：Atlas + 同尺寸二值 mask）
- background：背景 scene 修补（黑色诊断孔图 + 可选同尺寸二值 mask；
  mask 让生图线只重绘洞区——纯黑孔约定会被部分模型当"画框"独立构图）

生图模型文档未指定；gemini 实现走 Gemini 图像输出，local 为离线降级。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ImageRepairProvider(ABC):
    name: str = "base"
    degraded: bool = False  # True 表示未走生图的降级实现

    @abstractmethod
    def repair_elements(self, atlas_rgb: np.ndarray, atlas_mask: np.ndarray) -> np.ndarray:
        """mask 白色区域修补、黑色区域保持不变；返回同尺寸 RGB。"""

    @abstractmethod
    def repair_background(self, scene_hole_rgb: np.ndarray,
                          hole_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """填充黑色缺失区域使背景自然连续；返回同尺寸 RGB。

        hole_mask 白=待修补（与 scene_hole_rgb 同尺寸）；提供时生图线
        应以 mask 约束重绘范围，而不是靠黑孔猜。"""


class NullProvider(ImageRepairProvider):
    """不合成隐藏像素：等价于所有修补按原样返回（用于 none 模式兜底）。"""
    name = "null"
    degraded = True

    def repair_elements(self, atlas_rgb, atlas_mask):
        return atlas_rgb.copy()

    def repair_background(self, scene_hole_rgb, hole_mask=None):
        return scene_hole_rgb.copy()
