"""编辑图片面板 mediatransformation 语义引擎（README §8）。

Java 侧 ``TransformationJobService`` 为每个官方 ``/_ajax/mediatransformation`` 请求
建作业目录（``request.json`` + ``source.png`` + 引用的 ``mask-<id>.json``）后以
``python -m algorithm.transform --job-dir <dir>`` 拉起本模块：

- ``type=TRANSFORMATION_REQUEST``：按序执行 ``transformations[]`` 栈（官方非破坏
  栈语义：客户端每次提交的都是「原图 + 累积变换数组」），产物 ``output.png``。
- ``type=SEGMENTATION_MASK / DEPTHMAP_MASK / FACE_SEGMENTATION_MASK``：先执行
  栈内既有变换得到待分析图像，再产出 ``mask.png``（FACE 无本地人脸模型，按官方
  同款错误 FAILED/NO_FACES_FOUND 收场）。

生图类变换复用拆图的元素/背景修补通道（``providers.gemini_image`` 的 stage 端点
+ 冷却换线）；确定性变换（调整/放大/模糊/Alpha 抠图）纯 cv2/numpy 本地完成。
``status.json`` 由本模块独占写入，Java 轮询只读。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import runlog
from .common import Hint, clamp, decode_image, encode_png

# styleId -> 提示词预设（官方风格名 -> 我们生图通道的描述；未收录的风格用通用句式）
STYLE_PRESETS: Dict[str, str] = {
    "lofi": "Restyle the image into lo-fi aesthetics: soft grain, muted warm palette, "
            "gentle vignette, cozy nostalgic atmosphere. Keep the subject and layout intact.",
}

DEFAULT_STYLE_PROMPT = ("Restyle the image with the '{style_id}' art style. Keep the subject, "
                        "composition and layout intact.")

# 官方轮询失败样本（run 2026-09-02）里的 error.type 词表；未知错误统一 PROVIDER_FAILED
ERR_NO_FOREGROUND = "NO_FOREGROUND_FOUND"
ERR_NO_SEGMENTS = "NO_SEGMENTS_FOUND"
ERR_NO_FACES = "NO_FACES_FOUND"
ERR_PROVIDER = "PROVIDER_FAILED"
ERR_BAD_REQUEST = "BAD_TRANSFORMATION"


class TransformFailure(Exception):
    """带官方 failedStep/error.type 语义的失败。"""

    def __init__(self, step: str, error_type: str, detail: str = ""):
        super().__init__(f"{step}: {error_type} {detail}".strip())
        self.step = step
        self.error_type = error_type


@dataclass
class Frame:
    """变换栈的当前图像：RGB + 可选 alpha（BACKGROUND_REMOVAL 之后非空）。"""

    rgb: np.ndarray
    alpha: Optional[np.ndarray] = None

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Status:
    """status.json 独占写入（Java 轮询只读）。"""

    def __init__(self, job_dir: str):
        self._path = os.path.join(job_dir, "status.json")

    def write(self, **fields) -> None:
        payload = {"state": fields.pop("state"), "updated_at": _now()}
        payload.update(fields)
        with open(self._path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)

    def running(self, stage: str) -> None:
        self.write(state="running", stage=stage)

    def done(self, kind: str, mask_type: Optional[str] = None) -> None:
        self.write(state="done", kind=kind, **({"mask_type": mask_type} or {}))

    def failed(self, step: str, error_type: str) -> None:
        self.write(state="failed", failed_step=step, error_type=error_type)


# ---------------------------------------------------------------------------
# 基础几何 / 遮罩工具
# ---------------------------------------------------------------------------

def _fit(image: np.ndarray, ref: Frame) -> np.ndarray:
    """生图通道输出几何不保真：缩放回参考帧尺寸。"""
    if image.shape[:2] == (ref.height, ref.width):
        return image
    return cv2.resize(image, (ref.width, ref.height), interpolation=cv2.INTER_LANCZOS4)


def _f32_array(b64: str) -> np.ndarray:
    """官方 float32 小端 base64 数组（坐标/笔刷路径的 wire 形态）。"""
    raw = base64.b64decode(b64)
    return np.array(struct.unpack("<%df" % (len(raw) // 4), raw), dtype=np.float64)


def rasterize_strokes(strokes: Sequence[dict], width: int, height: int) -> np.ndarray:
    """maskselection 笔刷路径 -> 二值遮罩（白=选中）。

    stroke: ``{"A?": "A"|"C", "A": {"A": xs_f32, "B": ys_f32}, "B": 笔宽(归一化),
    "C": 颜色}``；坐标 0..1 归一化。A?/C 的语义差异无 wire 证据，均按笔刷折线处理。
    """
    mask = np.zeros((height, width), np.uint8)
    for stroke in strokes:
        inner = (stroke or {}).get("A") or {}
        xs = _f32_array(inner.get("A", "")) if inner.get("A") else np.array([])
        ys = _f32_array(inner.get("B", "")) if inner.get("B") else np.array([])
        if len(xs) < 1 or len(ys) < 1 or len(xs) != len(ys):
            continue
        points = np.stack([
            np.clip(xs, 0.0, 1.0) * (width - 1),
            np.clip(ys, 0.0, 1.0) * (height - 1),
        ], axis=1).astype(np.int32)
        radius = max(2, int(round(float(stroke.get("B") or 0.05) * min(width, height) / 2)))
        if len(points) == 1:
            cv2.circle(mask, tuple(points[0]), radius, 255, -1)
        else:
            cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 255,
                          thickness=radius * 2, lineType=cv2.LINE_AA)
    return mask


def _center_hint(width: int, height: int, extra_points: Optional[List[Tuple[int, int]]] = None) -> Hint:
    """中心点 + 可选用户点：主体分割的默认提示。bbox 给中央 2/3 框（SAM 质量重试
    的网格补点路径需要可解包的 bbox）。"""
    points = [(width // 2, height // 2)] + list(extra_points or [])
    box = (width // 6, height // 6, width * 5 // 6, height * 5 // 6)
    return Hint(bbox=box, positive_points=points, negative_points=[],
                bbox_norm=None, positive_points_norm=[], negative_points_norm=[])


def _subject_mask(frame: Frame, points: Optional[List[Tuple[int, int]]] = None) -> Optional[np.ndarray]:
    """SAM 主体遮罩；失败返回 None。"""
    try:
        from .segment.sam import Segmenter
    except Exception as exc:  # 权重/依赖缺失
        runlog.event("WARN", f"SAM 不可用: {exc}")
        return None
    try:
        segmenter = Segmenter(frame.rgb)
        mask, _detail = segmenter.segment_hint(
            _center_hint(frame.width, frame.height, points), frame.width, frame.height)
        return mask if mask is not None and mask.any() else None
    except Exception as exc:
        runlog.event("WARN", f"SAM 分割失败: {exc}")
        return None


def _points_from_mask_spec(spec: dict, width: int, height: int) -> List[Tuple[int, int]]:
    """PROMPT_EDITING/AUTOFOCUS 遮罩里的 ``points {A: xs, B: ys}`` -> 像素点。"""
    points_spec = (spec or {}).get("points") or {}
    xs = _f32_array(points_spec["A"]) if points_spec.get("A") else np.array([])
    ys = _f32_array(points_spec["B"]) if points_spec.get("B") else np.array([])
    out = []
    for x, y in zip(xs, ys):
        out.append((int(round(clamp(float(x), 0.0, 1.0) * (width - 1))),
                    int(round(clamp(float(y), 0.0, 1.0) * (height - 1)))))
    return out


# ---------------------------------------------------------------------------
# 生图通道（复用拆图修补通道：stage 端点 + 冷却换线）
# ---------------------------------------------------------------------------

def _image_edit(step: str, prompt: str, frame: Frame, mask: Optional[np.ndarray]) -> np.ndarray:
    """提示词图生图：带遮罩走 /images/edits，无遮罩走 chat 出图线。失败按官方
    failedStep=<变换类型> / error.type=PROVIDER_FAILED 上抛。"""
    from .providers.gemini_image import generate_edit

    images = [frame.rgb] + ([mask] if mask is not None else [])
    try:
        return _fit(generate_edit("BACKGROUND", prompt, images), frame)
    except TransformFailure:
        raise
    except Exception as exc:
        raise TransformFailure(step, ERR_PROVIDER, str(exc)) from exc


# ---------------------------------------------------------------------------
# 各变换类型
# ---------------------------------------------------------------------------

def op_upscale(frame: Frame, _: dict, max_edge: int) -> Frame:
    scale = 2
    longest = max(frame.width, frame.height) * scale
    if longest > max(max_edge * 2, 4096):
        scale = max(1, max(max_edge * 2, 4096) // max(frame.width, frame.height))
    rgb = cv2.resize(frame.rgb, (frame.width * scale, frame.height * scale),
                     interpolation=cv2.INTER_LANCZOS4)
    alpha = None
    if frame.alpha is not None:
        alpha = cv2.resize(frame.alpha, (frame.width * scale, frame.height * scale),
                           interpolation=cv2.INTER_NEAREST)
    return Frame(rgb, alpha)


def op_gaussian_blur(frame: Frame, spec: dict, masks: Dict[str, dict]) -> Frame:
    """VARIABLE_INTENSITY_GAUSSIAN_BLUR：遮罩区高斯模糊（无遮罩记录时全图模糊）。"""
    mask_id = ((spec.get("intensityMask") or {}).get("maskId") or "").strip()
    stroke_spec = masks.get(mask_id)
    sigma = 6.0
    blurred = cv2.GaussianBlur(frame.rgb, (0, 0), sigma)
    if stroke_spec is None:
        return Frame(blurred, frame.alpha)
    region = rasterize_strokes((stroke_spec.get("mask") or {}).get("A", []),
                               frame.width, frame.height) > 127
    out = frame.rgb.copy()
    out[region] = blurred[region]
    return Frame(out, frame.alpha)


def op_adjustment(frame: Frame, spec: dict) -> Frame:
    """ADJUSTMENT_V3 globalParams（-1..1）逐项调色，float 全程后 clip。"""
    p = spec.get("globalParams") or {}
    img = frame.rgb.astype(np.float32) / 255.0

    def applied(key: str) -> float:
        return clamp(float(p.get(key) or 0.0), -1.0, 1.0)

    brightness, contrast = applied("brightness"), applied("contrast")
    if brightness:
        img = img + brightness * 0.5
    if contrast:
        img = (img - 0.5) * (1.0 + contrast) + 0.5

    highlights, shadows = applied("highlights"), applied("shadows")
    luma = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if highlights:
        weight = np.clip((luma - 0.5) * 2.0, 0.0, 1.0)[..., None]
        img = img + highlights * 0.4 * weight
    if shadows:
        weight = np.clip((0.5 - luma) * 2.0, 0.0, 1.0)[..., None]
        img = img + shadows * 0.4 * weight

    whites, blacks = applied("whites"), applied("blacks")
    if whites:
        img = np.clip(img, 0.0, 1.0)
        img = img ** (1.0 / (1.0 + whites)) if whites > 0 else img ** (1.0 - whites * 0.5)
    if blacks:
        img = np.clip(img, 0.0, 1.0)
        img = 1.0 - (1.0 - img) ** (1.0 - blacks * 0.5) if blacks < 0 else img

    saturation, vibrance = applied("saturation"), applied("vibrance")
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)[..., None]
    if saturation:
        img = gray + (img - gray) * (1.0 + saturation)
    if vibrance:
        sat_now = np.clip((np.max(img, axis=-1, keepdims=True)
                           - np.min(img, axis=-1, keepdims=True)), 0.0, 1.0)
        boost = 1.0 + vibrance * (1.0 - sat_now)
        img = gray + (img - gray) * boost

    temperature, tint = applied("temperature"), applied("tint")
    if temperature:
        img[..., 0] += temperature * 0.15
        img[..., 2] -= temperature * 0.15
    if tint:
        img[..., 1] += tint * 0.15

    clarity, sharpness = applied("clarity"), applied("sharpness")
    if clarity:
        low = cv2.GaussianBlur(img, (0, 0), 9)
        img = img + clarity * (img - low)
    if sharpness:
        low = cv2.GaussianBlur(img, (0, 0), 1.5)
        img = img + sharpness * 0.8 * (img - low)

    vignette = applied("vignette")
    if vignette:
        yy, xx = np.mgrid[0:frame.height, 0:frame.width].astype(np.float32)
        radius = np.sqrt(((xx / frame.width - 0.5) ** 2 + (yy / frame.height - 0.5) ** 2) * 2.0)
        falloff = np.clip(radius - 0.35, 0.0, 1.0)[..., None] * vignette * 0.8
        img = img * (1.0 - falloff)

    out = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Frame(out, frame.alpha)


def op_background_removal(frame: Frame) -> Frame:
    """SAM 主体 -> alpha；无主体按官方错误词表失败。"""
    mask = _subject_mask(frame)
    if mask is None:
        raise TransformFailure("BACKGROUND_REMOVAL", ERR_NO_FOREGROUND)
    alpha = (mask.astype(np.uint8)) * 255
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
    return Frame(frame.rgb, alpha)


def op_inpainting(frame: Frame, spec: dict, masks: Dict[str, dict]) -> Frame:
    """擦除：MASK_SELECTION 笔刷遮罩，优先生图通道，失败回落 TELEA。"""
    mask_spec = spec.get("mask") or {}
    mask_id = (mask_spec.get("maskId") or "").strip()
    stroke_spec = masks.get(mask_id)
    if stroke_spec is None:
        raise TransformFailure("INPAINTING", ERR_BAD_REQUEST, f"maskId {mask_id} 未存储")
    region = rasterize_strokes((stroke_spec.get("mask") or {}).get("A", []),
                               frame.width, frame.height)
    hole = region > 127
    if not hole.any():
        raise TransformFailure("INPAINTING", ERR_BAD_REQUEST, "遮罩为空")
    base = frame.rgb
    if frame.alpha is not None:
        opaque = np.dstack([frame.rgb, np.full_like(frame.alpha, 255)])
        base = cv2.cvtColor(opaque, cv2.COLOR_RGBA2BGR)
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    prompt = ("Inpaint the white masked region so it blends with the surrounding content. "
              "Remove the masked object entirely and continue the background naturally. "
              "Do not add text, logos or new objects. Keep everything outside the mask unchanged.")
    try:
        rgb = _image_edit("INPAINTING", prompt, Frame(base, None), region)
    except TransformFailure as exc:
        runlog.event("WARN", f"INPAINTING 生图通道失败，回落 TELEA: {exc}")
        bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        patched = cv2.inpaint(bgr, region, 5, cv2.INPAINT_TELEA)
        rgb = cv2.cvtColor(patched, cv2.COLOR_BGR2RGB)
    return Frame(rgb, frame.alpha)


def op_prompt_editing(frame: Frame, spec: dict) -> Frame:
    """AI 改图 / AI 风格刷：提示词编辑；风格刷带 SEGMENTATION_MASK+points 只改选中区。"""
    text = ((spec.get("prompt") or {}).get("text") or "").strip()
    if not text:
        raise TransformFailure("PROMPT_EDITING", ERR_BAD_REQUEST, "prompt.text 为空")
    mask_spec = spec.get("mask") or {}
    region = None
    if mask_spec.get("maskType") == "SEGMENTATION_MASK":
        points = _points_from_mask_spec(mask_spec, frame.width, frame.height)
        mask = _subject_mask(frame, points)
        if mask is None:
            raise TransformFailure("SEGMENTATION_MASK", ERR_NO_SEGMENTS)
        region = mask.astype(np.uint8) * 255
    prompt = (f"Edit this image as instructed: {text}. Keep the subject identity, composition "
              f"and everything not mentioned by the instruction unchanged. Output one complete "
              f"image at the same size.")
    base = frame.rgb
    if frame.alpha is not None and region is None:
        base = cv2.cvtColor(cv2.cvtColor(np.dstack([frame.rgb, frame.alpha]),
                                         cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2RGB)
    try:
        rgb = _image_edit("PROMPT_EDITING", prompt, Frame(base, None), region)
    except TransformFailure:
        raise
    return Frame(rgb, frame.alpha)


def op_background_generation(frame: Frame, spec: dict) -> Frame:
    """背景生成：主体外区域按提示词重绘。"""
    text = ((spec.get("prompt") or {}).get("text") or "").strip() or "Regenerate the background."
    subject = _subject_mask(frame)
    if subject is None:
        # 找不到主体时整图交给提示词（官方 removeObjectBackground 场景可无主体）
        runlog.event("WARN", "BACKGROUND_GENERATION 未检出主体，整图重绘")
        region = None
    else:
        region = ((~subject).astype(np.uint8)) * 255
        region = cv2.morphologyEx(region, cv2.MORPH_DILATE,
                                  np.ones((9, 9), np.uint8))
    prompt = (f"Keep the main subject exactly unchanged. {text} Apply the change only to the "
              f"background; continue edges naturally. Do not add text or logos.")
    base = frame.rgb
    if frame.alpha is not None:
        base = cv2.cvtColor(cv2.cvtColor(np.dstack([frame.rgb, frame.alpha]),
                                         cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2RGB)
    rgb = _image_edit("BACKGROUND_GENERATION", prompt, Frame(base, None), region)
    return Frame(rgb, frame.alpha)


def op_style_transfer(frame: Frame, spec: dict) -> Frame:
    style_id = (spec.get("styleId") or "").strip()
    prompt = STYLE_PRESETS.get(style_id, DEFAULT_STYLE_PROMPT.format(style_id=style_id or "artistic"))
    base = frame.rgb
    if frame.alpha is not None:
        base = cv2.cvtColor(cv2.cvtColor(np.dstack([frame.rgb, frame.alpha]),
                                         cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2RGB)
    rgb = _image_edit("STYLE_TRANSFER", prompt, Frame(base, None), None)
    return Frame(rgb, frame.alpha)


def op_outpainting(frame: Frame, spec: dict) -> Frame:
    """扩图：outpaintedWidth/Height 为相对倍率，topLeftCornerX 为原图左缘相对位置。"""
    try:
        w_ratio = max(1.0, float(spec.get("outpaintedWidth") or 1.0))
        h_ratio = max(1.0, float(spec.get("outpaintedHeight") or 1.0))
        origin_x_ratio = float(spec.get("topLeftCornerX") or 0.0)
    except (TypeError, ValueError) as exc:
        raise TransformFailure("OUTPAINTING", ERR_BAD_REQUEST, str(exc)) from exc
    new_w, new_h = int(round(frame.width * w_ratio)), int(round(frame.height * h_ratio))
    offset_x = int(round((new_w - frame.width) * clamp(origin_x_ratio, 0.0, 1.0)))
    offset_y = (new_h - frame.height) // 2
    canvas = np.zeros((new_h, new_w, 3), np.uint8)
    canvas[offset_y:offset_y + frame.height, offset_x:offset_x + frame.width] = frame.rgb
    mask = np.full((new_h, new_w), 255, np.uint8)
    mask[offset_y:offset_y + frame.height, offset_x:offset_x + frame.width] = 0
    prompt = ("Outpaint the black padded border so the scene continues naturally outward. "
              "Keep the original center content exactly unchanged.")
    try:
        rgb = _image_edit("OUTPAINTING", prompt, Frame(canvas, None), mask)
    except TransformFailure as exc:
        runlog.event("WARN", f"OUTPAINTING 生图通道失败，回落镜像延拓: {exc}")
        rgb = _mirror_extend(frame.rgb, new_w, new_h, offset_x, offset_y)
    return Frame(rgb, frame.alpha)


def _mirror_extend(rgb: np.ndarray, new_w: int, new_h: int,
                   offset_x: int, offset_y: int) -> np.ndarray:
    canvas = np.zeros((new_h, new_w, 3), np.uint8)
    canvas[offset_y:offset_y + rgb.shape[0], offset_x:offset_x + rgb.shape[1]] = rgb
    left = cv2.flip(canvas[:, :offset_x], 1) if offset_x > 0 else None
    if left is not None:
        canvas[:, :offset_x] = cv2.resize(left, (offset_x, new_h)) if left.shape[0] != new_h else left
    right = cv2.flip(canvas[:, offset_x + rgb.shape[1]:], 1) if offset_x + rgb.shape[1] < new_w else None
    if right is not None:
        canvas[:, offset_x + rgb.shape[1]:] = cv2.resize(
            right, (new_w - offset_x - rgb.shape[1], new_h)) if right.shape[0] != new_h else right
    return cv2.GaussianBlur(canvas, (0, 0), 4.0)


def op_autofocus(frame: Frame, spec: dict) -> Frame:
    """景深：主体保持清晰，其余按 blurIntensity 高斯模糊。"""
    blur = clamp(float(spec.get("blurIntensity") or 0.0), 0.0, 1.0)
    subject = _subject_mask(frame)
    if subject is None:
        runlog.event("WARN", "AUTOFOCUS 未检出主体，整图轻微模糊")
        subject = np.zeros((frame.height, frame.width), bool)
    sigma = 1.0 + blur * 9.0
    blurred = cv2.GaussianBlur(frame.rgb, (0, 0), sigma)
    out = blurred.copy()
    out[subject] = frame.rgb[subject]
    return Frame(out, frame.alpha)


# ---------------------------------------------------------------------------
# 栈执行
# ---------------------------------------------------------------------------

OP_HANDLERS = {
    "UPSCALE": lambda frame, spec, ctx: op_upscale(frame, spec, ctx.max_edge),
    "ADJUSTMENT_V3": lambda frame, spec, ctx: op_adjustment(frame, spec),
    "VARIABLE_INTENSITY_GAUSSIAN_BLUR": lambda frame, spec, ctx: op_gaussian_blur(frame, spec, ctx.masks),
    "BACKGROUND_REMOVAL": lambda frame, spec, ctx: op_background_removal(frame),
    "INPAINTING": lambda frame, spec, ctx: op_inpainting(frame, spec, ctx.masks),
    "PROMPT_EDITING": lambda frame, spec, ctx: op_prompt_editing(frame, spec),
    "BACKGROUND_GENERATION": lambda frame, spec, ctx: op_background_generation(frame, spec),
    "STYLE_TRANSFER": lambda frame, spec, ctx: op_style_transfer(frame, spec),
    "OUTPAINTING": lambda frame, spec, ctx: op_outpainting(frame, spec),
    "AUTOFOCUS": lambda frame, spec, ctx: op_autofocus(frame, spec),
}

MASK_JOB_KINDS = {"SEGMENTATION_MASK", "DEPTHMAP_MASK", "FACE_SEGMENTATION_MASK"}


@dataclass
class JobContext:
    max_edge: int
    masks: Dict[str, dict]


def run_stack(frame: Frame, transformations: List[dict], ctx: JobContext) -> Frame:
    for spec in transformations or []:
        op_type = (spec or {}).get("type") or ""
        handler = OP_HANDLERS.get(op_type)
        if handler is None:
            raise TransformFailure(op_type or "UNKNOWN", ERR_BAD_REQUEST, "未知变换类型")
        runlog.event("OP", f"{op_type} 开始")
        started = time.time()
        frame = handler(frame, spec, ctx)
        runlog.event("OP", f"{op_type} 完成 {time.time() - started:.1f}s "
                           f"-> {frame.width}x{frame.height}")
    return frame


def produce_mask(job_type: str, frame: Frame) -> np.ndarray:
    """mask 作业：SEGMENTATION/DEPTHMAP 的产物遮罩（白=目标）。"""
    if job_type == "FACE_SEGMENTATION_MASK":
        raise TransformFailure("FACE_SEGMENTATION", ERR_NO_FACES)
    if job_type == "DEPTHMAP_MASK":
        # 无本地深度模型：中心径向深度代理（供 AUTOFOCUS 类消费），runlog 注明
        runlog.event("WARN", "DEPTHMAP 使用中心径向代理（无本地深度模型）")
        yy, xx = np.mgrid[0:frame.height, 0:frame.width].astype(np.float32)
        radius = np.sqrt(((xx / max(frame.width - 1, 1) - 0.5) ** 2
                          + (yy / max(frame.height - 1, 1) - 0.5) ** 2) * 2.0)
        return ((1.0 - np.clip(radius, 0.0, 1.0)) * 255).astype(np.uint8)
    mask = _subject_mask(frame)
    if mask is None:
        raise TransformFailure("SEGMENTATION_MASK", ERR_NO_SEGMENTS)
    return mask.astype(np.uint8) * 255


def _write_png(path: str, frame: Frame) -> None:
    if frame.alpha is None:
        with open(path, "wb") as handle:
            handle.write(encode_png(frame.rgb))
        return
    rgba = np.dstack([frame.rgb, frame.alpha])
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise RuntimeError("PNG encode failed")
    with open(path, "wb") as handle:
        handle.write(buf.tobytes())


def run_job(job_dir: str) -> None:
    status = _Status(job_dir)
    with open(os.path.join(job_dir, "request.json"), encoding="utf-8") as handle:
        request = json.load(handle)
    with open(os.path.join(job_dir, "source.png"), "rb") as handle:
        frame = Frame(decode_image(handle.read()))

    masks: Dict[str, dict] = {}
    for name in os.listdir(job_dir):
        if name.startswith("mask-") and name.endswith(".json"):
            with open(os.path.join(job_dir, name), encoding="utf-8") as handle:
                masks[name[len("mask-"):-len(".json")]] = json.load(handle)
    ctx = JobContext(max_edge=int(request.get("maxLongestEdge") or 1600), masks=masks)

    job_type = request.get("type") or "TRANSFORMATION_REQUEST"
    status.running("stack")
    frame = run_stack(frame, request.get("transformations") or [], ctx)

    if job_type in MASK_JOB_KINDS:
        status.running(job_type)
        mask = produce_mask(job_type, frame)
        cv2.imwrite(os.path.join(job_dir, "mask.png"), mask)
        mask_type = "DEPTHMAP_RESPONSE" if job_type == "DEPTHMAP_MASK" else "SEGMENTATION_RESPONSE"
        status.done("mask", mask_type)
        runlog.event("JOB", f"mask 作业完成 {job_type}")
        return

    _write_png(os.path.join(job_dir, "output.png"), frame)
    status.done("image")
    runlog.event("JOB", f"变换完成 {len(request.get('transformations') or [])} 步 "
                        f"-> {frame.width}x{frame.height} alpha={'有' if frame.alpha is not None else '无'}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="mediatransformation 语义引擎")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    job_dir = os.path.abspath(args.job_dir)
    runlog.install(job_dir, engine="transform", job=os.path.basename(job_dir))
    status = _Status(job_dir)
    try:
        status.running("boot")
        run_job(job_dir)
        runlog.finish({}, failed=None)
        return 0
    except TransformFailure as failure:
        status.failed(failure.step, failure.error_type)
        runlog.event("FAIL", f"{failure.step} {failure.error_type}")
        runlog.finish({}, failed=failure.step)
        return 0  # 失败也是合法终态：轮询要回 FAILED 信封
    except Exception as exc:  # 未预期异常：按当前栈类型报 PROVIDER_FAILED
        runlog.event("FAIL", f"引擎异常: {exc}")
        try:
            with open(os.path.join(job_dir, "request.json"), encoding="utf-8") as handle:
                job_type = (json.load(handle).get("type") or "TRANSFORMATION_REQUEST")
        except Exception:
            job_type = "TRANSFORMATION_REQUEST"
        status.failed(job_type, ERR_PROVIDER)
        runlog.finish({}, failed=job_type)
        return 0


if __name__ == "__main__":
    sys.exit(main())
