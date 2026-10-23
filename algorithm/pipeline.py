"""流水线编排（文档 §0.2 总流程）：

OCR -> 初次全去字 -> LLM 图层规划（文字分支 + 美术字恢复）-> SAM 逐实例
分割 -> 图层构建与遮挡修补 -> 背景修补 -> 组合输出。

产物按作业目录组织（README §4）：jobs/<job_id>/，status.json 状态机 +
job.json artifacts 清单 + 过程图。

用法：
  python -m algorithm.pipeline --image <ui.png> --job-id <id> \
      [--planner auto|gemini|mock --plan plan.json] \
      [--provider auto|gemini|local] [--device cpu] [--jobs-root <dir>] \
      [--bbox x0,y0,x1,y1]        # ROI 拆分（仅拆该区域，导出平移回根图坐标） \
      [--offset x,y --parent-job <id> --parent-layer <id>]  # 单层重拆 lineage
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import cv2
import numpy as np

from .common import (Instance, OcrRegion, Plan, TextLayer,
                     dump_json, norm_bbox_to_pixels, norm_points_to_pixels)
from .compose import compose_preview, save_outputs
from .ocr import engine as ocr_engine
from .repair.background import build_hidden_mask, repair_background
from .repair.elements import build_element_layers
from .runlog import enabled as runlog_enabled
from .runlog import event as runlog
from .runlog import finish as runlog_finish
from .runlog import install as runlog_install
from .runlog import model_calls as runlog_model_calls
from .segment.cleanup import refine_boundary
from .segment.sam import Segmenter
from .segment.split import split_wide_decorations
from .segment.widen import widen_hollow_mask
from .storage import JobStore, write_json_atomic
from .texterase.erase import (erase_all, erase_region, inpaint_mask_region,
                              restore_raster)
from .texterase.style import estimate_style, render_text_png


def semantic_mask_image(instances: List[Instance], w: int, h: int) -> np.ndarray:
    """语义 mask 示意（文档 §2 image16 右图）：每实例一色，按 z 自后向前覆盖。"""
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:] = (129, 36, 24)  # 深蓝底（RGB）
    for idx, inst in enumerate(sorted(instances, key=lambda i: i.query.z_order)):
        hue = int((idx * 47) % 180)
        b, g, r = cv2.cvtColor(np.uint8([[[hue, 216, 235]]]),
                               cv2.COLOR_HSV2BGR)[0, 0]
        canvas[inst.mask] = (r, g, b)
    return canvas


def text_mask_image(regions: List[OcrRegion], w: int, h: int) -> np.ndarray:
    """全图文字 removal mask 并集（文档 image7 右图：白字黑底）。"""
    out = np.zeros((h, w), np.uint8)
    for r in regions:
        if r.removal_mask is not None:
            out |= r.removal_mask.astype(np.uint8) * 255
    return out


def text_glyph_sheet(regions: List[OcrRegion], w: int, h: int) -> np.ndarray:
    """各区域字形 mask 按区域 bbox 拼到全图画布（glyph_mask 为裁片坐标）。"""
    canvas = np.zeros((h, w), np.uint8)
    for r in regions:
        if r.glyph_mask is None:
            continue
        x0, y0, x1, y1 = r.bbox
        canvas[y0:y1, x0:x1] |= r.glyph_mask.astype(np.uint8) * 255
    return canvas


def _prewarm_sam(device: str) -> None:
    """规划期间后台把 SAM 权重读进内存（约 10s 的纯 IO/CPU，与规划互不依赖）。"""
    try:
        from .segment.sam import warmup

        warmup(device)
    except Exception:
        pass  # 预热失败不阻断；真正的加载错误在 Segmenter 构造时暴露


class Pipeline:
    def __init__(self, planner: str = "auto", plan_path: Optional[str] = None,
                 provider: str = "auto", device: str = "cpu",
                 job_id: Optional[str] = None, jobs_root: Optional[str] = None,
                 out_dir: Optional[str] = None,
                 roi_bbox: Optional[tuple] = None,
                 origin: Optional[tuple] = None,
                 meta: Optional[dict] = None):
        self.planner_backend = planner
        self.plan_path = plan_path
        self.provider_kind = provider
        self.device = device
        self.job_id = job_id
        self.jobs_root = jobs_root
        self.out_dir = out_dir  # 测试覆盖：直接指定作业目录
        # 手动拆分扩展：roi_bbox=(x0,y0,x1,y1) 载入后先裁剪（全流程在 ROI 内进行）；
        # origin=(ox,oy) 是本作业在根图坐标系里的原点，导出时统一平移回去
        # （ROI 拆分 = bbox 原点，单层重拆 = 父图层 x/y）。
        self.roi_bbox = roi_bbox
        self.origin = origin
        self.meta = meta or {}  # mode / lineage，随 summary 落进 job.json
        self.timings: Dict[str, float] = {}
        self.warnings: List[str] = []
        self._sam_rows: List[str] = []
        # steps.json 的分步备注（修补降级等只归属单一阶段的警告）
        self._step_notes: Dict[str, List[str]] = {"segment": [],
                                                  "element_layers": [],
                                                  "background": []}
        self._steps_registered = False
        self.store: Optional[JobStore] = None

    # ---------- 阶段 ----------
    def _time(self, name: str, fn, *args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        self.timings[name] = round((time.time() - t0) * 1000)
        return out

    def _debug(self, name: str, image) -> None:
        if self.store is None or image is None:
            return
        os.makedirs(self.store.debug_dir, exist_ok=True)
        path = os.path.join(self.store.debug_dir, name)
        if isinstance(image, str):
            # 文本载荷（如 05_atlas_placements.json）：非图像 debug 记录
            with open(path, "w", encoding="utf-8") as f:
                f.write(image)
        elif image.ndim == 3:
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(path, image)
        self.store.register_artifact("debug", path)

    def _write_steps(self) -> None:
        """把当前进度汇总落成 debug/steps.json（面板分步回看用）。

        每步：标题/说明/状态/耗时 + 该步的输入输出图（存在的才列出）+ 送往
        模型的完整输入（提示词全文、线、参数、每张图的语义）+ 归属该步的
        警告。幂等，run() 各阶段推进点重写，失败时也保留已到的进度。
        """
        if self.store is None:
            return
        job = self.store.job_id
        base = f"/api/magiclayer/jobs/{job}"
        debug_dir = self.store.debug_dir

        def art(name, caption, role):
            if not os.path.exists(os.path.join(debug_dir, name)):
                return None
            return {"name": name, "caption": caption, "role": role,
                    "url": f"{base}/artifacts/{name}"}

        def root_png(name, caption, role):
            if not os.path.exists(os.path.join(self.store.dir, name)):
                return None
            return {"name": name, "caption": caption, "role": role,
                    "url": f"{base}/{name}"}

        def images(*entries):
            return [e for e in entries if e]

        stages = self.store.stage_states()
        calls = runlog_model_calls()

        def calls_of(*scopes):
            return [c for c in calls if c.get("scope") in scopes]

        def phase(timing_key, carried_by):
            """repair 阶段内并行的两个子步骤的状态推断。"""
            if timing_key in self.timings or self.store.state == "done":
                return "done"
            if stages.get(carried_by, {}).get("state") == "running":
                return "running"
            return "pending"

        seg_notes = [w for w in self.warnings if "分割为空" in w]
        doc = {
            "version": 1,
            "job_id": job,
            "mode": self.meta.get("mode", "full"),
            "job_state": self.store.state,
            "steps": [
                {"key": "input", "title": "输入原图",
                 "desc": "本作业要拆分的设计图（整图模式）"
                         if not self.roi_bbox else
                         f"父图 ROI 裁片 bbox={self.roi_bbox}，全流程在裁片内进行",
                 "state": "done", "duration_ms": None,
                 "images": images(root_png("input.png", "input.png", "输入")),
                 "calls": [], "notes": []},
                {"key": "ocr", "title": "① 文字识别 OCR",
                 "desc": "本地 OCR 引擎识别全部文字块的位置与内容，产出带稳定 ID 的"
                         "OCR 区域列表（去字与文字层重建的输入）。无云端调用。",
                 "state": stages.get("ocr", {}).get("state", "pending"),
                 "duration_ms": stages.get("ocr", {}).get("duration_ms"),
                 "images": images(
                     art("01_ocr.png", "OCR 结果可视化（识别框+文本）", "输出")),
                 "calls": [], "notes": []},
                {"key": "erase", "title": "② 初次全去字",
                 "desc": "按 OCR 文字遮罩用本地 inpaint 抹掉全部文字，得到去字图——"
                         "它是图层规划模型的第二张输入。",
                 "state": stages.get("erase", {}).get("state", "pending"),
                 "duration_ms": stages.get("erase", {}).get("duration_ms"),
                 "images": images(
                     art("01b_text_mask.png", "文字遮罩（白=文字像素）", "输出"),
                     art("01c_text_glyphs.png", "文字字形拼图", "输出"),
                     art("02_erased.png", "去字图（03 工作图的前身）", "输出")),
                 "calls": [], "notes": []},
                {"key": "planner", "title": "③ LLM 图层规划",
                 "desc": "原图 + 去字图 + OCR 区域列表（拼进提示词）一起送给规划模型，"
                         "产出图层查询计划（元素层级、修补模式、背景模式），"
                         "完整 JSON 落在 plan.json。",
                 "state": stages.get("planner", {}).get("state", "pending"),
                 "duration_ms": stages.get("planner", {}).get("duration_ms"),
                 "images": images(
                     root_png("input.png", "原图（第 1 张模型输入）", "输入"),
                     art("02_erased.png", "去字图（第 2 张模型输入）", "输入")),
                 "calls": calls_of("PLANNER"),
                 "text_files": ["plan.json"],
                 "notes": []},
                {"key": "segment", "title": "④ SAM 逐实例分割",
                 "desc": "按规划的每个图层查询，本地 SAM 以提示（点/框）在去字工作图"
                         "上分割出实例遮罩；遮罩决定元素裁片范围与背景遮挡区。",
                 "state": stages.get("segment", {}).get("state", "pending"),
                 "duration_ms": stages.get("segment", {}).get("duration_ms"),
                 "images": images(
                     art("03_work.png", "工作图（去字+美术字恢复+纠偏后）", "输入"),
                     art("04_masks.png", "每实例遮罩可视化（颜色区分实例）", "输出")),
                 "calls": [], "notes": seg_notes},
                {"key": "element_layers", "title": "⑤ 元素图层修补（生图 Atlas）",
                 "desc": "image 模式的元素裁片按原始尺寸零缩放拼进一张正方形 "
                         "Atlas（最小 1024×1024，位置/大小记 "
                         "05_atlas_placements.json），待补区涂黑后只送这一张图"
                         "（不另附 mask，mask 仅落 debug）；surface 模式走本地 "
                         "Telea，none 保持镂空。生成图按记录位置整块切回即为各"
                         "元素 PNG（天然原始尺寸），不和原裁片或涂黑图融合。"
                         "内容子层 mask 事先按几何剥离父框环。",
                 "state": phase("element_layers", "repair"),
                 "duration_ms": self.timings.get("element_layers"),
                 "images": images(
                     art("05_atlas_input.png",
                         "涂黑 Atlas 输入（正方形、零缩放，送给模型的唯一一张图）",
                         "输入"),
                     art("05_atlas_mask.png",
                         "硬遮罩（白=待补；只给本地 debug，不送模型）", "输入"),
                     art("05_atlas.png",
                         "生图输出的 Atlas（按 placements.json 切片即为各元素）",
                         "输出")),
                 "calls": calls_of("ELEMENT"),
                 "notes": list(self._step_notes.get("element_layers", []))},
                {"key": "background", "title": "⑥ 背景修补（生图 scene）",
                 "desc": "全部实例遮罩取并集=元素遮挡区，膨胀 2px 吞掉没扣干净的"
                         "残边；待补区涂黑后只送这一张图（不再另附 mask）。模型出图"
                         "即为最终背景，不和原图或涂黑图融合。"
                         "surface 模式走本地低频延续。",
                 "state": phase("background", "repair"),
                 "duration_ms": self.timings.get("background"),
                 "images": images(
                     art("06b_background_input.png",
                         "涂黑输入（送给模型的唯一一张图）", "输入"),
                     art("06b_background_mask.png",
                         "硬遮罩（白=待补；只给本地 debug，不送模型）", "输入"),
                     art("06_background.png", "修补后的完整背景", "输出")),
                 "calls": calls_of("BACKGROUND"),
                 "notes": list(self._step_notes.get("background", []))},
                {"key": "compose", "title": "⑦ 文字层重建与组合输出",
                 "desc": "普通文字按估算样式重建成文字层（美术字走图片层），与元素层、"
                         "修补背景组合成预览；导出 layers/ 每层 PNG 与 layers.json 契约。",
                 "state": stages.get("compose", {}).get("state", "pending"),
                 "duration_ms": stages.get("compose", {}).get("duration_ms"),
                 "images": images(
                     art("07_preview.png", "组合预览（背景+全部图层）", "输出")),
                 "calls": [], "notes": []},
            ],
        }
        try:
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(debug_dir, "steps.json")
            write_json_atomic(path, doc)
            if not self._steps_registered:
                self.store.register_artifact("debug", path)
                self._steps_registered = True
        except Exception as e:  # 过程可视化失败不阻断管线
            runlog("WARN", f"steps.json 写入失败: {e}")

    def run(self, image_path: str) -> dict:
        from PIL import Image

        self.store = JobStore(self.jobs_root, self.job_id, out_dir=self.out_dir)
        try:
            if self.meta:
                meta_path = os.path.join(self.store.dir, "meta.json")
                if not os.path.exists(meta_path):
                    write_json_atomic(meta_path, self.meta)
            original = np.asarray(Image.open(image_path).convert("RGB"))
            h, w = original.shape[:2]
            ox, oy = self.origin or (0, 0)
            if self.roi_bbox:
                x0, y0, x1, y1 = self.roi_bbox
                x0, y0 = max(int(x0), 0), max(int(y0), 0)
                x1, y1 = min(int(x1), w), min(int(y1), h)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    raise ValueError(f"ROI 过小或越界: {self.roi_bbox} (图 {w}x{h})")
                original = original[y0:y1, x0:x1]
                ox, oy = ox + x0, oy + y0
                h, w = original.shape[:2]
                runlog("INFO", f"ROI 拆分: bbox=({x0},{y0},{x1},{y1}) 根图原点=({ox},{oy})")
            self._origin = (int(ox), int(oy))
            if self.roi_bbox:
                # input.png 记录本作业实际处理的 ROI 裁片；完整原图看父作业
                cv2.imwrite(self.store.input_png,
                            cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
                self.store.register_artifact("input", self.store.input_png)
            else:
                self.store.create_input(image_path)
            runlog_install(self.store.dir, image=image_path,
                           size=f"{w}x{h}", job=self.store.job_id)
            self._write_steps()

            # 1. OCR 文字提取
            self.store.begin_stage("ocr")
            regions = self._timed_stage("ocr", ocr_engine.extract_regions, original)
            self._debug_ocr(original, regions)
            self._write_steps()

            # 2. 初次全 OCR 去字
            self.store.begin_stage("erase")
            cleaned = self._timed_stage("erase", erase_all, original, regions)
            self._debug("02_erased.png", cleaned)
            self._debug("01b_text_mask.png", text_mask_image(regions, w, h))
            self._debug("01c_text_glyphs.png", text_glyph_sheet(regions, w, h))
            self._write_steps()

            # 3. LLM 图层规划（SAM 权重同时在后台预热——加载不依赖规划结果）
            self.store.begin_stage("planner")
            sam_prewarm = threading.Thread(
                target=_prewarm_sam, args=(self.device,), daemon=True)
            sam_prewarm.start()
            plan = self._run_planner(original, cleaned, regions)
            dump_json(self.store.plan_json, plan.raw)
            self.store.register_artifact("plan", self.store.plan_json)
            self._corrections = plan.text_corrections
            self._apply_raster_and_corrections(plan, regions, original, cleaned)
            raster_regions = [r for r in regions if r.is_raster]
            if raster_regions:
                # 美术字恢复前工作图（与 03_work.png 构成恢复前后对比，文档 image15）
                self._debug("03a_work_before_raster.png", cleaned)
            work = restore_raster(cleaned, original, raster_regions)
            work = self._erase_corrections(work, regions)
            self._debug("03_work.png", work)
            self.store.end_stage("planner")
            self._write_steps()

            # 4. SAM 逐实例分割
            self.store.begin_stage("segment")
            self._apply_repair_overrides(plan)
            instances = self._pixelize(plan, w, h)
            segmenter = self._time("sam_load_encode", Segmenter, work, self.device)
            from .common import bbox_of_mask

            for i, inst in enumerate(instances):
                self.store.set_progress("segment",
                                        f"{inst.instance_id}（{inst.query.name}）",
                                        i + 1, len(instances))
                t0 = time.time()
                mask, detail = segmenter.segment_hint(inst.hint, w, h)
                ms = round((time.time() - t0) * 1000)
                self.timings["sam_segment"] = self.timings.get("sam_segment", 0) + ms
                inst.mask = mask
                if not mask.any():
                    self.warnings.append(f"实例 {inst.instance_id} 分割为空，已丢弃")
                    runlog("WARN", f"实例 {inst.instance_id}（{inst.query.name}）分割为空，已丢弃")
                    continue
                # 边界整形（二期问题三b）：SAM mask 边缘粗糙，导出 PNG 呈锯齿
                shaped, shape_info = refine_boundary(mask)
                if shape_info["applied"]:
                    inst.mask = mask = shaped
                    runlog("LAYER", f"边界整形 {inst.instance_id}："
                                    f"{shape_info['mode']} IoU={shape_info['iou']:.4f} "
                                    f"改写={shape_info['changed']}px")
                bbox = bbox_of_mask(mask)
                row = (f"{inst.instance_id}（{inst.query.name}）: {ms}ms bbox={bbox}")
                self._sam_rows.append(row)
                runlog("LAYER", f"分割 {row}")
            instances = [i for i in instances if i.mask is not None and i.mask.any()]
            # 环形外扩（框太细，2026-09-03）：SAM 框环 mask 只切到描边中线，外半圈
            # 笔画留在背景层。确定性外扩补回（自身开口的环内边界不动，算法面板
            # 可配）。实心父框需子实例 mask 齐备才能按其预演镂空量环带，故放循环后
            for inst in instances:
                kids = [o.mask for o in instances
                        if o.query.parent_query_id == inst.query.id
                        and o.mask is not None and o.mask.any()]
                widened, widen_info = widen_hollow_mask(inst.mask, kids)
                if widen_info["applied"]:
                    inst.mask = widened
                    band = (f" 环带≈{widen_info['band']:.0f}px"
                            if widen_info["method"] == "adaptive" else "")
                    runlog("LAYER", f"环形外扩 {inst.instance_id}："
                                    f"{widen_info['method']} +{widen_info['px']}px{band}")
            # 装饰大横带纵向分割（二期问题二）：planner 把左山群与右树丛合成
            # 一条全宽横带且裁掉向下延伸时，按列投影谷位切簇并逐簇下扩重跑
            before = len(instances)
            t_split = time.time()
            instances = split_wide_decorations(instances, segmenter, w, h)
            self.timings["sam_segment"] = self.timings.get("sam_segment", 0) + round(
                (time.time() - t_split) * 1000)   # 重跑的 SAM 计时并入分割
            if len(instances) != before:
                note = f"装饰大横带按簇间谷位分割：{before} -> {len(instances)} 个实例"
                self._step_notes["segment"].append(note)
                runlog("LAYER", note)
            self._debug("04_masks.png", semantic_mask_image(instances, w, h))
            self.store.end_stage("segment",
                                 {"instances": len(instances)})
            self._write_steps()

            # 5+6. 元素图层修补与背景修补并行（互不依赖：都只消费分割结果与 work 图，
            # 模型侧各自一次 edits 调用；总耗时由较慢一侧决定而非两者之和）
            self.store.begin_stage("repair")
            provider = self._make_provider()
            hidden = build_hidden_mask([i.mask for i in instances])
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="element") as pool:
                element_future = pool.submit(
                    self._timed_stage, "element_layers",
                    build_element_layers, instances, work, provider, self._debug,
                    self._element_progress)
                background, bg_degraded = self._time(
                    "background", self._background_with_progress, work, hidden,
                    plan.background_mode, provider, self._debug)
                element_results, diag = element_future.result()
            if diag.get("atlas_fallback"):
                note = "Atlas 生图修补退回传统修补: " + "; ".join(diag["atlas_fallback"])
                self.warnings.append(note)
                self._step_notes["element_layers"].append(note)
                runlog("WARN", note)
            if diag.get("none_coerced"):
                note = ("none 模式带子元素遮挡，已纠偏为 surface: "
                        + ", ".join(diag["none_coerced"]))
                self.warnings.append(note)
                self._step_notes["element_layers"].append(note)
            if diag.get("auto_occluded"):
                pairs = ", ".join(f"{item['instance_id']}←{'/'.join(item['by'])}"
                                  for item in diag["auto_occluded"])
                self._step_notes["element_layers"].append(
                    "无父子声明的压叠已按几何重叠自动镂空修补: " + pairs)
            if diag.get("child_clipped"):
                self._step_notes["element_layers"].append(
                    "内容子层 mask 已确定性剥离父框环: "
                    + ", ".join(diag["child_clipped"]))
            if bg_degraded and not provider.degraded:
                note = "背景 scene 生图修补未走生图（失败或冷却），已退回传统延续"
                self.warnings.append(note)
                self._step_notes["background"].append(note)
                runlog("WARN", note)
            self._debug("06_background.png", background)
            self._write_steps()

            # 7. 文字层（普通文字重建，美术字走图片层）
            text_layers, text_pngs = self._build_text_layers(original, regions)

            self._map_parents(element_results)
            preview = self._time("compose", compose_preview, background,
                                 [(lay, rgba) for _inst, rgba, lay in element_results],
                                 text_layers, text_pngs)
            if self._origin != (0, 0):
                # 手动拆分（ROI/单层重拆）：导出坐标平移回根图坐标系；
                # 预览/背景 PNG 保持作业局部坐标（预览已在平移前按局部坐标组合）。
                gox, goy = self._origin
                for _inst, _rgba, layer in element_results:
                    layer.x += gox
                    layer.y += goy
                for tl in text_layers:
                    tl.x += gox
                    tl.y += goy
            save_outputs(self.store.dir, background,
                         [(lay, rgba) for _inst, rgba, lay in element_results],
                         text_layers, text_pngs, preview)
            self._debug("07_preview.png", preview)
            self.store.end_stage("repair", {"background_mode": plan.background_mode})

            self.store.begin_stage("compose")
            summary = self._write_contract(original, element_results,
                                           text_layers, plan, diag, provider,
                                           bg_degraded)
            self.store.end_stage("compose")
            self.store.complete()
            self._write_steps()
            runlog_finish(self._summary_sections(diag, provider, bg_degraded))
            return summary
        except Exception as e:
            self.store.fail(e)
            self._write_steps()  # 失败也保留已到的进度供面板分步回看
            runlog("FAIL", repr(e)[:400])
            runlog_finish(self._summary_sections({}, None, None), failed=repr(e)[:200])
            raise

    def _summary_sections(self, diag, provider, bg_degraded) -> dict:
        """run.log 收尾 SUMMARY 的各小节内容。"""
        sections = {
            "阶段耗时(ms)": [f"{name}: {ms}" for name, ms in self.timings.items()],
            "SAM 每实例": self._sam_rows,
            "警告": self.warnings,
        }
        if diag:
            sections["修补构成"] = [
                f"atlas(生图): {diag.get('atlas_ids') or []}",
                f"surface(传统): {diag.get('surface_ids') or []}",
                f"none(未修补): {diag.get('none_ids') or []}",
                f"none 纠偏为 surface: {diag.get('none_coerced') or []}",
                f"子层剥离父框: {diag.get('child_clipped') or []}",
                f"空 mask 丢弃: {diag.get('empty_masks') or []}",
            ]
        if provider is not None:
            sections["provider"] = [
                f"name={provider.name} degraded={provider.degraded}",
                f"背景降级={bg_degraded}",
            ]
        return sections

    def _timed_stage(self, name: str, fn, *args, **kwargs):
        out = self._time(name, fn, *args, **kwargs)
        self.store.end_stage(name)
        return out

    def _apply_repair_overrides(self, plan) -> None:
        """设置面板的强制修补模式：只在环境变量显式给出时覆盖规划结果。

        MAGICLAYER_ELEMENT_REPAIR（image|surface|none）统一改写每个查询的元素
        修补模式；MAGICLAYER_BACKGROUND_REPAIR（scene|surface|none）改写背景
        修补模式。未设置/非法值 = 交给 LLM 规划（auto 语义）。
        """
        bg = os.environ.get("MAGICLAYER_BACKGROUND_REPAIR", "").strip().lower()
        if bg in ("scene", "surface", "none"):
            if bg != plan.background_mode:
                runlog("INFO", f"背景修补模式被设置覆盖: "
                               f"{plan.background_mode} -> {bg}")
            plan.background_mode = bg
        elem = os.environ.get("MAGICLAYER_ELEMENT_REPAIR", "").strip().lower()
        if elem in ("image", "surface", "none"):
            for q in plan.queries:
                q.element_repair_mode = elem
            runlog("INFO", f"元素修补模式被设置覆盖为 {elem}"
                           f"（{len(plan.queries)} 个查询）")

    def _element_progress(self, name: str, index=None, total=None) -> None:
        if self.store is not None:
            self.store.set_progress("repair", "元素 " + name, index, total)

    def _background_with_progress(self, work, hidden, mode, provider, debug):
        if self.store is not None:
            self.store.set_progress("repair", f"背景修补（{mode}）")
        return repair_background(work, hidden, mode, provider, debug)

    # ---------- 各步骤辅助 ----------
    def _run_planner(self, original, cleaned, regions) -> Plan:
        from .providers.stage import stage_endpoint

        backend = self.planner_backend
        if backend == "auto":
            if self.plan_path:
                backend = "mock"
            elif stage_endpoint("PLANNER").key or os.environ.get("GEMINI_API_KEY"):
                backend = "gemini"
            else:
                raise SystemExit("无 GEMINI_API_KEY 且未提供 --plan；"
                                 "离线联调请用 --planner mock --plan plan.json")
        t0 = time.time()
        from .planner import run_planner

        plan, warns = run_planner(original, cleaned, regions,
                                  backend=backend, plan_path=self.plan_path)
        self.timings["planner"] = round((time.time() - t0) * 1000)
        self.warnings.extend(warns)
        return plan

    def _apply_raster_and_corrections(self, plan: Plan, regions: List[OcrRegion],
                                      original, cleaned) -> None:
        by_id = {r.id: r for r in regions}
        for rid in plan.raster_text_ids:
            if rid in by_id:
                by_id[rid].is_raster = True
        emitted = {q.id for q in plan.queries}
        max_z = max((q.z_order for q in plan.queries), default=0)
        from .common import Hint, Query

        for r in [r for r in regions if r.is_raster]:
            qid = f"raster_{r.id}"
            if qid in emitted or any(
                    self._overlap(q.hints[0].bbox_norm, r, original.shape[1], original.shape[0]) > 0.8
                    for q in plan.queries):
                continue  # prompt 已要求 LLM 为美术字补发查询；避免重复
            x0, y0, x1, y1 = r.bbox
            W, H = original.shape[1], original.shape[0]
            plan.queries.append(Query(
                id=qid, name=f"raster text {r.id}", kind="logo", role="foreground",
                element_repair_mode="none", parent_query_id=None, z_order=max_z + 1,
                hints=[Hint(
                    bbox_norm=[x0 / W, y0 / H, x1 / W, y1 / H],
                    positive_points_norm=[[(x0 + x1) / 2 / W, (y0 + y1) / 2 / H]],
                    negative_points_norm=[],
                    bbox=(x0, y0, x1, y1),
                    positive_points=[((x0 + x1) // 2, (y0 + y1) // 2)],
                    negative_points=[])]))
            self.warnings.append(f"管线为美术字 {r.id} 补发非文本查询 {qid}")

    @staticmethod
    def _overlap(bbox_norm, region: OcrRegion, W: int, H: int) -> float:
        from .common import iou

        x0, y0, x1, y1 = region.bbox
        a = (x0, y0, x1, y1)
        px0, py0, px1, py1 = norm_bbox_to_pixels(bbox_norm, W, H)
        return iou(a, (px0, py0, px1, py1))

    def _erase_corrections(self, work: np.ndarray, regions: List[OcrRegion]) -> np.ndarray:
        """二次文字处理：LLM 发现的漏识普通文字（不重复运行整图去字）。"""
        corrections = getattr(self, "_corrections", [])
        for i, c in enumerate(corrections, 1):
            W = work.shape[1]
            H = work.shape[0]
            x0, y0, x1, y1 = norm_bbox_to_pixels(c["bbox_norm"], W, H)
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            region = OcrRegion(id=f"text_c{i:02d}", text=c["text"], conf=float(c["confidence"]),
                               quad=quad, bbox=(x0, y0, x1, y1), is_correction=True)
            erase_region(work, region)
            if region.removal_mask is not None and region.removal_mask.any():
                work = inpaint_mask_region(work, region.removal_mask)
            regions.append(region)
        return work

    def _pixelize(self, plan: Plan, w: int, h: int) -> List[Instance]:
        from .planner.schema import expand_instances

        instances = expand_instances(plan)
        for inst in instances:
            hb = inst.hint
            hb.bbox = norm_bbox_to_pixels(hb.bbox_norm, w, h)
            hb.positive_points = norm_points_to_pixels(hb.positive_points_norm, w, h)
            hb.negative_points = norm_points_to_pixels(hb.negative_points_norm, w, h)
        return instances

    def _make_provider(self):
        from .providers import create_provider
        from .providers.stage import stage_endpoint

        kind = self.provider_kind
        if kind == "auto":
            has_model_credentials = (
                stage_endpoint("ELEMENT").configured
                or stage_endpoint("BACKGROUND").configured
                or os.environ.get("GEMINI_API_KEY"))
            kind = "gemini" if has_model_credentials else "local"
        if kind == "local":
            self.warnings.append("使用离线降级 provider（surface 式修补，非生图）")
            runlog("WARN", "使用离线降级 provider（surface 式修补，非生图）")
        return create_provider(kind)

    def _build_text_layers(self, original, regions):
        text_layers: List[TextLayer] = []
        pngs: Dict[str, np.ndarray] = {}
        for r in regions:
            if r.is_raster or r.mask_mode == "empty":
                continue
            style = estimate_style(original, r)
            r.style = style
            x0, y0, x1, y1 = r.bbox
            text_layers.append(TextLayer(
                region_id=r.id, text=r.text, x=float(x0), y=float(y0),
                w=float(x1 - x0), h=float(y1 - y0), conf=r.conf, style=style))
            png = render_text_png(r.text, style)
            if png is not None:
                pngs[r.id] = png
        return text_layers, pngs

    def _map_parents(self, element_results) -> None:
        inst_by_query: Dict[str, str] = {}
        for inst, _, layer in element_results:
            inst_by_query.setdefault(inst.query.id, inst.instance_id)
        for inst, _, layer in element_results:
            pid = inst.query.parent_query_id
            layer.parent = inst_by_query.get(pid) if pid else None

    def _debug_ocr(self, original, regions) -> None:
        viz = original.copy()
        for r in regions:
            x0, y0, x1, y1 = r.bbox
            cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(viz, r.id, (x0, max(y0 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        self._debug("01_ocr.png", viz)

    # ---------- 输出 ----------
    def _write_contract(self, original, element_results, text_layers,
                        plan: Plan, diag, provider,
                        background_degraded: bool) -> dict:
        from .compose.compose import _safe

        store = self.store
        layers_json = []
        for inst, rgba, layer in element_results:
            # Contract paths are portable; filesystem paths are joined separately below.
            rel = f"layers/{_safe(layer.instance_id)}.png"
            store.register_artifact("layer", os.path.join(store.dir, rel),
                                    layer_id=layer.instance_id, name=layer.name,
                                    x=layer.x, y=layer.y, w=layer.w, h=layer.h,
                                    z=layer.z, parent=layer.parent,
                                    repair_mode=layer.repair_mode,
                                    repaired=layer.repaired, degraded=layer.degraded)
            layers_json.append({
                "kind": "image", "name": layer.name, "id": layer.instance_id,
                "x": layer.x, "y": layer.y, "w": layer.w, "h": layer.h,
                "z": layer.z, "parent": layer.parent,
                "png": rel, "repairMode": layer.repair_mode,
                "repaired": layer.repaired, "degraded": layer.degraded,
            })
        for tl in text_layers:
            rel = f"layers_text/{_safe(tl.region_id)}.png"
            store.register_artifact("text_layer", os.path.join(store.dir, rel),
                                    layer_id=tl.region_id, text=tl.text,
                                    x=tl.x, y=tl.y, w=tl.w, h=tl.h, style=tl.style)
            layers_json.append({
                "kind": "text", "id": tl.region_id, "text": tl.text,
                "x": tl.x, "y": tl.y, "w": tl.w, "h": tl.h,
                "style": tl.style,
                "png": rel,
            })
        contract = {
            "width": original.shape[1], "height": original.shape[0],
            # 手动拆分作业在根图坐标系的原点；图层 x/y 已平移为根图坐标
            "origin": list(getattr(self, "_origin", (0, 0))),
            "layers": layers_json,
            "backgroundPng": "background.png",
            # 降级标记供服务端复用判定：模型修补退回传统输出的产物不得被同图复用
            "backgroundMode": plan.background_mode,
            "backgroundDegraded": bool(background_degraded),
        }
        if self.meta:
            contract["job"] = self.meta
        dump_json(store.layers_json, contract)
        store.register_artifact("contract", store.layers_json)
        store.register_artifact("background", store.background_png,
                                mode=plan.background_mode)
        store.register_artifact("preview", store.preview_png)
        if runlog_enabled():
            store.register_artifact("run_log", os.path.join(store.dir, "run.log"))

        summary = {
            "input": {"source": store.input_png,
                      "width": original.shape[1], "height": original.shape[0]},
            "origin": list(getattr(self, "_origin", (0, 0))),
            "scene_summary": plan.scene_summary,
            "background_mode": plan.background_mode,
            "background_reason": plan.background_reason,
            "warnings": self.warnings,
            "timings_ms": self.timings,
            "provider": {"name": provider.name, "degraded": provider.degraded},
            "repair_diag": diag,
            "plan": plan.raw,
            "job": self.meta,
            "counts": {"image_layers": len(element_results),
                       "text_layers": len(text_layers)},
        }
        store.write_summary(summary)
        return summary


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="magiclayer 拆图算法流水线")
    ap.add_argument("--image", required=True,
                    help="输入图（单层重拆时传图层 PNG）")
    ap.add_argument("--job-id", help="作业 id（缺省按时间+短 uuid 生成）；接 server 后传作业 uuid")
    ap.add_argument("--jobs-root", help="作业产物根目录（缺省 magiclayer/jobs/）")
    ap.add_argument("--out", help="直接指定作业目录（测试用，正式勿用）")
    ap.add_argument("--planner", default="auto", choices=["auto", "gemini", "mock"])
    ap.add_argument("--plan", help="mock planner 的规划 JSON")
    ap.add_argument("--provider", default="auto", choices=["auto", "gemini", "local"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bbox", help="ROI 拆分：输入图内像素 bbox x0,y0,x1,y1，仅拆该区域")
    ap.add_argument("--offset", help="单层重拆：图层 PNG 在根图坐标系的原点 x,y（导出平移用）")
    ap.add_argument("--parent-job", help="lineage：父作业 id（重拆/ROI 记录用）")
    ap.add_argument("--parent-layer", help="lineage：父图层 id（单层重拆的来源图层）")
    args = ap.parse_args()

    roi_bbox = tuple(int(v) for v in args.bbox.split(",")) if args.bbox else None
    origin = tuple(int(v) for v in args.offset.split(",")) if args.offset else None
    # --bbox 与 --offset 可叠加：对重拆出的图层再做 ROI 时，
    # 原点 = 父图层根图原点 + ROI 在图层内的左上角。
    mode = "resplit" if args.parent_layer else ("roi" if roi_bbox else "full")
    job_id = args.job_id
    if job_id is None and args.parent_job and args.parent_layer:
        job_id = safe_job_id(f"{args.parent_job}-resplit-{args.parent_layer}")
    meta = {"mode": mode, "origin": list(origin or (0, 0)),
            "parentJob": args.parent_job, "parentLayer": args.parent_layer}
    if roi_bbox:
        meta["bbox"] = list(roi_bbox)

    pipe = Pipeline(planner=args.planner, plan_path=args.plan,
                    provider=args.provider, device=args.device,
                    job_id=job_id, jobs_root=args.jobs_root, out_dir=args.out,
                    roi_bbox=roi_bbox, origin=origin, meta=meta)
    summary = pipe.run(args.image)
    print("作业目录:", pipe.store.dir)
    print("完成。阶段耗时(ms):", pipe.timings)
    print("背景模式:", summary["background_mode"], "| provider:", summary["provider"]["name"])
    if pipe.warnings:
        print("警告:")
        for wn in pipe.warnings:
            print(" -", wn)


def safe_job_id(name: str) -> str:
    """作业 id 即目录名：仅保留安全字符。"""
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]


def _load_dotenv() -> None:
    """加载 magiclayer/.env（KEY=VALUE 行，已 gitignore），不覆盖已有环境变量。"""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


if __name__ == "__main__":
    main()
