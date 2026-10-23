"""作业资源存储（README §4）：作业目录、状态机与产物清单。

一次图片处理 = jobs/<job_id>/ 一个目录；status.json 阶段状态机供轮询对账，
job.json 的 artifacts 清单登记全部落盘文件；关键 JSON 原子写。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import runlog


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def default_job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def resolve_jobs_root(explicit: Optional[str] = None) -> str:
    if explicit:
        root = explicit
    else:
        root = os.environ.get("MAGICLAYER_JOBS_ROOT", "") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jobs")
    return os.path.abspath(root)


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, dict) and "$json" in o:
        return o
    raise TypeError(f"not serializable: {type(o)}")


def create_pending(jobs_root: Optional[str], job_id: str,
                   meta: Optional[dict] = None) -> str:
    """只登记 pending 作业目录，不拉起算法（手动拆分解耦编排）。

    服务器建目录写 status.json(state=pending) + meta.json(mode/lineage)；
    之后 trigger 拉起管线时 JobStore 构造会把状态覆写为 running。
    """
    job_dir = os.path.join(resolve_jobs_root(jobs_root), job_id)
    os.makedirs(job_dir, exist_ok=True)
    now = _now_iso()
    write_json_atomic(os.path.join(job_dir, "status.json"), {
        "job_id": job_id,
        "state": "pending",
        "stage": None,
        "progress": None,
        "attempt": 0,
        "started_at": now,
        "updated_at": now,
        "stages": {},
        "error": None,
    })
    write_json_atomic(os.path.join(job_dir, "meta.json"), meta or {"mode": "full"})
    return job_dir


class JobStore:
    """作业目录与状态的唯一写入方。"""

    STAGES = ["ocr", "erase", "planner", "segment", "repair", "compose"]

    def __init__(self, jobs_root: Optional[str] = None,
                 job_id: Optional[str] = None,
                 out_dir: Optional[str] = None):
        self.job_id = job_id or default_job_id()
        self.dir = os.path.abspath(out_dir) if out_dir else os.path.join(
            resolve_jobs_root(jobs_root), self.job_id)
        os.makedirs(self.dir, exist_ok=True)
        self.debug_dir = os.path.join(self.dir, "debug")
        self.layers_dir = os.path.join(self.dir, "layers")
        self.text_dir = os.path.join(self.dir, "layers_text")
        self.artifacts: list = []
        self._stage_t0: dict = {}
        # 元素/背景两线程并行更新进度：所有 status 变更持锁，避免序列化撞上并发写
        self._lock = threading.RLock()
        self._status = {
            "job_id": self.job_id,
            "state": "running",
            "stage": None,
            "progress": None,
            "attempt": 1,
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "stages": {},
            "error": None,
        }
        self.flush_status()

    # ---------- 路径 ----------
    @property
    def input_png(self) -> str:
        return os.path.join(self.dir, "input.png")
    @property
    def plan_json(self) -> str:
        return os.path.join(self.dir, "plan.json")

    @property
    def layers_json(self) -> str:
        return os.path.join(self.dir, "layers.json")

    @property
    def background_png(self) -> str:
        return os.path.join(self.dir, "background.png")

    @property
    def preview_png(self) -> str:
        return os.path.join(self.dir, "preview.png")

    @property
    def job_json(self) -> str:
        return os.path.join(self.dir, "job.json")

    # ---------- 生命周期 ----------
    def create_input(self, source_image_path: str) -> str:
        # 服务器编排时源图就是本作业目录里的 input.png（README §4.6），此时跳过拷贝。
        try:
            if os.path.samefile(source_image_path, self.input_png):
                self.register_artifact("input", self.input_png)
                return self.input_png
        except OSError:
            pass
        shutil.copyfile(source_image_path, self.input_png)
        self.register_artifact("input", self.input_png)
        return self.input_png

    def begin_stage(self, stage: str) -> None:
        assert stage in self.STAGES, stage
        with self._lock:
            self._stage_t0[stage] = time.time()
            self._status["stage"] = stage
            self._status["progress"] = {"stage": stage}
            self._status["updated_at"] = _now_iso()
            self._status["stages"][stage] = {"started_at": _now_iso()}
            self.flush_status()
        runlog.event("STAGE", f"开始 {stage}")

    def set_progress(self, stage: str, current: str,
                     index: Optional[int] = None, total: Optional[int] = None) -> None:
        """阶段内进度（正在处理的实例/图层），供面板实时展示。

        元素与背景两线程并行修补时后写覆盖先写——展示层面取最后一次即可。
        """
        with self._lock:
            prog = {"stage": stage, "current": current}
            if index is not None:
                prog["index"] = index
            if total is not None:
                prog["total"] = total
            self._status["progress"] = prog
            self._status["updated_at"] = _now_iso()
            self.flush_status()

    def end_stage(self, stage: str, detail: Optional[dict] = None) -> None:
        ms = (time.time() - self._stage_t0.pop(stage, time.time())) * 1000
        with self._lock:
            rec = self._status["stages"].get(stage, {})
            rec["ended_at"] = _now_iso()
            rec["duration_ms"] = round(ms)
            if detail:
                rec["detail"] = detail
            self._status["stages"][stage] = rec
            self._status["stage"] = None
            self._status.pop("progress", None)
            self._status["updated_at"] = _now_iso()
            self.flush_status()
        extra = f" detail={detail}" if detail else ""
        runlog.event("STAGE", f"完成 {stage} 耗时 {ms:.0f}ms{extra}")

    def stage_states(self) -> dict:
        """{阶段: {"state": running|done, "duration_ms": int|None}}——steps.json 组装用。"""
        with self._lock:
            out = {}
            for stage, rec in self._status["stages"].items():
                out[stage] = {"state": "done" if rec.get("ended_at") else "running",
                              "duration_ms": rec.get("duration_ms")}
            current = self._status.get("stage")
            if current and current not in out:
                out[current] = {"state": "running", "duration_ms": None}
            return out

    @property
    def state(self) -> str:
        with self._lock:
            return self._status["state"]

    def complete(self) -> None:
        with self._lock:
            self._status["state"] = "done"
            self._status["stage"] = None
            self._status.pop("progress", None)
            self._status["updated_at"] = _now_iso()
            self.flush_status()

    def fail(self, error: str) -> None:
        with self._lock:
            self._status["state"] = "failed"
            self._status["error"] = str(error)[:2000]
            self._status["updated_at"] = _now_iso()
            self.flush_status()

    def flush_status(self) -> None:
        write_json_atomic(self.status_path, self._status)

    @property
    def status_path(self) -> str:
        return os.path.join(self.dir, "status.json")

    # ---------- 产物 ----------
    def register_artifact(self, kind: str, path: str, **meta) -> str:
        rel = os.path.relpath(path, self.dir).replace("\\", "/")
        entry = {"kind": kind, "path": rel}
        entry.update(meta)
        self.artifacts.append(entry)
        return rel

    def write_summary(self, summary: dict) -> None:
        doc = {
            "job_id": self.job_id,
            "state": self._status["state"],
            "attempt": self._status["attempt"],
            "started_at": self._status["started_at"],
            "finished_at": _now_iso(),
        }
        doc.update(summary)
        doc["artifacts"] = self.artifacts
        write_json_atomic(self.job_json, doc)
