"""运行日志 run.log：一次拆图全程的人类可读流水账。

Pipeline.run 开头 install(作业目录)，此后各挂点写行，收尾 finish(summary)：

- STAGE   阶段开始/结束（JobStore.begin/end_stage 挂点，含耗时与 detail）
- LAYER   SAM 每实例分割、每层修补（模式/镂空像素/atlas 归属/none 纠偏）
- BG      背景修补的模式/上下文框/降级/耗时
- REQ     每次上游 HTTP 请求（wire/url/model/attempt/状态/耗时/请求响应字节数）
- SWITCH  长冷却换线（首选冷却 -> 备用图像条目）
- model_call() 并行积累的结构化模型调用记录（prompt/参数/时延），由 pipeline
  汇总落进 debug/steps.json 供面板按步骤展示"给模型的输入是什么"
- WARN    过程警告
- SUMMARY 收尾汇总（分阶段耗时、每实例 SAM、Atlas 构成、provider、警告）

元素/背景两线程并行写，故全程持锁追加；未安装（CLI 测试、单元测试）时所有
调用都是 no-op，调用方无需判断。
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Optional

_lock = threading.Lock()
_path: Optional[str] = None
_t0: Optional[float] = None
_model_calls: list = []


def install(job_dir: str, **header) -> None:
    """开始一份 run.log（截断旧文件），并写入头信息行。"""
    global _path, _t0
    _path = os.path.join(job_dir, "run.log")
    _t0 = time.time()
    with _lock:
        _model_calls.clear()
    with open(_path, "w", encoding="utf-8"):
        pass
    event("JOB", "run.log 开始 " + " ".join(f"{k}={v}" for k, v in header.items()))


def enabled() -> bool:
    """run.log 是否已安装（未安装时调用方可跳过昂贵的统计计算）。"""
    return _path is not None


def reset() -> None:
    """卸载当前 run.log（测试隔离用）。"""
    global _path, _t0
    with _lock:
        _path = None
        _t0 = None
        _model_calls.clear()


def model_call(**record) -> None:
    """记录一次模型调用的结构化信息（debug/steps.json 过程可视化用）。

    与 event 同语义：未安装时 no-op，任何自身异常不外抛。元素/背景两线程
    并行调用时持锁追加，at_s 取相对 run.log 起点的秒数。
    """
    if not _path:
        return
    try:
        rec = dict(record)
        rec.setdefault("at_s", round(time.time() - (_t0 or time.time()), 1))
        with _lock:
            _model_calls.append(rec)
    except Exception:
        pass


def model_calls() -> list:
    """已记录模型调用的浅拷贝（pipeline 落 steps.json 时消费）。"""
    with _lock:
        return [dict(c) for c in _model_calls]


def event(tag: str, message: str) -> None:
    """写一行 ``[时刻 +相对秒] [TAG] message``；任何自身异常都不外抛。"""
    if not _path:
        return
    try:
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        elapsed = time.time() - _t0 if _t0 is not None else 0.0
        line = f"[{now} +{elapsed:7.1f}s] [{tag:<6}] {message}\n"
        with _lock:
            with open(_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass  # 日志永不阻断管线


def finish(sections: dict, failed: Optional[str] = None) -> None:
    """收尾 SUMMARY 块：sections 为 {小节名: [行, ...]}。"""
    if not _path:
        return
    event("SUMMARY", "运行结束" + (f"（失败：{failed}）" if failed else "（成功）"))
    for name, rows in sections.items():
        event("SUMMARY", f"—— {name} ——")
        for row in rows or ["（无）"]:
            event("SUMMARY", f"  {row}")
