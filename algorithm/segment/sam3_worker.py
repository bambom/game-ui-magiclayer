"""SAM3 子进程 worker：跑在独立环境 .venv-sam3（torch 2.10+cu128，RTX 4090）。

主管线 .venv 的 torch 是 CPU 版（红线：不让 pip 升级 torch），SAM3 需要
torch>=2.10+CUDA，两边不能同进程，因此按行式 JSON 桥接（stdin/stdout，
图像与 mask 用 PNG base64）。本文件不 import algorithm 包，可被任意
Python 直接执行；协议：

  环境就绪后先输出一行 {"ready": true}
  请求 {"id":n,"op":"set_image","png_b64":...}
       {"id":n,"op":"predict","points":[[x,y],...],"labels":[0/1,...],
        "box":[x0,y0,x1,y1]}
  响应 {"id":n,"ok":true,...} / {"id":n,"ok":false,"error":"..."}

调用走官方 SAM1-task 流程（examples/sam3_for_sam1_task_example.ipynb）：
`Sam3Processor.set_image(PIL)` 算 state，`model.predict_inst(state, ...)` 出
三候选。实测注意：不能直接用 `inst_interactive_predictor.set_image`——
checkpoint 只有 detector.* 权重，tracker 的 backbone 未实例化（NoneType
forward_image）；state 由主模型 backbone 计算。set_image 只喂 PIL（ndarray
HWC 会被 shape[-2:] 误解析成 (W,C=3)）；推理在 torch.autocast('cuda',
bfloat16) 下（模型按 AMP 设计）。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3 line-JSON worker")
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    # sam3 内部 logging.info 很密（每次前向多行），压到 WARNING
    logging.getLogger().setLevel(logging.WARNING)

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "fatal": "CUDA 不可用：SAM3 需要 GPU（bfloat16 autocast）"}),
              flush=True)
        return 3

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=args.weights,
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model)
    state = {}
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            rid = req.get("id")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if op == "set_image":
                    from PIL import Image

                    png = base64.b64decode(req["png_b64"])
                    image = Image.open(io.BytesIO(png)).convert("RGB")
                    state = processor.set_image(image)
                    resp = {"id": rid, "ok": True, "size": list(image.size)}
                elif op == "predict":
                    import cv2

                    points = req.get("points") or []
                    labels = req.get("labels") or []
                    coords = (np.array(points, np.float32).reshape(-1, 2)
                              if points else None)
                    labs = np.array(labels, np.int32) if labels else None
                    box = (np.array(req["box"], np.float32)
                           if req.get("box") else None)
                    masks, scores, _ = model.predict_inst(
                        state,
                        point_coords=coords, point_labels=labs, box=box,
                        multimask_output=True)
                    encoded = []
                    for mask in masks:
                        ok, buf = cv2.imencode(
                            ".png", (mask > 0).astype(np.uint8) * 255)
                        if not ok:
                            raise RuntimeError("mask PNG 编码失败")
                        encoded.append(
                            base64.b64encode(buf.tobytes()).decode("ascii"))
                    resp = {"id": rid, "ok": True, "masks": encoded,
                            "scores": [float(s) for s in scores]}
                else:
                    resp = {"id": rid, "ok": False,
                            "error": f"未知 op {op!r}"}
        except Exception as exception:  # noqa: BLE001 —— 单请求失败不杀 worker
            rid = req.get("id") if isinstance(req, dict) else None
            resp = {"id": rid, "ok": False, "error": repr(exception)}
        print(json.dumps(resp), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
