# algorithm — 图片拆层核心

安装、模型下载和启动以 [根 README](../README.md) 为准。AI 应先运行 `scripts/bootstrap.py`，再由虚拟环境执行 `scripts/run.py --demo --model vit_b`。独立仓库不依赖原有 Server 工程。

| 模块 | 职责 |
| --- | --- |
| `pipeline.py` | OCR → 去字 → 规划 → 分割 → 修补 → 导出 |
| `config.py` | 算法参数及 MAGICLAYER 环境覆盖 |
| `ocr/engine.py` | RapidOCR 及文字过滤 |
| `texterase/` | 文字去除、样式估计与文字预览 |
| `planner/` | 视觉规划提示词、JSON 校验、本地 plan 读取 |
| `segment/` | SAM1/SAM2/SAM3 适配、候选评分、mask 清理/整形 |
| `repair/` | 元素遮挡修补、Atlas、背景延续、连续 alpha |
| `compose/` | 按父子链及 z 序组合 PNG |
| `providers/` | Gemini/OpenAI 兼容协议及本地修补 |
| `storage.py` / `runlog.py` | 作业状态、产物清单与过程记录 |

完整流程见 [拆图算法详细流程](../doc/拆图算法详细流程.md)。该文档保留算法设计与示例；SAM1 为默认后端，SAM2/SAM3 为后来增加的可选实现。实际行为以当前源码为准。

底层命令为 `python -m algorithm.pipeline --help`。直接导入 `Pipeline` 时，应在导入前配置环境变量；根目录启动器已经处理 `.env` 的加载时机。

`tests/` 保留原有评分、OCR、schema、几何与修补检查源码。安装过程不自动执行这组历史回归，也不将历史通过记录视为本次发布重新验证。新增独立初始化/启动验证见 [SETUP-VERIFICATION](../doc/SETUP-VERIFICATION.md)。
