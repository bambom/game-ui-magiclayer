# 独立仓库初始化与发布验证

日期：2026-10-23。范围：本仓库新增环境安装、模型下载、启动器与 Git 发布规则。

实机环境：Windows x64，Python 3.10.11；新建独立验证环境使用 torch 2.6.0+cpu / torchvision 0.21.0+cpu。Linux/macOS 的命令按跨平台脚本和上游安装方式提供，本次未作对应实机启动验收。

本次不运行插件测试及已有 `algorithm/tests` 回归；已有算法实现和历史测试仅整理/引用，不声称重新验证通过。新增本地演示用于检查独立启动及输出契约，在线 AI 与可选 GPU 后端单独列出验证边界。

| 记录 | 命令/范围 | 状态 | 结果 |
| --- | --- | --- | --- |
| S01 | `python scripts/bootstrap.py --venv .venv-bootstrap-check`；新建 Python 3.10 独立环境 | INTERRUPTED（保留） | CPU torch 安装成功；后续 OpenCV 下载低至 11.8 kB/s 并中断重连。本任务停止自己的 pip 子进程，保留环境，改用按官方 PyPI SHA256 校验的本地 wheel 缓存续装，结果见 S08 |
| S02 | `python scripts/check_publish.py`；初次 Git 索引检查 | FAILED（保留） | 原审计统一限制单文件 5 MiB，算法文档既有 image1.GIF 超出；没有发现密钥。后续将图片文档上限改为 25 MiB，源码仍为 5 MiB，保留该图片。待记录修复后的检查 |
| S03 | `python scripts/test_setup.py`；仅新增初始化工具用例 | PASS | 5/5，退出码 0。本机 HTTP 下载/校验与复用、损坏文件保留、错误下载不转正式模型、环境变量优先、发布审计阻止模型和凭据且不输出值 |
| S04 | 从 Git 索引导出的独立目录执行 `scripts/run.py --demo --model vit_b --job-id setup-standalone-20261023` | PASS | 退出码 0，state=done，3 个图层，preview.png/background.png/layers.json 均存在；真实 RapidOCR 及 SAM ViT-B CPU 推理，provider=local，无在线 AI 调用 |
| S05 | `python scripts/check_publish.py`；调整图片文档上限后的索引检查 | PASS | 114 个文件，21.79 MiB，0 findings；模型、.env、环境和作业目录全部排除 |
| S06 | AST 解析核心算法及新增脚本 | PASS | 72 个 Python 文件语法可解析，无需执行历史算法测试 |
| S07 | 官方 SAM ViT-B 下载地址 HTTP HEAD | PASS | HTTP 200，Content-Length=375042383，与配置大小一致；未重复下载大模型 |
| S08 | 同一 bootstrap 命令，临时设置 `PIP_FIND_LINKS=<本机已校验 wheel 目录>`、`PIP_NO_INDEX=1`，复用 S01 环境续装 | PASS | 退出码 0；固定运行依赖安装成功，`pip check` 无冲突，doctor 全部导入成功，OCR 三个 ONNX 文件齐全，SAM 模型大小及 SHA256 匹配。仅更换依赖传输来源，未修改系统 pip 配置 |
| S09 | 最终代码/文档索引的发布审计及 `git diff --cached --check` | PASS | 114 文件、21.79 MiB，0 findings，无空白错误；新增启动器编码调整通过语法检查。没有重跑 S03 或 S04 |

验证限制：SAM2/SAM3 安装路线依据现有代码、原本机依赖元数据及上游官方说明整理，本次未重新执行其推理。没有调用任何付费在线 AI，也没有验证具体供应商 Key/模型。默认模型下载地址来自 Meta 官方清单；现有完整权重用于真实推理，新增下载器的 HTTP 与校验处理另以本机短小夹具验证，避免重复下载大型文件。

S04 的核心算法及输入与最终提交一致。运行后仅为启动器的子进程增加 `PYTHONIOENCODING=utf-8`，避免 Windows 重定向日志乱码；未因此重复执行已经通过的拆层流程。S04 原启动器 SHA256：`ecd955f86cda5e530b30a2055d708a944f90afaabf38be408385a0faf63ba4c0`。该输出编码调整不构成在线功能或 GPU 后端的新证据。
