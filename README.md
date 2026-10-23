# game-ui-magiclayer

将游戏 UI 图片拆成可独立编辑的图片层、文字层和背景。核心流程是 OCR → 去字 → 视觉模型规划 → SAM 分割 → 元素/背景修补 → PNG 图层与 JSON 契约输出。

这是可独立运行的 Python 命令行仓库。启动入口是 `scripts/run.py`；一次运行处理一张图片并退出。无需 Java 服务、Docker、数据库或原工程目录，也不提供 Web 服务端口。

## 给 AI 的自动执行指令

可以把下面这段直接交给 AI：

> 请阅读本仓库 README.md 和 AGENTS.md，先确认系统及 Python 版本，按照默认 CPU/SAM ViT-B 路线自动初始化环境、下载并校验模型、执行 doctor，再启动本地 demo。使用仓库脚本完成操作，不依赖原作者电脑的目录、网关和模型。没有 API Key 时完成本地演示，不调用在线 API。成功后报告环境路径、作业路径、status.json 状态、图层数量、preview.png 和 layers.json 路径。遇到错误保留现场，修复后仅续跑失败步骤，不覆盖已有作业。不得输出或提交真实密钥及模型文件。

AI 执行顺序：

1. 进入仓库根目录。检查 `python --version`、64 位架构和可用磁盘空间，使用 **Python 3.10–3.12**，默认选 3.10。为依赖、环境及权重预留约 5 GB 空间，CPU 推理建议至少 8 GB 内存。首次需要访问 PyPI、PyTorch 和 Meta 模型下载站点。
2. 执行下文 `bootstrap.py`，等待进程成功退出。它创建 `.venv`、安装指定 CPU PyTorch 和依赖、下载并核对 SAM 模型、在 `.env` 不存在时复制模板、调用 doctor。已有模型校验通过就复用，已有 `.env` 不覆盖。
3. 执行 `run.py --demo --model vit_b`。它生成一张合成 UI 图片和固定规划，实际执行 RapidOCR、SAM 分割、本地修补及导出，全程不需要 Key。
4. 以命令退出码为 0、`status.json.state=done`、`layers.json` 有图层且 `preview.png` 存在作为启动成功条件。此演示只证明本地链路可运行；在线规划与生图质量需要用户提供服务后另行验收。

不要把安装完成或 `--help` 成功当作已经运行成功。没有 Key 是本地演示的正常条件；在线所需的账号凭据必须由用户提供。

## 1. 克隆、初始化和启动

### Windows PowerShell

```powershell
git clone https://github.com/bambom/game-ui-magiclayer.git
cd game-ui-magiclayer
py -3.10 scripts/bootstrap.py
.\.venv\Scripts\python.exe scripts/run.py --demo --model vit_b
```

如果已有合适版本的 `python`，可用 `python scripts/bootstrap.py`。不需要激活虚拟环境或修改 PowerShell 执行策略。缺少 Python 时，AI 可在有软件安装权限的机器上安装：

```powershell
winget install --exact --id Python.Python.3.10 --silent --accept-package-agreements --accept-source-agreements
```

安装后重新发现 `py`/Python 可执行文件；不要继续使用旧 PATH 中的 Windows Store 占位程序。

### Linux / macOS

```bash
git clone https://github.com/bambom/game-ui-magiclayer.git
cd game-ui-magiclayer
python3.10 scripts/bootstrap.py
.venv/bin/python scripts/run.py --demo --model vit_b
```

可使用已安装的 `python3.11` 或 `python3.12` 替代；不要用 Python 3.13+ 安装这组固定依赖。Linux 需系统的 venv/ensurepip 支持，例如 Ubuntu 22.04 安装 `python3.10 python3.10-venv`；Debian 12 可选 `python3.11 python3.11-venv`。macOS 可用 `brew install python@3.11`。系统包安装需要该机器已有的权限。默认 PyTorch CPU 索引面向 Windows/Linux x64；macOS 使用官方 PyPI wheel，见下方脚本说明。

### 脚本行为

```powershell
# 检查已有环境，不下载模型、不访问在线 AI
.\.venv\Scripts\python.exe scripts/doctor.py --model vit_b

# 只补齐模型；已存在且 hash 正确时跳过下载
python scripts/bootstrap.py --download-only --model vit_b

# 使用另一个独立环境目录，保留原环境
python scripts/bootstrap.py --venv .venv-other
```

重复初始化时，依赖文件、初始化脚本、Python 版本及所选模型均相同且上次安装成功，会跳过 pip 安装并继续检查模型与环境。首次安装或上述输入变化会重新处理依赖。下载写入 `.part`，完整大小和 SHA256 均匹配后才改为正式文件。发现已有正式模型损坏会报错并保留它，需明确移开后续传；中断下载重试会重新下载，不支持断点续传。HTTP 代理可在本机 `.env` 设置，脚本启动时会读取。

不要将已有生产环境指定给 `--venv`。默认安装的 torch/torchvision 是 `2.6.0/0.21.0`，运行依赖见 `requirements.txt`，测试依赖另在 `requirements-dev.txt`。

pip 下载异常慢时，可以先准备与当前系统/Python 匹配、按官方 PyPI SHA256 核对过的完整 wheel 缓存，在本机通过 `PIP_FIND_LINKS` 指定目录、`PIP_NO_INDEX=1` 禁用在线索引，再运行同一 bootstrap 命令续装。缓存需覆盖尚未安装的全部依赖；这两个变量仅用于当次安装，不写入共享配置或仓库。此次安装的网络问题及续装结果保留在验证记录中。

## 2. 处理自己的图片

### 本地运行：自己提供图层规划

```powershell
.\.venv\Scripts\python.exe scripts/run.py --image "C:\images\ui.png" --plan "C:\images\plan.json" --local --model vit_b
```

Linux/macOS 将 Python 路径换成 `.venv/bin/python`。规划格式见 `algorithm/planner/schema.py`、`algorithm/planner/prompt.py` 和 `scripts/run.py` 的演示生成逻辑。`scripts/make_sample_plan.py` 是特定 840×1872 卡牌布局的历史示例，不适用于任意图片。

### 在线运行：视觉模型规划与生图修补

初始化时会生成被 Git 忽略的 `.env`；按 `.env.example` 选择一种协议，填写真实服务信息。值使用纯 `KEY=VALUE`，不要加引号、`export` 或行尾注释；进程环境变量优先。

| 变量 | 填写内容 |
| --- | --- |
| `MAGICLAYER_API_STYLE` | `openai` 或 `gemini` |
| `MAGICLAYER_OPENAI_BASE` | OpenAI 兼容 API 根地址，例如 `https://your-gateway.example/v1`；不是网页聊天地址 |
| `OPENAI_API_KEY` | OpenAI 兼容服务密钥，留在本机；也兼容全局 `GEMINI_API_KEY` |
| `GEMINI_API_KEY` | 选择 Gemini 原生协议时的密钥 |
| `MAGICLAYER_PLANNER_MODEL` | 服务实际支持的视觉理解模型 ID，必须能看图并返回 JSON |
| `MAGICLAYER_IMAGE_MODEL` | 服务实际支持的图片生成/编辑模型 ID |

```powershell
.\.venv\Scripts\python.exe scripts/run.py --image "C:\images\ui.png" --model vit_b
```

`--plan` 缺省才调用在线规划；不带 `--local` 时调用在线图片修补。脚本显式选择在线 planner/provider，因此使用 `OPENAI_API_KEY` 也能进入在线流程。仓库中的历史模型默认名不代表任一供应商当前提供该模型，在线使用应填完整的实际模型名。

可分别配置 `MAGICLAYER_{PLANNER|ELEMENT|BACKGROUND}_{BASE,KEY,MODEL}`。显式阶段 BASE 使用 OpenAI 兼容协议：PLANNER 调 `/chat/completions`，ELEMENT/BACKGROUND 调 `/images/edits`；全局 Gemini 配置走原生内容生成接口，全局 OpenAI 配置走 chat 图片响应。阶段质量、代理等示例均在 `.env.example`。修补失败可能退回本地 Telea，需查看 `job.json` 的 warnings/degraded 和 `layers.json.backgroundDegraded`；规划失败会使作业失败。

真实图片与密钥不会因初始化而上传；选择在线运行时，原图、去字图或修补裁片会按算法阶段发往所配置服务。

## 3. 模型文件从哪里下载

权重和模型包不提交到 Git，也不通过 Git LFS 上传。默认只下载一个 SAM ViT-B；无需下载全部模型。

| 后端 | 权重及目标位置 | 下载方式 |
| --- | --- | --- |
| 默认 `vit_b` | `algorithm/models/sam_vit_b_01ec64.pth`，约 375 MB | `python scripts/bootstrap.py` 自动下载 |
| `sam2.1-hiera-tiny` | `algorithm/models/sam2.1_hiera_tiny.pt`，约 156 MB | `bootstrap.py --model sam2.1-hiera-tiny` |
| `sam2.1-hiera-small` | `algorithm/models/sam2.1_hiera_small.pt`，约 184 MB | 同上，将模型名替换为 small |
| `sam2.1-hiera-base_plus` | `algorithm/models/sam2.1_hiera_base_plus.pt`，约 324 MB | 同上，将模型名替换为 base_plus |
| `sam2.1-hiera-large` | `algorithm/models/sam2.1_hiera_large.pt`，约 898 MB | 同上，将模型名替换为 large |
| 可选 `sam3` | `algorithm/models/sam3.pt`，约 3.45 GB | 需账号下载权限及独立 CUDA 环境，见下一节 |
| RapidOCR | `.venv/.../site-packages/rapidocr_onnxruntime/models/` 下三个 ONNX 文件 | `pip install rapidocr-onnxruntime==1.4.4` 的包自带，由 bootstrap 自动安装，doctor 检查 |

官方来源：[SAM](https://github.com/facebookresearch/segment-anything)、[SAM2](https://github.com/facebookresearch/sam2)、[SAM3](https://github.com/facebookresearch/sam3)。SAM1/SAM2 的精确 URL、大小和 SHA256 在 `config/models.json`；脚本不从第三方镜像下载或猜测链接。RapidOCR 权重不需要手工拷入 `algorithm/models/`。

### 可选 SAM2.1

```powershell
# 需要 Git。下载所选权重并从固定版本安装 SAM2 Python 包，禁用可选 CUDA 扩展构建。
python scripts/bootstrap.py --venv .venv-sam2 --model sam2.1-hiera-small
.\.venv-sam2\Scripts\python.exe scripts/run.py --demo --model sam2.1-hiera-small
```

SAM2 Python 源码固定到 `2b90b9f5ceec907a1c18123530e92e794ad901a4`，不将源码副本提交到本仓库。`--model` 是当次配置，不会覆盖 `.env`；下一次启动仍需传同一模型名，或手动在本机 `.env` 中设置 `MAGICLAYER_SAM_MODEL`。默认 CPU 演示不依赖此选项。

### 可选 SAM3：单独的 Python 3.12 + NVIDIA CUDA 环境

此后端采用子进程桥，不改变主管线 `.venv` 的 CPU torch。需要兼容 NVIDIA 驱动、CUDA GPU、Python 3.12，以及已获准下载 [facebook/sam3](https://huggingface.co/facebook/sam3) 的 Hugging Face 账号。账号许可无法由 AI 自动代为批准；没有许可或 GPU 时保持默认 `vit_b`，不要绕过访问控制。

Windows 示例（Linux 对应替换解释器路径，并按官方方式安装 Triton）：

```powershell
py -3.12 -m venv .venv-sam3
.\.venv-sam3\Scripts\python.exe -m pip install --upgrade pip setuptools
.\.venv-sam3\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
.\.venv-sam3\Scripts\python.exe -m pip install -e third_party/sam3
.\.venv-sam3\Scripts\python.exe -m pip install opencv-python==4.11.0.86 scipy==1.13.1 einops pycocotools "huggingface_hub[cli]" triton-windows==3.8.0.post28
# 先由用户在本机完成 hf auth login，或通过本机环境注入 HF_TOKEN；不将 token 写入命令或文档。
.\.venv-sam3\Scripts\hf.exe download facebook/sam3 sam3.pt --local-dir algorithm/models
```

在本机 `.env` 中设置 `MAGICLAYER_SAM_MODEL=sam3`、`MAGICLAYER_SAM3_PYTHON=<独立环境 Python 绝对路径>`、`MAGICLAYER_SAM3_WEIGHTS=<sam3.pt 绝对路径>`。Linux 必须显式设置解释器路径，因为核心桥的历史默认路径是 Windows 路径。然后由主管线环境执行 `scripts/run.py --demo --model sam3`。当前桥使用 SAM3 的图像交互模型，不使用 `sam3.1_multiplex.pt`。可用 `algorithm/segment/sam3_worker.py` 单独诊断；调试时设置 `MAGICLAYER_SAM3_DEBUG=1`。

SAM3/SAM2 为可选配置。可复用现有源码中的历史实现，默认环境初始化的验收不代表在另一台机器上完成了 CUDA/GPU 验证。

## 4. 输出与代码地图

每次生成唯一作业目录 `jobs/<job-id>/`。也可传 `--jobs-root` 和 `--job-id`；启动器拒绝覆盖已有作业。

| 文件 | 用途 |
| --- | --- |
| `status.json` | 运行状态和阶段，成功时 state 为 done |
| `job.json` | 产物清单、warnings、耗时与降级说明 |
| `layers.json` | 图层几何、文本样式、PNG 相对路径及背景契约 |
| `layers/*.png` | 透明图片图层 |
| `layers_text/*.png` | 文字预览 |
| `background.png` / `preview.png` | 修补背景与重组预览 |
| `debug/` | OCR、去字、mask、Atlas 等中间结果 |

`algorithm/config.py` 管理算法参数，`MAGICLAYER_*` 环境变量可覆盖；`scripts/run.py` 会先读 `.env` 再启动算法，确保 import 时读取的参数生效。`algorithm/pipeline.py` 编排；`ocr/`、`texterase/`、`planner/`、`segment/`、`repair/`、`compose/` 分别负责各阶段；`providers/` 适配在线接口；`storage.py` 管理作业状态和产物。底层 CLI 的 ROI/重拆选项可用环境 Python 执行 `-m algorithm.pipeline --help` 查看。

详细算法见 [拆图算法详细流程](doc/拆图算法详细流程.md)，模块说明见 [algorithm/README.md](algorithm/README.md)。现有 `algorithm/tests/` 保留为源码参考；环境初始化不自动执行历史回归。此次独立发布的验证范围与结果单独记录在 [SETUP-VERIFICATION](doc/SETUP-VERIFICATION.md)。

## 5. 错误处理与发布前检查

- 缺少 `venv/ensurepip`：安装当前 Python 对应的系统 venv 包，再重试 bootstrap。
- 安装/下载超时：检查外网和本机 `.env` 的 HTTPS_PROXY；不要把代理密码输出到日志或上传。pip 失败继续运行同一 bootstrap 命令；模型步骤不加载 `.part` 文件。
- 缺少或损坏模型：按 doctor 输出下载对应模型。完整模型 SHA256 不匹配时，移开该文件后重试，不覆盖已有证据。
- `No module named ...`：确认执行的是 `.venv` 中的 Python，再检查 bootstrap 的失败步骤。
- 内存不足：先默认 `vit_b` 和较小图片，避免自动切换更大的 SAM2/SAM3。
- 在线服务 401/403、模型不存在：只在本机检查 Key、BASE 和模型 ID；不要换填虚构凭据。无在线凭据仍可执行 demo。

发布前先显式暂存需要的文件，然后执行：

```bash
python scripts/check_publish.py
git diff --cached --stat
```

审计检查 Git 索引中的模型、运行数据、密钥文件、常见凭据格式和超大文件，报错仅输出路径及行号。`.gitignore` 额外排除了 `.env`、虚拟环境、全部模型目录/权重扩展名、下载缓存、作业及历史本机诊断脚本。原有本机文件保留，但不进入这个独立仓库的提交历史。
