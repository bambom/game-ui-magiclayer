"""算法常数集中管理。

所有数值均来自 doc/拆图算法详细流程.md（下称「文档」）；文档未给出的取值
在注释里标注「文档未定」，属于实现补齐，可经实验调整。

带 MAGICLAYER_ 前缀环境变量的常数可被服务器设置面板覆盖（launch 时注入
子进程环境）；未设置或值非法时回落到文档/实现默认值。
"""
from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_str(name: str, default: str, choices: tuple) -> str:
    value = os.environ.get(name, default)
    return value if value in choices else default


# ---------- OCR 过滤（文档 §1.1） ----------
OCR_MIN_CONF = _env_float("MAGICLAYER_OCR_MIN_CONF", 0.35)  # 最低置信度
SINGLE_LATIN_MIN_CONF = 0.85   # 单个拉丁字母低于该值视为图标误识别
# 「低置信度且高宽比>=2.4 的竖排装饰候选剔除」：文档未给出低置信度的分界值，
# 借用文档中唯一的高置信度门槛 0.85，可调。
VERTICAL_DECOR_MAX_CONF = 0.85
VERTICAL_DECOR_MIN_ASPECT = 2.4  # 高宽比（h/w）

# ---------- 去字（文档 §1.2.1） ----------
ERASE_BAND_DIV = 4              # 底色边带宽度 = clamp(min(h,w)/4, 1, 3)
ERASE_BAND_MIN = 1
ERASE_BAND_MAX = 3
ERASE_DIST_SCALE = 3            # 颜色距离放大倍数，3d 截断到 0..255
ERASE_OTSU_FLOOR = 12           # 字形条件 3d >= max(12, 0.65T)
ERASE_OTSU_FACTOR = 0.65
ERASE_AREA_MIN = 0.015          # 字形面积占比下限，1.5%
ERASE_AREA_MAX = 0.68           # 面积占比超上限回退 coarse mask
ERASE_DILATE_CJK = (0.16, 3, 8)   # 单个中文字符：clamp(round(0.16h), 3, 8)
ERASE_DILATE_OTHER = (0.09, 2, 6)  # 其他文字：clamp(round(0.09h), 2, 6)
INPAINT_RADIUS = 5              # Telea radius

# ---------- 文字样式（文档 §1.3） ----------
STYLE_QUANT_STEP = 32           # 前景色按 32 宽度量化
STYLE_BUCKET_BASE_SCORE = 24    # 颜色桶得分 = 像素数 x (24 + 与底色距离)
STYLE_SIZE_CJK = 0.88           # 单个中文字符 round(0.88h)
STYLE_SIZE_OTHER = 0.82         # 其他文字 round(0.82h)
STYLE_SIZE_MIN = 8
STYLE_WEIGHT_COVERAGE = 0.17    # coverage >= 0.17 -> 700
STYLE_WEIGHT_BOLD = 700
STYLE_WEIGHT_NORMAL = 600
STYLE_STROKE_COVERAGE = 0.12    # coverage > 0.12 -> round(0.045h)
STYLE_STROKE_WIDE = 0.045
STYLE_STROKE_NARROW = 0.03
STYLE_STROKE_MAX = 2
STYLE_STROKE_DARK = "#1e2322"   # 前景不暗于背景时
STYLE_STROKE_LIGHT = "#f0f4f1"
FONT_CJK = "Microsoft YaHei"
FONT_LATIN = "Arial"

# ---------- SAM 候选评分（文档 §2.3） ----------
SCORE_POS_WEIGHT = 0.22
SCORE_NEG_WEIGHT = 0.38
SCORE_CONTAIN_WEIGHT = 0.10
SCORE_IOU_WEIGHT = 0.06

# ---------- mask 清理（文档 §2.4） ----------
CLOSE_KERNEL = 3                # 3x3 全 1 核，闭运算（先膨胀后腐蚀）
COMPONENT_KEEP_RATIO = 0.08     # 面积 >= 最大组件 8% 的组件保留

# ---------- 装饰大横带纵向分割（2026-09-03 二期问题二，文档未定） ----------
# planner 规则 v2 已写"每个独立簇一条查询"但模型不守（run 4968ff94 的
# scenery_relic_cluster 只有 1 条，bbox_norm 全宽横带 [0.052,0.088,0.893,0.165]），
# 且横带下缘把向下延伸的树冠/山脚裁掉。规划是概率行为，兜底必须是确定性的：
# 按列像素投影找簇间谷位切分，再逐簇下扩 bbox 重跑一次 SAM 把延伸部分收回。
SPLIT_DISABLE = _env_int("MAGICLAYER_SPLIT_DISABLE", 0)
SPLIT_MIN_WIDTH_RATIO = _env_float("MAGICLAYER_SPLIT_MIN_WIDTH_RATIO", 0.55)
SPLIT_MIN_ASPECT = _env_float("MAGICLAYER_SPLIT_MIN_ASPECT", 3.0)
SPLIT_VALLEY_RATIO = _env_float("MAGICLAYER_SPLIT_VALLEY_RATIO", 0.15)
SPLIT_VALLEY_MIN_W = _env_int("MAGICLAYER_SPLIT_VALLEY_MIN_W", 12)
SPLIT_MIN_CLUSTER_PX = _env_int("MAGICLAYER_SPLIT_MIN_CLUSTER_PX", 400)
SPLIT_EXTEND_DOWN = _env_float("MAGICLAYER_SPLIT_EXTEND_DOWN", 1.0)
# 下扩重跑的接受条件：新 mask 必须仍盖住原簇（覆盖率下限）且没吞进背景（面积上限）
SPLIT_KEEP_MIN_COVER = _env_float("MAGICLAYER_SPLIT_KEEP_MIN_COVER", 0.6)
SPLIT_KEEP_MAX_GROWTH = _env_float("MAGICLAYER_SPLIT_KEEP_MAX_GROWTH", 2.5)

# ---------- mask 边界整形（2026-09-03 二期问题三，文档未定） ----------
# SAM mask 边界本就粗糙（vit_b 在 841px 宽图上 illustration_card_frames#1 的
# alpha 边界 x 跳动 159 次、navigation_tab_bar 106 次，平滑矩形应≈0），3x3 闭
# 运算只补 1-2px 裂缝，导出 PNG 呈锯齿/缺口。两档确定性整形：
# ①圆角矩形吸附——bbox 内按 IoU 拟合圆角半径重光栅，**以重建 IoU 自证**
#   （≥SNAP_MIN_IOU 才认为"这个矩形就是它的平滑版"），非矩形形状自动落空；
# ②轮廓高斯重光栅——沿轮廓序列平滑 sigma，只对 compactness=周长²/(4π面积)
#   不超上限的形状做。实测标尺：圆 1.1、L 形 1.7、十字 2.6、带锯齿的矩形 4.2
#   （噪声本身抬高 compactness，门槛必须给足余量）、多刺植被剪影 58。
# 两档都受"位移不出 MAX_SHIFT_PX + IoU ≥ MIN_IOU"守卫，超出即放弃整形。
SHAPE_DISABLE = _env_int("MAGICLAYER_SHAPE_DISABLE", 0)
SHAPE_SNAP_MIN_IOU = _env_float("MAGICLAYER_SHAPE_SNAP_MIN_IOU", 0.97)
SHAPE_MIN_IOU = _env_float("MAGICLAYER_SHAPE_MIN_IOU", 0.95)
SHAPE_MAX_SHIFT_PX = _env_int("MAGICLAYER_SHAPE_MAX_SHIFT_PX", 4)
SHAPE_SMOOTH_SIGMA = _env_float("MAGICLAYER_SHAPE_SMOOTH_SIGMA", 2.0)
SHAPE_MAX_COMPACTNESS = _env_float("MAGICLAYER_SHAPE_MAX_COMPACTNESS", 8.0)

# ---------- SAM 分割质量重试（文档未定，实现补齐） ----------
SAM_RETRY_SCORE = _env_float("MAGICLAYER_SAM_RETRY_SCORE", 0.75)  # clean 后低于该值则网格补点重跑一次
SAM_RETRY_MAX_COMPONENTS = _env_int("MAGICLAYER_SAM_RETRY_MAX_COMPONENTS", 3)  # 连通域数超过该值视为碎裂
SAM_RETRY_LARGEST_RATIO = _env_float("MAGICLAYER_SAM_RETRY_LARGEST_RATIO", 0.8)  # 碎裂判定：最大组件占比低于该比例才重试
SAM3_TIMEOUT = _env_int("MAGICLAYER_SAM3_TIMEOUT", 180)  # SAM3 子进程单请求超时秒数（build~7s + 3.45G 权重首载）

# ---------- 环形 mask 外扩（框太细，2026-09-03 用户裁定，文档未定） ----------
# SAM 对框环只切到描边中线，导出框层缺外半圈笔画（README「Atlas 洞区涂黑」
# 节记载的同一偏差）。确定性外扩补回：只作用于有内开口的环形 mask（框/面板，
# 几何判定，不看 planner 声明的 kind），内开口尺寸不动，外边界外推。
MASK_WIDEN_METHOD = _env_str("MAGICLAYER_MASK_WIDEN_METHOD", "adaptive",
                             ("off", "fixed", "adaptive"))
MASK_WIDEN_PX = _env_int("MAGICLAYER_MASK_WIDEN_PX", 2)      # fixed：固定外扩像素
MASK_WIDEN_RATIO = _env_float("MAGICLAYER_MASK_WIDEN_RATIO", 1.0)  # adaptive：外扩 = 环带宽度 × 比例（1.0 ≈ 补回被切掉的外半圈）
MASK_WIDEN_MAX_PX = _env_int("MAGICLAYER_MASK_WIDEN_MAX_PX", 12)   # adaptive：外扩上限，防大框过冲吃背景

# ---------- Atlas / 背景修补 ----------
# 2026-09-03 用户裁定：元素原尺寸零缩放拼放，画布 1:1 正方形且最小 1024
# （gpt-image 线宽高比归一，方进方出形变最小）。位置/大小记
# debug/05_atlas_placements.json，切回按记录整块取图。
ATLAS_MIN_SIDE = _env_int("MAGICLAYER_ATLAS_MIN_SIDE", 1024)
# ATLAS_MAX_SIDE 不再触发缩放（旧的等比缩小+切回放大是"元素被拉伸/大小改变"
# 的来源），仅作超大画布的告警阈值：超过时 runlog 提醒网关出图≈1.5MP、
# 切回分辨率可能低于原始。
ATLAS_MAX_SIDE = _env_int("MAGICLAYER_ATLAS_MAX_SIDE", 1600)
ATLAS_GAP = 8                   # 灰色间隔宽度（生图 prompt 提到 gray spacing）
ATLAS_GAP_COLOR = (128, 128, 128)
BG_SOFT_FEATHER_SIGMA = 3.0     # 硬 mask -> 软 mask 的高斯 sigma（极窄软边）
SURFACE_BG_INPAINT_RADIUS = 15  # surface 背景延续用的 Telea 半径（文档未定）
SURFACE_BG_DOWNSCALE = 4        # surface 背景修补先降采样提速（文档未定）
BG_REPAIR_DILATE_ITERS = _env_int("MAGICLAYER_BG_REPAIR_DILATE_ITERS", 2)  # 隐藏区 mask 膨胀迭代（3x3）：吞掉没扣干净的残边再送模型/融合
BG_SEAM_BAND_ITERS = _env_int("MAGICLAYER_BG_SEAM_BAND_ITERS", 2)  # 接缝带起始宽度（3x3 迭代）：只在 mask 内侧
BG_SEAM_BLUR_SIGMA = _env_float("MAGICLAYER_BG_SEAM_BLUR_SIGMA", 1.2)  # 接缝带高斯混合强度
# 接缝带自适应加宽（2026-09-03 二期问题五a）：val-postproc-3 实测洞边界向外
# 1px 带差 28.7 / P95 63、2-3px 带 3.6、≥4px 为 0——残留一条 ≤3px 过渡带，
# 固定 2px 不够吃。平滑后量接缝阶跃（洞缘内外 1px 环的中位色差），超阈就加宽
# 重做（每轮都从原图重算，不叠加）。加满仍超阈时可选梯度域调和收尾。
BG_SEAM_RESID_THR = _env_float("MAGICLAYER_BG_SEAM_RESID_THR", 10.0)
BG_SEAM_MAX_ITERS = _env_int("MAGICLAYER_BG_SEAM_MAX_ITERS", 4)
BG_SEAM_POISSON = _env_int("MAGICLAYER_BG_SEAM_POISSON", 0)
# 低频锚定合成（2026-09-03，run 4968ff94：宽松 img2img 网关洞内自行构图出暖褐
# 卡板）的低频/高频分离 sigma；文档未定，实现补齐。
BG_ANCHOR_SIGMA = _env_float("MAGICLAYER_BG_ANCHOR_SIGMA", 12.0)
# probe-bg-r1 用户裁定合格的背景图都是网关原始出图，没走锚定。默认关：
# MAGICLAYER_BG_ANCHOR=0。打开(=1)才把洞内颜色场换成 Telea 低频（旧卡板防御）。
BG_ANCHOR_ENABLE = _env_int("MAGICLAYER_BG_ANCHOR", 0)
# 背景是否把硬洞图作为第二张图送出。auto=不送（待补区已涂黑，只送一张图）。
# 0=从不送，1=有洞就送。调用时读 env，不在 import 冻结。
# MAGICLAYER_BG_ATTACH_MASK=auto|0|1
# MAGICLAYER_BG_PROMPT=auto|scene|short  （auto/scene=SCENE 长提示；short=中文口语）
# MAGICLAYER_IMAGE_URL_PROXY：Grok edits 结果 JPEG 在 imgen.x.ai，直连超时时
# 用该代理下载（与 OpenRouter 同一 VPN 口，例如 http://127.0.0.1:13808）。
# 洞内过平的环带纹理注入（二期问题五b）：洞内总高频 RMS 低于环带的这个比例
# 才注入，补到环带水平（实测轴线洞边缘密度只有真纸环带的 0.65，而验收目标是
# 0.8~1.2，故门槛就取 0.8）。0=关。整体密度比均值≈1.05 已达标，只治这类例外。
# 口径必须是"总高频"而不是细颗粒带：k_grain=min(1,环/洞) 一旦 <1，洞内细颗粒
# 就恰好等于环带、永无亏空，真正被压掉的是 k_struct 管的结构带（证据里
# k_struct=0.16 才是密度比 0.65 的来源）；模型返回纯平填充时 hole_grain≈0
# 反而得 k_grain=1.0——按 k_grain 设门槛会把最平的洞漏掉。
BG_GRAIN_INJECT_BELOW = _env_float("MAGICLAYER_BG_GRAIN_INJECT_BELOW", 0.8)
# 洞间一致性（二期问题五c）：k_grain 逐洞独立估计引入方差（密度比散布
# 0.65~1.60）。环带颗粒统计按 全局:局部 混合消方差——全靠全局会丢局部材质
# 差异（纸面与插画区颗粒本就不同），故取中间值。
BG_ANN_GLOBAL_MIX = _env_float("MAGICLAYER_BG_ANN_GLOBAL_MIX", 0.5)
# 洞内均值向环带均值的微平移上限（逐通道）：实测洞内均值与环带均值差 8~32，
# 大头是合法材质差，只削峰不硬拉。0=关。
BG_HOLE_MEAN_SHIFT_MAX = _env_float("MAGICLAYER_BG_HOLE_MEAN_SHIFT_MAX", 3.0)
# Atlas 洞填再对齐（linework.relevel：洞填色场不接边界，如节点洞白盘）的低频
# 场 sigma；文档未定，实现补齐。
ATLAS_RELEVEL_SIGMA = _env_float("MAGICLAYER_ATLAS_RELEVEL_SIGMA", 8.0)
# 子 mask 镂空膨胀（2026-09-03 二期问题一）：圆徽章的光晕/外环落在固定 2px
# 之外，在父层留成弧斑。紧凑团块按 min 边比例自适应膨胀（实测光晕半径 ≈
# 徽章尺寸 10~15%），细长/稀疏形状仍取 2px。RATIO=0 即回退固定 2px。
HALO_DILATE_RATIO = _env_float("MAGICLAYER_HALO_DILATE_RATIO", 0.12)
HALO_DILATE_MAX = _env_int("MAGICLAYER_HALO_DILATE_MAX", 8)
# 洞邻域残晕吸收（同上问题一b）：洞外环带宽度与"算一块残晕"的面积下限。
# RING_PX=0 关闭吸收。
ATLAS_HALO_RING_PX = _env_int("MAGICLAYER_ATLAS_HALO_RING_PX", 6)
ATLAS_HALO_MIN_PX = _env_int("MAGICLAYER_ATLAS_HALO_MIN_PX", 12)

# ---------- 半透明连续 alpha（2026-09-03 二期问题四，文档未定） ----------
# 导出层 alpha 一直是 SAM 二值 mask（0/255），植被雾化边缘、玻璃条、光晕被硬
# 切，叠回背景时边缘生硬。SAM 系（含 SAM2）本质都不产连续 alpha，但我们手里
# 同时有"合成图"与"背景估计"：观测 = α·F + (1−α)·B，对近均匀前景可逐像素解 α。
# 不依赖网关能力（gpt-image 的 background=transparent 支持性未知），符合分工原则。
MATTING_DISABLE = _env_int("MAGICLAYER_MATTING_DISABLE", 0)
# trimap 未知带宽 = clamp(RATIO·min(bbox边), MIN, MAX)
MAT_TRIMAP_RATIO = _env_float("MAGICLAYER_MAT_TRIMAP_RATIO", 0.04)
MAT_TRIMAP_MIN = _env_int("MAGICLAYER_MAT_TRIMAP_MIN", 2)
MAT_TRIMAP_MAX = _env_int("MAGICLAYER_MAT_TRIMAP_MAX", 12)
# |F−B| 低于此值时 α 在数值上不可解（分母趋零），该像素回退二值 mask
MAT_MIN_CONTRAST = _env_float("MAGICLAYER_MAT_MIN_CONTRAST", 8.0)
# 软边判据：边界过渡的等效宽度（px）达此值才算"软边"（硬边元素如卡框过渡只
# 有 1-2px，做 matting 只会把干净的直边啃出毛边）。等效宽度 = 8·跨界色差 /
# 带内 p90 梯度，见 matting.edge_softness——直接用梯度大小判是错的，梯度 =
# 对比度/宽度，高对比雾边的梯度比低对比硬边还大。实测标尺：硬阶跃 2.0、
# 1px 斜坡 2.0、2px 2.35、4px 4.0、8px 7.9，故门槛取 3.0（≥3px 过渡才值得解 α）。
MAT_SOFT_EDGE_WIDTH = _env_float("MAGICLAYER_MAT_SOFT_EDGE_WIDTH", 3.0)
# 未走 matting 的层：导出前对二值 alpha 边界做 1px 羽化，缓解视觉锯齿。0=关
EDGE_FEATHER_SIGMA = _env_float("MAGICLAYER_EDGE_FEATHER_SIGMA", 0.7)

# ---------- planner / 生图模型 ----------
# 官方端点；本地网关等经 MAGICLAYER_GEMINI_BASE 覆盖。
# OpenAI 兼容协议经 MAGICLAYER_API_STYLE=openai + MAGICLAYER_OPENAI_BASE 启用。
GEMINI_BASE = "https://generativelanguage.googleapis.com"
OPENAI_BASE = "http://127.0.0.1:8317/v1"
PLANNER_MODEL_DEFAULT = "gemini-3.7-flash"     # 文档指定（openai 网关用 -high 后缀，经 .env 覆盖）
IMAGE_MODEL_DEFAULT = "gemini-3.1-flash-image"  # 文档未定，取 3.1 flash 系（3-pro-image 可作质量升级）
GEMINI_TIMEOUT = 300
GEMINI_RETRIES = 3
