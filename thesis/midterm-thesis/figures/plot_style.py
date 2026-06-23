"""统一画图模板 — 支持 paper / ppt / a4 三种输出格式

使用:
    from plot_style import apply_style, COLORS
    cfg = apply_style('ppt')          # 'paper' | 'ppt' | 'a4'
    fig, ax = plt.subplots(figsize=cfg['figsize'], dpi=cfg['dpi'])
    ...
    fig.savefig('out.png', dpi=cfg['dpi'], bbox_inches='tight')

或命令行:
    python any_plot.py --style ppt
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================================
# 预设样式
# ============================================================================

PRESETS = {
    # 学位论文 / 答辩报告 — A4 纸张, 中等字号
    "paper": {
        "dpi": 200,
        "figsize": (12, 8),
        "figsize_wide": (14, 6),
        "figsize_square": (8, 8),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "savefig.bbox": "tight",
    },
    # PPT 答辩用 — 16:9, 大字号, 300 DPI
    "ppt": {
        "dpi": 300,
        "figsize": (16, 9),
        "figsize_wide": (18, 6),
        "figsize_square": (10, 10),
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "savefig.bbox": "tight",
    },
    # 嵌入论文 (LaTeX) — textwidth = 6.7 inch
    "a4": {
        "dpi": 300,
        "figsize": (6.7, 4.5),
        "figsize_wide": (6.7, 3.0),
        "figsize_square": (4.5, 4.5),
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "savefig.bbox": "tight",
    },
}


# ============================================================================
# 项目统一配色 (用于方法对比)
# ============================================================================

# 方法配色 (语义化, 4 种主方案)
COLORS = {
    "MLP+LIFT (CP 11d)":  "#2E7D32",    # 深绿 — 推荐方案
    "MLP+LIFT (sec 47d)": "#66BB6A",    # 浅绿 — 同类对照
    "MoE+LIFT (sec 47d)": "#1976D2",    # 蓝 — 次优
    "MoE+MDN (sec 47d)":  "#D32F2F",    # 红 — 否决方案
    # 输出维度色 (语义化)
    "T":   "#1F77B4",   # 推力 — 蓝
    "H":   "#FF7F0E",   # 侧向力 — 橘
    "My":  "#D62728",   # 俯仰矩 — 红 (难点)
    "Q":   "#2CA02C",   # 扭矩 — 绿
    # OOD 等级
    "in_dist":  "#4CAF50",
    "ood_mild": "#FF9800",
    "ood_far":  "#F44336",
    # 通用
    "grid":     "#E0E0E0",
    "best":     "#388E3C",   # 标注最佳值
    "ref_line": "#888888",   # 对角参考线
}


# 学术风格颜色映射 — 红-黄-绿,用于 heatmap
CMAP_RYG = LinearSegmentedColormap.from_list(
    "ryg",
    [(0.0, "#D73027"),    # 红 (最差)
     (0.25, "#FDAE61"),   # 橘
     (0.5, "#FFFFBF"),    # 浅黄
     (0.75, "#A6D96A"),   # 浅绿
     (1.0, "#1A9850")],   # 深绿 (最佳)
    N=256
)


# ============================================================================
# 主入口
# ============================================================================

def _find_cjk_font():
    """探测系统可用的中文字体 (macOS / Linux 通用)。"""
    candidates = [
        "PingFang SC", "PingFang HK", "Heiti SC", "STHeiti",
        "Songti SC", "STSong",
        "Source Han Sans CN", "Source Han Sans SC", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
        "Microsoft YaHei", "SimHei",
    ]
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None


def apply_style(mode: str = "paper") -> dict:
    """应用预设样式并返回配置字典。

    Parameters
    ----------
    mode : str
        'paper' (默认): 论文 / 答辩文档,A4 中等字号
        'ppt'        : PPT 大字号, 16:9, 300 DPI
        'a4'         : 学位论文嵌入, textwidth=6.7in
    """
    if mode not in PRESETS:
        raise ValueError(f"unknown mode: {mode}, choose from {list(PRESETS)}")
    p = PRESETS[mode]

    cjk = _find_cjk_font()
    sans_list = ["Helvetica", "Arial", "DejaVu Sans"]
    if cjk:
        sans_list = [cjk] + sans_list   # CJK 字体放最前面,确保中文用它渲染

    # 应用 matplotlib rcParams
    matplotlib.rcParams.update({
        "figure.dpi": p["dpi"],
        "savefig.dpi": p["dpi"],
        "font.family": ["sans-serif"],
        "font.sans-serif": sans_list,
        "axes.unicode_minus": False,           # 负号正常显示
        "font.size": p["font.size"],
        "axes.titlesize": p["axes.titlesize"],
        "axes.labelsize": p["axes.labelsize"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": COLORS["grid"],
        "xtick.labelsize": p["xtick.labelsize"],
        "ytick.labelsize": p["ytick.labelsize"],
        "legend.fontsize": p["legend.fontsize"],
        "legend.frameon": False,
        "lines.linewidth": p["lines.linewidth"],
        "lines.markersize": p["lines.markersize"],
        "savefig.bbox": p["savefig.bbox"],
        "savefig.pad_inches": 0.1,
    })
    return p


def parse_style_arg(argv):
    """从 sys.argv 解析 --style <mode> 参数, 默认 'paper'。"""
    if "--style" in argv:
        idx = argv.index("--style")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return "paper"
