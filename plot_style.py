"""统一论文 / PPT 绘图样式

用法:
    from plot_style import apply_style, COLORS, savefig
    apply_style()              # 默认论文模式
    apply_style("ppt")         # PPT 模式 (大字体, 粗线条)
    apply_style("poster")      # 海报模式

    fig, axes = make_fig(2, 2)                # 论文双栏 2x2
    fig, axes = make_fig(1, 3, target="ppt")  # PPT 宽幅 1x3
    label_subplots(axes)                      # 自动标注 (a) (b) (c)...
    savefig(fig, "output.png")                # 300dpi 保存
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from itertools import product as _product

# ================================================================
#  配色方案
# ================================================================

COLORS = {
    "primary":    "#2563EB",
    "secondary":  "#DC2626",
    "accent1":    "#059669",
    "accent2":    "#D97706",
    "accent3":    "#7C3AED",
    "accent4":    "#DB2777",
    "gray":       "#6B7280",
    "light_gray": "#D1D5DB",
    "train":      "#2563EB",
    "test":       "#DC2626",
    "reference":  "#DC2626",
}

PALETTE = [
    COLORS["primary"], COLORS["secondary"], COLORS["accent1"],
    COLORS["accent2"], COLORS["accent3"], COLORS["accent4"],
]

# ================================================================
#  预设参数 (论文 / PPT / 海报)
# ================================================================

_PRESETS = {
    "paper": {
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":   "stix",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "axes.linewidth":     0.8,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "lines.linewidth":    1.2,
        "lines.markersize":   4,
        "legend.fontsize":    8,
        "figure.dpi":         150,
        "savefig.dpi":        300,
    },
    "ppt": {
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset":   "dejavusans",
        "font.size":          16,
        "axes.titlesize":     18,
        "axes.labelsize":     16,
        "axes.linewidth":     1.2,
        "xtick.labelsize":    14,
        "ytick.labelsize":    14,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "lines.linewidth":    2.0,
        "lines.markersize":   7,
        "legend.fontsize":    13,
        "figure.dpi":         100,
        "savefig.dpi":        200,
    },
    "poster": {
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset":   "dejavusans",
        "font.size":          20,
        "axes.titlesize":     22,
        "axes.labelsize":     20,
        "axes.linewidth":     1.5,
        "xtick.labelsize":    18,
        "ytick.labelsize":    18,
        "xtick.major.width":  1.2,
        "ytick.major.width":  1.2,
        "lines.linewidth":    2.5,
        "lines.markersize":   9,
        "legend.fontsize":    16,
        "figure.dpi":         100,
        "savefig.dpi":        300,
    },
}

_COMMON_RC = {
    "axes.grid":          True,
    "axes.grid.which":    "major",
    "axes.prop_cycle":    plt.cycler("color", PALETTE),
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#CCCCCC",
    "scatter.edgecolors": "none",
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "figure.constrained_layout.use": True,
}

# ================================================================
#  预设尺寸 (英寸)
# ================================================================

_FIGURE_SIZES = {
    "paper": {
        "single_col":  3.5,    # 单栏宽度 (约 88mm)
        "double_col":  7.2,    # 双栏宽度 (约 183mm)
        "full_page":   7.2,
        "cell_w":      3.5,    # 每个子图单元宽度
        "cell_h":      3.0,    # 每个子图单元高度
    },
    "ppt": {
        "single_col":  6.0,
        "double_col":  12.0,
        "full_page":   13.0,
        "cell_w":      5.0,
        "cell_h":      4.0,
    },
    "poster": {
        "single_col":  8.0,
        "double_col":  16.0,
        "full_page":   16.0,
        "cell_w":      7.0,
        "cell_h":      5.5,
    },
}

_current_target = "paper"

# ================================================================
#  公共接口
# ================================================================

def apply_style(target="paper"):
    """应用绘图样式预设。

    Args:
        target: "paper" | "ppt" | "poster"
    """
    global _current_target
    if target not in _PRESETS:
        raise ValueError(f"未知预设: {target}，可选: {list(_PRESETS.keys())}")
    _current_target = target
    rc = {}
    rc.update(_COMMON_RC)
    rc.update(_PRESETS[target])
    plt.rcParams.update(rc)


def make_fig(nrows=1, ncols=1, target=None, scale=1.0, **kwargs):
    """创建适配目标媒介尺寸的 Figure。

    Args:
        nrows, ncols: 子图行列数
        target: 覆盖当前预设 ("paper" / "ppt" / "poster")，None 则用 apply_style 设定的
        scale: 整体缩放因子
        **kwargs: 传递给 plt.subplots 的其他参数

    Returns:
        fig, axes (与 plt.subplots 一致)

    示例:
        fig, axes = make_fig(2, 3)                   # 论文 2x3
        fig, axes = make_fig(1, 4, target="ppt")     # PPT 1x4
        fig, (ax1, ax2) = make_fig(1, 2, scale=1.2)  # 稍大一点
    """
    t = target or _current_target
    sz = _FIGURE_SIZES.get(t, _FIGURE_SIZES["paper"])
    w = sz["cell_w"] * ncols * scale
    h = sz["cell_h"] * nrows * scale
    kwargs.setdefault("figsize", (w, h))
    kwargs.setdefault("squeeze", True)
    return plt.subplots(nrows, ncols, **kwargs)


def savefig(fig, path, dpi=None, formats=None):
    """统一保存图片。

    Args:
        fig: matplotlib Figure
        path: 保存路径 (如 "output.png")
        dpi: 覆盖默认 DPI，None 则使用预设值
        formats: 同时保存为多种格式，如 ["png", "pdf", "svg"]
    """
    save_dpi = dpi or _PRESETS.get(_current_target, _PRESETS["paper"])["savefig.dpi"]

    if formats:
        import os
        base, _ = os.path.splitext(path)
        for fmt in formats:
            fig.savefig(f"{base}.{fmt}", dpi=save_dpi, bbox_inches="tight", pad_inches=0.05)
    else:
        fig.savefig(path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ================================================================
#  子图标注
# ================================================================

def label_subplots(axes, fmt="({letter})", x=-0.08, y=1.06, fontweight="bold", fontsize=None):
    """为子图添加 (a) (b) (c)... 标签。

    Args:
        axes: 单个 Axes、1D/2D 数组
        fmt: 格式字符串，{letter} 会被替换为 a/b/c...
             也支持 {LETTER} → A/B/C，{number} → 1/2/3
        x, y: 标签位置 (axes 坐标)
        fontweight: 字重
        fontsize: 字号，None 则使用 axes.titlesize

    示例:
        label_subplots(axes)                          # (a) (b) (c) ...
        label_subplots(axes, fmt="({LETTER})")        # (A) (B) (C) ...
        label_subplots(axes, fmt="{letter})")          # a) b) c) ...
    """
    if fontsize is None:
        fontsize = plt.rcParams.get("axes.titlesize", 11)

    if hasattr(axes, "flat"):
        ax_list = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        ax_list = list(axes)
    else:
        ax_list = [axes]

    for i, ax in enumerate(ax_list):
        if not ax.get_visible():
            continue
        letter = chr(ord("a") + i)
        text = fmt.format(letter=letter, LETTER=letter.upper(), number=i + 1)
        ax.text(x, y, text, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, va="top", ha="left")


# ================================================================
#  常用绘图辅助
# ================================================================

def add_identity_line(ax, **kwargs):
    """在散点图上添加 y=x 参考线。"""
    lo, hi = ax.get_xlim()
    defaults = dict(color=COLORS["reference"], linestyle="--", linewidth=1,
                    label="$y = x$", zorder=0)
    defaults.update(kwargs)
    ax.plot([lo, hi], [lo, hi], **defaults)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def format_r2(val):
    """格式化 R² 值。"""
    return f"$R^2 = {val:.4f}$"


def format_metric(name, val, unit=""):
    """格式化指标值。"""
    if unit:
        return f"{name} = {val:.4f} {unit}"
    return f"{name} = {val:.4f}"


def annotate_stats(ax, text, loc="upper left"):
    """在图上添加统计信息文本框。

    Args:
        ax: matplotlib Axes
        text: 文本内容（支持多行和 LaTeX）
        loc: "upper left" | "upper right" | "lower left" | "lower right"
    """
    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CCCCCC")
    loc_map = {
        "upper left":  (0.03, 0.97, "top",    "left"),
        "upper right": (0.97, 0.97, "top",    "right"),
        "lower left":  (0.03, 0.03, "bottom", "left"),
        "lower right": (0.97, 0.03, "bottom", "right"),
    }
    lx, ly, va, ha = loc_map.get(loc, loc_map["upper left"])
    ax.text(lx, ly, text, transform=ax.transAxes, fontsize=plt.rcParams.get("legend.fontsize", 8),
            verticalalignment=va, horizontalalignment=ha, bbox=props)
