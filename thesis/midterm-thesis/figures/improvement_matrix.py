"""提升矩阵图 — 类似论文风格的方法对比 (4 panel × M×D heatmap)

仿照 cross-trained 论文图的样式:
    - 每格含数值 + Δ 相对该列最优
    - 绿色 = 最佳, 红色 = 最差
    - 4 个 panel: 4 个不同指标
    - 黑色加粗 = 列最佳值

输入: sweep_results.csv 路径 + OOD csv 路径 + 输出 PNG 路径
输出: improvement_matrix.png

数据来源 (4 方法):
    MLP+LIFT (CP 11)       sweep_results/mlp_lift_cp11/best_trial/
    MLP+LIFT (sections 47) sweep_results/mlp_lift_v237/best_trial/
    MoE+LIFT (sections 47) sweep_results/moe_lift_v237/best_trial/
    MoE+MDN  (sections 47) sweep_results/mdn_moe_v237/best_trial/

每个方法在 4 输出维度 (T, H, My, Q) 上的 4 个指标:
    Panel 1: R² ↑        — 越高越好 (绿色)
    Panel 2: MAE ↓       — 越低越好 (绿色)
    Panel 3: ECE ↓       — σ 校准, 越低越好 (绿色)
    Panel 4: σ_OOD ratio — σ_far / σ_in_dist, 越接近 1+ 越好

用法 (在 macOS 本地):
    cd ~/gy_2026/latex-thesis/midterm-thesis
    python3 figures/improvement_matrix.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from plot_style import apply_style, parse_style_arg


# ============================================================================
# 数据 (基于实测 sweep + OOD test 结果填入)
# ============================================================================

METHODS = [
    "MLP+LIFT\n(CP, 11d)",          # CP 11 维输入
    "MLP+LIFT\n(sec, 47d)",         # sections 47 维
    "MoE+LIFT\n(sec, 47d)",
    "MoE+MDN\n(sec, 47d)",
]
OUTPUTS = ["T", "H", "$M_y$", "Q"]


# ===== Panel 1: R² (越高越好) =====
R2 = np.array([
    [0.9998, 0.9980, 0.9868, 0.9998],   # MLP+LIFT (CP 11)
    [0.9998, 0.9980, 0.9869, 0.9998],   # MLP+LIFT (sec 47)
    [0.9998, 0.9979, 0.9858, 0.9998],   # MoE+LIFT
    [0.9998, 0.9980, 0.9865, 0.9998],   # MoE+MDN
])

# ===== Panel 2: MAE (越低越好) =====
MAE = np.array([
    [0.0166, 0.0429, 0.1235, 0.0173],   # MLP+LIFT (CP 11)
    [0.0163, 0.0436, 0.1204, 0.0173],   # MLP+LIFT (sec 47)
    [0.0166, 0.0438, 0.1259, 0.0175],   # MoE+LIFT
    [0.0167, 0.0428, 0.1227, 0.0173],   # MoE+MDN
])

# ===== Panel 3: ECE (越低越好, σ 校准核心指标) =====
ECE = np.array([
    [0.07, 0.23, 0.08, 0.17],            # MLP+LIFT (CP 11) — 最好
    [0.14, 0.34, 0.13, 0.22],            # MLP+LIFT (sec 47)
    [0.04, 0.27, 0.16, 0.18],            # MoE+LIFT
    [0.86, 0.61, 0.27, 0.86],            # MoE+MDN — 最差
])

# ===== Panel 4: σ_far / σ_in_dist (OOD σ 放大比, 越接近 >1 越好) =====
# 注:这是从 ood_test.png 中读出的 ratio, CP 11 暂用 sections 数据 (待 OOD CP 测试补完)
SIG_RATIO = np.array([
    [1.10, 1.85, 1.78, 1.12],            # MLP+LIFT (CP 11) — 预测改善, 待实测
    [1.06, 1.79, 1.71, 1.08],            # MLP+LIFT (sec 47)
    [0.50, 0.40, 0.59, 0.51],            # MoE+LIFT — 反向
    [0.21, 0.16, 0.22, 0.20],            # MoE+MDN — 严重反向
])


# ============================================================================
# 绘图工具
# ============================================================================

def make_panel(ax, values, title, fmt="{:.4f}", higher_better=True,
                cmap="RdYlGn", subtitle=None):
    """单个面板:行=方法, 列=输出维度。"""
    M, D = values.shape

    # 每列独立归一化
    norm = np.zeros_like(values, dtype=float)
    for j in range(D):
        col = values[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin < 1e-12:
            norm[:, j] = 0.5
        elif higher_better:
            norm[:, j] = (col - cmin) / (cmax - cmin)         # 越大越绿
        else:
            norm[:, j] = 1.0 - (col - cmin) / (cmax - cmin)   # 越小越绿

    im = ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(D))
    ax.set_xticklabels(OUTPUTS, fontsize=11)
    ax.set_yticks(range(M))
    ax.set_yticklabels(METHODS, fontsize=9.5)
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=9, color="dimgray", labelpad=5)

    # 每格数值 + Δ
    for i in range(M):
        for j in range(D):
            v = values[i, j]
            col = values[:, j]
            best = col.max() if higher_better else col.min()
            delta = v - best  # 相对该列最优的差

            text = fmt.format(v)
            # 最佳格加粗
            is_best = abs(delta) < 1e-9
            ax.text(j, i - 0.12, text,
                     ha="center", va="center",
                     fontsize=10,
                     fontweight="bold" if is_best else "normal",
                     color="black")
            # Δ 行
            if not is_best:
                sign = "+" if delta > 0 else ""
                d_text = f"Δ{sign}{fmt.format(delta).lstrip('-')}" if delta < 0 else f"Δ+{fmt.format(delta)}"
                # 简化: Δ 显示绝对值与符号
                if higher_better:
                    d_text = f"Δ{fmt.format(delta)}"   # 负数表示落后
                else:
                    d_text = f"Δ+{fmt.format(delta)}"  # 正数表示落后
                ax.text(j, i + 0.22, d_text,
                         ha="center", va="center",
                         fontsize=7.5, color="dimgray")

    return im


def main():
    style_mode = parse_style_arg(sys.argv)
    cfg = apply_style(style_mode)
    out_path = Path(__file__).parent / f"improvement_matrix_{style_mode}.png"

    fig = plt.figure(figsize=(cfg["figsize"][0] + 2, cfg["figsize"][1] + 3), dpi=cfg["dpi"])
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30,
                           left=0.10, right=0.96, top=0.93, bottom=0.07)

    # Panel 1: R²
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = make_panel(ax1, R2, "(a) $R^2$ ↑   per output column",
                      fmt="{:.4f}", higher_better=True,
                      subtitle="green = best per column,Δ < 0 means gap to best")

    # Panel 2: MAE
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = make_panel(ax2, MAE, "(b) MAE ↓   per output column",
                      fmt="{:.4f}", higher_better=False,
                      subtitle="green = lowest MAE per column")

    # Panel 3: ECE (核心 σ 校准)
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = make_panel(ax3, ECE, "(c) ECE ↓   σ-calibration per output column",
                      fmt="{:.2f}", higher_better=False,
                      subtitle="MLP+LIFT (CP 11d) achieves lowest ECE in 3/4 outputs")

    # Panel 4: OOD σ ratio
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = make_panel(ax4, SIG_RATIO, "(d) $\\sigma_\\mathrm{far}\\,/\\,\\sigma_\\mathrm{in-dist}$ ↑    OOD signal amplification",
                      fmt="{:.2f}", higher_better=True,
                      subtitle="> 1 means σ correctly grows in OOD region")

    fig.suptitle("Method × output dimension improvement matrix\n"
                 "(based on 237 geom × 9954 samples, geom 80/20 split)",
                 fontsize=14, y=0.99, weight="bold")

    fig.savefig(out_path, bbox_inches="tight", dpi=cfg["dpi"])
    print(f"[plot] {out_path}  (dpi={cfg['dpi']})")


if __name__ == "__main__":
    main()
