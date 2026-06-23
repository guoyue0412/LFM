"""OOD σ 失效专题图 — 突出反直觉发现

主题: σ 在数据范围外几何上不放大反而缩小, 论证三层防御机制的必要性。

显示:
    左: 三方法在 4 输出维度上的 σ_far / σ_in 比值条形图
         (期望值 > 1, 实测 MoE/MDN 反向)
    右: 三方法 σ 在三组几何 (in-dist / ood-mild / ood-far) 的分布
         (M_y 维度 — 最敏感)

用法:
    python ood_sigma_failure.py --style paper   # 默认
    python ood_sigma_failure.py --style ppt     # 16:9 PPT 版
    python ood_sigma_failure.py --style a4      # 嵌入论文
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import apply_style, parse_style_arg, COLORS


# ============================================================================
# 实测数据 (来自 ood_test.csv)
# ============================================================================

METHODS = ["MLP+LIFT\n(sec 47d)", "MoE+LIFT\n(sec 47d)", "MoE+MDN\n(sec 47d)"]
OUTPUTS = ["T", "H", "$M_y$", "Q"]

# σ_far / σ_in_dist 比值 (实测, 来自 ood_test.csv)
RATIO_FAR = np.array([
    [1.06, 1.79, 1.71, 1.08],   # MLP+LIFT
    [0.50, 0.40, 0.59, 0.51],   # MoE+LIFT
    [0.21, 0.16, 0.22, 0.20],   # MoE+MDN
])
RATIO_MILD = np.array([
    [1.02, 0.91, 1.29, 1.01],
    [0.73, 0.74, 0.90, 0.75],
    [0.35, 0.31, 0.50, 0.32],
])

# σ 值 (in-dist / mild / far) 仅 M_y 维度 (难点)
SIGMA_MY_IN = [0.1129, 0.1183, 0.0897]
SIGMA_MY_MILD = [0.1456, 0.1061, 0.0452]
SIGMA_MY_FAR = [0.1926, 0.0696, 0.0200]


def main():
    style_mode = parse_style_arg(sys.argv)
    cfg = apply_style(style_mode)
    out_dir = Path(__file__).parent

    # ---------- (Left) 三方法在 4 输出上的 σ_far/σ_in 比值 ----------
    fig, axes = plt.subplots(1, 2, figsize=(cfg["figsize"][0], cfg["figsize"][1] * 0.7),
                              gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    x = np.arange(len(OUTPUTS))
    w = 0.25
    bar_colors = ["#66BB6A", "#1976D2", "#D32F2F"]
    for i, (method, color) in enumerate(zip(METHODS, bar_colors)):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, RATIO_FAR[i], w, label=method, color=color, alpha=0.85)
        # 数值标注
        for j, b in enumerate(bars):
            v = RATIO_FAR[i, j]
            ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=cfg["xtick.labelsize"] - 1,
                     fontweight="bold" if v > 1 else "normal",
                     color="darkgreen" if v > 1 else "darkred")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
                label="理想 σ_far = σ_in (no signal)", alpha=0.7)
    ax.fill_between([-0.5, len(OUTPUTS) - 0.5], 0, 1, color="red", alpha=0.06,
                     label="σ 反向区 — OOD 时 σ 错误地缩小")
    ax.set_xticks(x)
    ax.set_xticklabels(OUTPUTS, fontsize=cfg["xtick.labelsize"] + 1)
    ax.set_ylabel(r"$\sigma_{\mathrm{far}} \;/\; \sigma_{\mathrm{in\text{-}dist}}$")
    ax.set_title("(a) OOD-far 几何上 σ 的放大比 (期望 > 1)", weight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=cfg["legend.fontsize"] - 1)
    ax.set_ylim(0, max(RATIO_FAR.max() + 0.4, 2.0))
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.5, len(OUTPUTS) - 0.5)

    # ---------- (Right) M_y 维度上三组几何的 σ 演变 ----------
    ax = axes[1]
    groups = ["in-dist", "ood-mild", "ood-far"]
    group_colors = [COLORS["in_dist"], COLORS["ood_mild"], COLORS["ood_far"]]
    data = {
        "MLP+LIFT": [SIGMA_MY_IN[0], SIGMA_MY_MILD[0], SIGMA_MY_FAR[0]],
        "MoE+LIFT": [SIGMA_MY_IN[1], SIGMA_MY_MILD[1], SIGMA_MY_FAR[1]],
        "MoE+MDN":  [SIGMA_MY_IN[2], SIGMA_MY_MILD[2], SIGMA_MY_FAR[2]],
    }
    line_colors = ["#66BB6A", "#1976D2", "#D32F2F"]
    line_styles = ["-", "--", ":"]
    markers = ["o", "s", "D"]
    for (method, vals), c, ls, m in zip(data.items(), line_colors, line_styles, markers):
        ax.plot(groups, vals, marker=m, linestyle=ls, color=c, label=method,
                 linewidth=cfg["lines.linewidth"], markersize=cfg["lines.markersize"] + 2)
        # 标注 in-dist 与 far 的比值
        ratio = vals[2] / max(vals[0], 1e-9)
        ax.annotate(f"×{ratio:.2f}",
                     xy=(2, vals[2]), xytext=(2 + 0.05, vals[2]),
                     fontsize=cfg["xtick.labelsize"] - 1,
                     fontweight="bold",
                     color="darkgreen" if ratio > 1 else "darkred")

    ax.set_xlabel("几何分布")
    ax.set_ylabel(r"$\sigma_{M_y}$ (predicted)")
    ax.set_title("(b) $M_y$ 维度 σ 在 OOD 上的演变", weight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=cfg["legend.fontsize"] - 1)
    ax.grid(True, alpha=0.3)

    # 总标题
    fig.suptitle(
        "异方差 σ 在分布外几何上的失效现象 — LIFT 不能单独防 OOD\n"
        "MoE/MDN 的 σ 不放大反而缩小 → 必须配合 CP_BOUNDS 硬约束 + Mahalanobis 距离",
        fontsize=cfg["axes.titlesize"] + 1, y=1.02, weight="bold")
    fig.tight_layout()

    out_path = out_dir / f"ood_sigma_failure_{style_mode}.png"
    fig.savefig(out_path, dpi=cfg["dpi"], bbox_inches="tight")
    print(f"[plot] {out_path}  (dpi={cfg['dpi']})")
    plt.close(fig)


if __name__ == "__main__":
    main()
