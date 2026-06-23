"""超参数扫描热力图 — 针对最终方案 MLP+LIFT (CP 11d) 的 16 trials

读取 sweep_results/mlp_lift_cp11/results.csv,展示 4 个 (hidden, depth) × (lr, wd) 热图:
    (a) MAE                — 越低越绿
    (b) best_test_loss     — 越低 (越负) 越绿
    (c) R²_My              — 越高越绿
    (d) best_epoch         — 收敛速度

用法:
    python hyperparam_heatmap.py --style paper|ppt|a4 --csv path/to/results.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import apply_style, parse_style_arg, CMAP_RYG


DEFAULT_CSV = Path(__file__).parent.parent.parent / "graduation" / "LFM" / \
              "sweep_results" / "mlp_lift_cp11" / "results.csv"


def panel_heatmap(ax, df, x_col, y_col, val_col, fmt="{:.4f}",
                   lower_better=True, title="", cmap=None, cbar_label=""):
    """单个 (x, y) heatmap, 取 mean(val) 跨其余维度聚合。"""
    if cmap is None:
        cmap = CMAP_RYG if not lower_better else CMAP_RYG.reversed()

    piv = df.pivot_table(index=y_col, columns=x_col, values=val_col, aggfunc="mean")
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto")

    # 数值标注
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            # 决定字体颜色 (背景深浅自适应)
            ax.text(j, i, fmt.format(v),
                     ha="center", va="center",
                     fontsize=plt.rcParams["axes.labelsize"] - 1,
                     color="black", weight="bold")

    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([str(r) for r in piv.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, weight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=plt.rcParams["axes.labelsize"] - 1)
    return piv


def main():
    style_mode = parse_style_arg(sys.argv)
    cfg = apply_style(style_mode)

    csv_path = None
    if "--csv" in sys.argv:
        idx = sys.argv.index("--csv")
        if idx + 1 < len(sys.argv):
            csv_path = Path(sys.argv[idx + 1])
    if csv_path is None:
        csv_path = DEFAULT_CSV

    if not csv_path.exists():
        # fallback: 尝试本地 LFM repo
        alt = Path("/Users/guoyue/gy_2026/graduation/LFM/sweep_results/mlp_lift_cp11/results.csv")
        if alt.exists():
            csv_path = alt
        else:
            print(f"[error] {csv_path} not found")
            sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[data] loaded {len(df)} trials from {csv_path}")
    print(df.columns.tolist())

    out_dir = Path(__file__).parent
    fig, axes = plt.subplots(2, 2, figsize=cfg["figsize"], dpi=cfg["dpi"])

    # (a) hidden_dim × depth → MAE (主性能指标)
    panel_heatmap(axes[0, 0], df, "depth", "hidden_dim", "final_mae",
                   fmt="{:.4f}", lower_better=True,
                   title="(a) MAE ↓  vs  (depth, hidden_dim)",
                   cbar_label="test MAE")

    # (b) lr × weight_decay → best_test_loss
    panel_heatmap(axes[0, 1], df, "lr", "weight_decay", "best_test_loss",
                   fmt="{:.2f}", lower_better=True,
                   title="(b) best test loss ↓  vs  (lr, weight_decay)",
                   cbar_label="NLL")

    # (c) hidden_dim × lr → R²_My
    panel_heatmap(axes[1, 0], df, "lr", "hidden_dim", "r2_My",
                   fmt="{:.4f}", lower_better=False,
                   title="(c) $R^2_{M_y}$ ↑  vs  (lr, hidden_dim)",
                   cbar_label="$R^2_{M_y}$")

    # (d) depth × dropout → best_epoch (收敛速度)
    if "dropout" in df.columns and df["dropout"].nunique() > 1:
        panel_heatmap(axes[1, 1], df, "dropout", "depth", "best_epoch",
                       fmt="{:.0f}", lower_better=True,
                       title="(d) best epoch ↓  vs  (dropout, depth)",
                       cbar_label="epoch")
    else:
        # 退化:  weight_decay × hidden_dim → time_s
        panel_heatmap(axes[1, 1], df, "weight_decay", "hidden_dim", "time_s",
                       fmt="{:.0f}", lower_better=True,
                       title="(d) wall time (s) ↓  vs  (wd, hidden)",
                       cbar_label="seconds")

    fig.suptitle(
        f"MLP+LIFT (CP 11d) 超参数扫描热力图 — {len(df)} trials × 1500 epoch / CPU\n"
        f"best: MAE={df['final_mae'].min():.4f}, "
        f"R²_avg={df[[c for c in df.columns if c.startswith('r2_')]].mean(axis=1).max():.4f}",
        fontsize=cfg["axes.titlesize"] + 1, y=1.02, weight="bold"
    )
    fig.tight_layout()

    out_path = out_dir / f"hyperparam_heatmap_{style_mode}.png"
    fig.savefig(out_path, dpi=cfg["dpi"], bbox_inches="tight")
    print(f"[plot] {out_path}  (dpi={cfg['dpi']})")
    plt.close(fig)


if __name__ == "__main__":
    main()
