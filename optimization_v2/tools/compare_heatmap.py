"""方法对比热图 (类似论文 FD-PSNR / Policy-MSE 风格)

模仿用户提供的 cross-trained 热图风格,把三方法 × 多指标的性能矩阵
可视化为热力图。

用法:
    python -m optimization_v2.tools.compare_heatmap \
        --sweeps mlp_lift_v237 moe_lift_v237 mdn_moe_v237 \
        --data ./data_for_train/processed_lhs/norm/data_normalized.npz \
        --geom-idx-csv ./data_for_train/processed_lhs/dataset_v2_20260617.csv \
        --out ./sweep_results/comparison.png

输出:
    每个方法对每个测试集 (T/H/M_y/Q + 总体) 的:
        R² (越大越好,绿色)
        MAE (越小越好,红→绿)
        ECE (越小越好,红→绿)
        相对 best 的 Δ
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# 从 sweep_moe.py 复用模型定义
from optimization_v2.tools.sweep_moe import (
    HeteroscedasticMLP, MoEPredictor, TrialConfig,
    load_normalized, split_by_geometry,
    CONDITION_COLS, SECTION_COLS, OUTPUT_COLS,
)


def load_model_from_sweep(sweep_dir: Path, X_dim: int = 47, Y_dim: int = 4):
    """从 sweep best_trial 目录加载模型 + 配置。"""
    cfg = json.load(open(sweep_dir / "best_trial" / "config.json"))
    state = torch.load(sweep_dir / "best_trial" / "model.pth",
                       map_location="cpu", weights_only=True)

    # 通过 sweep 目录名推断 arch + loss
    name = sweep_dir.name
    arch = "mlp" if "mlp" in name else "moe"
    loss = "mdn" if "mdn" in name else "lift"

    if arch == "mlp":
        model = HeteroscedasticMLP(X_dim, Y_dim,
                                    hidden_dim=cfg["hidden_dim"],
                                    depth=cfg["depth"])
    else:
        model = MoEPredictor(X_dim, Y_dim,
                              n_experts=cfg["n_experts"],
                              shared_dim=cfg["shared_dim"],
                              hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(state)
    model.eval()
    return model, cfg, arch, loss


@torch.no_grad()
def evaluate_one(model, X_te, Y_te):
    """返回 (mu, sigma) 数组及标量指标。"""
    X_t = torch.tensor(X_te).float()
    Y_t = torch.tensor(Y_te).float()
    mu, sig, *_ = model(X_t)
    mu = mu.numpy(); sig = sig.numpy()
    err = mu - Y_te

    # per-col 指标
    mae_j = np.abs(err).mean(axis=0)
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((Y_te - Y_te.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    r2_j = 1 - ss_res / ss_tot

    # ECE per col (10-bin reliability)
    ece_j = []
    for j in range(Y_te.shape[1]):
        order = np.argsort(sig[:, j])
        bins = np.array_split(order, 10)
        sig_bin = np.array([sig[b, j].mean() for b in bins])
        rmse_bin = np.array([np.sqrt(((mu[b, j] - Y_te[b, j]) ** 2).mean()) for b in bins])
        ece = float(np.mean(np.abs(rmse_bin - sig_bin)) / (sig[:, j].mean() + 1e-8))
        ece_j.append(ece)
    ece_j = np.array(ece_j)

    return {
        "mae_total": float(np.abs(err).mean()),
        "mae_per_col": mae_j,
        "r2_per_col": r2_j,
        "r2_total": float(np.mean(r2_j)),
        "ece_per_col": ece_j,
        "ece_total": float(np.mean(ece_j)),
        "sigma_per_col": sig.mean(axis=0),
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def annotated_heatmap(ax, values, row_labels, col_labels,
                       fmt="{:.4f}", cmap="RdYlGn", lower_better=True,
                       vmin=None, vmax=None, title="", show_delta=True):
    """单矩阵热图,带数值标注 + 与列最优的 Δ。

    lower_better=True 时,列最小值是 best(绿色);
    lower_better=False 时,列最大值是 best(绿色)。
    """
    # 标准化:每列独立归一,把 best 朝绿色
    norm_vals = np.zeros_like(values, dtype=float)
    for j in range(values.shape[1]):
        col = values[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin < 1e-12:
            norm_vals[:, j] = 0.5
        elif lower_better:
            norm_vals[:, j] = 1.0 - (col - cmin) / (cmax - cmin)  # 0~1, 越小越绿
        else:
            norm_vals[:, j] = (col - cmin) / (cmax - cmin)         # 0~1, 越大越绿

    im = ax.imshow(norm_vals, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=11)

    # 标注每格的数值 + Δ
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            col = values[:, j]
            best = col.min() if lower_better else col.max()
            delta = v - best
            text = fmt.format(v)
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8.5, weight="bold" if delta == 0 else "normal",
                    color="black")
            if show_delta and delta != 0:
                d_fmt = ("Δ+" if delta > 0 else "Δ") + fmt.format(delta)
                ax.text(j, i + 0.32, d_fmt, ha="center", va="center",
                        fontsize=6.5, color="dimgray")
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", nargs="+", required=True,
                   help="sweep 子目录名列表 (位于 ./sweep_results/), 第一个 = MLP, 其余 MoE")
    p.add_argument("--sweep-root", default="./sweep_results")
    p.add_argument("--data", required=True)
    p.add_argument("--geom-idx-csv", required=True)
    p.add_argument("--out", default="./sweep_results/comparison.png")
    args = p.parse_args()

    # 加载测试集 (与 sweep 用相同的几何分割)
    X, Y, in_cols, out_cols = load_normalized(Path(args.data))
    gi = pd.read_csv(args.geom_idx_csv)["geom_idx"].to_numpy()
    if len(gi) != len(X):
        gi = np.arange(len(X)) % max(1, len(X) // 50)
    _, _, X_te, Y_te = split_by_geometry(X, Y, gi)
    print(f"[data] test set: {X_te.shape}")

    # 评估每个方法
    sweep_root = Path(args.sweep_root)
    results = []
    for sw in args.sweeps:
        sw_dir = sweep_root / sw
        model, cfg, arch, loss = load_model_from_sweep(sw_dir)
        m = evaluate_one(model, X_te, Y_te)
        n_params = count_params(model)
        time_s = cfg.get("time_s", 0)
        results.append({
            "method": f"{arch.upper()}+{loss.upper()}",
            "sweep_dir": sw,
            "metrics": m,
            "params": n_params / 1e3,   # K
            "time_min": time_s / 60.0,
            "config": cfg,
        })
        print(f"[{arch}+{loss}] MAE={m['mae_total']:.4f} R²_avg={m['r2_total']:.4f} "
              f"ECE_avg={m['ece_total']:.3f} params={n_params/1e3:.0f}K time={time_s/60:.1f}min")

    method_names = [r["method"] for r in results]
    M = len(method_names)
    D = len(OUTPUT_COLS)

    # ============== 构造 4 个矩阵 ==============
    r2_mat = np.array([r["metrics"]["r2_per_col"] for r in results])             # (M, D)
    mae_mat = np.array([r["metrics"]["mae_per_col"] for r in results])
    ece_mat = np.array([r["metrics"]["ece_per_col"] for r in results])
    cmp_mat = np.array([
        [r["metrics"]["r2_total"], r["metrics"]["mae_total"],
         r["metrics"]["ece_total"], r["params"], r["time_min"]]
        for r in results
    ])  # (M, 5)
    cmp_cols = ["R² avg", "MAE", "ECE avg", "Params (K)", "Time (min)"]
    cmp_lower_better = [False, True, True, True, True]

    # ============== 画图 ==============
    fig = plt.figure(figsize=(16, 11), dpi=130)
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.30,
                           height_ratios=[1, 1, 1.1])

    # (a) R² per col (越大越好)
    ax = fig.add_subplot(gs[0, 0])
    annotated_heatmap(ax, r2_mat, method_names, OUTPUT_COLS,
                       fmt="{:.4f}", lower_better=False,
                       title="R² ↑  per output column")
    ax.set_xlabel("Output dim"); ax.set_ylabel("Method")

    # (b) MAE per col (越小越好)
    ax = fig.add_subplot(gs[0, 1])
    annotated_heatmap(ax, mae_mat, method_names, OUTPUT_COLS,
                       fmt="{:.4f}", lower_better=True,
                       title="MAE ↓  per output column")
    ax.set_xlabel("Output dim"); ax.set_ylabel("Method")

    # (c) ECE per col (越小越好)
    ax = fig.add_subplot(gs[1, 0])
    annotated_heatmap(ax, ece_mat, method_names, OUTPUT_COLS,
                       fmt="{:.3f}", lower_better=True,
                       title="ECE ↓  σ-calibration per output column")
    ax.set_xlabel("Output dim"); ax.set_ylabel("Method")

    # (d) σ 平均尺度
    ax = fig.add_subplot(gs[1, 1])
    sigma_mat = np.array([r["metrics"]["sigma_per_col"] for r in results])
    annotated_heatmap(ax, sigma_mat, method_names, OUTPUT_COLS,
                       fmt="{:.3f}", lower_better=False,  # 不评好坏,仅展示
                       title="Predicted σ  (mean over test set)")
    ax.set_xlabel("Output dim"); ax.set_ylabel("Method")

    # (e) 综合对比 (含模型复杂度)
    ax = fig.add_subplot(gs[2, :])
    # 每列分别 lower_better/upper_better — 自己写循环
    norm_vals = np.zeros_like(cmp_mat, dtype=float)
    for j, lb in enumerate(cmp_lower_better):
        col = cmp_mat[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin < 1e-12:
            norm_vals[:, j] = 0.5
        elif lb:
            norm_vals[:, j] = 1.0 - (col - cmin) / (cmax - cmin)
        else:
            norm_vals[:, j] = (col - cmin) / (cmax - cmin)
    im = ax.imshow(norm_vals, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cmp_cols)))
    ax.set_xticklabels(cmp_cols, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(M))
    ax.set_yticklabels(method_names, fontsize=10)
    ax.set_title("Overall — accuracy / calibration / complexity", fontsize=11)
    for i in range(M):
        for j in range(cmp_mat.shape[1]):
            v = cmp_mat[i, j]
            col = cmp_mat[:, j]
            best = col.min() if cmp_lower_better[j] else col.max()
            delta = v - best
            if j == 3:   # params
                text = f"{v:.0f} K"
            elif j == 4: # time
                text = f"{v:.1f} min"
            else:
                text = f"{v:.4f}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10, weight="bold" if delta == 0 else "normal")

    fig.suptitle("Method comparison — MLP+LIFT vs MoE+LIFT vs MoE+MDN\n"
                 "(green = best per column,Δ = gap to best)",
                 fontsize=12, y=0.995)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[plot] {args.out}")


if __name__ == "__main__":
    main()
