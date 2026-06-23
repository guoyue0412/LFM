"""OOD (out-of-distribution) 几何测试 — 验证 LIFT 不确定度是否真的放大。

设计思路:
    1. 从训练数据反推每个 8-CP 控制点的实际分布范围 [cp_min, cp_max]。
    2. 三组测试集:
        - in_dist  : 在 [cp_min, cp_max] 内随机采样几何 (应该 σ 小)
        - ood_mild : 在 [cp_min - 0.5·R, cp_max + 0.5·R] 但避开 in_dist 区间 (R=cp_max-cp_min)
        - ood_far  : 在 [cp_min - 1.5·R, cp_max + 1.5·R] 但避开前两段
    3. 每个几何用 B-Spline 展开为 22 chord + 22 twist sections,加上典型工况
       (RPM=5250, WIND=10, ANGLE=85) 输入模型,对比三组的 σ 分布。
    4. 期望: σ_ood 显著大于 σ_in_dist,体现 "模型知道自己不会"。

用法:
    python -m optimization_v2.tools.ood_test \
        --sweeps mlp_lift_v237 moe_lift_v237 mdn_moe_v237 \
        --data ./data_for_train/processed_lhs/dataset_v2_20260617.csv \
        --normalizer ./data_for_train/processed_lhs/norm/normalizer.pkl \
        --out ./sweep_results/ood_test.png \
        --n-per-group 200

输出:
    ood_test.png — 三方法 × 4 输出维度 × 3 OOD 等级 的 σ 箱型图
    ood_table.csv — σ 统计 + 信号比 (σ_ood / σ_in_dist)
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import BSpline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_v2.tools.sweep_moe import (
    HeteroscedasticMLP, MoEPredictor,
)
# pickle 反序列化 normalizer.pkl 需要 HybridNormalizer 类在当前 module 路径下
from optimization_v2.tools.analyze_and_normalize import HybridNormalizer, ColumnStats  # noqa: F401


# =========================================================
# B-Spline 展开 (与 optimization_v2/geometry.py 一致)
# =========================================================

N_SECTIONS = 22
# 22 个径向站位 (与训练数据采样一致)
RADII = np.array([
    0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46,
    0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86,
    0.90, 0.94,
]) * 0.127  # R_prop = 0.127m

# B-Spline knot 向量 (4 CP + 3 阶)
_KNOTS_CHORD = np.array([0.10, 0.10, 0.10, 0.10, 0.94, 0.94, 0.94, 0.94]) * 0.127
_KNOTS_TWIST = _KNOTS_CHORD.copy()


def cp_to_sections(cp8):
    """8 CP (4 chord + 4 twist) → 22 chord + 22 twist sections。"""
    cp_chord = cp8[:4]
    cp_twist = cp8[4:]
    # 4 CP + 3 阶 = 8 knots (clamped 形式), 简化用 cubic spline 插值
    # 这里用 simpler scipy 实现避免依赖项目内部模块
    from scipy.interpolate import CubicSpline
    cs_chord = CubicSpline(np.linspace(RADII[0], RADII[-1], 4), cp_chord)
    cs_twist = CubicSpline(np.linspace(RADII[0], RADII[-1], 4), cp_twist)
    return cs_chord(RADII), cs_twist(RADII)


# =========================================================
# CP 范围 (从训练数据反推, 与 config.OptConfig.cp_bounds 一致)
# =========================================================

CP_BOUNDS = np.array([
    [0.013599, 0.026293],   # chord cp 0
    [0.022239, 0.043821],
    [0.010052, 0.019808],
    [0.004783, 0.009199],
    [37.4, 72.7],           # twist cp 0 (deg)
    [13.4, 26.5],
    [11.3, 22.1],
    [8.4, 16.6],
])


def sample_cp(n, kind, seed=0):
    """采样 n 个 8-CP 几何。

    kind:
        in_dist  — 完全在 bounds 内 (训练分布)
        ood_mild — 在 bounds 外 0.0~0.5·R, 任意维度 (中等外推)
        ood_far  — 在 bounds 外 0.5~1.5·R, 任意维度 (强外推)
    """
    rng = np.random.RandomState(seed)
    R = CP_BOUNDS[:, 1] - CP_BOUNDS[:, 0]
    out = np.zeros((n, 8))
    for i in range(n):
        if kind == "in_dist":
            out[i] = CP_BOUNDS[:, 0] + rng.rand(8) * R
        elif kind == "ood_mild":
            # 每个维度随机选 below/above, 偏移 0~0.5·R
            for d in range(8):
                side = rng.choice([-1, 1])
                offset = rng.rand() * 0.5 * R[d]
                if side > 0:
                    out[i, d] = CP_BOUNDS[d, 1] + offset
                else:
                    out[i, d] = CP_BOUNDS[d, 0] - offset
        elif kind == "ood_far":
            for d in range(8):
                side = rng.choice([-1, 1])
                offset = 0.5 * R[d] + rng.rand() * 1.0 * R[d]
                if side > 0:
                    out[i, d] = CP_BOUNDS[d, 1] + offset
                else:
                    out[i, d] = CP_BOUNDS[d, 0] - offset
        else:
            raise ValueError(kind)
    return out


# =========================================================
# 构造 47 维输入 + 归一化
# =========================================================

INPUT_COLS = (["RPM", "WIND", "ANGLE"]
              + [f"chord_{i}" for i in range(22)]
              + [f"twist_{i}" for i in range(22)])
OUTPUT_COLS = ["T", "H", "My", "Q"]


def cp_grid_to_inputs(cp_array, rpms=[5000, 5500, 6000], angles=[83, 85, 87], wind=10.0):
    """对 (N, 8) CP 数组,在多工况下展开为 (N·K, 47) 输入表 (raw, 未归一化)。"""
    rows = []
    for cp in cp_array:
        chord, twist = cp_to_sections(cp)
        for rpm in rpms:
            for ang in angles:
                rows.append([rpm, wind, ang, *chord, *twist])
    return pd.DataFrame(rows, columns=INPUT_COLS)


def normalize(df, normalizer):
    """用 HybridNormalizer 归一化输入列。

    normalizer.columns 包含 51 列(47 输入 + 4 输出),我们只关心输入列。
    补 4 个输出列占位 0,然后挑出输入列的归一化结果。
    """
    df2 = df.copy()
    for c in OUTPUT_COLS:
        if c not in df2.columns:
            df2[c] = 0.0
    df2 = df2[normalizer.columns]
    arr = normalizer.transform(df2)
    # 取与 INPUT_COLS 顺序对齐的列索引
    input_idx = [normalizer.columns.index(c) for c in INPUT_COLS if c in normalizer.columns]
    return arr[:, input_idx]


# =========================================================
# 模型加载 (复用 compare_heatmap)
# =========================================================

def load_model(sweep_dir: Path, X_dim=47, Y_dim=4):
    cfg = json.load(open(sweep_dir / "best_trial" / "config.json"))
    state = torch.load(sweep_dir / "best_trial" / "model.pth",
                       map_location="cpu", weights_only=True)
    name = sweep_dir.name
    arch = "mlp" if "mlp" in name else "moe"
    loss = "mdn" if "mdn" in name else "lift"

    if arch == "mlp":
        model = HeteroscedasticMLP(X_dim, Y_dim,
                                    hidden_dim=cfg["hidden_dim"],
                                    depth=cfg["depth"],
                                    dropout=cfg.get("dropout", 0.0))
    else:
        model = MoEPredictor(X_dim, Y_dim,
                              n_experts=cfg["n_experts"],
                              shared_dim=cfg["shared_dim"],
                              hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(state)
    model.eval()
    return model, arch, loss


@torch.no_grad()
def predict_sigma(model, X_norm):
    """返回 (N, 4) σ 数组。"""
    X_t = torch.tensor(X_norm).float()
    mu, sig, *_ = model(X_t)
    return sig.numpy()


# =========================================================
# 主流程
# =========================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", nargs="+", required=True)
    p.add_argument("--sweep-root", default="./sweep_results")
    p.add_argument("--normalizer", required=True,
                   help="normalizer.pkl path")
    p.add_argument("--out", default="./sweep_results/ood_test.png")
    p.add_argument("--n-per-group", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # 加载归一化器
    bundle = pickle.load(open(args.normalizer, "rb"))
    normalizer = bundle["normalizer"]
    print(f"[norm] loaded, columns={len(normalizer.columns)}")

    # 三组几何
    print(f"[gen] sampling {args.n_per_group} per group × 3 groups...")
    cp_in   = sample_cp(args.n_per_group, "in_dist", seed=args.seed)
    cp_mild = sample_cp(args.n_per_group, "ood_mild", seed=args.seed + 1)
    cp_far  = sample_cp(args.n_per_group, "ood_far", seed=args.seed + 2)

    # 每组展开为多工况输入
    df_in   = cp_grid_to_inputs(cp_in)
    df_mild = cp_grid_to_inputs(cp_mild)
    df_far  = cp_grid_to_inputs(cp_far)
    print(f"[gen] in_dist:{len(df_in)} mild:{len(df_mild)} far:{len(df_far)} 行")

    # 归一化
    X_in   = normalize(df_in,   normalizer)
    X_mild = normalize(df_mild, normalizer)
    X_far  = normalize(df_far,  normalizer)

    # 评估三模型
    sweep_root = Path(args.sweep_root)
    method_results = {}
    for sw in args.sweeps:
        sw_dir = sweep_root / sw
        model, arch, loss = load_model(sw_dir)
        method = f"{arch.upper()}+{loss.upper()}"
        sig_in   = predict_sigma(model, X_in)
        sig_mild = predict_sigma(model, X_mild)
        sig_far  = predict_sigma(model, X_far)
        method_results[method] = {
            "in":   sig_in,
            "mild": sig_mild,
            "far":  sig_far,
        }
        # 信号比
        for j, oc in enumerate(OUTPUT_COLS):
            r_mild = sig_mild[:, j].mean() / (sig_in[:, j].mean() + 1e-12)
            r_far  = sig_far[:, j].mean()  / (sig_in[:, j].mean() + 1e-12)
            print(f"  [{method}] σ_{oc}: in={sig_in[:,j].mean():.4f}, "
                  f"mild={sig_mild[:,j].mean():.4f} (x{r_mild:.2f}), "
                  f"far={sig_far[:,j].mean():.4f} (x{r_far:.2f})")

    # ============== 绘图 ==============
    M = len(method_results)
    D = len(OUTPUT_COLS)
    fig, axes = plt.subplots(M, D, figsize=(4 * D, 3.5 * M), dpi=120, sharey="row")
    if M == 1:
        axes = axes[None, :]

    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green=in, orange=mild, red=far
    labels = ["in-dist", "ood-mild", "ood-far"]
    groups_order = ["in", "mild", "far"]

    rows_csv = []

    for i, (method, sig_dict) in enumerate(method_results.items()):
        for j, oc in enumerate(OUTPUT_COLS):
            ax = axes[i, j]
            data = [sig_dict[g][:, j] for g in groups_order]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                             tick_labels=labels, showfliers=False)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.set_title(f"{method} — σ_{oc}", fontsize=10)
            ax.tick_params(axis='x', labelsize=8, rotation=20)
            ax.tick_params(axis='y', labelsize=8)
            if j == 0:
                ax.set_ylabel("predicted σ")

            means = [d.mean() for d in data]
            r_mild = means[1] / (means[0] + 1e-12)
            r_far  = means[2] / (means[0] + 1e-12)
            ax.text(0.5, 0.95,
                     f"σ_mild/σ_in = {r_mild:.2f}\nσ_far/σ_in = {r_far:.2f}",
                     transform=ax.transAxes, ha="center", va="top",
                     fontsize=7.5, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

            rows_csv.append({
                "method": method, "output": oc,
                "sigma_in_mean":   means[0],
                "sigma_mild_mean": means[1],
                "sigma_far_mean":  means[2],
                "ratio_mild_over_in": r_mild,
                "ratio_far_over_in":  r_far,
            })

    fig.suptitle(
        "OOD test — does σ grow on out-of-distribution geometries?\n"
        f"({args.n_per_group}×3 groups × 9 conditions each, σ aggregated over conditions)",
        fontsize=12, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[plot] {args.out}")

    csv_path = Path(args.out).with_suffix(".csv")
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
    print(f"[csv]  {csv_path}")


if __name__ == "__main__":
    main()
