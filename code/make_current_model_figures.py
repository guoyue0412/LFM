#!/usr/bin/env python3
"""Generate current thesis figures for model comparison and MoE sweep status."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch

from optimization_v2.tools.sweep_moe import load_normalized, split_by_geometry, MoEPredictor


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT.parent / "mid" / "hithesis-master" / "examples" / "hitart" / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#4E79A7"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
RED = "#E15759"
PURPLE = "#B07AA1"
GRAY = "#6E6E6E"
LIGHT = "#F7F7F7"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def normalize_curve(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return y
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def load_best_history(results_csv: Path, histories_npz: Path, metric_col: str = "best_test_loss") -> np.ndarray | None:
    if not results_csv.exists() or not histories_npz.exists():
        return None
    df = pd.read_csv(results_csv)
    if df.empty or metric_col not in df.columns:
        return None
    trial = int(df.sort_values(metric_col).iloc[0]["trial"])
    z = np.load(histories_npz, allow_pickle=True)
    key = f"trial_{trial}"
    if key not in z.files:
        return None
    arr = np.asarray(z[key], dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    return arr[1]


def readable_cell_text(ax, x: int, y: int, text: str, fontsize: int = 8) -> None:
    t = ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.84),
    )
    t.set_path_effects([pe.withStroke(linewidth=0.35, foreground="white")])


def plot_training_curve_comparison() -> Path:
    base = ROOT / "remote_3660_merged_latest_20260628_1425"
    dnn_csv = base / "current_snapshot_baselines" / "dnn_mse_history.csv"
    mlp_csv = base / "current_snapshot_baselines" / "mlp_lift_history.csv"
    moe_results = base / "results_partial.csv"
    moe_hist = base / "histories_partial.npz"

    curves: list[tuple[str, np.ndarray, str]] = []
    if dnn_csv.exists():
        dnn = pd.read_csv(dnn_csv)["test_loss"].to_numpy(dtype=float)
        curves.append(("DNN (MSE)", dnn, GRAY))
    if mlp_csv.exists():
        mlp = pd.read_csv(mlp_csv)["test_loss"].to_numpy(dtype=float)
        curves.append(("MLP + LIFT-NLL", mlp, ORANGE))
    moe = load_best_history(moe_results, moe_hist)
    if moe is not None:
        curves.append(("MoE + LIFT-NLL", moe, BLUE))

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    for label, y, color in curves:
        yn = normalize_curve(y)
        if yn.size > 900:
            idx = np.linspace(0, yn.size - 1, 900).astype(int)
            x = idx + 1
            yn = yn[idx]
        else:
            x = np.arange(1, yn.size + 1)
        ax.plot(x, yn, label=label, color=color, linewidth=2.0)

    ax.set_title("Architecture comparison: normalized validation objective", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized validation objective\n(lower is better)", fontsize=9)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.01,
        -0.25,
        "All curves use the current 508-geometry data. DNN uses MSE; LIFT models use NLL, so values are normalized for trend comparison.",
        transform=ax.transAxes,
        fontsize=8,
        color=GRAY,
        va="top",
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.27)
    out = FIG_DIR / "current_model_training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def current_test_split() -> tuple[np.ndarray, np.ndarray, list[str]]:
    base = ROOT / "remote_3660_merged_latest_20260628_1425"
    data = base / "processed_lhs_merged_latest_20260628_1425" / "norm" / "data_normalized.npz"
    csv = base / "processed_lhs_merged_latest_20260628_1425" / "dataset_merged_latest_20260628_1425.csv"
    X, Y, _, out_cols = load_normalized(data, use_cp=False)
    gi = pd.read_csv(csv)["geom_idx"].to_numpy()
    _, _, X_te, Y_te = split_by_geometry(X, Y, gi, seed=606281425)
    return X_te, Y_te, out_cols


def evaluate_current_moe_predictions() -> dict:
    base = ROOT / "remote_3660_merged_latest_20260628_1425"
    data = base / "processed_lhs_merged_latest_20260628_1425" / "norm" / "data_normalized.npz"
    model_dir = base / "best_so_far"
    cfg = pd.read_json(model_dir / "config.json", typ="series").to_dict()
    X, Y, _, out_cols = load_normalized(data, use_cp=False)
    X_te, Y_te, _ = current_test_split()
    model = MoEPredictor(
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        n_experts=int(cfg["n_experts"]),
        shared_dim=int(cfg["shared_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
    )
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        mu, sig, *_ = model(torch.tensor(X_te))
    pred = mu.numpy()
    sigma = sig.numpy()
    err = np.abs(pred - Y_te)
    ss_res = ((pred - Y_te) ** 2).sum(axis=0)
    ss_tot = ((Y_te - Y_te.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {
        "pred": pred,
        "sigma": sigma,
        "Y_te": Y_te,
        "r2": r2,
        "out_cols": out_cols,
        "cfg": cfg,
        "err": err,
    }


def evaluate_current_moe() -> dict:
    ev = evaluate_current_moe_predictions()
    err = ev["err"]
    sigma = ev["sigma"]
    r2 = ev["r2"]
    return {
        "model": "MoE+LIFT-NLL",
        "final_mae": float(err.mean()),
        "r2_mean": float(r2.mean()),
        "r2_My": float(r2[2]),
        "sigma_error_corr": float(np.corrcoef(err.mean(axis=1), sigma.mean(axis=1))[0, 1]),
        "best_test_loss": float(ev["cfg"]["best_test_loss"]),
    }


def load_current_model_metrics() -> pd.DataFrame:
    base = ROOT / "remote_3660_merged_latest_20260628_1425"
    rows = []
    baseline = base / "current_snapshot_baselines" / "baseline_metrics.csv"
    if baseline.exists():
        bdf = pd.read_csv(baseline)
        for r in bdf.to_dict("records"):
            r2 = np.array(eval(r["r2"]) if isinstance(r["r2"], str) else r["r2"], dtype=float)
            rows.append(
                {
                    "model": r["model"],
                    "final_mae": float(r["final_mae"]),
                    "r2_mean": float(r2.mean()),
                    "r2_My": float(r2[2]),
                    "sigma_error_corr": np.nan if pd.isna(r.get("sigma_error_corr")) else float(r["sigma_error_corr"]),
                    "best_test_loss": float(r["best_test_loss"]),
                }
            )
    rows.append(evaluate_current_moe())
    return pd.DataFrame(rows)


def plot_current_model_metric_heatmap() -> Path:
    df = load_current_model_metrics()
    raw = df.set_index("model")[["final_mae", "r2_mean", "r2_My", "sigma_error_corr"]]
    score = raw.copy()
    score["final_mae"] = 1.0 - normalize_curve(raw["final_mae"].to_numpy())
    for col in ["r2_mean", "r2_My", "sigma_error_corr"]:
        vals = raw[col].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        norm = np.zeros_like(vals, dtype=float)
        if finite.any():
            norm[finite] = normalize_curve(vals[finite])
        score[col] = norm
    labels = ["MAE score\n(lower MAE better)", "Mean $R^2$", "$R^2(M_y)$", "Uncertainty\nerror corr."]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    im = ax.imshow(score.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(score.index)))
    ax.set_yticklabels(score.index)
    ax.set_title("Current 508-geometry model comparison: accuracy and uncertainty")
    for i in range(raw.shape[0]):
        for j, col in enumerate(raw.columns):
            v = raw.iloc[i, j]
            txt = "N/A" if not np.isfinite(v) else f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
            readable_cell_text(ax, j, i, txt, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="normalized score")
    fig.tight_layout()
    out = FIG_DIR / "current_model_metric_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(ROOT / "remote_3660_merged_latest_20260628_1425" / "current_model_metric_summary.csv", index=False)
    return out


def plot_current_moe_sweep_dotplot() -> Path:
    csv = ROOT / "remote_3660_merged_latest_20260628_1425" / "results_partial.csv"
    df = pd.read_csv(csv).copy().sort_values("best_test_loss", ascending=False).reset_index(drop=True)
    df["score"] = -df["best_test_loss"]
    mae = df["final_mae"].to_numpy(dtype=float)
    mae_score = 1.0 - normalize_curve(mae)
    sizes = 90 + 420 * mae_score
    y = np.arange(len(df))

    fig_h = max(4.2, 0.35 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    sc = ax.scatter(
        df["score"],
        y,
        s=sizes,
        c=df["score"],
        cmap="RdYlGn",
        edgecolors="#222222",
        linewidths=0.7,
        alpha=0.92,
        zorder=3,
    )
    xmax = float(df["score"].max())
    xmin = float(df["score"].min())
    span = max(xmax - xmin, 0.1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"T{int(t)}" for t in df["trial"]])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel("- best validation NLL (higher is better)")
    ax.set_title("Current MoE sweep dot plot (completed trials)")
    ax.grid(True, axis="x", color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.set_xlim(xmin - 0.08 * span, xmax + 0.16 * span)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("- best validation NLL")

    handles = [
        ax.scatter([], [], s=90, color="#BBBBBB", edgecolors="#222222", label="higher MAE"),
        ax.scatter([], [], s=510, color="#BBBBBB", edgecolors="#222222", label="lower MAE"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", title="point size")
    fig.subplots_adjust(left=0.10, right=0.86, top=0.90, bottom=0.12)
    out = FIG_DIR / "current_moe_sweep_dotplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def load_r2_by_output() -> pd.DataFrame:
    base = ROOT / "remote_3660_merged_latest_20260628_1425"
    rows = []
    baseline = base / "current_snapshot_baselines" / "baseline_metrics.csv"
    out_cols = ["T", "H", "My", "Q"]
    if baseline.exists():
        bdf = pd.read_csv(baseline)
        for r in bdf.to_dict("records"):
            vals = np.array(eval(r["r2"]) if isinstance(r["r2"], str) else r["r2"], dtype=float)
            for col, val in zip(out_cols, vals):
                rows.append({"model": r["model"], "output": col, "r2": float(val)})
    ev = evaluate_current_moe_predictions()
    for col, val in zip(out_cols, ev["r2"]):
        rows.append({"model": "MoE+LIFT-NLL", "output": col, "r2": float(val)})
    return pd.DataFrame(rows)


def plot_current_prediction_fit_2d() -> Path:
    ev = evaluate_current_moe_predictions()
    pred = ev["pred"]
    y = ev["Y_te"]
    r2 = ev["r2"]
    out_cols = ["T", "H", r"$M_y$", "Q"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for ax, i, name in zip(axes.ravel(), range(4), out_cols):
        ax.scatter(y[:, i], pred[:, i], s=8, alpha=0.35, color=GREEN, edgecolors="none", rasterized=True)
        lo = float(min(y[:, i].min(), pred[:, i].min()))
        hi = float(max(y[:, i].max(), pred[:, i].max()))
        pad = 0.03 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=RED, lw=1.2, ls="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("True value (normalized)")
        ax.set_ylabel("Predicted value (normalized)")
        ax.set_title(f"{name}: 2-D fit")
        ax.text(
            0.05,
            0.92,
            rf"$R^2={r2[i]:.4f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB", alpha=0.9),
        )
    fig.suptitle("MoE+LIFT-NLL prediction fit on held-out geometries", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "current_moe_prediction_fit_2d.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_current_r2_by_output() -> Path:
    df = load_r2_by_output()
    models = ["DNN-MSE", "MLP+LIFT-NLL", "MoE+LIFT-NLL"]
    outputs = ["T", "H", "My", "Q"]
    colors = [GRAY, ORANGE, GREEN]
    width = 0.23
    x = np.arange(len(outputs))

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, model in enumerate(models):
        vals = []
        for output in outputs:
            row = df[(df["model"] == model) & (df["output"] == output)]
            vals.append(float(row["r2"].iloc[0]) if not row.empty else np.nan)
        xpos = x + (k - 1) * width
        ax.bar(xpos, vals, width=width, label=model, color=colors[k], edgecolor="white", linewidth=0.5)
        for xi, v in zip(xpos, vals):
            if np.isfinite(v):
                t = ax.text(xi, v - 0.003, f"{v:.3f}", ha="center", va="top", fontsize=7, rotation=90, color="white")
                t.set_path_effects([pe.withStroke(linewidth=1.1, foreground="#333333")])

    ax.set_xticks(x)
    ax.set_xticklabels(["T", "H", r"$M_y$", "Q"])
    ax.set_ylabel(r"$R^2$ on held-out geometries")
    ax.set_ylim(0.88, 1.005)
    ax.set_title(r"Output-wise $R^2$ comparison on the current 508-geometry data", pad=12)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.25)
    out = FIG_DIR / "current_model_r2_by_output.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_box(ax, xy, wh, text, fc, ec="#333333", fontsize=10):
    box = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.03,rounding_size=0.035",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p1, p2, color="#333333"):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", lw=1.3, color=color, shrinkA=3, shrinkB=3))


def plot_input_dimension_comparison() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Final surrogate input design: 47-D full section vs 11-D control-point form", ha="center", fontsize=12, weight="bold")

    draw_box(ax, (0.05, 0.63), (0.28, 0.20), "47-D full section input\n3 conditions\n+ 22 chord samples\n+ 22 twist samples", "#E8EEF8", fontsize=9)
    draw_box(ax, (0.05, 0.22), (0.28, 0.20), "11-D compact input\n3 conditions\n+ 4 chord CP\n+ 4 twist CP", "#EAF4E6", fontsize=9)

    draw_box(ax, (0.42, 0.63), (0.22, 0.20), "High-resolution\nsection descriptor", "#F7F2E8", fontsize=9)
    draw_box(ax, (0.42, 0.22), (0.22, 0.20), "Optimization-native\nB-spline controls", "#F7F2E8", fontsize=9)

    draw_box(ax, (0.73, 0.43), (0.22, 0.22), "MoE + LIFT-NLL\nsurrogate\nT, H, My, Q + uncertainty", "#F4E6E6", fontsize=9)

    arrow(ax, (0.33, 0.73), (0.42, 0.73))
    arrow(ax, (0.33, 0.32), (0.42, 0.32))
    arrow(ax, (0.64, 0.73), (0.73, 0.57))
    arrow(ax, (0.64, 0.32), (0.73, 0.51))

    ax.text(0.19, 0.56, "Best for data-rich section learning", ha="center", fontsize=8, color=GRAY)
    ax.text(0.19, 0.15, "Best for CMA-ES geometry optimization", ha="center", fontsize=8, color=GRAY)
    ax.text(0.50, 0.08, "Current thesis route keeps the 11-D CP representation for design variables and reports 47-D as the full-section surrogate baseline.", ha="center", fontsize=8, color=GRAY)

    out = FIG_DIR / "input_dimension_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_current_sweep_partial() -> Path:
    csv = ROOT / "remote_3660_merged_latest_20260628_1425" / "results_partial.csv"
    df = pd.read_csv(csv).sort_values("best_test_loss").reset_index(drop=True)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2))
    fig.suptitle("Current MoE sweep status on 508-geometry snapshot (8/32 trials complete)", fontsize=12, weight="bold")

    ax = axs[0, 0]
    ax.bar(np.arange(len(df)), -df["best_test_loss"], color=BLUE)
    ax.set_xlabel("Rank by validation NLL")
    ax.set_ylabel("- best validation NLL")
    ax.set_title("Validation objective")

    ax = axs[0, 1]
    ax.scatter(df["final_mae"], df["r2_My"], s=70, c=df["n_experts"], cmap="viridis", edgecolor="#333333")
    ax.set_xlabel("Validation MAE")
    ax.set_ylabel(r"$R^2(M_y)$")
    ax.set_title("Accuracy vs pitch-moment fit")

    ax = axs[1, 0]
    groups = df.groupby("n_experts")["best_test_loss"].min().sort_index()
    ax.plot(groups.index, -groups.values, marker="o", color=ORANGE, linewidth=2)
    ax.set_xlabel("Number of experts K")
    ax.set_ylabel("Best -NLL observed")
    ax.set_title("Expert-count effect")

    ax = axs[1, 1]
    ax.scatter(df["time_s"] / 60, -df["best_test_loss"], s=70, color=GREEN, edgecolor="#333333")
    ax.set_xlabel("Training time / min")
    ax.set_ylabel("- best validation NLL")
    ax.set_title("Cost-performance tradeoff")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "current_moe_sweep_partial.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_parameter_comparison() -> Path:
    csv = ROOT / "remote_3660_merged_latest_20260628_1425" / "results_partial.csv"
    df = pd.read_csv(csv).sort_values("best_test_loss").head(5).copy()
    labels = [
        f"T{int(r.trial)}: K{int(r.n_experts)} H{int(r.hidden_dim)} S{int(r.shared_dim)}"
        for r in df.itertuples()
    ]

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axs[0]
    y = np.arange(len(df))
    ax.barh(y, df["final_mae"], color=BLUE)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Validation MAE")
    ax.set_title("Top configurations by validation NLL")

    ax = axs[1]
    outputs = ["r2_T", "r2_H", "r2_My", "r2_Q"]
    best = df.iloc[0][outputs].to_numpy(dtype=float)
    ax.bar(["T", "H", "My", "Q"], best, color=[BLUE, ORANGE, GREEN, PURPLE])
    ax.set_ylim(0.96, 1.001)
    ax.set_ylabel(r"$R^2$")
    ax.set_title("Current best output-wise fit")
    for i, v in enumerate(best):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Current best MoE parameter comparison (508-geometry partial sweep)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "current_moe_parameter_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    set_style()
    outs = [
        plot_training_curve_comparison(),
        plot_current_model_metric_heatmap(),
        plot_current_prediction_fit_2d(),
        plot_current_r2_by_output(),
        plot_current_moe_sweep_dotplot(),
    ]
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()
