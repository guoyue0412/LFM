"""超参数扫描脚本 — 寻找最佳 DNN 结构、学习率和正则化系数

扫描维度:
  1. 网络结构 (hidden_dims): 深度和宽度的组合
  2. 学习率 (lr): 对数均匀采样
  3. 正则化 (weight_decay): 对数均匀采样
  4. dropout 率

用法:
    python sweep.py                          # 运行完整网格扫描
    python sweep.py --max-trials 20          # 限制最大试验数
    python sweep.py --epochs 1000            # 每次试验的 epoch 数
    python sweep.py --quick                  # 快速模式 (少量组合, 短训练)
"""

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from train import PropellerPredictor, load_tensors
from plot_style import apply_style, COLORS, PALETTE, savefig, annotate_stats

apply_style()


def _compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list) -> dict:
    """计算论文级别的详细指标 (整体 + 逐输出)。"""
    result = {"overall": {}}
    result["overall"]["mse"] = float(mean_squared_error(y_true, y_pred))
    result["overall"]["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    result["overall"]["mae"] = float(mean_absolute_error(y_true, y_pred))
    result["overall"]["r2"] = float(r2_score(y_true, y_pred))

    for i, name in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        errors = yp - yt
        abs_errors = np.abs(errors)
        rel_errors = np.where(np.abs(yt) > 1e-8,
                              np.abs(errors) / np.abs(yt) * 100, 0.0)
        result[name] = {
            "mse": float(mean_squared_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
            "max_abs_error": float(abs_errors.max()),
            "mean_rel_error_pct": float(np.mean(rel_errors)),
            "median_rel_error_pct": float(np.median(rel_errors)),
            "p95_rel_error_pct": float(np.percentile(rel_errors, 95)),
            "p99_rel_error_pct": float(np.percentile(rel_errors, 99)),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
        }
    return result


def train_one_trial(cfg: Config, trial_id: int, verbose: bool = False) -> dict:
    """训练一次试验，返回指标字典。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PropellerPredictor(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_lr,
    )

    X_train, X_test, y_train, y_test = load_tensors(cfg)
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.batch_size, shuffle=True,
    )

    best_test_loss = float("inf")
    best_test_r2 = -float("inf")
    best_epoch = 0
    no_improve = 0
    train_loss_history = []
    test_loss_history = []
    lr_history = []
    train_r2_history = []
    test_r2_history = []

    t_start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item() * bx.size(0)

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train.to(device)).cpu().numpy()
            test_pred = model(X_test.to(device)).cpu().numpy()

        train_mse = mean_squared_error(y_train.numpy(), train_pred)
        test_mse = mean_squared_error(y_test.numpy(), test_pred)
        test_r2 = r2_score(y_test.numpy(), test_pred)
        train_r2 = r2_score(y_train.numpy(), train_pred)

        train_loss_history.append(train_mse)
        test_loss_history.append(test_mse)
        lr_history.append(optimizer.param_groups[0]["lr"])
        train_r2_history.append(train_r2)
        test_r2_history.append(test_r2)

        scheduler.step(test_mse)

        if test_mse < best_test_loss:
            best_test_loss = test_mse
            best_test_r2 = test_r2
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= cfg.early_stop_patience:
            if verbose:
                print(f"    [Trial {trial_id}] Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t_start

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_best = model(X_test.to(device)).cpu().numpy()
        train_pred_best = model(X_train.to(device)).cpu().numpy()

    test_np = y_test.numpy()
    train_np = y_train.numpy()

    test_metrics = _compute_detailed_metrics(test_np, test_pred_best, cfg.output_columns)
    train_metrics = _compute_detailed_metrics(train_np, train_pred_best, cfg.output_columns)

    return {
        "trial_id": trial_id,
        "hidden_dims": cfg.hidden_dims,
        "n_layers": len(cfg.hidden_dims),
        "max_width": max(cfg.hidden_dims),
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "param_count": param_count,
        "input_dim": cfg.input_dim,
        "output_dim": cfg.output_dim,
        "geometry_mode": cfg.geometry_mode,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "best_test_mse": float(best_test_loss),
        "best_test_r2": float(best_test_r2),
        "train_mse": train_metrics["overall"]["mse"],
        "train_r2": train_metrics["overall"]["r2"],
        "overfit_gap": train_metrics["overall"]["r2"] - best_test_r2,
        "elapsed_sec": round(elapsed, 1),
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "per_output": {
            name: {
                "test_r2": test_metrics[name]["r2"],
                "test_mae": test_metrics[name]["mae"],
                "test_rmse": test_metrics[name]["rmse"],
                "test_mape": test_metrics[name]["mean_rel_error_pct"],
                "train_r2": train_metrics[name]["r2"],
            }
            for name in cfg.output_columns
        },
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "lr_history": lr_history,
        "train_r2_history": train_r2_history,
        "test_r2_history": test_r2_history,
        "best_model_state": best_state,
    }


def generate_sweep_configs(args):
    """生成超参数组合列表。"""
    if args.quick:
        hidden_dims_list = [
            [64, 128, 64],
            [128, 256, 128],
            [64, 128, 256, 128],
            [128, 256, 512, 256, 128],
        ]
        lr_list = [1e-3, 5e-4]
        wd_list = [1e-4, 1e-3]
        dropout_list = [0.1]
    else:
        hidden_dims_list = [
            # 3 层
            [64, 128, 64],
            [128, 256, 128],
            [256, 512, 256],
            # 4 层
            [64, 128, 128, 64],
            [128, 256, 256, 128],
            [64, 128, 256, 128],
            [128, 256, 512, 256],
            [256, 512, 512, 256],
            # 5 层
            [64, 128, 256, 128, 64],
            [128, 256, 512, 256, 128],
            [64, 128, 256, 256, 128],
            [128, 256, 256, 128, 64],
        ]
        lr_list = [5e-3, 1e-3, 5e-4, 1e-4]
        wd_list = [0, 1e-5, 1e-4, 1e-3]
        dropout_list = [0.0, 0.1, 0.2]

    combos = list(itertools.product(hidden_dims_list, lr_list, wd_list, dropout_list))

    if args.max_trials and len(combos) > args.max_trials:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(combos), size=args.max_trials, replace=False)
        combos = [combos[i] for i in sorted(indices)]

    return combos


def run_sweep(args):
    """执行超参数扫描。"""
    combos = generate_sweep_configs(args)
    n_trials = len(combos)

    print("=" * 70)
    print(f"  超参数扫描: {n_trials} 组合")
    print(f"  每次训练: {args.epochs} epochs (早停 patience={args.patience})")
    print("=" * 70)

    results = []
    base_cfg = Config()

    for idx, (hidden, lr, wd, dropout) in enumerate(combos):
        cfg = Config(
            hidden_dims=list(hidden),
            lr=lr,
            weight_decay=wd,
            dropout=dropout,
            epochs=args.epochs,
            early_stop_patience=args.patience,
            scheduler_patience=args.patience // 3,
            print_every=args.epochs + 1,
            plot_every=args.epochs + 1,
        )

        depth = len(hidden)
        width = max(hidden)
        print(f"\n[{idx+1}/{n_trials}] hidden={hidden} lr={lr:.0e} wd={wd:.0e} "
              f"drop={dropout} (depth={depth}, width={width})")

        try:
            result = train_one_trial(cfg, trial_id=idx, verbose=True)
            results.append(result)

            print(f"  → Test MSE={result['best_test_mse']:.6f}  "
                  f"R²={result['best_test_r2']:.4f}  "
                  f"Epoch={result['best_epoch']}  "
                  f"Params={result['param_count']:,}  "
                  f"Gap={result['overfit_gap']:.4f}  "
                  f"Time={result['elapsed_sec']:.0f}s")

        except Exception as e:
            print(f"  [失败] {e}")
            results.append({
                "trial_id": idx,
                "hidden_dims": list(hidden),
                "lr": lr, "weight_decay": wd, "dropout": dropout,
                "best_test_mse": float("inf"),
                "best_test_r2": -float("inf"),
                "error": str(e),
            })

    return results


def save_results(results: list, output_dir: str):
    """保存扫描结果为多种论文级格式。

    输出文件:
      - sweep_<ts>.csv       主对比表
      - sweep_<ts>_detail.json  完整指标 (含逐输出 MSE/RMSE/MAE/MAPE/R2/Max Error)
      - sweep_<ts>_curves.npz   训练曲线 (loss / R2 / lr vs epoch)
      - sweep_<ts>_per_output.csv  逐输出指标展开表
      - best_model_trial<id>.pt    最佳模型权重
      - sweep_<ts>.png             总览图
      - sweep_<ts>_best_pred.png   最佳模型 预测 vs 真实散点图
      - sweep_<ts>_error_dist.png  最佳模型 逐输出误差分布
      - sweep_<ts>_curves.png      Top-N 训练曲线
      - sweep_<ts>.tex             LaTeX 表格片段
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("[警告] 没有成功的试验")
        return

    valid_sorted = sorted(valid, key=lambda r: r["best_test_mse"])

    # ===== 1. 主对比 CSV =====
    rows = []
    for r in valid_sorted:
        row = {
            "rank": len(rows) + 1,
            "hidden_dims": str(r["hidden_dims"]),
            "depth": len(r["hidden_dims"]),
            "width": max(r["hidden_dims"]),
            "lr": r["lr"],
            "weight_decay": r["weight_decay"],
            "dropout": r["dropout"],
            "params": r["param_count"],
            "best_epoch": r["best_epoch"],
            "total_epochs": r["total_epochs"],
            "n_train": r.get("n_train", ""),
            "n_test": r.get("n_test", ""),
            "test_mse": r["best_test_mse"],
            "test_rmse": r["test_metrics"]["overall"]["rmse"],
            "test_mae": r["test_metrics"]["overall"]["mae"],
            "test_r2": r["best_test_r2"],
            "train_mse": r["train_mse"],
            "train_r2": r["train_r2"],
            "overfit_gap": r["overfit_gap"],
            "time_sec": r["elapsed_sec"],
        }
        for name in r.get("per_output", {}):
            m = r["per_output"][name]
            row[f"{name}_R2"] = m["test_r2"]
            row[f"{name}_MAE"] = m["test_mae"]
            row[f"{name}_RMSE"] = m["test_rmse"]
            row[f"{name}_MAPE%"] = m["test_mape"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"sweep_{timestamp}.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n主对比表已保存: {csv_path}")

    # ===== 2. 逐输出指标展开表 =====
    per_out_rows = []
    for r in valid_sorted:
        for name in r.get("test_metrics", {}):
            if name == "overall":
                continue
            tm = r["test_metrics"][name]
            per_out_rows.append({
                "trial_id": r["trial_id"],
                "hidden_dims": str(r["hidden_dims"]),
                "output": name,
                "MSE": tm["mse"],
                "RMSE": tm["rmse"],
                "MAE": tm["mae"],
                "R2": tm["r2"],
                "MaxAbsErr": tm["max_abs_error"],
                "MeanRelErr%": tm["mean_rel_error_pct"],
                "P95RelErr%": tm["p95_rel_error_pct"],
                "P99RelErr%": tm["p99_rel_error_pct"],
                "MeanBias": tm["mean_error"],
                "StdErr": tm["std_error"],
            })
    df_per = pd.DataFrame(per_out_rows)
    per_csv = os.path.join(output_dir, f"sweep_{timestamp}_per_output.csv")
    df_per.to_csv(per_csv, index=False, float_format="%.6f")
    print(f"逐输出指标表已保存: {per_csv}")

    # ===== 3. 完整 JSON (不含权重和原始曲线) =====
    json_path = os.path.join(output_dir, f"sweep_{timestamp}_detail.json")
    json_results = []
    for r in valid_sorted:
        entry = {k: v for k, v in r.items()
                 if k not in ("train_loss_history", "test_loss_history",
                              "lr_history", "train_r2_history",
                              "test_r2_history", "best_model_state")}
        json_results.append(entry)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"详细 JSON 已保存: {json_path}")

    # ===== 4. 训练曲线 NPZ =====
    curves = {}
    for r in valid_sorted:
        tid = r["trial_id"]
        curves[f"t{tid}_train_loss"] = np.array(r["train_loss_history"])
        curves[f"t{tid}_test_loss"] = np.array(r["test_loss_history"])
        curves[f"t{tid}_lr"] = np.array(r.get("lr_history", []))
        curves[f"t{tid}_train_r2"] = np.array(r.get("train_r2_history", []))
        curves[f"t{tid}_test_r2"] = np.array(r.get("test_r2_history", []))
    npz_path = os.path.join(output_dir, f"sweep_{timestamp}_curves.npz")
    np.savez_compressed(npz_path, **curves)
    print(f"训练曲线数据已保存: {npz_path}")

    # ===== 5. 最佳模型权重 =====
    best = valid_sorted[0]
    if "best_model_state" in best:
        model_path = os.path.join(
            output_dir, f"best_model_trial{best['trial_id']}.pt")
        torch.save(best["best_model_state"], model_path)
        print(f"最佳模型权重已保存: {model_path}")

    # ===== 6. LaTeX 表格 =====
    _save_latex_table(valid_sorted, output_dir, timestamp)

    # ===== 7. 可视化: 总览 =====
    _plot_overview(valid, valid_sorted, output_dir, timestamp)

    # ===== 8. 可视化: 训练曲线 =====
    _plot_training_curves(valid_sorted, output_dir, timestamp)

    # ===== 打印 Top 10 =====
    print("\n" + "=" * 100)
    print("  Top 10 Results")
    print("=" * 100)
    header = (f"{'Rank':>4} {'Architecture':<25} {'LR':>8} {'WD':>8} {'Drop':>5} "
              f"{'Params':>8} {'TestMSE':>10} {'TestRMSE':>10} {'TestR2':>8} {'Gap':>6}")
    print(header)
    print("-" * 100)
    for i, r in enumerate(valid_sorted[:10]):
        rmse = r["test_metrics"]["overall"]["rmse"]
        print(f"{i+1:>4} {str(r['hidden_dims']):<25} {r['lr']:>8.0e} "
              f"{r['weight_decay']:>8.0e} {r['dropout']:>5.2f} "
              f"{r['param_count']:>8,} {r['best_test_mse']:>10.6f} "
              f"{rmse:>10.6f} "
              f"{r['best_test_r2']:>8.4f} {r['overfit_gap']:>6.4f}")
    print("=" * 100)

    print(f"\n{'='*40}")
    print(f"  Best Configuration")
    print(f"{'='*40}")
    print(f"  hidden_dims  = {best['hidden_dims']}")
    print(f"  lr           = {best['lr']}")
    print(f"  weight_decay = {best['weight_decay']}")
    print(f"  dropout      = {best['dropout']}")
    print(f"  Test R2      = {best['best_test_r2']:.6f}")
    print(f"  Test MSE     = {best['best_test_mse']:.8f}")
    print(f"  Test RMSE    = {best['test_metrics']['overall']['rmse']:.8f}")
    print(f"  Test MAE     = {best['test_metrics']['overall']['mae']:.8f}")
    for name in best.get("per_output", {}):
        m = best["per_output"][name]
        print(f"    {name:>8}: R2={m['test_r2']:.6f}  MAE={m['test_mae']:.6f}  "
              f"RMSE={m['test_rmse']:.6f}  MAPE={m['test_mape']:.2f}%")

    return valid_sorted[0]


# =========================================================================
#  辅助函数: LaTeX 表格
# =========================================================================
def _save_latex_table(valid_sorted: list, output_dir: str, timestamp: str):
    """生成 LaTeX 格式的结果表格 (可直接用于论文)。"""
    tex_path = os.path.join(output_dir, f"sweep_{timestamp}.tex")
    top = valid_sorted[:min(10, len(valid_sorted))]

    output_names = list(top[0].get("per_output", {}).keys())
    n_out = len(output_names)

    col_spec = "c" * (5 + n_out)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Hyperparameter sweep results (top configurations)}",
        r"  \label{tab:sweep}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    header_parts = [r"Rank", r"Architecture", r"LR", r"$R^2$", r"RMSE"]
    for n in output_names:
        header_parts.append(f"${n}$ $R^2$")
    lines.append("    " + " & ".join(header_parts) + r" \\")
    lines.append(r"    \midrule")

    for i, r in enumerate(top):
        arch = str(r["hidden_dims"]).replace("[", "").replace("]", "")
        rmse = r["test_metrics"]["overall"]["rmse"]
        parts = [
            str(i + 1),
            f"[{arch}]",
            f"{r['lr']:.0e}",
            f"{r['best_test_r2']:.4f}",
            f"{rmse:.4f}",
        ]
        for n in output_names:
            parts.append(f"{r['per_output'][n]['test_r2']:.4f}")
        lines.append("    " + " & ".join(parts) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"LaTeX 表格已保存: {tex_path}")


# =========================================================================
#  辅助函数: 总览可视化
# =========================================================================
def _plot_overview(valid: list, valid_sorted: list, output_dir: str, timestamp: str):
    """生成论文级别的总览图。"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    top_n = min(20, len(valid_sorted))
    labels = [str(r["hidden_dims"]) for r in valid_sorted[:top_n]]
    r2_vals = [r["best_test_r2"] for r in valid_sorted[:top_n]]
    cmap_vals = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))

    ax = axes[0, 0]
    ax.barh(range(top_n), r2_vals, color=cmap_vals)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Test $R^2$")
    ax.set_title(f"Top {top_n} Architectures")
    ax.invert_yaxis()

    ax = axes[0, 1]
    lrs = [r["lr"] for r in valid]
    mses = [r["best_test_mse"] for r in valid]
    ax.scatter(lrs, mses, alpha=0.6, c=COLORS["primary"], edgecolors="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Test MSE")
    ax.set_title("Learning Rate vs Test MSE")

    ax = axes[0, 2]
    wds = [max(r["weight_decay"], 1e-7) for r in valid]
    sc = ax.scatter(wds, mses, alpha=0.6, c=[r["best_test_r2"] for r in valid],
                    cmap="RdYlGn", edgecolors="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel("Test MSE")
    ax.set_title("Weight Decay vs Test MSE (color=$R^2$)")
    plt.colorbar(sc, ax=ax, label="$R^2$")

    ax = axes[1, 0]
    params = [r["param_count"] for r in valid]
    r2s = [r["best_test_r2"] for r in valid]
    ax.scatter(params, r2s, alpha=0.6, c=COLORS["primary"], edgecolors="white", linewidth=0.3)
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Test $R^2$")
    ax.set_title("Model Size vs Test $R^2$")

    ax = axes[1, 1]
    gaps = [r["overfit_gap"] for r in valid]
    wds_valid = [max(r["weight_decay"], 1e-8) for r in valid]
    norm = plt.matplotlib.colors.LogNorm()
    ax.scatter(r2s, gaps, alpha=0.6, c=wds_valid,
               cmap="coolwarm", edgecolors="white", linewidth=0.3, norm=norm)
    ax.set_xlabel("Test $R^2$")
    ax.set_ylabel("Overfit Gap (Train $R^2$ $-$ Test $R^2$)")
    ax.set_title("Generalization Gap")
    ax.axhline(0, color=COLORS["gray"], linestyle="--", alpha=0.5)

    ax = axes[1, 2]
    best = valid_sorted[0]
    output_names = list(best.get("per_output", {}).keys())
    if output_names:
        x_pos = np.arange(len(output_names))
        width = 0.35
        train_r2 = [best["per_output"][n]["train_r2"] for n in output_names]
        test_r2_ = [best["per_output"][n]["test_r2"] for n in output_names]
        ax.bar(x_pos - width / 2, train_r2, width, label="Train",
               alpha=0.85, color=COLORS["train"])
        ax.bar(x_pos + width / 2, test_r2_, width, label="Test",
               alpha=0.85, color=COLORS["test"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(output_names)
        ax.set_ylabel("$R^2$")
        ax.set_title("Best Model: Per-Output $R^2$")
        ax.legend()
        ax.set_ylim(min(min(train_r2), min(test_r2_)) - 0.01, 1.001)

    fig.suptitle("Hyperparameter Sweep Results", fontsize=13)
    fig_path = os.path.join(output_dir, f"sweep_{timestamp}.png")
    savefig(fig, fig_path)
    print(f"总览可视化已保存: {fig_path}")


# =========================================================================
#  辅助函数: 训练曲线
# =========================================================================
def _plot_training_curves(valid_sorted: list, output_dir: str, timestamp: str):
    """Top-N 训练曲线 (Loss + R2 + LR)。"""
    top_n = min(5, len(valid_sorted))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, r in enumerate(valid_sorted[:top_n]):
        c = PALETTE[idx % len(PALETTE)]
        lbl = f"{r['hidden_dims']} lr={r['lr']:.0e}"

        axes[0].plot(r["test_loss_history"], alpha=0.8, label=lbl, color=c)
        axes[0].plot(r["train_loss_history"], alpha=0.3, linestyle="--", color=c)

        if r.get("test_r2_history"):
            axes[1].plot(r["test_r2_history"], alpha=0.8, label=lbl, color=c)

        if r.get("lr_history"):
            axes[2].plot(r["lr_history"], alpha=0.8, label=lbl, color=c)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Loss Curves (solid=test, dashed=train)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("$R^2$")
    axes[1].set_title("Test $R^2$ Curves")
    axes[1].legend(fontsize=7)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=7)

    fig.suptitle("Top Training Curves", fontsize=12)
    fig_path = os.path.join(output_dir, f"sweep_{timestamp}_curves.png")
    savefig(fig, fig_path)
    print(f"训练曲线图已保存: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="DNN 超参数扫描")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="每次试验的最大 epoch 数")
    parser.add_argument("--patience", type=int, default=300,
                        help="早停耐心值")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="最大试验数 (从全组合中随机采样)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 少量组合 + 短训练")
    parser.add_argument("--output-dir", type=str, default="./sweep_results",
                        help="结果输出目录")
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 500)
        args.patience = min(args.patience, 100)

    results = run_sweep(args)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
