"""螺旋桨气动力预测模型 — 训练脚本

用法:
    python train.py

TensorBoard 查看:
    tensorboard --logdir=./runs
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from config import Config
from plot_style import apply_style, COLORS, PALETTE, savefig, add_identity_line, annotate_stats

apply_style()


# ================================================================
#  模型定义
# ================================================================

class PropellerPredictor(nn.Module):
    """
    螺旋桨气动力预测网络
    输入: [RPM, WIND, ANGLE]   → 3维
    输出: [Fx, Fy, Fz, Torque] → 4维
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        dims = [cfg.input_dim] + cfg.hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if i >= 2:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(dims[-1], cfg.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ================================================================
#  工具函数
# ================================================================

def load_tensors(cfg: Config):
    """加载标准化后的数据并转为 Tensor。"""
    d = cfg.processed_data_dir
    X_train = torch.tensor(pd.read_pickle(os.path.join(d, "X_train_scaled")).values, dtype=torch.float32)
    X_test = torch.tensor(pd.read_pickle(os.path.join(d, "X_test_scaled")).values, dtype=torch.float32)
    y_train = torch.tensor(pd.read_pickle(os.path.join(d, "y_train_scaled")).values, dtype=torch.float32)
    y_test = torch.tensor(pd.read_pickle(os.path.join(d, "y_test_scaled")).values, dtype=torch.float32)
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """计算整体和逐输出的指标，返回字典。"""
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rel": np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-8)),
    }
    return metrics


def compute_per_output_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list):
    """逐输出计算 MAE / MSE / R²，返回 {name: {mae, mse, r2, rel_mean}}。"""
    result = {}
    for i, name in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        rel = np.abs((yp - yt) / (np.abs(yt) + 1e-8))
        result[name] = {
            "mae": mean_absolute_error(yt, yp),
            "mse": mean_squared_error(yt, yp),
            "r2": r2_score(yt, yp),
            "rel_mean": float(np.mean(rel)),
        }
    return result


# ================================================================
#  可视化
# ================================================================

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray,
                      names: list, epoch: int, tag: str = "Test"):
    """
    生成预测 vs 真实的散点图 + 误差分布直方图。
    返回两个 matplotlib Figure 对象。
    """
    n = len(names)

    # ---- 散点图: pred vs true ----
    fig_scatter, axes_s = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes_s = [axes_s]
    for i, (ax, name) in enumerate(zip(axes_s, names)):
        yt, yp = y_true[:, i], y_pred[:, i]
        r2 = r2_score(yt, yp)
        ax.scatter(yt, yp, alpha=0.5, s=12, edgecolors="none")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", linewidth=1.2, label="y=x")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}  (R²={r2:.4f})")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    fig_scatter.suptitle(f"[{tag}] Epoch {epoch} — Pred vs True", fontsize=13)
    fig_scatter.tight_layout()

    # ---- 误差直方图 ----
    fig_hist, axes_h = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes_h = [axes_h]
    for i, (ax, name) in enumerate(zip(axes_h, names)):
        error = y_pred[:, i] - y_true[:, i]
        ax.hist(error, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Count")
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        ax.set_title(f"{name}  (MAE={mae:.5f})")
    fig_hist.suptitle(f"[{tag}] Epoch {epoch} — Error Distribution", fontsize=13)
    fig_hist.tight_layout()

    return fig_scatter, fig_hist


def plot_pred_vs_true_original(y_true_s: np.ndarray, y_pred_s: np.ndarray,
                                scaler_Y, names: list, epoch: int, tag: str = "Test"):
    """反标准化后绘制原始物理量的预测图。"""
    y_true_orig = scaler_Y.inverse_transform(y_true_s)
    y_pred_orig = scaler_Y.inverse_transform(y_pred_s)

    n = len(names)
    units = {"Fx": "N", "Fy": "N", "Fz": "N", "Torque": "N·m", "My": "N·m", "Mz": "N·m"}

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, name in enumerate(names):
        yt, yp = y_true_orig[:, i], y_pred_orig[:, i]
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        unit = units.get(name, "")

        # 上排: 散点图
        ax = axes[0, i]
        ax.scatter(yt, yp, alpha=0.5, s=12, edgecolors="none", c="steelblue")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", linewidth=1.2)
        ax.set_xlabel(f"True ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(f"{name}  R²={r2:.4f}")
        ax.set_aspect("equal", adjustable="datalim")

        # 下排: 逐样本对比折线
        ax2 = axes[1, i]
        idx = np.argsort(yt)
        ax2.plot(yt[idx], label="True", linewidth=1.2)
        ax2.plot(yp[idx], label="Pred", linewidth=1.2, alpha=0.8)
        ax2.set_xlabel("Sample (sorted by True)")
        ax2.set_ylabel(f"{name} ({unit})")
        ax2.set_title(f"MAE={mae:.5f} {unit}")
        ax2.legend(fontsize=8)

    fig.suptitle(f"[{tag}] Epoch {epoch} — Original Scale", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ================================================================
#  TensorBoard 记录
# ================================================================

def log_scalars(writer: SummaryWriter, epoch: int,
                train_metrics: dict, test_metrics: dict,
                train_per: dict, test_per: dict, lr: float):
    """记录所有标量指标到 TensorBoard。"""
    writer.add_scalar("Loss/Train_MSE", train_metrics["mse"], epoch)
    writer.add_scalar("Loss/Test_MSE", test_metrics["mse"], epoch)
    writer.add_scalar("Loss/Train_MAE", train_metrics["mae"], epoch)
    writer.add_scalar("Loss/Test_MAE", test_metrics["mae"], epoch)
    writer.add_scalar("R2/Train", train_metrics["r2"], epoch)
    writer.add_scalar("R2/Test", test_metrics["r2"], epoch)
    writer.add_scalar("LR", lr, epoch)

    for name in train_per:
        writer.add_scalar(f"MSE_per_output/Train/{name}", train_per[name]["mse"], epoch)
        writer.add_scalar(f"MSE_per_output/Test/{name}", test_per[name]["mse"], epoch)
        writer.add_scalar(f"MAE_per_output/Train/{name}", train_per[name]["mae"], epoch)
        writer.add_scalar(f"MAE_per_output/Test/{name}", test_per[name]["mae"], epoch)
        writer.add_scalar(f"R2_per_output/Train/{name}", train_per[name]["r2"], epoch)
        writer.add_scalar(f"R2_per_output/Test/{name}", test_per[name]["r2"], epoch)
        writer.add_scalar(f"RelError/Train/{name}", train_per[name]["rel_mean"], epoch)
        writer.add_scalar(f"RelError/Test/{name}", test_per[name]["rel_mean"], epoch)


def log_figures(writer: SummaryWriter, epoch: int,
                y_true_s: np.ndarray, y_pred_s: np.ndarray,
                scaler_Y, names: list, tag: str = "Test"):
    """生成预测图并写入 TensorBoard。"""
    fig_scatter, fig_hist = plot_pred_vs_true(y_true_s, y_pred_s, names, epoch, tag)
    writer.add_figure(f"PredVsTrue/{tag}/scatter", fig_scatter, epoch)
    writer.add_figure(f"PredVsTrue/{tag}/error_hist", fig_hist, epoch)
    plt.close(fig_scatter)
    plt.close(fig_hist)

    fig_orig = plot_pred_vs_true_original(y_true_s, y_pred_s, scaler_Y, names, epoch, tag)
    writer.add_figure(f"PredVsTrue/{tag}/original_scale", fig_orig, epoch)
    plt.close(fig_orig)


# ================================================================
#  训练主函数
# ================================================================

def train(cfg: Config):
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

    scaler_Y = joblib.load(os.path.join(cfg.processed_data_dir, "scaler_Y.pkl"))

    log_dir = os.path.join(cfg.log_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir=log_dir)

    best_test_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_path = os.path.join(cfg.model_dir, "best_model.pth")

    train_loss_history = []
    test_loss_history = []
    lr_history = []
    train_r2_history = []
    test_r2_history = []

    import time as _time
    _t_start = _time.time()

    print(f"设备: {device} | 参数量: {param_count:,}")
    print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")
    print(f"TensorBoard: {log_dir}")
    print("-" * 70)

    for epoch in range(cfg.epochs):
        # ---------- 训练 ----------
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

        # ---------- 验证 ----------
        model.eval()
        with torch.no_grad():
            train_out = model(X_train.to(device)).cpu().numpy()
            test_out = model(X_test.to(device)).cpu().numpy()

        train_np = y_train.numpy()
        test_np = y_test.numpy()

        train_metrics = compute_metrics(train_np, train_out)
        test_metrics = compute_metrics(test_np, test_out)
        train_per = compute_per_output_metrics(train_np, train_out, cfg.output_columns)
        test_per = compute_per_output_metrics(test_np, test_out, cfg.output_columns)

        test_loss = test_metrics["mse"]
        lr_now = optimizer.param_groups[0]["lr"]

        train_loss_history.append(float(train_metrics["mse"]))
        test_loss_history.append(float(test_loss))
        lr_history.append(float(lr_now))
        train_r2_history.append(float(train_metrics["r2"]))
        test_r2_history.append(float(test_metrics["r2"]))

        # ---------- TensorBoard: 标量 ----------
        log_scalars(writer, epoch, train_metrics, test_metrics,
                    train_per, test_per, lr_now)

        # ---------- 学习率调度 ----------
        scheduler.step(test_loss)

        # ---------- 最佳模型保存 + 预测图 ----------
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            log_figures(writer, epoch, test_np, test_out,
                        scaler_Y, cfg.output_columns, tag="Test")
            log_figures(writer, epoch, train_np, train_out,
                        scaler_Y, cfg.output_columns, tag="Train")
        else:
            no_improve += 1

        # ---------- 定期绘图 ----------
        if epoch % cfg.plot_every == 0 and epoch > 0:
            log_figures(writer, epoch, test_np, test_out,
                        scaler_Y, cfg.output_columns, tag="Test")

        # ---------- 早停 ----------
        if no_improve >= cfg.early_stop_patience:
            print(f"[早停] Epoch {epoch}, 连续 {cfg.early_stop_patience} 轮未改善")
            log_figures(writer, epoch, test_np, test_out,
                        scaler_Y, cfg.output_columns, tag="Test_Final")
            break

        if epoch % cfg.print_every == 0:
            per_info = " | ".join(
                f"{n}: R²={test_per[n]['r2']:.3f}" for n in cfg.output_columns
            )
            print(
                f"[Epoch {epoch:>4d}] "
                f"Train: {train_metrics['mse']:.6f} | Test: {test_loss:.6f} | "
                f"R²: {test_metrics['r2']:.4f} | LR: {lr_now:.2e}\n"
                f"           {per_info}"
            )

    # ---------- 保存最终模型 ----------
    model.load_state_dict(torch.load(best_path, weights_only=True))
    final_path = os.path.join(cfg.model_dir, "propeller_predictor.pth")
    torch.save(model.state_dict(), final_path)

    # ---------- 最终评估 ----------
    model.eval()
    with torch.no_grad():
        train_out_final = model(X_train.to(device)).cpu().numpy()
        test_out_final = model(X_test.to(device)).cpu().numpy()

    log_figures(writer, epoch, y_test.numpy(), test_out_final,
                scaler_Y, cfg.output_columns, tag="Test_Final")
    writer.close()

    _elapsed = round(_time.time() - _t_start, 1)

    # ---------- 保存训练报告 ----------
    save_training_report(
        cfg=cfg,
        model=model,
        scaler_Y=scaler_Y,
        y_train=y_train.numpy(),
        y_test=y_test.numpy(),
        train_pred=train_out_final,
        test_pred=test_out_final,
        train_loss_history=train_loss_history,
        test_loss_history=test_loss_history,
        lr_history=lr_history,
        train_r2_history=train_r2_history,
        test_r2_history=test_r2_history,
        best_epoch=best_epoch,
        total_epochs=epoch + 1,
        elapsed_sec=_elapsed,
        log_dir=log_dir,
    )

    print("-" * 70)
    print(f"训练完成 | 最佳 Test Loss: {best_test_loss:.6f}")
    print(f"模型已保存至 {final_path}")
    return model


# ================================================================
#  训练报告保存 (论文级)
# ================================================================

def _detailed_per_output_metrics(y_true, y_pred, names):
    """计算逐输出的详细指标 (供论文引用)。"""
    rows = []
    for i, name in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        err = yp - yt
        abs_err = np.abs(err)
        denom = np.abs(yt)
        rel = np.where(denom > 1e-8, abs_err / denom * 100, 0.0)
        rows.append({
            "output": name,
            "MSE": float(mean_squared_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp)),
            "MaxAbsError": float(abs_err.max()),
            "MeanRelError_pct": float(np.mean(rel)),
            "MedianRelError_pct": float(np.median(rel)),
            "P95RelError_pct": float(np.percentile(rel, 95)),
            "P99RelError_pct": float(np.percentile(rel, 99)),
            "MeanBias": float(np.mean(err)),
            "StdError": float(np.std(err)),
        })
    return rows


def save_training_report(
    cfg, model, scaler_Y,
    y_train, y_test, train_pred, test_pred,
    train_loss_history, test_loss_history,
    lr_history, train_r2_history, test_r2_history,
    best_epoch, total_epochs, elapsed_sec, log_dir,
):
    """训练结束后保存完整的论文级数据报告。

    输出到 cfg.model_dir/report/ 下:
      - training_report.json    超参数 + 整体指标 + 逐输出指标
      - test_metrics.csv        测试集逐输出指标表
      - train_metrics.csv       训练集逐输出指标表
      - predictions_test.csv    测试集预测 vs 真实 (原始量纲)
      - predictions_train.csv   训练集预测 vs 真实 (原始量纲)
      - training_curves.csv     逐 epoch loss / R2 / LR
      - training_curves.npz     同上 (numpy 格式)
      - pred_vs_true.png        散点图 (原始量纲)
      - error_distribution.png  误差分布图
      - training_curves.png     训练曲线图
      - report.tex              LaTeX 指标表格片段
    """
    report_dir = os.path.join(cfg.model_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    param_count = sum(p.numel() for p in model.parameters())

    # ---- 逐输出指标 ----
    test_detail = _detailed_per_output_metrics(y_test, test_pred, cfg.output_columns)
    train_detail = _detailed_per_output_metrics(y_train, train_pred, cfg.output_columns)

    df_test_m = pd.DataFrame(test_detail)
    df_train_m = pd.DataFrame(train_detail)
    df_test_m.to_csv(os.path.join(report_dir, "test_metrics.csv"),
                     index=False, float_format="%.6f")
    df_train_m.to_csv(os.path.join(report_dir, "train_metrics.csv"),
                      index=False, float_format="%.6f")

    # ---- 整体指标 ----
    overall_test = {
        "MSE": float(mean_squared_error(y_test, test_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "MAE": float(mean_absolute_error(y_test, test_pred)),
        "R2": float(r2_score(y_test, test_pred)),
    }
    overall_train = {
        "MSE": float(mean_squared_error(y_train, train_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "MAE": float(mean_absolute_error(y_train, train_pred)),
        "R2": float(r2_score(y_train, train_pred)),
    }

    # ---- 综合 JSON 报告 ----
    report = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "hidden_dims": cfg.hidden_dims,
            "n_layers": len(cfg.hidden_dims),
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "batch_size": cfg.batch_size,
            "epochs_max": cfg.epochs,
            "early_stop_patience": cfg.early_stop_patience,
            "scheduler_factor": cfg.scheduler_factor,
            "scheduler_patience": cfg.scheduler_patience,
            "grad_clip": cfg.grad_clip,
            "geometry_mode": cfg.geometry_mode,
        },
        "model": {
            "input_dim": cfg.input_dim,
            "output_dim": cfg.output_dim,
            "input_columns": cfg.input_columns,
            "output_columns": cfg.output_columns,
            "param_count": param_count,
            "architecture": str(model),
        },
        "data": {
            "n_train": len(y_train),
            "n_test": len(y_test),
            "test_ratio": cfg.test_size,
        },
        "training": {
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "elapsed_sec": elapsed_sec,
            "final_lr": lr_history[-1] if lr_history else None,
        },
        "test_overall": overall_test,
        "train_overall": overall_train,
        "overfit_gap_R2": overall_train["R2"] - overall_test["R2"],
        "test_per_output": test_detail,
        "train_per_output": train_detail,
    }
    with open(os.path.join(report_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- 预测值 CSV (原始量纲) ----
    y_test_orig = scaler_Y.inverse_transform(y_test)
    y_pred_test_orig = scaler_Y.inverse_transform(test_pred)
    y_train_orig = scaler_Y.inverse_transform(y_train)
    y_pred_train_orig = scaler_Y.inverse_transform(train_pred)

    cols_true = [f"{c}_true" for c in cfg.output_columns]
    cols_pred = [f"{c}_pred" for c in cfg.output_columns]
    cols_err = [f"{c}_error" for c in cfg.output_columns]

    df_pred_test = pd.DataFrame(
        np.hstack([y_test_orig, y_pred_test_orig, y_pred_test_orig - y_test_orig]),
        columns=cols_true + cols_pred + cols_err,
    )
    df_pred_test.to_csv(os.path.join(report_dir, "predictions_test.csv"),
                        index=False, float_format="%.6f")

    df_pred_train = pd.DataFrame(
        np.hstack([y_train_orig, y_pred_train_orig, y_pred_train_orig - y_train_orig]),
        columns=cols_true + cols_pred + cols_err,
    )
    df_pred_train.to_csv(os.path.join(report_dir, "predictions_train.csv"),
                         index=False, float_format="%.6f")

    # ---- 训练曲线 CSV + NPZ ----
    n_epochs = len(train_loss_history)
    df_curves = pd.DataFrame({
        "epoch": range(n_epochs),
        "train_mse": train_loss_history,
        "test_mse": test_loss_history,
        "train_r2": train_r2_history[:n_epochs],
        "test_r2": test_r2_history[:n_epochs],
        "lr": lr_history[:n_epochs],
    })
    df_curves.to_csv(os.path.join(report_dir, "training_curves.csv"),
                     index=False, float_format="%.8f")
    np.savez_compressed(
        os.path.join(report_dir, "training_curves.npz"),
        train_mse=np.array(train_loss_history),
        test_mse=np.array(test_loss_history),
        train_r2=np.array(train_r2_history),
        test_r2=np.array(test_r2_history),
        lr=np.array(lr_history),
    )

    # ---- 可视化: 预测 vs 真实 (原始量纲) ----
    _plot_report_pred_vs_true(y_test_orig, y_pred_test_orig, cfg.output_columns,
                              os.path.join(report_dir, "pred_vs_true.png"))
    _plot_report_error_distribution(y_test_orig, y_pred_test_orig, cfg.output_columns,
                                    os.path.join(report_dir, "error_distribution.png"))
    _plot_report_training_curves(
        train_loss_history, test_loss_history,
        train_r2_history, test_r2_history, lr_history,
        best_epoch,
        os.path.join(report_dir, "training_curves.png"),
    )

    # ---- LaTeX 表格 ----
    _save_report_latex(test_detail, overall_test, cfg,
                       os.path.join(report_dir, "report.tex"))

    print(f"训练报告已保存至: {report_dir}")


def _plot_report_pred_vs_true(y_true, y_pred, names, save_path):
    """论文级预测 vs 真实散点图。"""
    units = {"Fx": "N", "Fy": "N", "Fz": "N", "Torque": "N$\\cdot$m",
             "My": "N$\\cdot$m", "Mz": "N$\\cdot$m"}
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, names)):
        yt, yp = y_true[:, i], y_pred[:, i]
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        unit = units.get(name, "")

        ax.scatter(yt, yp, alpha=0.35, s=8, c=COLORS["primary"])
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        margin = (hi - lo) * 0.05
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        add_identity_line(ax)
        ax.set_xlabel(f"CFD ({unit})")
        ax.set_ylabel(f"DNN ({unit})")
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="datalim")
        annotate_stats(ax, f"$R^2 = {r2:.4f}$\nRMSE = {rmse:.4f}")

    savefig(fig, save_path)


def _plot_report_error_distribution(y_true, y_pred, names, save_path):
    """论文级误差分布图 (直方图 + 相对误差 CDF)。"""
    units = {"Fx": "N", "Fy": "N", "Fz": "N", "Torque": "N$\\cdot$m",
             "My": "N$\\cdot$m", "Mz": "N$\\cdot$m"}
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, name in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        err = yp - yt
        unit = units.get(name, "")
        mae = mean_absolute_error(yt, yp)

        ax = axes[0, i]
        ax.hist(err, bins=40, edgecolor="white", linewidth=0.4,
                alpha=0.85, color=COLORS["primary"])
        ax.axvline(0, color=COLORS["secondary"], linestyle="--", linewidth=1)
        ax.set_xlabel(f"Error ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(name)
        annotate_stats(ax, f"MAE = {mae:.5f}")

        ax2 = axes[1, i]
        denom = np.abs(yt)
        rel = np.where(denom > 1e-8, np.abs(err) / denom * 100, np.nan)
        rel_clean = rel[~np.isnan(rel)]
        if len(rel_clean) > 0:
            sorted_rel = np.sort(rel_clean)
            cdf = np.arange(1, len(sorted_rel) + 1) / len(sorted_rel) * 100
            ax2.plot(sorted_rel, cdf, color=COLORS["primary"])
            p95 = np.percentile(rel_clean, 95)
            p99 = np.percentile(rel_clean, 99)
            ax2.axhline(95, color=COLORS["accent2"], linestyle="--", alpha=0.7,
                        label=f"P95 = {p95:.2f}%")
            ax2.axhline(99, color=COLORS["secondary"], linestyle="--", alpha=0.7,
                        label=f"P99 = {p99:.2f}%")
            ax2.set_xlabel("Relative Error (%)")
            ax2.set_ylabel("Cumulative %")
            ax2.set_title(f"{name} CDF")
            ax2.legend()
            ax2.set_xlim(left=0)

    fig.suptitle("Error Distribution (Original Scale)")
    savefig(fig, save_path)


def _plot_report_training_curves(train_loss, test_loss, train_r2, test_r2,
                                  lr_hist, best_epoch, save_path):
    """论文级训练曲线图 (3 panel: Loss, R2, LR)。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(len(train_loss))

    ax = axes[0]
    ax.plot(epochs, train_loss, alpha=0.8, label="Train", color=COLORS["train"])
    ax.plot(epochs, test_loss, alpha=0.8, label="Test", color=COLORS["test"])
    ax.axvline(best_epoch, color=COLORS["gray"], linestyle=":", alpha=0.6,
               label=f"Best (ep {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Loss Convergence")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    if train_r2:
        ax.plot(epochs, train_r2, alpha=0.8, label="Train", color=COLORS["train"])
        ax.plot(epochs, test_r2, alpha=0.8, label="Test", color=COLORS["test"])
        ax.axvline(best_epoch, color=COLORS["gray"], linestyle=":", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$R^2$")
    ax.set_title("$R^2$ Convergence")
    ax.legend()

    ax = axes[2]
    if lr_hist:
        ax.plot(epochs, lr_hist, color=COLORS["accent1"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")

    savefig(fig, save_path)


def _save_report_latex(test_detail, overall_test, cfg, save_path):
    """生成 LaTeX 表格 (逐输出指标)。"""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{DNN model prediction accuracy on test set}",
        r"  \label{tab:dnn_accuracy}",
        r"  \begin{tabular}{lcccccc}",
        r"    \toprule",
        r"    Output & MSE & RMSE & MAE & $R^2$ & MAPE(\%) & MaxErr \\",
        r"    \midrule",
    ]
    for row in test_detail:
        lines.append(
            f"    {row['output']} & {row['MSE']:.6f} & {row['RMSE']:.6f} "
            f"& {row['MAE']:.6f} & {row['R2']:.4f} "
            f"& {row['MeanRelError_pct']:.2f} & {row['MaxAbsError']:.4f} \\\\"
        )
    lines += [
        r"    \midrule",
        f"    Overall & {overall_test['MSE']:.6f} & {overall_test['RMSE']:.6f} "
        f"& {overall_test['MAE']:.6f} & {overall_test['R2']:.4f} & -- & -- \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ================================================================
#  入口
# ================================================================

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
