"""MoE 模型训练脚本

用法:
    python -m optimization_v2.train_moe
    python -m optimization_v2.train_moe --epochs 5000 --lr 5e-4
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import OptConfig
from .moe_model import MoEPredictor, MoELoss


def _column_mapping_from_config(cfg: OptConfig):
    """从 OptConfig 构建 ColumnMapping。"""
    from .data_loader import ColumnMapping
    return ColumnMapping(
        rpm_col=cfg.rpm_col,
        wind_col=cfg.wind_col,
        angle_col=cfg.angle_col,
        T_col=cfg.T_col,
        H_col=cfg.H_col,
        My_col=cfg.My_col,
        Q_col=cfg.Q_col,
        cp_prefix=cfg.cp_prefix,
        chord_prefix=cfg.chord_prefix,
        twist_prefix=cfg.twist_prefix,
        chord_cp_prefix=cfg.chord_cp_prefix,
        twist_cp_prefix=cfg.twist_cp_prefix,
    )


def prepare_data(cfg: OptConfig):
    """加载数据并标准化。

    优先级:
        1. cfg.excel_path 指定的 Excel/CSV 文件 (真实数据)
        2. cfg.data_dir/synthetic_aero_data.csv (合成数据, fallback)
        3. 自动生成合成数据
    """
    from .data_loader import load_excel_dataset
    from .geometry import INPUT_COLS, OUTPUT_COLS
    from .synthetic_data import load_dataset, generate_synthetic_dataset, save_dataset

    if cfg.excel_path is not None and os.path.exists(cfg.excel_path):
        print(f"从 Excel 加载真实数据: {cfg.excel_path}")
        mapping = _column_mapping_from_config(cfg)
        df = load_excel_dataset(cfg.excel_path, cfg.excel_sheet, mapping=mapping)
    else:
        try:
            df = load_dataset(cfg)
            print(f"使用合成数据集: {len(df)} 条")
        except FileNotFoundError:
            print("无可用数据,生成合成数据...")
            df = generate_synthetic_dataset(cfg)
            save_dataset(df, cfg)

    X = df[INPUT_COLS].values.astype(np.float64)
    y = df[OUTPUT_COLS].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    return (X_train_s, X_test_s, y_train_s, y_test_s,
            scaler_X, scaler_y, OUTPUT_COLS)


def train_moe(cfg: OptConfig, verbose: bool = True):
    """训练 MoE 模型。"""
    os.makedirs(cfg.model_dir, exist_ok=True)

    (X_train, X_test, y_train, y_test,
     scaler_X, scaler_y, output_cols) = prepare_data(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"设备: {device}")
        print(f"训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=cfg.moe_batch_size, shuffle=True)

    model = MoEPredictor(
        input_dim=cfg.moe_input_dim,
        output_dim=cfg.moe_output_dim,
        n_experts=cfg.moe_n_experts,
        shared_dim=cfg.moe_shared_dim,
        hidden_dim=cfg.moe_hidden_dim,
    ).to(device)

    criterion = MoELoss(
        balance_weight=cfg.moe_balance_weight,
        smooth_weight=cfg.moe_smooth_weight,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.moe_lr,
        weight_decay=cfg.moe_weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6
    )

    best_test_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "test_loss": [], "test_mape": []}

    t0 = time.time()
    for epoch in range(1, cfg.moe_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, sigma, gate_weights = model(xb)
            loss, _ = criterion(mu, sigma, gate_weights, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # 测试集评估
        model.eval()
        with torch.no_grad():
            mu_test, sigma_test, gate_test = model(X_test_t)
            test_loss, test_details = criterion(mu_test, sigma_test, gate_test, y_test_t)

            # MAPE (在原始尺度)
            mu_np = mu_test.cpu().numpy()
            y_test_orig = scaler_y.inverse_transform(y_test)
            mu_orig = scaler_y.inverse_transform(mu_np)
            mape = np.mean(np.abs((y_test_orig - mu_orig) / (np.abs(y_test_orig) + 1e-8))) * 100

        scheduler.step(test_loss.item())

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(test_loss.item())
        history["test_mape"].append(mape)

        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(cfg.model_dir, "moe_best.pth"))

        if verbose and (epoch % 500 == 0 or epoch == 1):
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:5d} | train={avg_train_loss:.4f} "
                  f"test={test_loss.item():.4f} MAPE={mape:.2f}% "
                  f"lr={lr_now:.1e} [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n训练完成: {elapsed:.1f}s, best_epoch={best_epoch}, "
              f"best_test_loss={best_test_loss:.4f}")

    # 保存 scalers 和配置
    import joblib
    joblib.dump(scaler_X, os.path.join(cfg.model_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(cfg.model_dir, "scaler_y.pkl"))

    with open(os.path.join(cfg.model_dir, "train_history.json"), "w") as f:
        json.dump(history, f)

    # 最终评估
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, "moe_best.pth"),
                                     map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        mu_test, sigma_test, _ = model(X_test_t)
        mu_np = mu_test.cpu().numpy()
        sigma_np = sigma_test.cpu().numpy()

    mu_orig = scaler_y.inverse_transform(mu_np)
    y_test_orig = scaler_y.inverse_transform(y_test)

    if verbose:
        print(f"\n{'='*50}")
        print("  最终测试集指标 (原始尺度)")
        print(f"{'='*50}")
        for i, name in enumerate(output_cols):
            mape_i = np.mean(np.abs((y_test_orig[:, i] - mu_orig[:, i]) /
                                    (np.abs(y_test_orig[:, i]) + 1e-8))) * 100
            r2 = 1.0 - np.sum((y_test_orig[:, i] - mu_orig[:, i])**2) / \
                        np.sum((y_test_orig[:, i] - np.mean(y_test_orig[:, i]))**2)
            sigma_mean = np.mean(sigma_np[:, i])
            print(f"  {name:>4s}: MAPE={mape_i:6.2f}%  R²={r2:.4f}  σ_mean={sigma_mean:.4f}")

    return model, scaler_X, scaler_y


def load_trained_moe(cfg: OptConfig, device=None):
    """加载训练好的 MoE 模型和 scalers。"""
    import joblib

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MoEPredictor(
        input_dim=cfg.moe_input_dim,
        output_dim=cfg.moe_output_dim,
        n_experts=cfg.moe_n_experts,
        shared_dim=cfg.moe_shared_dim,
        hidden_dim=cfg.moe_hidden_dim,
    ).to(device)

    model_path = os.path.join(cfg.model_dir, "moe_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scaler_X = joblib.load(os.path.join(cfg.model_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(cfg.model_dir, "scaler_y.pkl"))

    return model, scaler_X, scaler_y, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 MoE 气动代理模型")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-experts", type=int, default=None)
    args = parser.parse_args()

    cfg = OptConfig()
    if args.epochs:
        cfg.moe_epochs = args.epochs
    if args.lr:
        cfg.moe_lr = args.lr
    if args.n_experts:
        cfg.moe_n_experts = args.n_experts

    train_moe(cfg)
