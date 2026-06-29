#!/usr/bin/env python3
"""Train current-snapshot DNN and MLP+LIFT baselines for thesis figures.

The formal MoE sweep is already running on 3660. This script uses the same
2026-06-28 14:25 normalized 508-geometry snapshot locally to produce fair
baseline evidence without touching the remote GPU job.
"""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from optimization_v2.tools.sweep_moe import (
    HeteroscedasticMLP,
    MoELoss,
    load_normalized,
    split_by_geometry,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "remote_3660_merged_latest_20260628_1425" / "processed_lhs_merged_latest_20260628_1425"
DATA = DATA_DIR / "norm" / "data_normalized.npz"
CSV = DATA_DIR / "dataset_merged_latest_20260628_1425.csv"
OUT = ROOT / "remote_3660_merged_latest_20260628_1425" / "current_snapshot_baselines"
OUT.mkdir(parents=True, exist_ok=True)


class DeterministicDNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden_dim), nn.GELU()]
            last = hidden_dim
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def r2_score(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    ss_res = ((pred - y) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def train_dnn(X_tr, Y_tr, X_te, Y_te, device, epochs=1500, seed=20260628):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeterministicDNN(X_tr.shape[1], Y_tr.shape[1], hidden_dim=128, depth=3).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=80, min_lr=1e-6)
    crit = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)),
        batch_size=2048,
        shuffle=True,
    )
    X_te_t = torch.tensor(X_te).to(device)
    Y_te_t = torch.tensor(Y_te).to(device)
    hist = {"epoch": [], "train_loss": [], "test_loss": [], "test_mae": [], "sigma_error_corr": []}
    best = {"loss": float("inf"), "state": None, "epoch": 0}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        s = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s += float(loss.item())
            n += 1
        model.eval()
        with torch.no_grad():
            pred = model(X_te_t)
            loss = crit(pred, Y_te_t)
            err = (pred - Y_te_t).detach().cpu().numpy()
            mae = float(np.mean(np.abs(err)))
        sched.step(float(loss.item()))
        hist["epoch"].append(ep)
        hist["train_loss"].append(s / max(n, 1))
        hist["test_loss"].append(float(loss.item()))
        hist["test_mae"].append(mae)
        hist["sigma_error_corr"].append(np.nan)
        if loss.item() < best["loss"]:
            best = {"loss": float(loss.item()), "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": ep}
        if ep % 20 == 0:
            print(f"[DNN] ep={ep} test_mse={loss.item():.5f} mae={mae:.5f}", flush=True)
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pred = model(X_te_t).detach().cpu().numpy()
    torch.save(best["state"], OUT / "dnn_mse_model.pth")
    return hist, {
        "model": "DNN-MSE",
        "best_epoch": best["epoch"],
        "best_test_loss": best["loss"],
        "final_mae": float(np.mean(np.abs(pred - Y_te))),
        "r2": r2_score(pred, Y_te).tolist(),
        "sigma_error_corr": None,
        "time_s": time.time() - t0,
        "epochs": epochs,
    }


def train_mlp_lift(X_tr, Y_tr, X_te, Y_te, device, epochs=1500, seed=20260628):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HeteroscedasticMLP(
        input_dim=X_tr.shape[1],
        output_dim=Y_tr.shape[1],
        hidden_dim=128,
        depth=3,
        dropout=0.0,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=80, min_lr=1e-6)
    crit = MoELoss(balance_weight=0.0, smooth_weight=0.0, kind="lift")
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)),
        batch_size=2048,
        shuffle=True,
    )
    X_te_t = torch.tensor(X_te).to(device)
    Y_te_t = torch.tensor(Y_te).to(device)
    hist = {"epoch": [], "train_loss": [], "test_loss": [], "test_mae": [], "sigma_error_corr": []}
    best = {"loss": float("inf"), "state": None, "epoch": 0}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        s = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, sig, gw, mus, lv = model(xb)
            loss, _ = crit(mu, sig, gw, yb, mus=mus, log_vars=lv)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s += float(loss.item())
            n += 1
        model.eval()
        with torch.no_grad():
            mu, sig, gw, mus, lv = model(X_te_t)
            loss, _ = crit(mu, sig, gw, Y_te_t, mus=mus, log_vars=lv)
            err = (mu - Y_te_t).detach().cpu().numpy()
            sigma = sig.detach().cpu().numpy()
            mae = float(np.mean(np.abs(err)))
            corr = float(np.corrcoef(np.abs(err).mean(axis=1), sigma.mean(axis=1))[0, 1])
        sched.step(float(loss.item()))
        hist["epoch"].append(ep)
        hist["train_loss"].append(s / max(n, 1))
        hist["test_loss"].append(float(loss.item()))
        hist["test_mae"].append(mae)
        hist["sigma_error_corr"].append(corr)
        if loss.item() < best["loss"]:
            best = {"loss": float(loss.item()), "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": ep}
        if ep % 20 == 0:
            print(f"[MLP+LIFT] ep={ep} test_nll={loss.item():.5f} mae={mae:.5f} corr={corr:.3f}", flush=True)
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        mu, sig, *_ = model(X_te_t)
        pred = mu.detach().cpu().numpy()
        sigma = sig.detach().cpu().numpy()
    torch.save(best["state"], OUT / "mlp_lift_model.pth")
    return hist, {
        "model": "MLP+LIFT-NLL",
        "best_epoch": best["epoch"],
        "best_test_loss": best["loss"],
        "final_mae": float(np.mean(np.abs(pred - Y_te))),
        "r2": r2_score(pred, Y_te).tolist(),
        "sigma_error_corr": float(np.corrcoef(np.abs(pred - Y_te).mean(axis=1), sigma.mean(axis=1))[0, 1]),
        "time_s": time.time() - t0,
        "epochs": epochs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1500, help="每个基线模型训练轮数，默认与正式 sweep 保持一致")
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    if args.epochs != 1500:
        print(f"[warn] current thesis sweep standard is 1500 epochs; got --epochs {args.epochs}", flush=True)

    device = choose_device()
    print(f"[device] {device}", flush=True)
    print(f"[epochs] {args.epochs}", flush=True)
    X, Y, _, _ = load_normalized(DATA, use_cp=False)
    gi = pd.read_csv(CSV)["geom_idx"].to_numpy()
    X_tr, Y_tr, X_te, Y_te = split_by_geometry(X, Y, gi, seed=606281425)
    print(f"[data] X={X.shape} Y={Y.shape} train={X_tr.shape[0]} test={X_te.shape[0]}", flush=True)

    dnn_hist, dnn_metrics = train_dnn(X_tr, Y_tr, X_te, Y_te, device, epochs=args.epochs, seed=args.seed)
    mlp_hist, mlp_metrics = train_mlp_lift(X_tr, Y_tr, X_te, Y_te, device, epochs=args.epochs, seed=args.seed)

    pd.DataFrame(dnn_hist).to_csv(OUT / "dnn_mse_history.csv", index=False)
    pd.DataFrame(mlp_hist).to_csv(OUT / "mlp_lift_history.csv", index=False)
    pd.DataFrame([dnn_metrics, mlp_metrics]).to_csv(OUT / "baseline_metrics.csv", index=False)
    (OUT / "baseline_metrics.json").write_text(
        json.dumps([dnn_metrics, mlp_metrics], indent=2, ensure_ascii=False)
    )
    print(pd.DataFrame([dnn_metrics, mlp_metrics]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
