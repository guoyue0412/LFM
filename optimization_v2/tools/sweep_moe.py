"""MoE 超参 sweep + 综合可视化

用法:
    # 1) 跑 sweep (默认网格,可用 --grid sample 限规模)
    python -m optimization_v2.tools.sweep_moe \
        --data ./data_for_train/processed_lhs/norm/data_normalized.npz \
        --norm ./data_for_train/processed_lhs/norm/normalizer.pkl \
        --out  ./sweep_results/moe_v2 \
        --epochs 1500 --device cuda

    # 2) 仅出图 (从已有 results.csv)
    python -m optimization_v2.tools.sweep_moe --plot-only \
        --out ./sweep_results/moe_v2

==== Sweep 维度 (默认网格) ====
    n_experts        : [2, 4, 6, 8]
    hidden_dim       : [64, 128, 256]
    shared_dim       : [32, 64, 128]
    lr               : [1e-3, 5e-4, 1e-4]
    weight_decay     : [1e-4, 1e-3]
    balance_weight   : [0.01, 0.1, 0.3]
    smooth_weight    : [0.0, 0.01]
    全网格 = 4×3×3×3×2×3×2 = 1296 → 默认用 random sample 32 组

==== 输出 ====
    results.csv            — 每个 trial 一行的最终指标
    histories.npz          — 每个 trial 的 epoch 级 train/test/mape
    sweep_overview.png     — 多面板综合图 (见 plot_sweep)
    best_trial/            — 最优 trial 的模型/混淆/残差等细图
"""

import argparse
import itertools
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_COLS = ["RPM", "WIND", "ANGLE"]
SECTION_COLS = [f"chord_{i}" for i in range(22)] + [f"twist_{i}" for i in range(22)]
INPUT_COLS = CONDITION_COLS + SECTION_COLS
OUTPUT_COLS = ["T", "H", "My", "Q"]


# ============================================================================
# 模型 (复刻 optimization_v2/moe_model.py,本工具自包含,避免循环依赖)
# ============================================================================

class HeteroscedasticMLP(nn.Module):
    """对比基线: 单一 MLP + 异方差头, 同样用 LIFT 风格 NLL 训练。
       — 无 gate, 无专家分歧, σ 只反映 aleatoric (单网络的自评噪声)。
       支持可选 dropout 正则 (默认 0.0 关闭)。
    """
    def __init__(self, input_dim: int = 47, output_dim: int = 4,
                 hidden_dim: int = 128, depth: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        last = input_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, output_dim)
        self.logvar_head = nn.Linear(last, output_dim)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu_head(h)
        log_var = torch.clamp(self.logvar_head(h), min=-10.0, max=5.0)
        sigma = torch.sqrt(torch.exp(log_var) + 1e-6)
        # 接口对齐 MoE 的 5-tuple: gw/mus/log_vars 用 None 占位
        return mu, sigma, None, None, None


class Expert(nn.Module):
    """每个 expert 输出 (μ_k, log σ²_k),log σ² 直接作为对数方差(LIFT 约定)。"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim * 2),
        )

    def forward(self, x):
        h = self.net(x)
        d = h.shape[-1] // 2
        mu = h[..., :d]
        log_var = h[..., d:]                       # LIFT: 直接预测 log σ² 数值稳定
        # 裁剪防爆炸,log σ² ∈ [-10, 5] → σ ∈ [e^-5, e^2.5] ≈ [0.007, 12]
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mu, log_var


class Gate(nn.Module):
    def __init__(self, input_dim: int, n_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, n_experts),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class MoEPredictor(nn.Module):
    def __init__(self, input_dim=47, output_dim=4, n_experts=4,
                 shared_dim=64, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_dim), nn.GELU(),
            nn.Linear(shared_dim, shared_dim), nn.GELU(),
        )
        self.experts = nn.ModuleList([
            Expert(shared_dim, hidden_dim, output_dim) for _ in range(n_experts)
        ])
        self.gate = Gate(shared_dim, n_experts)

    def forward(self, x):
        """前向输出 5-tuple: (μ_mix, σ_mix, gw, mus, log_vars)。

        前 3 项: 给 LIFT-NLL 损失 / 推理 / 下游 CMA-ES 用 (混合分布的均值与标准差)。
        后 2 项: 给 MDN-NLL 损失用 (每个 expert 自己的 μ_k 与 log σ²_k)。

        混合方差遵循 law of total variance:
            σ²_mix = Σ_k π_k σ²_k          ← aleatoric (每个专家自评噪声)
                   + Σ_k π_k (μ_k - μ_mix)²  ← epistemic (专家分歧)
        """
        h = self.shared(x)
        outs = [exp(h) for exp in self.experts]
        mus = torch.stack([m for m, _ in outs], dim=1)             # (B,K,D)
        log_vars = torch.stack([s for _, s in outs], dim=1)        # (B,K,D), 已 clamp
        gw = self.gate(h)                                           # (B,K)

        mu = (mus * gw.unsqueeze(-1)).sum(dim=1)                   # (B,D)
        var_k = torch.exp(log_vars)                                 # (B,K,D)
        aleatoric = (var_k * gw.unsqueeze(-1)).sum(dim=1)
        diff = mus - mu.unsqueeze(1)
        epistemic = ((diff ** 2) * gw.unsqueeze(-1)).sum(dim=1)
        var = aleatoric + epistemic + 1e-6
        sigma = torch.sqrt(var)
        return mu, sigma, gw, mus, log_vars


class MoELoss(nn.Module):
    """LIFT 风格异方差 NLL + MoE 辅助正则。

    主损失 (LIFT Eq. 5):
        L_NLL = mean[ (y - μ)² / σ² + log σ² ]
              = mean[ (y - μ)² · exp(-log σ²) + log σ² ]    (数值更稳)

    σ 是混合分布的标准差(已含 aleatoric + epistemic),NLL 会同时优化:
      - 预测准 → (y-μ)² 小
      - 不确定度合理 → σ 不能太小(否则 (y-μ)²/σ² 爆炸),
                       也不能太大(否则 log σ² 大)
    这正是 LIFT 论文中"自信度"机制的核心。
    """
    def __init__(self, balance_weight=0.01, smooth_weight=0.0, kind="lift"):
        super().__init__()
        self.bw = balance_weight
        self.sw = smooth_weight
        assert kind in ("lift", "mdn"), f"unknown loss kind: {kind}"
        self.kind = kind

    def forward(self, mu, sigma, gw, y, mus=None, log_vars=None):
        # ============ MDN-NLL: 混合似然,gate 自动 load balance ============
        if self.kind == "mdn" and gw is not None and mus is not None and log_vars is not None:
            # log N(y_j | μ_kj, σ²_kj) 在每个 D 维独立累加
            #   mus, log_vars: (B,K,D); y:(B,D); gw:(B,K)
            sq = (y.unsqueeze(1) - mus) ** 2                              # (B,K,D)
            log_norm = -0.5 * (sq * torch.exp(-log_vars)
                               + log_vars
                               + math.log(2.0 * math.pi))                 # (B,K,D)
            log_lik_k = log_norm.sum(dim=-1)                              # (B,K) 4 维联合
            log_pi = torch.log(gw + 1e-12)                                # (B,K)
            nll = -torch.logsumexp(log_pi + log_lik_k, dim=-1).mean()
            ent = -(gw * torch.log(gw + 1e-8)).sum(dim=-1).mean()
            return nll, {"nll": nll.item(), "bal": 0.0, "smooth": ent.item()}

        # ============ LIFT-NLL: 在 (μ_mix, σ_mix) 上做单高斯 NLL ============
        log_var = 2.0 * torch.log(sigma + 1e-8)
        nll = ((y - mu) ** 2 * torch.exp(-log_var) + log_var).mean()
        if gw is None:
            return nll, {"nll": nll.item(), "bal": 0.0, "smooth": 0.0}
        mean_gw = gw.mean(dim=0)
        K = gw.shape[-1]
        bal = ((mean_gw - 1.0 / K) ** 2).sum() * K
        smooth = -(gw * torch.log(gw + 1e-8)).sum(dim=-1).mean()
        loss = nll + self.bw * bal - self.sw * smooth
        return loss, {"nll": nll.item(), "bal": bal.item(), "smooth": smooth.item()}


# ============================================================================
# 数据
# ============================================================================

def load_normalized(npz_path: Path, use_cp: bool = False):
    """加载归一化后的训练矩阵。

    use_cp=False (默认): 输入 47 维 = 3 工况 + 22 chord + 22 twist sections
    use_cp=True : 输入 11 维 = 3 工况 + 4 chord (索引 0,7,14,21) + 4 twist (同)
                  - 模拟 V1 风格的 CP 输入,验证维度降低对 σ OOD 信号的影响
    """
    z = np.load(npz_path, allow_pickle=True)
    arr = z["X_y"]
    cols = list(z["columns"])

    if use_cp:
        # 从 22 sections 中均匀选 4 个索引作为"伪 CP"
        cp_idx_chord = [f"chord_{i}" for i in (0, 7, 14, 21)]
        cp_idx_twist = [f"twist_{i}" for i in (0, 7, 14, 21)]
        in_cols_active = CONDITION_COLS + cp_idx_chord + cp_idx_twist
    else:
        in_cols_active = INPUT_COLS

    in_idx = [cols.index(c) for c in in_cols_active if c in cols]
    out_idx = [cols.index(c) for c in OUTPUT_COLS if c in cols]
    X = arr[:, in_idx]
    Y = arr[:, out_idx]
    return X.astype(np.float32), Y.astype(np.float32), [cols[i] for i in in_idx], [cols[i] for i in out_idx]


def split_by_geometry(X, Y, geom_idx, test_ratio=0.2, seed=42):
    """按几何索引分组划分,避免训练/测试集间几何泄露。"""
    rng = np.random.RandomState(seed)
    uniq = np.unique(geom_idx)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_ratio))
    test_geoms = set(uniq[:n_test])
    test_mask = np.array([g in test_geoms for g in geom_idx])
    return (X[~test_mask], Y[~test_mask], X[test_mask], Y[test_mask])


# ============================================================================
# 训练单个 trial
# ============================================================================

@dataclass
class TrialConfig:
    arch: str = "moe"               # "moe" | "mlp"
    loss_kind: str = "lift"         # "lift" | "mdn" (mdn 仅对 moe 生效)
    n_experts: int = 4              # 仅 moe 使用
    hidden_dim: int = 128
    shared_dim: int = 64            # 仅 moe 使用
    depth: int = 3                  # 仅 mlp 使用
    dropout: float = 0.0            # 仅 mlp 使用
    lr: float = 1e-3
    weight_decay: float = 1e-3
    balance_weight: float = 0.1     # 仅 lift 损失下生效
    smooth_weight: float = 0.01     # 仅 lift 损失下生效
    batch_size: int = 64
    epochs: int = 1500
    seed: int = 42


def train_one_trial(cfg: TrialConfig, X_tr, Y_tr, X_te, Y_te, device, verbose=False):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.arch == "mlp":
        model = HeteroscedasticMLP(
            input_dim=X_tr.shape[1],
            output_dim=Y_tr.shape[1],
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
        ).to(device)
    else:
        model = MoEPredictor(
            input_dim=X_tr.shape[1],
            output_dim=Y_tr.shape[1],
            n_experts=cfg.n_experts,
            shared_dim=cfg.shared_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(device)

    crit = MoELoss(cfg.balance_weight, cfg.smooth_weight, kind=cfg.loss_kind)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=100, min_lr=1e-6)

    X_tr_t = torch.tensor(X_tr); Y_tr_t = torch.tensor(Y_tr)
    X_te_t = torch.tensor(X_te).to(device); Y_te_t = torch.tensor(Y_te).to(device)
    loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=cfg.batch_size, shuffle=True)

    hist = {"epoch": [], "train": [], "test": [], "test_mae": [], "test_r2": []}
    best = {"loss": float("inf"), "epoch": 0, "state": None}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        s = 0.0; n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, sig, gw, mus_k, log_vars_k = model(xb)
            loss, _ = crit(mu, sig, gw, yb, mus=mus_k, log_vars=log_vars_k)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            s += loss.item(); n += 1
        tr = s / n

        model.eval()
        with torch.no_grad():
            mu_te, sig_te, gw_te, mus_te, lv_te = model(X_te_t)
            te, _ = crit(mu_te, sig_te, gw_te, Y_te_t, mus=mus_te, log_vars=lv_te)
            err = (mu_te - Y_te_t).cpu().numpy()
            mae = float(np.mean(np.abs(err)))
            y = Y_te_t.cpu().numpy()
            ss_res = ((mu_te.cpu().numpy() - y) ** 2).sum(axis=0)
            ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
            r2 = 1 - ss_res / ss_tot
        sched.step(te.item())

        hist["epoch"].append(ep)
        hist["train"].append(tr)
        hist["test"].append(te.item())
        hist["test_mae"].append(mae)
        hist["test_r2"].append(r2.tolist())

        if te.item() < best["loss"]:
            best["loss"] = te.item()
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and ep % 200 == 0:
            print(f"    ep{ep:5d} train={tr:.3f} test={te.item():.3f} mae={mae:.3f} r2={r2}")

    return {
        "history": hist,
        "best_epoch": best["epoch"],
        "best_test_loss": best["loss"],
        "final_mae": hist["test_mae"][-1],
        "final_r2": hist["test_r2"][-1],
        "model_state": best["state"],
    }


# ============================================================================
# Sweep 主控
# ============================================================================

DEFAULT_GRID_MOE = {
    # LIFT 路线发现: lr=1e-3 几乎是必然,balance_weight 应小 → 收窄网格
    "n_experts":      [2, 4, 6, 8],
    "hidden_dim":     [64, 128, 256],
    "shared_dim":     [32, 64, 128],
    "lr":             [1e-3, 5e-4],
    "weight_decay":   [1e-4, 1e-3],
    "balance_weight": [0.0, 0.01, 0.1],
    "smooth_weight":  [0.0, 0.01],
}

DEFAULT_GRID_MLP = {
    # MLP 基线网格: 无 gate / experts, 但 trunk 深度可调
    "hidden_dim":     [64, 128, 256, 512],
    "depth":          [2, 3, 4, 5],
    "dropout":        [0.0],
    "lr":             [1e-3, 5e-4],
    "weight_decay":   [1e-4, 1e-3],
}

# 专用于 dropout/lr 调参的 sub-sweep 网格
GRID_MLP_DROPOUT_LR = {
    "hidden_dim":     [256],         # 锁定 best
    "depth":          [2],           # 锁定 best
    "dropout":        [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "lr":             [1e-3, 5e-4, 2e-4, 1e-4],
    "weight_decay":   [1e-4],
}

# 向后兼容
DEFAULT_GRID = DEFAULT_GRID_MOE


def sample_grid(grid: dict, n: int, seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    all_combos = list(itertools.product(*grid.values()))
    if len(all_combos) <= n:
        return [dict(zip(grid.keys(), c)) for c in all_combos]
    chosen = rng.sample(all_combos, n)
    return [dict(zip(grid.keys(), c)) for c in chosen]


def run_sweep(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    arch = args.arch
    if arch not in ("moe", "mlp"):
        raise ValueError(f"--arch 应为 'moe' 或 'mlp', 当前 {arch}")

    X, Y, in_cols, out_cols = load_normalized(Path(args.data), use_cp=args.use_cp)
    print(f"[data] arch={arch} use_cp={args.use_cp} X={X.shape} Y={Y.shape}, inputs={len(in_cols)}, outputs={out_cols}")

    # 几何索引 (按原 csv 推断,若无则随机分)
    if args.geom_idx_csv and Path(args.geom_idx_csv).exists():
        gi = pd.read_csv(args.geom_idx_csv)["geom_idx"].to_numpy()
        if len(gi) != len(X):
            print("[warn] geom_idx 长度不匹配,退回随机划分")
            gi = np.arange(len(X)) % max(1, len(X) // 50)
    else:
        gi = np.arange(len(X)) % max(1, len(X) // 50)
    X_tr, Y_tr, X_te, Y_te = split_by_geometry(X, Y, gi)
    print(f"[split] train={X_tr.shape[0]}, test={X_te.shape[0]}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.grid and args.grid.endswith(".json"):
        grid = json.load(open(args.grid))
    elif args.grid == "dropout_lr":
        grid = GRID_MLP_DROPOUT_LR
    else:
        grid = DEFAULT_GRID_MLP if arch == "mlp" else DEFAULT_GRID_MOE
    trials = sample_grid(grid, n=args.n_trials, seed=args.seed)
    print(f"[sweep] arch={arch} {len(trials)} trials × {args.epochs} epochs on {device}")

    records = []
    histories = {}
    best_overall = {"loss": float("inf"), "idx": -1, "trial": None, "state": None}

    t0 = time.time()
    for i, params in enumerate(trials):
        cfg = TrialConfig(arch=arch, loss_kind=args.loss, **params, epochs=args.epochs)
        print(f"\n[trial {i+1}/{len(trials)}] {params}")
        ts = time.time()
        r = train_one_trial(cfg, X_tr, Y_tr, X_te, Y_te, device, verbose=args.verbose)
        elapsed = time.time() - ts
        rec = {
            "trial": i,
            **params,
            "best_epoch": r["best_epoch"],
            "best_test_loss": r["best_test_loss"],
            "final_mae": r["final_mae"],
            **{f"r2_{c}": v for c, v in zip(out_cols, r["final_r2"])},
            "time_s": elapsed,
        }
        records.append(rec)
        histories[i] = r["history"]
        print(f"  ✓ best_loss={r['best_test_loss']:.4f} mae={r['final_mae']:.4f} "
              f"r2={r['final_r2']} ({elapsed:.0f}s)")

        if r["best_test_loss"] < best_overall["loss"]:
            best_overall.update(loss=r["best_test_loss"], idx=i, trial=rec, state=r["model_state"])

    total = time.time() - t0
    print(f"\n[done] {len(trials)} trials in {total:.0f}s, best trial = #{best_overall['idx']}")

    # 落盘
    df = pd.DataFrame(records).sort_values("best_test_loss").reset_index(drop=True)
    df.to_csv(out_dir / "results.csv", index=False)
    np.savez_compressed(out_dir / "histories.npz", **{f"trial_{k}": np.array([
        v["train"], v["test"], v["test_mae"]
    ]) for k, v in histories.items()})

    # 保存最优模型
    if best_overall["state"] is not None:
        best_dir = out_dir / "best_trial"
        best_dir.mkdir(exist_ok=True)
        torch.save(best_overall["state"], best_dir / "model.pth")
        with open(best_dir / "config.json", "w") as f:
            json.dump(best_overall["trial"], f, indent=2)
        # 残差/对角图
        plot_best_diagnostics(best_overall, X_te, Y_te, out_cols, device, best_dir, args)

    plot_sweep_overview(df, histories, out_cols, out_dir)
    print(f"[done] 结果 → {out_dir}/")


# ============================================================================
# 综合可视化
# ============================================================================

def plot_sweep_overview(df: pd.DataFrame, histories: Dict[int, dict],
                         out_cols: List[str], out_dir: Path):
    """8-panel 综合图。"""
    fig = plt.figure(figsize=(16, 11), dpi=120)
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # (1) 训练曲线 (所有 trial)
    ax = fig.add_subplot(gs[0, 0])
    for k, h in histories.items():
        ax.plot(h["epoch"], h["test"], lw=0.6, alpha=0.5, color="C0")
    # 最优 trial 突出
    best_idx = df.iloc[0]["trial"] if len(df) else 0
    h = histories.get(int(best_idx))
    if h:
        ax.plot(h["epoch"], h["test"], lw=1.6, color="C3", label=f"best #{best_idx}")
    ax.set_xlabel("epoch"); ax.set_ylabel("test loss"); ax.set_yscale("log")
    ax.set_title("Test loss curves"); ax.legend(fontsize=8)

    # (2) trial 排名
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(len(df)), df["best_test_loss"], color="#4a6ea5")
    ax.set_xlabel("trial rank"); ax.set_ylabel("best test loss")
    ax.set_title(f"Trials ranked  (best={df['best_test_loss'].min():.3f})")

    # (3) 各超参的边际效应
    ax = fig.add_subplot(gs[0, 2:])
    hp_cols = ["n_experts", "hidden_dim", "shared_dim", "lr", "weight_decay",
               "balance_weight", "smooth_weight"]
    available = [c for c in hp_cols if c in df.columns]
    means = {c: df.groupby(c)["best_test_loss"].mean().to_dict() for c in available}
    text = " | ".join(
        f"{c}: " + ", ".join(f"{k}→{v:.2f}" for k, v in sorted(d.items()))
        for c, d in means.items()
    )
    ax.text(0.02, 0.5, text, fontsize=8, va="center", wrap=True,
            transform=ax.transAxes, family="monospace")
    ax.axis("off"); ax.set_title("Hyperparam marginal mean(test loss)")

    # (4) 输出列 R²
    ax = fig.add_subplot(gs[1, 0])
    r2_cols = [f"r2_{c}" for c in out_cols if f"r2_{c}" in df.columns]
    if r2_cols:
        best_row = df.iloc[0]
        ax.bar(out_cols[:len(r2_cols)], [best_row[c] for c in r2_cols], color="#3a8a3a")
        ax.set_ylim(0, 1.02); ax.set_ylabel("R²")
        ax.set_title(f"Best trial #{int(best_row['trial'])} R²")
        for i, c in enumerate(r2_cols):
            ax.text(i, best_row[c] + 0.01, f"{best_row[c]:.3f}", ha="center", fontsize=8)

    # (5) 各输出列 R² 分布
    ax = fig.add_subplot(gs[1, 1])
    if r2_cols:
        data = [df[c].clip(-1, 1).values for c in r2_cols]
        ax.boxplot(data, labels=out_cols[:len(r2_cols)])
        ax.set_ylim(-0.1, 1.02); ax.set_ylabel("R² across trials")
        ax.set_title("R² distribution")
        ax.axhline(0, color="red", lw=0.5, ls="--")

    # (6) lr vs loss 散点 — color 用第二个超参 (MoE: n_experts; MLP: depth)
    ax = fig.add_subplot(gs[1, 2])
    if "lr" in df.columns:
        color_col = next((c for c in ["n_experts", "depth", "hidden_dim"] if c in df.columns), None)
        if color_col:
            sc = ax.scatter(df["lr"], df["best_test_loss"], c=df[color_col], cmap="viridis", s=40)
            plt.colorbar(sc, ax=ax, label=color_col)
            ax.set_title(f"lr vs loss (color: {color_col})")
        else:
            ax.scatter(df["lr"], df["best_test_loss"], s=40, color="#4a6ea5")
            ax.set_title("lr vs loss")
        ax.set_xscale("log"); ax.set_xlabel("lr"); ax.set_ylabel("best loss")

    # (7) hidden × 第二超参 heatmap
    ax = fig.add_subplot(gs[1, 3])
    second = next((c for c in ["n_experts", "depth"] if c in df.columns), None)
    if "hidden_dim" in df.columns and second:
        piv = df.pivot_table(index="hidden_dim", columns=second,
                              values="best_test_loss", aggfunc="mean")
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel(second); ax.set_ylabel("hidden_dim")
        ax.set_title("Mean test loss")
        plt.colorbar(im, ax=ax)

    # (8) MAE 排名
    ax = fig.add_subplot(gs[2, 0])
    if "final_mae" in df.columns:
        ax.bar(range(len(df)), df["final_mae"], color="#a85d5d")
        ax.set_xlabel("trial rank"); ax.set_ylabel("test MAE (normalized)")
        ax.set_title("Final MAE")

    # (9) 训练时间分布
    ax = fig.add_subplot(gs[2, 1])
    if "time_s" in df.columns:
        ax.hist(df["time_s"], bins=12, color="#888888")
        ax.set_xlabel("trial time (s)"); ax.set_ylabel("count")
        ax.set_title("Trial wall time")

    # (10) Top-10 trial 超参表 — 动态选列
    ax = fig.add_subplot(gs[2, 2:])
    base_cols = ["trial", "best_test_loss"]
    arch_cols = [c for c in ["n_experts", "depth", "hidden_dim", "shared_dim",
                              "lr", "weight_decay", "balance_weight"] if c in df.columns]
    show_cols = base_cols + arch_cols
    top = df.head(10)[show_cols]
    ax.axis("off")
    tab = ax.table(cellText=np.round(top.values, 4), colLabels=top.columns,
                   loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1, 1.1)
    ax.set_title("Top 10 trials", pad=10)

    fig.suptitle("MoE Sweep — Overview", fontsize=13)
    fig.savefig(out_dir / "sweep_overview.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] sweep_overview → {out_dir / 'sweep_overview.png'}")


def plot_best_diagnostics(best, X_te, Y_te, out_cols, device, out_dir: Path, args):
    """对最优模型在测试集上的诊断:对角图 + 残差 + 不确定度校准 + 专家分配(仅 moe)。"""
    arch = args.arch
    if arch == "mlp":
        cfg_keys = ["hidden_dim", "depth", "lr", "weight_decay"]
        cfg_dict = {k: best["trial"][k] for k in cfg_keys if k in best["trial"]}
        cfg = TrialConfig(arch="mlp", **cfg_dict, epochs=args.epochs)
        model = HeteroscedasticMLP(
            input_dim=X_te.shape[1], output_dim=Y_te.shape[1],
            hidden_dim=cfg.hidden_dim, depth=cfg.depth,
        ).to(device)
    else:
        cfg_keys = ["n_experts", "hidden_dim", "shared_dim", "lr", "weight_decay",
                    "balance_weight", "smooth_weight"]
        cfg_dict = {k: best["trial"][k] for k in cfg_keys}
        cfg = TrialConfig(arch="moe", **cfg_dict, epochs=args.epochs)
        model = MoEPredictor(
            input_dim=X_te.shape[1], output_dim=Y_te.shape[1],
            n_experts=cfg.n_experts, shared_dim=cfg.shared_dim, hidden_dim=cfg.hidden_dim,
        ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    model.eval()

    with torch.no_grad():
        mu, sig, gw, _, _ = model(torch.tensor(X_te).to(device))
    mu = mu.cpu().numpy(); sig = sig.cpu().numpy()
    gw = gw.cpu().numpy() if gw is not None else None

    n_out = len(out_cols)
    fig, axes = plt.subplots(4, n_out, figsize=(3.2 * n_out, 12), dpi=120)

    for i, col in enumerate(out_cols):
        # ---- 行 0:对角图 ----
        ax = axes[0, i]
        ax.scatter(Y_te[:, i], mu[:, i], s=8, alpha=0.6, c="#3a6ea5")
        lo, hi = min(Y_te[:, i].min(), mu[:, i].min()), max(Y_te[:, i].max(), mu[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8)
        ss_res = ((mu[:, i] - Y_te[:, i]) ** 2).sum()
        ss_tot = ((Y_te[:, i] - Y_te[:, i].mean()) ** 2).sum() + 1e-12
        r2 = 1 - ss_res / ss_tot
        ax.set_title(f"{col}  R²={r2:.3f}"); ax.set_xlabel("true"); ax.set_ylabel("pred")

        # ---- 行 1:残差直方图 ----
        ax = axes[1, i]
        res = mu[:, i] - Y_te[:, i]
        ax.hist(res, bins=30, color="#aa4a4a", alpha=0.8)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"residual {col}  μ={res.mean():.2e} σ={res.std():.2e}")

        # ---- 行 2:σ vs |error| 散点 ----
        ax = axes[2, i]
        ax.scatter(sig[:, i], np.abs(res), s=8, alpha=0.5, c="#3a6ea5")
        m = max(sig[:, i].max(), np.abs(res).max())
        ax.plot([0, m], [0, m], "r--", lw=0.6, label="perfect")
        ax.set_xlabel("predicted σ"); ax.set_ylabel("|error|")
        ax.set_title(f"σ-scatter {col}")
        ax.legend(fontsize=7)

        # ---- 行 3:reliability diagram (LIFT 风格) ----
        # 按 σ 分 10 箱,每箱计算 RMSE 与平均 σ,理想情况 RMSE ≈ σ_bin
        ax = axes[3, i]
        n_bins = 10
        order = np.argsort(sig[:, i])
        bins = np.array_split(order, n_bins)
        sig_bin = np.array([sig[b, i].mean() for b in bins])
        rmse_bin = np.array([np.sqrt(((mu[b, i] - Y_te[b, i]) ** 2).mean()) for b in bins])
        # 标准化 calibration error (ECE-like): 平均 |rmse - σ| / σ_全集
        ece = float(np.mean(np.abs(rmse_bin - sig_bin)) / (sig[:, i].mean() + 1e-8))
        ax.plot(sig_bin, rmse_bin, "o-", color="#c44d4d", lw=1.4, ms=4)
        m = max(sig_bin.max(), rmse_bin.max())
        ax.plot([0, m], [0, m], "k--", lw=0.6, alpha=0.5)
        ax.set_xlabel("avg predicted σ (bin)"); ax.set_ylabel("RMSE in bin")
        ax.set_title(f"reliability {col}  ECE={ece:.2f}")

    fig.suptitle("Diagnostics — LIFT-style heteroscedastic MoE", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    # 专家分配热图(仅 moe)
    if gw is not None:
        fig2, ax = plt.subplots(figsize=(6, 4), dpi=120)
        im = ax.imshow(gw.T, aspect="auto", cmap="magma")
        ax.set_xlabel("sample idx"); ax.set_ylabel("expert")
        ax.set_title("Expert gate weights (test set)")
        plt.colorbar(im, ax=ax)
        fig2.savefig(out_dir / "gate_assignment.png", bbox_inches="tight")
        plt.close(fig2)
        print(f"[plot] gate → {out_dir / 'gate_assignment.png'}")

    print(f"[plot] diagnostics → {out_dir / 'diagnostics.png'}")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use-cp", action="store_true",
                   help="使用 11 维 CP 输入 (3 工况 + 4 chord + 4 twist 均匀采样) 而非 47 维 sections")
    p.add_argument("--arch", choices=["moe", "mlp"], default="moe",
                   help="MoE (默认): K-expert + gate + LIFT 不确定度;  MLP: 单 MLP + 异方差头作对比基线")
    p.add_argument("--loss", choices=["lift", "mdn"], default="lift",
                   help="lift (默认): 在 (μ_mix, σ_mix) 上单高斯 NLL + balance/smooth 正则; "
                        "mdn: 混合高斯似然 logsumexp_k, gate 自动 load balance, 不需要 balance_weight")
    p.add_argument("--data", help="data_normalized.npz")
    p.add_argument("--norm", help="normalizer.pkl (供 inverse_transform 还原)")
    p.add_argument("--geom-idx-csv", help="原始 csv (用于按几何分割)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--n-trials", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", default=None, help="自定义 grid json")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--plot-only", action="store_true",
                   help="仅根据 out 目录已有 results.csv + histories.npz 重出图")
    args = p.parse_args()

    if args.plot_only:
        out_dir = Path(args.out)
        df = pd.read_csv(out_dir / "results.csv")
        hz = np.load(out_dir / "histories.npz")
        histories = {}
        for k in hz.files:
            arr = hz[k]
            i = int(k.split("_")[1])
            histories[i] = {"epoch": list(range(1, arr.shape[1] + 1)),
                            "train": arr[0].tolist(), "test": arr[1].tolist(),
                            "test_mae": arr[2].tolist()}
        out_cols = [c.replace("r2_", "") for c in df.columns if c.startswith("r2_")]
        plot_sweep_overview(df, histories, out_cols, out_dir)
        return

    if not args.data:
        raise SystemExit("需要 --data data_normalized.npz")
    run_sweep(args)


if __name__ == "__main__":
    main()
