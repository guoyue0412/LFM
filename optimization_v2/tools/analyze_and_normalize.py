"""数据归一化分析 + 应用器

用法:
    # 1) 分析 + 决策 + 拟合归一化器 + 出图
    python -m optimization_v2.tools.analyze_and_normalize \
        --csv ./data_for_train/processed_lhs/dataset_v2_20260615.csv \
        --out ./data_for_train/processed_lhs/norm \
        --plot

    # 2) 仅应用已保存的 normalizer.pkl 到新数据
    python -m optimization_v2.tools.analyze_and_normalize \
        --apply ./data_for_train/processed_lhs/norm/normalizer.pkl \
        --csv new_data.csv --out_csv new_data_normalized.csv

==== 量纲范围决策规则 (针对每一列) ====
1. **零方差列**: std < 1e-12        → 跳过 (常量,无信息)
2. **强右偏 + 全正**: skew > 2 且 min > 0
   → log1p + standard           (适用: T, Q 等推力/扭矩,常数倍量级波动)
3. **重尾**: |skew| > 1.5 或 |kurtosis| > 8
   → quantile (output: normal)  (适用: H, My 等可能含异常值的输出)
4. **多峰离散**: nunique <= 20 且 nunique/len < 0.05
   → standard                   (适用: RPM/Angle 离散网格,无需 quantile)
5. **默认**: 单峰近似 + 弱偏
   → standard                   (适用: chord, twist 等几何 sections)

输入列 X (47): RPM, WIND, ANGLE, chord_0..21, twist_0..21
输出列 y (4) : T, H, My, Q

报告输出:
    norm/normalizer.pkl       — sklearn-compatible 双向变换器
    norm/decisions.json       — 每列决策 + 统计量
    norm/distribution_grid.png — 全列分布图 (raw + transformed)
    norm/qq_outputs.png       — 输出列 Q-Q 图 (检查归一化后正态性)
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler, QuantileTransformer


CONDITION_COLS = ["RPM", "WIND", "ANGLE"]
SECTION_COLS = [f"chord_{i}" for i in range(22)] + [f"twist_{i}" for i in range(22)]
INPUT_COLS = CONDITION_COLS + SECTION_COLS
OUTPUT_COLS = ["T", "H", "My", "Q"]

PLOT_STYLE = dict(figsize_per=2.4, dpi=120)


# ============================================================================
# 决策逻辑
# ============================================================================

@dataclass
class ColumnStats:
    n: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    skew: float
    kurtosis: float
    nunique: int


def column_stats(x: np.ndarray) -> ColumnStats:
    x = x[np.isfinite(x)]
    return ColumnStats(
        n=int(x.size),
        mean=float(np.mean(x)) if x.size else float("nan"),
        std=float(np.std(x)) if x.size else float("nan"),
        min=float(np.min(x)) if x.size else float("nan"),
        max=float(np.max(x)) if x.size else float("nan"),
        median=float(np.median(x)) if x.size else float("nan"),
        skew=float(stats.skew(x)) if x.size > 2 else 0.0,
        kurtosis=float(stats.kurtosis(x)) if x.size > 2 else 0.0,
        nunique=int(np.unique(x).size),
    )


def decide_method(s: ColumnStats) -> str:
    """根据统计量返回 standard / quantile / log_standard / skip。"""
    if s.std < 1e-12 or s.n == 0:
        return "skip"
    if s.skew > 2.0 and s.min > 0:
        return "log_standard"
    if abs(s.skew) > 1.5 or abs(s.kurtosis) > 8.0:
        return "quantile"
    if s.nunique <= 20 and s.n > 200 and (s.nunique / s.n) < 0.05:
        return "standard"  # 离散网格不需要 quantile
    return "standard"


# ============================================================================
# 双向归一化器 (sklearn 兼容)
# ============================================================================

class HybridNormalizer:
    """每列独立选择 standard / quantile / log_standard / skip。"""

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.methods: Dict[str, str] = {}
        self.stats: Dict[str, ColumnStats] = {}
        self.transformers: Dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "HybridNormalizer":
        for col in self.columns:
            if col not in df.columns:
                raise KeyError(f"列 '{col}' 不在 DataFrame 中")
            x = df[col].to_numpy(dtype=np.float64)
            s = column_stats(x)
            method = decide_method(s)
            self.stats[col] = s
            self.methods[col] = method
            self.transformers[col] = self._build_transformer(method, x)
        return self

    @staticmethod
    def _build_transformer(method: str, x: np.ndarray):
        x_finite = x[np.isfinite(x)].reshape(-1, 1)
        if method == "skip":
            return None
        if method == "standard":
            return StandardScaler().fit(x_finite)
        if method == "log_standard":
            # log1p 后再 standard, 输入须严格正
            xl = np.log1p(x_finite)
            return ("log1p", StandardScaler().fit(xl))
        if method == "quantile":
            n_q = min(1000, max(100, x_finite.shape[0] // 4))
            return QuantileTransformer(
                n_quantiles=n_q, output_distribution="normal",
                subsample=int(1e6), random_state=42,
            ).fit(x_finite)
        raise ValueError(f"unknown method {method}")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.columns:
            x = df[col].to_numpy(dtype=np.float64).reshape(-1, 1)
            t = self.transformers[col]
            method = self.methods[col]
            if t is None:                            # skip 列 → 输出 0 占位
                cols.append(np.zeros_like(x))
            elif method == "log_standard":
                _, scaler = t
                cols.append(scaler.transform(np.log1p(x)))
            else:
                cols.append(t.transform(x))
        return np.concatenate(cols, axis=1)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.float64)
        for i, col in enumerate(self.columns):
            x = arr[:, i:i+1]
            t = self.transformers[col]
            method = self.methods[col]
            if t is None:
                out[:, i:i+1] = self.stats[col].mean
            elif method == "log_standard":
                _, scaler = t
                xl = scaler.inverse_transform(x)
                out[:, i:i+1] = np.expm1(xl)
            else:
                out[:, i:i+1] = t.inverse_transform(x)
        return out


# ============================================================================
# 可视化
# ============================================================================

def plot_distribution_grid(df: pd.DataFrame, normalizer: HybridNormalizer,
                            arr: np.ndarray, columns: List[str], out_path: Path):
    """每列两行: raw 直方图 + 归一化后直方图。"""
    n = len(columns)
    n_col = 6
    n_row = (n + n_col - 1) // n_col * 2  # 每列占 2 行 (raw + transformed)

    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(n_col * PLOT_STYLE["figsize_per"], n_row * PLOT_STYLE["figsize_per"] * 0.65),
        dpi=PLOT_STYLE["dpi"],
    )

    for i, col in enumerate(columns):
        r_raw = (i // n_col) * 2
        r_trf = r_raw + 1
        c = i % n_col

        ax_raw = axes[r_raw, c]
        ax_trf = axes[r_trf, c]

        x = df[col].to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        ax_raw.hist(x, bins=40, color="#3a6ea5", alpha=0.8)
        method = normalizer.methods.get(col, "?")
        ax_raw.set_title(f"{col}\n[{method}]", fontsize=8)
        ax_raw.tick_params(axis='both', labelsize=6)

        z = arr[:, i]
        z = z[np.isfinite(z)]
        ax_trf.hist(z, bins=40, color="#c44d4d", alpha=0.8)
        ax_trf.set_xlim(-4, 4)
        ax_trf.tick_params(axis='both', labelsize=6)

    # 关闭剩余子图
    total = n_row * n_col
    used = (((n - 1) // n_col) + 1) * 2 * n_col
    for k in range(used, total):
        axes.flatten()[k].axis("off")

    fig.suptitle("Distribution grid — top: raw, bottom: normalized", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_qq_outputs(df: pd.DataFrame, normalizer: HybridNormalizer,
                    arr: np.ndarray, columns: List[str], out_path: Path):
    """输出列归一化后 Q-Q 图,检查是否接近 N(0,1)。"""
    output_cols = [c for c in columns if c in OUTPUT_COLS]
    if not output_cols:
        return
    fig, axes = plt.subplots(1, len(output_cols), figsize=(3 * len(output_cols), 3), dpi=120)
    if len(output_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, output_cols):
        idx = columns.index(col)
        z = arr[:, idx]
        z = z[np.isfinite(z)]
        stats.probplot(z, dist="norm", plot=ax)
        ax.set_title(f"Q-Q: {col}  [{normalizer.methods[col]}]", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 主流程
# ============================================================================

def run_analyze(csv: Path, out_dir: Path, plot: bool):
    df = pd.read_csv(csv)
    print(f"[加载] {csv}: {len(df)} 行, {len(df.columns)} 列")

    use_cols = [c for c in INPUT_COLS + OUTPUT_COLS if c in df.columns]
    missing = [c for c in INPUT_COLS + OUTPUT_COLS if c not in df.columns]
    if missing:
        print(f"[警告] 缺失列: {missing[:5]}...")

    norm = HybridNormalizer(use_cols).fit(df)
    arr = norm.transform(df)

    # 决策汇总
    decisions = {
        col: {
            "method": norm.methods[col],
            "stats": asdict(norm.stats[col]),
        } for col in use_cols
    }
    method_counts = pd.Series([norm.methods[c] for c in use_cols]).value_counts().to_dict()
    print(f"[决策] {method_counts}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) 归一化器
    pkl_path = out_dir / "normalizer.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"normalizer": norm, "columns": use_cols}, f)
    print(f"[落盘] normalizer  → {pkl_path}")

    # 2) 决策报告
    json_path = out_dir / "decisions.json"
    with open(json_path, "w") as f:
        json.dump({
            "method_counts": method_counts,
            "columns": decisions,
        }, f, indent=2, ensure_ascii=False)
    print(f"[落盘] decisions   → {json_path}")

    # 3) 归一化后的训练数据 (供 sweep 直接用)
    norm_csv = out_dir / "data_normalized.npz"
    np.savez_compressed(norm_csv, X_y=arr, columns=np.array(use_cols))
    print(f"[落盘] data        → {norm_csv}")

    if plot:
        p1 = out_dir / "distribution_grid.png"
        plot_distribution_grid(df, norm, arr, use_cols, p1)
        print(f"[落盘] dist grid   → {p1}")
        p2 = out_dir / "qq_outputs.png"
        plot_qq_outputs(df, norm, arr, use_cols, p2)
        print(f"[落盘] qq outputs  → {p2}")

    return norm


def run_apply(normalizer_path: Path, csv: Path, out_csv: Path):
    with open(normalizer_path, "rb") as f:
        bundle = pickle.load(f)
    norm: HybridNormalizer = bundle["normalizer"]
    cols = bundle["columns"]
    df = pd.read_csv(csv)
    arr = norm.transform(df[cols])
    out = pd.DataFrame(arr, columns=cols)
    out.to_csv(out_csv, index=False)
    print(f"[apply] {csv} → {out_csv}  ({arr.shape})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", help="分析模式输出目录 (含 normalizer.pkl 等)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--apply", help="如指定,则使用该 normalizer.pkl 对 --csv 做变换")
    p.add_argument("--out_csv", help="apply 模式的输出 csv")
    args = p.parse_args()

    if args.apply:
        if not args.out_csv:
            raise SystemExit("--apply 需要 --out_csv")
        run_apply(Path(args.apply), Path(args.csv), Path(args.out_csv))
    else:
        if not args.out:
            raise SystemExit("分析模式需要 --out")
        run_analyze(Path(args.csv), Path(args.out), args.plot)


if __name__ == "__main__":
    main()
