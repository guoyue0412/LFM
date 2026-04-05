"""螺旋桨气动力预测模型 — 数据处理

支持两种数据源:
  1. 原始 pkl 文件（递归扫描 data*.pkl 和 sample_*.pkl）
  2. 已有 dealed_data.xlsx（旧流程兼容）

几何模式:
  "none"           → 仅工况 (RPM, WIND, ANGLE)
  "control_points" → 工况 + 8个控制点
  "sections"       → 工况 + 22 chord + 22 twist
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import Config
from plot_style import apply_style, COLORS, savefig

apply_style()


# ================================================================
#  几何数据提取
# ================================================================

def _extract_geometry(data: dict, cfg: Config) -> dict:
    """从一个 sample dict 中提取几何列，返回 {列名: 值} 字典。"""
    if cfg.geometry_mode == "none":
        return {}

    geo = {}

    if cfg.geometry_mode == "control_points":
        cp = data.get("control_points")
        if cp is not None:
            cp = np.asarray(cp, dtype=float).ravel()
            for i in range(min(len(cp), cfg.n_control_points)):
                geo[f"cp_{i}"] = float(cp[i])
        return geo

    if cfg.geometry_mode == "sections":
        geometry = data.get("geometry")
        if geometry is not None:
            geometry = np.asarray(geometry, dtype=float)
            n = min(len(geometry), cfg.n_sections)
            for i in range(n):
                geo[f"chord_{i}"] = float(geometry[i, 1])
                geo[f"twist_{i}"] = float(geometry[i, 2])
        return geo

    return geo


# ================================================================
#  工况数据提取
# ================================================================

def _extract_from_case_results(case_dict: dict, pattern: re.Pattern,
                                records: list, skipped_log: list,
                                source: str = "",
                                geometry: dict = None) -> int:
    """从一个 case_results 字典中提取工况数据，附加几何列，返回提取条数。"""
    count = 0
    required_cols = {"Thrust", "Torque"}
    if geometry is None:
        geometry = {}

    for key, val in case_dict.items():
        if not isinstance(val, pd.DataFrame):
            continue
        match = pattern.search(key)
        if not match:
            continue

        rpm = float(match.group(1))
        wind = float(match.group(2))
        angle = float(match.group(3))

        fy_col = "Thrust_y" if "Thrust_y" in val.columns else "Trust_y"
        fz_col = "Thrust_z" if "Thrust_z" in val.columns else "Trust_z"

        missing = required_cols - set(val.columns)
        if fy_col not in val.columns:
            missing.add("Thrust_y/Trust_y")
        if fz_col not in val.columns:
            missing.add("Thrust_z/Trust_z")

        if missing:
            skipped_log.append({
                "source": source, "case": key,
                "RPM": rpm, "WIND": wind, "ANGLE": angle,
                "reason": f"缺少列: {sorted(missing)}",
                "available_columns": sorted(val.columns.tolist()),
            })
            continue

        if val["Thrust"].isna().all() or len(val) == 0:
            skipped_log.append({
                "source": source, "case": key,
                "RPM": rpm, "WIND": wind, "ANGLE": angle,
                "reason": "数据为空或全为 NaN",
                "available_columns": sorted(val.columns.tolist()),
            })
            continue

        my_col = "Moment_y" if "Moment_y" in val.columns else None
        mz_col = "Moment_z" if "Moment_z" in val.columns else None

        record = {
            "RPM": rpm,
            "WIND": wind,
            "ANGLE": angle,
            **geometry,
            "Fx": -val["Thrust"].mean(),
            "Fy": val[fy_col].mean(),
            "Fz": val[fz_col].mean(),
            "Torque": -val["Torque"].mean(),
            "My": val[my_col].mean() if my_col else 0.0,
            "Mz": val[mz_col].mean() if mz_col else 0.0,
        }
        records.append(record)
        count += 1
    return count


# ================================================================
#  数据加载
# ================================================================

def load_raw_pkl(data_dir: str, cfg: Config) -> pd.DataFrame:
    """从 pkl 文件夹批量加载仿真数据。

    自动递归扫描 data_dir 下所有 data*.pkl / sample_*.pkl 文件，并自动识别格式:

    格式A — sample_XXXX.pkl (geometry_run_results 生成):
        {"id": int, "control_points": ndarray(8,),
         "geometry": ndarray(22,3),
         "case_results": {"RPM4000_Wind6_Angle82": DataFrame, ...}}

    格式B — SIMULATION 类生成的 geometry_simulation_dict.pkl:
        {"geometry_0": {"geometry": ndarray(22,3),
                        "RPM4000_Wind6_Angle82": DataFrame, ...}, ...}

    格式C — 顶层直接就是工况键 → DataFrame
    """
    case_pattern = re.compile(r"RPM([\d.]+)_Wind([\d.]+)_Angle([\d.]+)")

    pkl_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".pkl"):
                continue
            fl = f.lower()
            if fl.startswith("data") or fl.startswith("sample_"):
                pkl_files.append(os.path.join(root, f))
    pkl_files.sort()

    if not pkl_files:
        raise FileNotFoundError(
            f"在 {data_dir} 及其子目录中未找到 data*.pkl 或 sample_*.pkl 文件"
        )

    records = []
    skipped_log = []
    loaded_count = 0

    for fpath in pkl_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            skipped_log.append({
                "source": fpath, "case": "-", "RPM": "-", "WIND": "-", "ANGLE": "-",
                "reason": f"文件读取失败: {e}", "available_columns": [],
            })
            print(f"[跳过] 无法读取 {fpath}: {e}")
            continue

        if not isinstance(data, dict):
            skipped_log.append({
                "source": fpath, "case": "-", "RPM": "-", "WIND": "-", "ANGLE": "-",
                "reason": f"pkl 内容不是 dict，而是 {type(data).__name__}",
                "available_columns": [],
            })
            continue

        # 格式A: 含 "case_results" 键（几何在 sample 级别）
        if "case_results" in data:
            geo = _extract_geometry(data, cfg)
            n = _extract_from_case_results(
                data["case_results"], case_pattern, records, skipped_log,
                source=fname, geometry=geo)
            if n > 0:
                loaded_count += 1

        # 格式B: 顶层键为 "geometry_X"，每个子 dict 有自己的 geometry
        elif any(k.startswith("geometry") for k in data.keys()):
            for geo_key, geo_val in data.items():
                if not isinstance(geo_val, dict):
                    continue
                geo = _extract_geometry(geo_val, cfg)
                n = _extract_from_case_results(
                    geo_val, case_pattern, records, skipped_log,
                    source=f"{fname}/{geo_key}", geometry=geo)
                if n > 0:
                    loaded_count += 1

        # 格式C: 顶层直接是工况键（无几何数据）
        else:
            geo = _extract_geometry(data, cfg)
            n = _extract_from_case_results(
                data, case_pattern, records, skipped_log,
                source=fname, geometry=geo)
            if n > 0:
                loaded_count += 1

    # ---------- 保存跳过日志 ----------
    if skipped_log:
        log_path = os.path.join(data_dir, "skipped_cases_log.csv")
        pd.DataFrame(skipped_log).to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"[警告] {len(skipped_log)} 条数据被跳过，详见: {log_path}")
        for entry in skipped_log[:5]:
            print(f"  - [{entry['source']}] {entry['case']}: {entry['reason']}")
        if len(skipped_log) > 5:
            print(f"  ... 及其他 {len(skipped_log) - 5} 条")

    if not records:
        raise ValueError(f"在 {len(pkl_files)} 个 pkl 文件中未提取到有效数据")

    df_all = pd.DataFrame(records)

    # 检查几何列完整性
    if cfg.geometry_mode != "none":
        geo_missing = [c for c in cfg.geometry_columns if c not in df_all.columns]
        if geo_missing:
            print(f"[警告] 几何模式 '{cfg.geometry_mode}' 要求列 {geo_missing}，"
                  f"但 pkl 中未找到对应数据。请检查数据源或切换 geometry_mode。")
        else:
            geo_null_count = df_all[cfg.geometry_columns].isna().any(axis=1).sum()
            if geo_null_count > 0:
                print(f"[警告] {geo_null_count} 条数据的几何列存在 NaN，将被丢弃")
                df_all = df_all.dropna(subset=cfg.geometry_columns).reset_index(drop=True)

    print(f"从 {loaded_count} 个有效 pkl 文件中加载了 {len(df_all)} 条数据 "
          f"(共扫描 {len(pkl_files)} 个文件, 几何模式: {cfg.geometry_mode})")
    return df_all


def load_from_excel(excel_path: str, cfg: Config) -> pd.DataFrame:
    """从已有的 dealed_data.xlsx 加载（旧流程兼容）。

    Excel 中的几何列约定:
      - control_points 模式: cp_0 ~ cp_7
      - sections 模式: chord_0 ~ chord_21, twist_0 ~ twist_21
    """
    df = pd.read_excel(excel_path)

    required = cfg.condition_columns + cfg.output_columns
    if cfg.geometry_mode != "none":
        required += cfg.geometry_columns

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Excel 缺少列: {missing}")

    print(f"从 Excel 加载了 {len(df)} 条数据 (几何模式: {cfg.geometry_mode})")
    return df[required]


# ================================================================
#  数据处理 & 保存
# ================================================================

def process_and_save(cfg: Config, raw_df: pd.DataFrame = None) -> None:
    """切分 → 标准化 → 保存 pickle + scaler。

    数据源优先级: pkl 原始数据 > Excel
    pkl 数据文件约定: 文件名以 "data" 或 "sample_" 开头且后缀为 .pkl（递归扫描子目录）
    """
    if raw_df is None:
        has_pkl = False
        for root, _, files in os.walk(cfg.raw_data_dir):
            for f in files:
                if not f.endswith(".pkl"):
                    continue
                fl = f.lower()
                if fl.startswith("data") or fl.startswith("sample_"):
                    has_pkl = True
                    break
            if has_pkl:
                break

        if has_pkl:
            raw_df = load_raw_pkl(cfg.raw_data_dir, cfg)
        else:
            excel_path = os.path.join(cfg.processed_data_dir, "dealed_data.xlsx")
            if os.path.exists(excel_path):
                raw_df = load_from_excel(excel_path, cfg)
            else:
                raise FileNotFoundError(
                    f"未找到数据源: {cfg.raw_data_dir} 中无 data*.pkl / sample_*.pkl，"
                    f"{cfg.processed_data_dir} 中无 dealed_data.xlsx"
                )

    d = cfg.processed_data_dir

    raw_csv_path = os.path.join(d, "raw_data.csv")
    raw_df[cfg.input_columns + cfg.output_columns].to_csv(
        raw_csv_path, index=False, encoding="utf-8-sig"
    )
    print(f"原始汇总数据已保存: {raw_csv_path} ({len(raw_df)} 条)")

    X = raw_df[cfg.input_columns]
    Y = raw_df[cfg.output_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.fit_transform(y_train)
    Y_test_s = scaler_Y.transform(y_test)

    pd.DataFrame(X_train_s, columns=cfg.input_columns).to_pickle(os.path.join(d, "X_train_scaled"))
    pd.DataFrame(X_test_s, columns=cfg.input_columns).to_pickle(os.path.join(d, "X_test_scaled"))
    pd.DataFrame(Y_train_s, columns=cfg.output_columns).to_pickle(os.path.join(d, "y_train_scaled"))
    pd.DataFrame(Y_test_s, columns=cfg.output_columns).to_pickle(os.path.join(d, "y_test_scaled"))

    joblib.dump(scaler_X, os.path.join(d, "scaler_X.pkl"))
    joblib.dump(scaler_Y, os.path.join(d, "scaler_Y.pkl"))

    train_df = pd.concat([X_train.reset_index(drop=True),
                           y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True),
                          y_test.reset_index(drop=True)], axis=1)
    train_csv_path = os.path.join(d, "train_data.csv")
    test_csv_path = os.path.join(d, "test_data.csv")
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")

    print(f"训练集: {len(X_train_s)} 条 → {train_csv_path}")
    print(f"测试集: {len(X_test_s)} 条 → {test_csv_path}")
    print(f"输入维度: {X_train_s.shape[1]} ({cfg.geometry_mode}) | 输出维度: {Y_train_s.shape[1]}")

    plot_data_distribution(cfg, X_train, X_test, y_train, y_test)


# ================================================================
#  数据分布可视化
# ================================================================

def plot_data_distribution(cfg: Config, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.DataFrame, y_test: pd.DataFrame,
                           save_dir: str = None) -> None:
    """绘制训练集 / 测试集的输入输出分布图并保存为 PNG。"""
    if save_dir is None:
        save_dir = cfg.processed_data_dir
    os.makedirs(save_dir, exist_ok=True)

    input_cols = cfg.input_columns
    output_cols = cfg.output_columns
    cond_cols = cfg.condition_columns
    geo_cols = cfg.geometry_columns

    units_in = {"RPM": "rpm", "WIND": "m/s", "ANGLE": "°"}
    units_out = {"Fx": "N", "Fy": "N", "Fz": "N", "Torque": "N·m", "My": "N·m", "Mz": "N·m"}

    # ---- 图1: 工况特征分布 ----
    n_cond = len(cond_cols)
    fig1, axes1 = plt.subplots(1, n_cond, figsize=(5 * n_cond, 4))
    if n_cond == 1:
        axes1 = [axes1]
    for ax, col in zip(axes1, cond_cols):
        unit = units_in.get(col, "")
        ax.hist(X_train[col], bins=25, alpha=0.7, label=f"Train ({len(X_train)})",
                edgecolor="white", linewidth=0.4, color=COLORS["train"])
        ax.hist(X_test[col], bins=25, alpha=0.7, label=f"Test ({len(X_test)})",
                edgecolor="white", linewidth=0.4, color=COLORS["test"])
        ax.set_xlabel(f"{col} ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"{col} Distribution")
        ax.legend()
    fig1.suptitle("Condition Features Distribution", fontsize=12)
    path1 = os.path.join(save_dir, "dist_condition_features.png")
    savefig(fig1, path1)
    print(f"工况分布图已保存: {path1}")

    # ---- 图1b: 几何特征分布（仅在非 none 模式时绘制）----
    if geo_cols:
        n_geo = len(geo_cols)
        n_cols_geo = min(n_geo, 4)
        n_rows = (n_geo + n_cols_geo - 1) // n_cols_geo
        fig1b, axes1b = plt.subplots(n_rows, n_cols_geo,
                                      figsize=(4.5 * n_cols_geo, 3.5 * n_rows),
                                      squeeze=False)
        for idx, col in enumerate(geo_cols):
            ax = axes1b[idx // n_cols_geo, idx % n_cols_geo]
            ax.hist(X_train[col], bins=25, alpha=0.7, label="Train",
                    edgecolor="white", linewidth=0.4, color=COLORS["train"])
            ax.hist(X_test[col], bins=25, alpha=0.7, label="Test",
                    edgecolor="white", linewidth=0.4, color=COLORS["test"])
            ax.set_xlabel(col)
            ax.set_title(col)
            ax.legend(fontsize=7)
        for idx in range(n_geo, n_rows * n_cols_geo):
            axes1b[idx // n_cols_geo, idx % n_cols_geo].set_visible(False)
        fig1b.suptitle(f"Geometry Features Distribution ({cfg.geometry_mode})", fontsize=12)
        path1b = os.path.join(save_dir, "dist_geometry_features.png")
        savefig(fig1b, path1b)
        print(f"几何分布图已保存: {path1b}")

    # ---- 图2: 输出特征分布 ----
    n_out = len(output_cols)
    fig2, axes2 = plt.subplots(1, n_out, figsize=(4.5 * n_out, 4))
    if n_out == 1:
        axes2 = [axes2]
    for ax, col in zip(axes2, output_cols):
        unit = units_out.get(col, "")
        ax.hist(y_train[col], bins=30, alpha=0.7, label=f"Train ({len(y_train)})",
                edgecolor="white", linewidth=0.4, color=COLORS["train"])
        ax.hist(y_test[col], bins=30, alpha=0.7, label=f"Test ({len(y_test)})",
                edgecolor="white", linewidth=0.4, color=COLORS["test"])
        ax.set_xlabel(f"{col} ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"{col} Distribution")
        ax.legend()
    fig2.suptitle("Output Features Distribution", fontsize=12)
    path2 = os.path.join(save_dir, "dist_output_features.png")
    savefig(fig2, path2)
    print(f"输出分布图已保存: {path2}")

    # ---- 图3: 工况 vs 输出散点矩阵 ----
    fig3, axes3 = plt.subplots(n_out, n_cond, figsize=(4.5 * n_cond, 3.8 * n_out))
    if n_out == 1 and n_cond == 1:
        axes3 = np.array([[axes3]])
    elif n_out == 1:
        axes3 = axes3[np.newaxis, :]
    elif n_cond == 1:
        axes3 = axes3[:, np.newaxis]

    for row, out_col in enumerate(output_cols):
        for col_idx, in_col in enumerate(cond_cols):
            ax = axes3[row, col_idx]
            ax.scatter(X_train[in_col], y_train[out_col],
                       s=6, alpha=0.35, label="Train", color=COLORS["train"])
            ax.scatter(X_test[in_col], y_test[out_col],
                       s=6, alpha=0.35, label="Test", color=COLORS["test"])
            u_in = units_in.get(in_col, "")
            u_out = units_out.get(out_col, "")
            ax.set_xlabel(f"{in_col} ({u_in})")
            ax.set_ylabel(f"{out_col} ({u_out})")
            if row == 0:
                ax.set_title(in_col)
            if col_idx == 0:
                ax.legend(fontsize=7, loc="best")

    fig3.suptitle("Condition vs Output Scatter", fontsize=12)
    path3 = os.path.join(save_dir, "dist_condition_vs_output.png")
    savefig(fig3, path3)
    print(f"散点矩阵图已保存: {path3}")

    # ---- 图4: 输出相关性 ----
    if n_out >= 2:
        pairs = [(i, j) for i in range(n_out) for j in range(i + 1, n_out)]
        n_pairs = len(pairs)
        fig4, axes4 = plt.subplots(1, n_pairs, figsize=(4.5 * n_pairs, 4))
        if n_pairs == 1:
            axes4 = [axes4]
        for ax, (i, j) in zip(axes4, pairs):
            ci, cj = output_cols[i], output_cols[j]
            ui, uj = units_out.get(ci, ""), units_out.get(cj, "")
            ax.scatter(y_train[ci], y_train[cj], s=6, alpha=0.35,
                       label="Train", color=COLORS["train"])
            ax.scatter(y_test[ci], y_test[cj], s=6, alpha=0.35,
                       label="Test", color=COLORS["test"])
            ax.set_xlabel(f"{ci} ({ui})")
            ax.set_ylabel(f"{cj} ({uj})")
            ax.set_title(f"{ci} vs {cj}")
            ax.legend(fontsize=7)
        fig4.suptitle("Output Correlations", fontsize=12)
        path4 = os.path.join(save_dir, "dist_output_correlations.png")
        savefig(fig4, path4)
        print(f"输出相关性图已保存: {path4}")


# ================================================================
#  入口
# ================================================================

if __name__ == "__main__":
    cfg = Config()
    process_and_save(cfg)
