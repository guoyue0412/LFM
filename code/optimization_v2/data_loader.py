"""Excel 数据加载器 — 自动识别 CP / sections 格式 + 可配置列名映射

Excel 格式约定:
    每行一个样本,列包括:
        - 工况列: RPM, WIND, ANGLE (或可配置别名)
        - 几何列:
            * CP 模式: cp_0..cp_7 (或 chord_cp_0..3 + twist_cp_0..3)
            * Sections 模式: chord_0..21 + twist_0..21
        - 输出列: T, H, My, Q (或可配置别名)

加载器自动检测几何格式,如果是 CP 则通过 B-Spline 展开为 sections。
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .geometry import (
    cp_to_sections, ALL_COLS, CHORD_COLS, TWIST_COLS, N_SECTIONS,
)


# ============================================================================
# 列名别名表
# ============================================================================

CONDITION_ALIASES = {
    "RPM": ["RPM", "rpm", "rotor_speed", "Omega", "N", "n_rpm"],
    "WIND": ["WIND", "wind", "V", "V_wind", "v", "velocity"],
    "ANGLE": ["ANGLE", "angle", "Angle", "alpha_deg", "VERTANGLE", "tilt_angle"],
}

OUTPUT_ALIASES = {
    "T": ["T", "Thrust", "thrust", "Fx", "FX", "T_p", "Fz_axial"],
    "H": ["H", "H_force", "Hp", "H_p", "Fz", "FZ", "Fy_inplane", "Fxy"],
    "My": ["My", "M_y", "Mp_y", "Pitching_Moment", "M_pitch", "Moment"],
    "Q": ["Q", "Torque", "torque", "Q_p", "TORQUE", "Mz"],
}


@dataclass
class ColumnMapping:
    """列名映射配置 — 用户可显式指定 Excel 中的列名。"""
    # 工况列 (Excel 列名)
    rpm_col: Optional[str] = None
    wind_col: Optional[str] = None
    angle_col: Optional[str] = None
    # 输出列 (Excel 列名)
    T_col: Optional[str] = None
    H_col: Optional[str] = None
    My_col: Optional[str] = None
    Q_col: Optional[str] = None
    # 几何列前缀 (用于自动匹配)
    cp_prefix: str = "cp_"           # CP 模式前缀
    chord_prefix: str = "chord_"     # sections 模式前缀
    twist_prefix: str = "twist_"     # sections 模式前缀
    chord_cp_prefix: str = "chord_cp_"
    twist_cp_prefix: str = "twist_cp_"


def _find_column(df_columns: List[str], aliases: List[str]) -> Optional[str]:
    """在 DataFrame 列名中查找第一个匹配的别名 (大小写不敏感)。"""
    cols_lower = {c.lower(): c for c in df_columns}
    for alias in aliases:
        if alias in df_columns:
            return alias
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None


def _detect_geometry_mode(df: pd.DataFrame, mapping: ColumnMapping) -> Tuple[str, dict]:
    """自动检测几何格式。

    Returns:
        (mode, info) where mode ∈ {"cp", "cp_split", "sections", "unknown"}
    """
    cols = list(df.columns)

    # 1. 检测 sections 模式: chord_0..N + twist_0..N
    chord_pattern = re.compile(rf"^{re.escape(mapping.chord_prefix)}(\d+)$")
    twist_pattern = re.compile(rf"^{re.escape(mapping.twist_prefix)}(\d+)$")

    chord_indices = sorted(
        int(m.group(1)) for c in cols
        if (m := chord_pattern.match(c)) and not c.startswith(mapping.chord_cp_prefix)
    )
    twist_indices = sorted(
        int(m.group(1)) for c in cols
        if (m := twist_pattern.match(c)) and not c.startswith(mapping.twist_cp_prefix)
    )

    if len(chord_indices) >= 20 and len(twist_indices) >= 20:
        return "sections", {
            "n_chord": len(chord_indices),
            "n_twist": len(twist_indices),
            "chord_cols": [f"{mapping.chord_prefix}{i}" for i in chord_indices],
            "twist_cols": [f"{mapping.twist_prefix}{i}" for i in twist_indices],
        }

    # 2. 检测 CP 拆分模式: chord_cp_0..3 + twist_cp_0..3
    chord_cp_pattern = re.compile(rf"^{re.escape(mapping.chord_cp_prefix)}(\d+)$")
    twist_cp_pattern = re.compile(rf"^{re.escape(mapping.twist_cp_prefix)}(\d+)$")

    chord_cp_idx = sorted(int(m.group(1)) for c in cols if (m := chord_cp_pattern.match(c)))
    twist_cp_idx = sorted(int(m.group(1)) for c in cols if (m := twist_cp_pattern.match(c)))

    if len(chord_cp_idx) >= 4 and len(twist_cp_idx) >= 4:
        return "cp_split", {
            "n_chord_cp": len(chord_cp_idx),
            "n_twist_cp": len(twist_cp_idx),
            "chord_cp_cols": [f"{mapping.chord_cp_prefix}{i}" for i in chord_cp_idx[:4]],
            "twist_cp_cols": [f"{mapping.twist_cp_prefix}{i}" for i in twist_cp_idx[:4]],
        }

    # 3. 检测 CP 合并模式: cp_0..7
    cp_pattern = re.compile(rf"^{re.escape(mapping.cp_prefix)}(\d+)$")
    cp_idx = sorted(int(m.group(1)) for c in cols if (m := cp_pattern.match(c)))

    if len(cp_idx) >= 8:
        return "cp", {
            "n_cp": len(cp_idx),
            "cp_cols": [f"{mapping.cp_prefix}{i}" for i in cp_idx[:8]],
        }

    return "unknown", {
        "available_cols": cols,
        "hint": "未找到 chord_0..21+twist_0..21 / chord_cp_0..3+twist_cp_0..3 / cp_0..7",
    }


def _expand_cp_to_sections(df: pd.DataFrame, cp_cols: List[str]) -> pd.DataFrame:
    """将 CP 列展开为 sections (44 列)。

    每行调用 cp_to_sections,生成 chord_0..21 + twist_0..21 列。
    """
    cp_data = df[cp_cols].values  # (N, 8)

    sections_data = np.zeros((len(df), N_SECTIONS * 2))
    for i, cp in enumerate(cp_data):
        sections_data[i] = cp_to_sections(cp)

    sections_df = pd.DataFrame(
        sections_data,
        columns=CHORD_COLS + TWIST_COLS,
        index=df.index,
    )

    return pd.concat([df, sections_df], axis=1)


def load_excel_dataset(
    path: str,
    sheet_name: int = 0,
    mapping: Optional[ColumnMapping] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """从 Excel 文件加载数据集,自动识别几何格式。

    Args:
        path: Excel 文件路径 (.xlsx / .xls / .csv)
        sheet_name: 工作表名或索引 (Excel 时有效)
        mapping: 列名映射配置, None 则使用默认别名
        verbose: 打印检测信息

    Returns:
        DataFrame with columns:
            RPM, WIND, ANGLE,
            chord_0..21, twist_0..21,
            T, H, My, Q
        (CP 模式下会自动展开为 sections)

    Raises:
        ValueError: 缺少必需列或几何格式无法识别
    """
    if mapping is None:
        mapping = ColumnMapping()

    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}, 仅支持 .xlsx/.xls/.csv")

    if verbose:
        print(f"加载数据: {path}")
        print(f"  样本数: {len(df)}")
        print(f"  列数: {len(df.columns)}")

    # ===== 1. 解析工况列 =====
    rpm_col = mapping.rpm_col or _find_column(df.columns, CONDITION_ALIASES["RPM"])
    wind_col = mapping.wind_col or _find_column(df.columns, CONDITION_ALIASES["WIND"])
    angle_col = mapping.angle_col or _find_column(df.columns, CONDITION_ALIASES["ANGLE"])

    missing = []
    if rpm_col is None:
        missing.append("RPM")
    if wind_col is None:
        missing.append("WIND")
    if angle_col is None:
        missing.append("ANGLE")
    if missing:
        raise ValueError(
            f"缺少工况列: {missing}. "
            f"可用列: {list(df.columns)[:20]}... "
            f"使用 ColumnMapping 显式指定列名"
        )

    if verbose:
        print(f"  工况列: RPM={rpm_col}, WIND={wind_col}, ANGLE={angle_col}")

    # ===== 2. 解析输出列 =====
    T_col = mapping.T_col or _find_column(df.columns, OUTPUT_ALIASES["T"])
    H_col = mapping.H_col or _find_column(df.columns, OUTPUT_ALIASES["H"])
    My_col = mapping.My_col or _find_column(df.columns, OUTPUT_ALIASES["My"])
    Q_col = mapping.Q_col or _find_column(df.columns, OUTPUT_ALIASES["Q"])

    missing = []
    if T_col is None:
        missing.append("T (推力)")
    if H_col is None:
        missing.append("H (面内力)")
    if My_col is None:
        missing.append("My (俯仰力矩)")
    if Q_col is None:
        missing.append("Q (扭矩)")
    if missing:
        raise ValueError(
            f"缺少输出列: {missing}. "
            f"使用 ColumnMapping 显式指定列名"
        )

    if verbose:
        print(f"  输出列: T={T_col}, H={H_col}, My={My_col}, Q={Q_col}")

    # ===== 3. 检测并解析几何 =====
    mode, info = _detect_geometry_mode(df, mapping)

    if verbose:
        print(f"  几何格式: {mode}")
        if mode != "unknown":
            for k, v in info.items():
                if not isinstance(v, list):
                    print(f"    {k}: {v}")

    if mode == "unknown":
        raise ValueError(
            f"几何格式无法识别。\n  {info['hint']}\n"
            f"  可用列示例: {info['available_cols'][:30]}\n"
            f"  使用 ColumnMapping 自定义前缀,或确保列命名符合约定"
        )

    if mode == "sections":
        chord_cols = info["chord_cols"][:22]
        twist_cols = info["twist_cols"][:22]
        # 重命名为标准格式
        rename_map = {c: f"chord_{i}" for i, c in enumerate(chord_cols)}
        rename_map.update({c: f"twist_{i}" for i, c in enumerate(twist_cols)})
        df = df.rename(columns=rename_map)

    elif mode == "cp_split":
        cp_cols = info["chord_cp_cols"] + info["twist_cp_cols"]
        df = _expand_cp_to_sections(df, cp_cols)

    elif mode == "cp":
        cp_cols = info["cp_cols"]
        df = _expand_cp_to_sections(df, cp_cols)

    # ===== 4. 重命名工况和输出列为标准格式 =====
    rename_map = {
        rpm_col: "RPM",
        wind_col: "WIND",
        angle_col: "ANGLE",
        T_col: "T",
        H_col: "H",
        My_col: "My",
        Q_col: "Q",
    }
    df = df.rename(columns=rename_map)

    # ===== 5. 提取标准列 =====
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"标准化后仍缺少列: {missing}")

    out_df = df[ALL_COLS].copy()

    if verbose:
        print(f"  标准化数据: shape={out_df.shape}")
        print(f"  T 范围: [{out_df['T'].min():.3f}, {out_df['T'].max():.3f}]")
        print(f"  Q 范围: [{out_df['Q'].min():.4f}, {out_df['Q'].max():.4f}]")

    return out_df


def save_excel_dataset(df: pd.DataFrame, path: str, sheet_name: str = "data"):
    """保存数据集为 Excel/CSV。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df.to_excel(path, sheet_name=sheet_name, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"已保存: {path} ({len(df)} 条)")


def inspect_excel(path: str, sheet_name: int = 0, n_rows: int = 5):
    """快速检视 Excel 结构 (诊断用)。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet_name, nrows=n_rows)
    else:
        df = pd.read_csv(path, nrows=n_rows)

    print(f"文件: {path}")
    print(f"列数: {len(df.columns)}")
    print(f"前 {n_rows} 行预览:")
    print(df)
    print(f"\n所有列名:")
    for i, c in enumerate(df.columns):
        print(f"  [{i:3d}] {c} (dtype={df[c].dtype})")

    # 自动检测
    mapping = ColumnMapping()
    mode, info = _detect_geometry_mode(df, mapping)
    print(f"\n自动检测几何格式: {mode}")
    print(f"  详情: {info}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Excel 数据加载器诊断工具")
    parser.add_argument("path", help="Excel/CSV 文件路径")
    parser.add_argument("--sheet", default=0, help="工作表名或索引")
    parser.add_argument("--inspect", action="store_true", help="仅检视结构")
    args = parser.parse_args()

    if args.inspect:
        inspect_excel(args.path, args.sheet)
    else:
        df = load_excel_dataset(args.path, args.sheet)
        print(f"\n加载完成,共 {len(df)} 条样本")
