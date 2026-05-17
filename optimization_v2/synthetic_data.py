"""合成数据生成 — 模拟 LLFVW 输出

在真实 QBlade/LLFVW 数据就绪前,用基于物理的参数化模型生成训练数据。
数据格式与真实 LLFVW 输出对齐: 输入 (RPM, WIND, ANGLE, sections_44) → 输出 (T, H, My, Q)。
后续只需替换数据源,无需修改下游代码。
"""

import math
import os

import numpy as np
import pandas as pd

from .config import OptConfig
from .geometry import (
    cp_to_sections, random_cp, split_sections,
    CHORD_COLS, TWIST_COLS,
)


def _synthetic_rotor_aero(
    rpm: float, wind: float, angle_deg: float,
    chord_sections: np.ndarray, twist_sections: np.ndarray,
    rho: float = 1.225, D: float = 0.254,
) -> tuple:
    """基于物理的合成旋翼气动模型。

    比 dummy_rotor_aero 更精细: 考虑几何对推力系数、扭矩系数的影响。

    Returns:
        (T, H, My, Q) — 推力, 面内力, 俯仰力矩, 扭矩
    """
    R = D / 2.0
    n = rpm / 60.0
    if abs(n) < 1e-6:
        return 0.0, 0.0, 0.0, 0.0

    alpha_rad = math.radians(90.0 - angle_deg)
    V_axial = wind * math.sin(alpha_rad)
    V_inplane = wind * math.cos(alpha_rad)

    J = V_axial / (n * D) if abs(n * D) > 1e-6 else 0.0
    omega = rpm * 2.0 * math.pi / 60.0

    chord_mean = np.mean(chord_sections)
    chord_tip = chord_sections[-1]
    twist_root = twist_sections[0]
    twist_tip = twist_sections[-1]
    twist_mean = np.mean(twist_sections)
    twist_gradient = (twist_root - twist_tip) / R

    chord_ref = 0.026
    solidity = chord_mean / chord_ref

    # 推力系数 — 基准值 C_T0 ≈ 0.30 → 5000 RPM, 10" 桨产生 ~10 N
    C_T0 = 0.30 * np.clip(solidity, 0.3, 2.5)
    twist_opt = 35.0
    twist_eff = 1.0 - 0.25 * ((twist_mean - twist_opt) / 40.0) ** 2
    twist_eff = np.clip(twist_eff, 0.5, 1.2)

    k_J = 0.30 * (1.0 + 0.3 * (chord_tip / chord_mean))
    C_T = max(C_T0 * twist_eff - k_J * J ** 2, 0.02)
    T = C_T * rho * n**2 * D**4

    # 扭矩系数 — C_Q/C_T ≈ 0.05~0.08 (典型 10" 桨)
    C_Q_base = 0.020 * solidity
    induced_factor = 1.0 + 0.5 * abs(J)
    profile_factor = 1.0 + 0.15 * (1.0 - twist_eff)
    C_Q = C_Q_base * induced_factor * profile_factor
    Q = C_Q * rho * n**2 * D**5

    mu = V_inplane / (omega * R) if abs(omega * R) > 1e-6 else 0.0
    H = 0.05 * T * mu * (1.0 + 0.3 * solidity)

    k_m = 0.02 * (1.0 + 0.5 * abs(twist_gradient) / 500.0)
    My = k_m * T * R * mu

    return float(T), float(H), float(My), float(Q)


def generate_synthetic_dataset(
    cfg: OptConfig,
    n_geometries: int = 100,
    noise_std: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成训练数据集。

    采样策略:
        - n_geometries 个随机几何 (在 CP 边界内 LHS 采样)
        - 每个几何 72 个工况 (RPM × WIND × ANGLE 网格)
        - 加入高斯噪声模拟 LLFVW 数据散布

    Returns:
        DataFrame with columns: RPM, WIND, ANGLE, chord_0..21, twist_0..21, T, H, My, Q
    """
    rng = np.random.default_rng(seed)

    rpm_values = np.array([4000, 4500, 5000, 5500, 6000, 6500])
    wind_values = np.array([6, 8, 10, 12, 15])
    angle_values = np.array([82, 83, 84, 85, 86, 87, 88])

    # 工况采样: RPM × WIND 笛卡尔积,每个组合采 angle_values 中部分 angle
    conditions = []
    rng_local = np.random.default_rng(seed + 1)
    for rpm in rpm_values:
        for wind in wind_values:
            # 随机选 ~3 个 angle (避免 6×5×7=210 太多,也避免太少)
            n_angles = max(2, len(angle_values) // 3)
            chosen = rng_local.choice(angle_values, size=n_angles, replace=False)
            for angle in chosen:
                conditions.append((int(rpm), int(wind), int(angle)))

    # 截取到 72 个条件
    if len(conditions) > 72:
        idx = rng_local.choice(len(conditions), size=72, replace=False)
        conditions = [conditions[i] for i in sorted(idx)]
    conditions = conditions[:72]

    noise_scales = (1.0, 1.5, 2.0, 1.0)  # T, H, My, Q

    records = []
    for geo_idx in range(n_geometries):
        cp = random_cp(cfg.cp_bounds, rng)
        sections = cp_to_sections(cp)
        chord_sections, twist_sections = split_sections(sections)

        for rpm, wind, angle in conditions:
            outputs = _synthetic_rotor_aero(
                rpm, wind, angle, chord_sections, twist_sections,
                rho=cfg.rho, D=cfg.D_prop,
            )
            T, H, My, Q = (v * (1.0 + rng.normal(0, noise_std * s))
                           for v, s in zip(outputs, noise_scales))

            row = {"RPM": rpm, "WIND": wind, "ANGLE": angle}
            row.update(dict(zip(CHORD_COLS, chord_sections)))
            row.update(dict(zip(TWIST_COLS, twist_sections)))
            row["T"] = T
            row["H"] = H
            row["My"] = My
            row["Q"] = Q
            records.append(row)

    df = pd.DataFrame(records)
    return df


def save_dataset(df: pd.DataFrame, cfg: OptConfig):
    """保存数据集到 CSV。"""
    os.makedirs(cfg.data_dir, exist_ok=True)
    path = os.path.join(cfg.data_dir, "synthetic_aero_data.csv")
    df.to_csv(path, index=False)
    print(f"合成数据已保存: {path} ({len(df)} 条)")
    return path


def load_dataset(cfg: OptConfig) -> pd.DataFrame:
    """加载数据集。"""
    path = os.path.join(cfg.data_dir, "synthetic_aero_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}, 请先运行 generate_synthetic_dataset")
    return pd.read_csv(path)


if __name__ == "__main__":
    cfg = OptConfig()
    df = generate_synthetic_dataset(cfg, n_geometries=100, noise_std=0.02)
    save_dataset(df, cfg)
    print(f"\n数据统计:")
    print(df[["T", "H", "My", "Q"]].describe())
