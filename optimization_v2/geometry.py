"""B-Spline 几何参数化 — 8 CP → 44 sections (22 chord + 22 twist)

与 ppo_optimize/env.py 中的 cp_to_sections 保持一致的 knot 向量和径向站位,
确保 MoE 训练数据和优化器使用相同的几何映射。
"""

import numpy as np
from scipy.interpolate import BSpline

# 22 个径向站位 (与 QBlade .bld 文件对齐)
GEOMETRY_R = np.array([
    0.0201254, 0.0252054, 0.0315579, 0.0379079, 0.0442579, 0.0506079,
    0.0569579, 0.0633079, 0.0696579, 0.0760079, 0.0823586, 0.0887148,
    0.0950769, 0.1014424, 0.107804,  0.1141792, 0.1173761, 0.1205864,
    0.1238169, 0.1255,    0.12625,   0.127,
])

R_TIP = 0.127
R_NORM = GEOMETRY_R / R_TIP

# 三次 B-Spline knot 向量: 4 控制点, degree=3, 内部 knots 来源于训练数据反推
CHORD_KNOTS = np.concatenate(([0] * 3, [0.3984874, 0.89904882], [1] * 3))
TWIST_KNOTS = np.concatenate(([0] * 3, [0.2, 0.89904882], [1] * 3))

# 标准列名 schema (Excel/CSV/DataFrame 共享)
N_SECTIONS = 22
CONDITION_COLS = ["RPM", "WIND", "ANGLE"]
CHORD_COLS = [f"chord_{i}" for i in range(N_SECTIONS)]
TWIST_COLS = [f"twist_{i}" for i in range(N_SECTIONS)]
SECTION_COLS = CHORD_COLS + TWIST_COLS
INPUT_COLS = CONDITION_COLS + SECTION_COLS
OUTPUT_COLS = ["T", "H", "My", "Q"]
ALL_COLS = INPUT_COLS + OUTPUT_COLS

# 几何平滑惩罚中 twist 量纲归一化系数 (twist 单位 deg, chord 单位 m)
TWIST_NORM_SCALE = 10.0


def split_sections(sections: np.ndarray):
    """sections (44,) → (chord (22,), twist (22,))。"""
    return sections[:N_SECTIONS], sections[N_SECTIONS:]


def cp_to_sections(cp: np.ndarray) -> np.ndarray:
    """将 8 个 B-Spline 控制点展开为 44 个截面值。

    Args:
        cp: shape (8,) — [chord_cp_0..3, twist_cp_0..3]

    Returns:
        sections: shape (44,) — [chord_0..21, twist_0..21]
    """
    cp = np.asarray(cp, dtype=np.float64)
    chord_cp = cp[:4]
    twist_cp = cp[4:8]

    chord_spline = BSpline(CHORD_KNOTS, chord_cp, 3)
    twist_spline = BSpline(TWIST_KNOTS, twist_cp, 3)

    chord_vals = chord_spline(R_NORM)
    twist_vals = twist_spline(R_NORM)

    return np.concatenate([chord_vals, twist_vals])


def sections_to_input(sections: np.ndarray, rpm: float, wind: float, angle: float) -> np.ndarray:
    """组装 MoE 模型输入向量。

    Args:
        sections: shape (44,) — [chord_0..21, twist_0..21]
        rpm: 转速
        wind: 来流速度 m/s
        angle: QBlade 角度 (90° - alpha_deg)

    Returns:
        x: shape (47,) — [RPM, WIND, ANGLE, chord_0..21, twist_0..21]
    """
    return np.concatenate([[rpm, wind, angle], sections])


def validate_geometry(cp: np.ndarray, cp_bounds: np.ndarray) -> dict:
    """验证控制点边界 + 检查几何平滑性。"""
    cp = np.asarray(cp, dtype=np.float64)
    violations = []

    for i in range(len(cp)):
        if cp[i] < cp_bounds[i, 0]:
            violations.append(f"cp[{i}]={cp[i]:.6f} < lower={cp_bounds[i, 0]:.6f}")
        elif cp[i] > cp_bounds[i, 1]:
            violations.append(f"cp[{i}]={cp[i]:.6f} > upper={cp_bounds[i, 1]:.6f}")

    sections = cp_to_sections(cp)
    chord_vals, twist_vals = split_sections(sections)

    chord_grad = np.diff(chord_vals) / np.diff(GEOMETRY_R)
    twist_grad = np.diff(twist_vals) / np.diff(GEOMETRY_R)

    smoothness = float(np.sum(chord_grad**2) + np.sum(twist_grad**2))

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "smoothness": smoothness,
        "chord_grad_max": float(np.max(np.abs(chord_grad))),
        "twist_grad_max": float(np.max(np.abs(twist_grad))),
    }


def geometry_smoothness_penalty(cp: np.ndarray) -> float:
    """C_geo = Σ(c_{i+1}-c_i)² + Σ(θ_{i+1}-θ_i)² (twist 量纲归一化)。"""
    chord_cp = cp[:4]
    twist_cp = cp[4:8]
    return float(
        np.sum(np.diff(chord_cp)**2) +
        np.sum((np.diff(twist_cp) / TWIST_NORM_SCALE)**2)
    )


def random_cp(cp_bounds: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """在边界内均匀随机生成一组控制点。"""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(cp_bounds[:, 0], cp_bounds[:, 1])


def cp_to_normalized(cp: np.ndarray, cp_bounds: np.ndarray) -> np.ndarray:
    """将 CP 归一化到 [0, 1]。"""
    return (cp - cp_bounds[:, 0]) / (cp_bounds[:, 1] - cp_bounds[:, 0])


def normalized_to_cp(x_norm: np.ndarray, cp_bounds: np.ndarray) -> np.ndarray:
    """将 [0, 1] 归一化值还原为 CP。"""
    return x_norm * (cp_bounds[:, 1] - cp_bounds[:, 0]) + cp_bounds[:, 0]
