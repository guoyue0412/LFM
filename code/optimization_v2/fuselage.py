"""机身半经验气动模型

基于 cross-flow 分解,输入 (V, α) → 输出 (D_f, L_f, M_f_y)。
参数来源: NASA Russell et al. (2016), Theys & De Schutter (2020)。
对应 3.5 kg / 10" / 450mm 轴距四旋翼。
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FuselageParams:
    """机身气动参数。"""
    rho: float = 1.225          # 空气密度 kg/m³
    S_front: float = 0.035      # 正面投影面积 m²
    S_top: float = 0.080        # 顶面投影面积 m²
    Cd_front: float = 0.65      # 正面阻力系数
    Cd_top: float = 1.28        # 顶面阻力系数 (平板)
    f_arms: float = 0.005       # 机臂等效阻力面积 m²
    CL_alpha: float = 0.12      # 升力线斜率 (基于 S_top)
    l_body: float = 0.35        # 机身参考长度 m
    Cm_alpha: float = -0.03     # 俯仰力矩系数斜率
    S_ref: float = 0.035        # 力矩参考面积 (= S_front)


def fuselage_aero(V: float, alpha_rad: float,
                  params: FuselageParams = None) -> tuple:
    """机身气动力计算。

    Args:
        V: 来流速度 m/s
        alpha_rad: 机身迎角 rad

    Returns:
        (D_f, L_f, M_f_y) — 阻力 N, 升力 N, 俯仰力矩 N·m
    """
    if params is None:
        params = FuselageParams()

    q = 0.5 * params.rho * V * V
    if q < 1e-12:
        return 0.0, 0.0, 0.0

    sa = math.sin(alpha_rad)
    ca = math.cos(alpha_rad)

    # 阻力: cross-flow 分解
    f_total = (params.Cd_front * params.S_front * ca**2 +
               params.Cd_top * params.S_top * sa**2 +
               params.f_arms)
    D_f = q * f_total

    # 升力: sin(2α) 模型
    L_f = q * params.S_top * params.CL_alpha * math.sin(2.0 * alpha_rad)

    # 俯仰力矩: 线性模型
    M_f_y = q * params.S_ref * params.l_body * params.Cm_alpha * alpha_rad

    return float(D_f), float(L_f), float(M_f_y)


def fuselage_aero_batch(V: np.ndarray, alpha_rad: np.ndarray,
                        params: FuselageParams = None) -> np.ndarray:
    """批量计算机身气动力。

    Args:
        V: shape (N,) 来流速度
        alpha_rad: shape (N,) 迎角

    Returns:
        shape (N, 3) — [D_f, L_f, M_f_y] per row
    """
    if params is None:
        params = FuselageParams()

    q = 0.5 * params.rho * V**2
    sa = np.sin(alpha_rad)
    ca = np.cos(alpha_rad)

    f_total = (params.Cd_front * params.S_front * ca**2 +
               params.Cd_top * params.S_top * sa**2 +
               params.f_arms)
    D_f = q * f_total
    L_f = q * params.S_top * params.CL_alpha * np.sin(2.0 * alpha_rad)
    M_f_y = q * params.S_ref * params.l_body * params.Cm_alpha * alpha_rad

    return np.column_stack([D_f, L_f, M_f_y])
