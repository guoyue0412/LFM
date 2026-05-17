"""功率与航程计算模块

P_aero = 2·Q1·Ω1 + 2·Q2·Ω2
P_elec = P_aero / (η_m · η_esc)
Range = V · E_b / P_elec
"""

import math

from .config import OptConfig
from .trim_solver import TrimResultV2


def compute_power(trim_result: TrimResultV2, cfg: OptConfig) -> dict:
    """从配平结果计算功率。"""
    if not trim_result.converged:
        return {
            "P_aero": float("inf"),
            "P_elec": float("inf"),
            "P_elec_per_V": float("inf"),
            "valid": False,
        }

    omega1_rad = trim_result.omega_front * 2.0 * math.pi / 60.0
    omega2_rad = trim_result.omega_rear * 2.0 * math.pi / 60.0

    P_aero = 2.0 * abs(trim_result.Q_front) * omega1_rad + \
             2.0 * abs(trim_result.Q_rear) * omega2_rad
    P_elec = P_aero / cfg.eta_motor_esc
    P_elec_per_V = P_elec / cfg.V_cruise if cfg.V_cruise > 0 else float("inf")

    return {
        "P_aero": float(P_aero),
        "P_elec": float(P_elec),
        "P_elec_per_V": float(P_elec_per_V),
        "valid": True,
    }


def compute_range(P_elec: float, V: float, E_battery: float) -> float:
    """Range = V · E_b / P_elec (m)。"""
    if P_elec <= 0 or not math.isfinite(P_elec):
        return 0.0
    return V * E_battery / P_elec


def compute_endurance(P_elec: float, E_battery: float) -> float:
    """t = E_b / P_elec (s)。"""
    if P_elec <= 0 or not math.isfinite(P_elec):
        return 0.0
    return E_battery / P_elec


def compute_fm(trim_result: TrimResultV2, cfg: OptConfig) -> float:
    """品质因数 FM = mg·V / P_elec (与 V1 兼容指标)。"""
    power = compute_power(trim_result, cfg)
    if not power["valid"] or power["P_elec"] <= 0:
        return 0.0
    return cfg.weight * cfg.V_cruise / power["P_elec"]
