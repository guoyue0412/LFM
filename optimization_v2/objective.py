"""优化目标函数 — 将 CP → MoE → 配平 → 功率 串联为单一评估函数

J(x_g) = P_elec/V + λ_R·|R|² + λ_U·U + λ_G·C_geo
"""

import numpy as np

from .config import OptConfig
from .geometry import cp_to_sections, geometry_smoothness_penalty, validate_geometry
from .trim_solver import TrimSolverV2, TrimResultV2
from .power import compute_power


def uncertainty_penalty(trim_result: TrimResultV2, cfg: OptConfig) -> float:
    """U = w_T·σ_T + w_H·σ_H + w_M·σ_My + w_Q·σ_Q (前后桨平均)。"""
    if not trim_result.converged:
        return 1.0

    sigma_avg = (trim_result.sigma_front + trim_result.sigma_rear) / 2.0
    weights = np.asarray(cfg.uncertainty_weights, dtype=np.float64)
    return float(np.dot(weights, np.abs(sigma_avg)))


def evaluate_design(
    cp: np.ndarray,
    trim_solver: TrimSolverV2,
    cfg: OptConfig,
    V: float = None,
    prev_trim: TrimResultV2 = None,
) -> dict:
    """评估单个设计方案。

    Returns:
        dict with: J, P_elec, P_elec_per_V, trim_result, breakdown
    """
    if V is None:
        V = cfg.V_cruise

    cp = np.asarray(cp, dtype=np.float64)

    geo_check = validate_geometry(cp, cfg.cp_bounds)
    if not geo_check["valid"]:
        return {
            "J": cfg.penalty_trim_fail,
            "P_elec": float("inf"),
            "P_elec_per_V": float("inf"),
            "trim_result": TrimResultV2(converged=False, message="CP 越界"),
            "breakdown": {"penalty": "geometry_bounds"},
        }

    sections = cp_to_sections(cp)
    trim_result = trim_solver.solve_with_warmstart(V, sections, prev_trim)

    if not trim_result.converged:
        C_geo = geometry_smoothness_penalty(cp)
        return {
            "J": cfg.penalty_trim_fail + cfg.lambda_geometry * C_geo,
            "P_elec": float("inf"),
            "P_elec_per_V": float("inf"),
            "trim_result": trim_result,
            "breakdown": {"penalty": "trim_failed", "C_geo": C_geo},
        }

    power = compute_power(trim_result, cfg)
    P_elec_per_V = power["P_elec_per_V"]

    residual_penalty = float(np.sum(trim_result.residual**2))
    U = uncertainty_penalty(trim_result, cfg)
    C_geo = geometry_smoothness_penalty(cp)

    J = (P_elec_per_V +
         cfg.lambda_residual * residual_penalty +
         cfg.lambda_uncertainty * U +
         cfg.lambda_geometry * C_geo)

    return {
        "J": float(J),
        "P_elec": power["P_elec"],
        "P_elec_per_V": float(P_elec_per_V),
        "trim_result": trim_result,
        "breakdown": {
            "P_elec_per_V": float(P_elec_per_V),
            "residual_penalty": residual_penalty,
            "uncertainty_penalty": U,
            "geometry_penalty": C_geo,
            "lambda_R_contrib": cfg.lambda_residual * residual_penalty,
            "lambda_U_contrib": cfg.lambda_uncertainty * U,
            "lambda_G_contrib": cfg.lambda_geometry * C_geo,
        },
    }


def batch_evaluate(
    cp_array: np.ndarray,
    trim_solver: TrimSolverV2,
    cfg: OptConfig,
) -> list:
    """批量评估多个设计方案。"""
    results = []
    prev_trim = None
    for i in range(len(cp_array)):
        result = evaluate_design(cp_array[i], trim_solver, cfg, prev_trim=prev_trim)
        if result["trim_result"].converged:
            prev_trim = result["trim_result"]
        results.append(result)
    return results
