"""前飞配平求解器 V2

求解 [α, Ω1, Ω2] 满足 R_x = R_z = R_m = 0。
气动量 (T, H, My, Q) 全部来自 MoE,M_p_y 不再用解析估算。
配平结果附带 MoE 不确定度 σ。
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import root

from .config import OptConfig
from .fuselage import fuselage_aero, FuselageParams


@dataclass
class TrimResultV2:
    alpha_rad: float = 0.0
    alpha_deg: float = 0.0
    omega_front: float = 0.0
    omega_rear: float = 0.0
    converged: bool = False
    residual: np.ndarray = field(default_factory=lambda: np.zeros(3))
    residual_norm: float = 0.0

    T_front: float = 0.0
    H_front: float = 0.0
    My_front: float = 0.0
    Q_front: float = 0.0
    T_rear: float = 0.0
    H_rear: float = 0.0
    My_rear: float = 0.0
    Q_rear: float = 0.0

    sigma_mean: float = 0.0
    sigma_max: float = 0.0
    sigma_front: np.ndarray = field(default_factory=lambda: np.zeros(4))
    sigma_rear: np.ndarray = field(default_factory=lambda: np.zeros(4))

    D_f: float = 0.0
    L_f: float = 0.0
    M_f_y: float = 0.0

    message: str = ""


class TrimSolverV2:
    """前飞配平求解器。

    给定几何 sections 和速度 V,求解 [α, Ω1, Ω2] 使三轴平衡。
    """

    def __init__(self, cfg: OptConfig, moe_predict_fn: Callable):
        """
        Args:
            cfg: 全局配置
            moe_predict_fn: (x: (N, 47)) → (mu: (N, 4), sigma: (N, 4)) 物理量空间
        """
        self.cfg = cfg
        self.moe_predict_fn = moe_predict_fn
        self.fuselage_params = FuselageParams(rho=cfg.rho)

    def _query_moe_pair(
        self, omega1: float, omega2: float, V: float, angle: float,
        x_buf: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """单次前向预测前后桨气动量。

        x_buf: shape (2, 47), 调用前已写入 sections 部分。
        """
        x_buf[0, 0] = omega1
        x_buf[1, 0] = omega2
        x_buf[:, 1] = V
        x_buf[:, 2] = angle

        mu, sigma = self.moe_predict_fn(x_buf)
        return mu, sigma  # 各 (2, 4)

    def _make_residuals_fn(self, V: float, sections: np.ndarray):
        """构造残差函数闭包。

        预分配 (2, 47) 输入缓冲区,sections 仅写入一次。
        每次残差评估只覆写 [RPM, V, ANGLE] 三列。
        """
        cfg = self.cfg
        fuse = self.fuselage_params

        x_buf = np.empty((2, 47), dtype=np.float64)
        x_buf[:, 3:] = sections  # 广播到两行

        last = {"mu": None, "sigma": None, "alpha_rad": None}

        def residuals(z: np.ndarray) -> np.ndarray:
            alpha_rad, omega1, omega2 = z
            angle = 90.0 - math.degrees(alpha_rad)

            mu, sigma = self._query_moe_pair(omega1, omega2, V, angle, x_buf)
            last["mu"] = mu
            last["sigma"] = sigma
            last["alpha_rad"] = alpha_rad

            T1, H1, My1, Q1 = mu[0]
            T2, H2, My2, Q2 = mu[1]

            D_f, L_f, M_f_y = fuselage_aero(V, alpha_rad, fuse)

            sa = math.sin(alpha_rad)
            ca = math.cos(alpha_rad)

            R_x = 2 * (T1 + T2) * sa - 2 * (H1 + H2) * ca - D_f
            R_z = 2 * (T1 + T2) * ca + 2 * (H1 + H2) * sa - cfg.weight - L_f
            R_m = (2 * (My1 + My2) - M_f_y +
                   2 * (T2 * cfg.l2 - T1 * cfg.l1) +
                   2 * (H1 * cfg.d1 + H2 * cfg.d2))

            return np.array([R_x, R_z, R_m])

        return residuals, last

    def solve(self, V: float, sections: np.ndarray,
              x0: np.ndarray = None) -> TrimResultV2:
        """求解配平。"""
        cfg = self.cfg
        residuals_fn, last = self._make_residuals_fn(V, sections)

        if x0 is not None:
            initial_guesses = [x0]
        else:
            initial_guesses = self._generate_initial_guesses(V)

        best_result = None
        best_residual_norm = float("inf")
        EARLY_STOP_TOL = 1e-3

        for x0_i in initial_guesses:
            try:
                sol = root(residuals_fn, x0_i, method="hybr",
                           options={"maxfev": 500})
            except Exception:
                continue

            alpha_rad, omega1, omega2 = sol.x
            res_norm = float(np.linalg.norm(sol.fun))

            in_bounds = (
                cfg.alpha_min_rad <= alpha_rad <= cfg.alpha_max_rad and
                cfg.omega_min <= omega1 <= cfg.omega_max and
                cfg.omega_min <= omega2 <= cfg.omega_max
            )

            if sol.success and in_bounds and res_norm < best_residual_norm:
                best_residual_norm = res_norm
                best_result = sol
                if res_norm < EARLY_STOP_TOL:
                    break

        if best_result is None or best_residual_norm > 1.0:
            return TrimResultV2(
                converged=False,
                residual_norm=best_residual_norm if best_result else float("inf"),
                message="配平未收敛或超出约束范围",
            )

        alpha_rad, omega1, omega2 = best_result.x

        # 若最后一次残差评估对应 best_result,直接复用其 mu/sigma
        if last["mu"] is not None and last["alpha_rad"] == alpha_rad:
            mu, sigma = last["mu"], last["sigma"]
        else:
            residuals_fn(best_result.x)
            mu, sigma = last["mu"], last["sigma"]

        D_f, L_f, M_f_y = fuselage_aero(V, alpha_rad, self.fuselage_params)
        sigma_all = sigma.reshape(-1)

        return TrimResultV2(
            alpha_rad=float(alpha_rad),
            alpha_deg=float(math.degrees(alpha_rad)),
            omega_front=float(omega1),
            omega_rear=float(omega2),
            converged=True,
            residual=best_result.fun,
            residual_norm=best_residual_norm,
            T_front=float(mu[0, 0]),
            H_front=float(mu[0, 1]),
            My_front=float(mu[0, 2]),
            Q_front=float(mu[0, 3]),
            T_rear=float(mu[1, 0]),
            H_rear=float(mu[1, 1]),
            My_rear=float(mu[1, 2]),
            Q_rear=float(mu[1, 3]),
            sigma_mean=float(np.mean(sigma_all)),
            sigma_max=float(np.max(sigma_all)),
            sigma_front=sigma[0].copy(),
            sigma_rear=sigma[1].copy(),
            D_f=float(D_f),
            L_f=float(L_f),
            M_f_y=float(M_f_y),
            message="收敛",
        )

    def _generate_initial_guesses(self, V: float) -> list:
        cfg = self.cfg
        alpha_mid = (cfg.alpha_min_rad + cfg.alpha_max_rad) / 2.0
        omega_mid = (cfg.omega_min + cfg.omega_max) / 2.0

        return [
            np.array([alpha_mid, omega_mid, omega_mid]),
            np.array([alpha_mid * 0.7, omega_mid * 0.9, omega_mid * 1.1]),
            np.array([alpha_mid * 1.3, omega_mid * 1.1, omega_mid * 0.9]),
            np.array([cfg.alpha_min_rad * 1.5, omega_mid * 0.8, omega_mid * 0.8]),
            np.array([cfg.alpha_max_rad * 0.8, omega_mid * 1.2, omega_mid * 1.2]),
            np.array([alpha_mid, cfg.omega_min * 1.2, cfg.omega_max * 0.8]),
        ]

    def solve_with_warmstart(self, V: float, sections: np.ndarray,
                             prev_result: TrimResultV2 = None) -> TrimResultV2:
        x0 = None
        if prev_result is not None and prev_result.converged:
            x0 = np.array([prev_result.alpha_rad,
                           prev_result.omega_front,
                           prev_result.omega_rear])
        return self.solve(V, sections, x0=x0)
