"""
quadcopter_trim_solver.py
=========================
多旋翼无人机前飞配平动力学求解器（基础版）

基于 Ye et al. (2021) "Propulsion optimization of a quadcopter in forward state"
的纵向平面配平方程组，使用 scipy.optimize.root 求解定高定速平飞状态下的
机身迎角 α、前旋翼转速 Ω₁ 和后旋翼转速 Ω₂。

配平方程组（θ = 0°，加速度项为 0）：
  F_x : 2(T_p1 + T_p2) sin α − 2(H_p1 + H_p2) cos α − D_f = 0
  F_z : 2(T_p1 + T_p2) cos α + 2(H_p1 + H_p2) sin α − mg − L_f = 0
  M_y : 2M_p1^y + 2M_p2^y − M_f^y + 2(T_p2 l₂ − T_p1 l₁) + 2(H_p1 d₁ + H_p2 d₂) = 0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import root


@dataclass
class TrimResult:
    alpha_deg: float = 0.0
    omega_front: float = 0.0
    omega_rear: float = 0.0
    converged: bool = False
    residual: np.ndarray = field(default_factory=lambda: np.zeros(3))
    message: str = ""
    aero_details: Dict = field(default_factory=dict)


FuselageAeroFunc = Callable[[float, float], Tuple[float, float, float]]
RotorAeroFunc = Callable[[float, float, float], Tuple[float, float, float]]


def dummy_fuselage_aero(V: float, alpha_rad: float) -> Tuple[float, float, float]:
    """机身气动力模型: (V, alpha_rad) -> (D_f, L_f, M_f_y)

    基于 cross-flow 分解的半经验模型, 参数来源:
      - NASA Russell et al. (2016) 多旋翼风洞实验
      - Theys & De Schutter (2020) 四旋翼阻力实测
      - RotorPy hummingbird_params (AscTec Hummingbird)

    默认参数对应 3.5 kg / 10" 四旋翼。
    完整参数化模型见 uav_model.py。
    """
    rho = 1.225
    q = 0.5 * rho * V * V

    if q < 1e-12:
        return 0.0, 0.0, 0.0

    sa = math.sin(alpha_rad)
    ca = math.cos(alpha_rad)

    S_front = 0.035
    S_top = 0.080
    Cd_front = 0.65
    Cd_top = 1.28
    f_arms = 0.005

    f_total = Cd_front * S_front * ca**2 + Cd_top * S_top * sa**2 + f_arms
    D_f = q * f_total

    CL_alpha = 0.12
    L_f = q * S_top * CL_alpha * math.sin(2.0 * alpha_rad)

    l_body = 0.35
    Cm_alpha = -0.03
    M_f_y = q * S_front * l_body * Cm_alpha * alpha_rad

    return D_f, L_f, M_f_y


def dummy_rotor_aero(V: float, alpha_rad: float, omega_rpm: float) -> Tuple[float, float, float]:
    """旋翼气动力占位函数: (V, alpha_rad, omega_rpm) -> (T_p, H_p, M_p_y)"""
    rho = 1.225
    D_prop = 0.254
    R_prop = D_prop / 2

    n = omega_rpm / 60.0
    if abs(n) < 1e-6:
        return 0.0, 0.0, 0.0

    V_axial = V * math.sin(alpha_rad) if abs(alpha_rad) > 1e-6 else 0.0
    V_inplane = V * math.cos(alpha_rad)

    J = V_axial / (n * D_prop) if abs(n * D_prop) > 1e-6 else 0.0

    C_T0 = 0.12
    k_J = 0.15
    C_T = max(C_T0 - k_J * J**2, 0.01)

    T_p = C_T * rho * n**2 * D_prop**4

    omega_rad = omega_rpm * 2 * math.pi / 60.0
    mu = V_inplane / (omega_rad * R_prop) if abs(omega_rad * R_prop) > 1e-6 else 0.0
    H_p = 0.05 * T_p * mu
    M_p_y = 0.02 * T_p * R_prop * mu

    return T_p, H_p, M_p_y


class QuadcopterTrimSolver:
    """多旋翼无人机前飞配平动力学求解器。

    力臂约定 (纵向平面, 参见 uav_model.py 详细说明):
        l1 : CG 到前旋翼纵向距离 (正值)
        l2 : CG 到后旋翼纵向距离 (正值)
        d1 : 前旋翼桨毂高于 CG 的垂直距离 (正值)
        d2 : 后旋翼桨毂高于 CG 的垂直距离 (正值)

    默认值对应 3.5 kg / 10" X 构型四旋翼, 450 mm 轴距。
    """

    def __init__(
        self,
        mass: float = 3.5,
        g: float = 9.81,
        l1: float = 0.16,
        l2: float = 0.16,
        d1: float = 0.06,
        d2: float = 0.06,
        fuselage_aero_func: Optional[FuselageAeroFunc] = None,
        rotor_aero_func: Optional[RotorAeroFunc] = None,
    ) -> None:
        self.mass = mass
        self.g = g
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2
        self.weight = mass * g
        self._fuselage_aero = fuselage_aero_func or dummy_fuselage_aero
        self._rotor_aero = rotor_aero_func or dummy_rotor_aero

    def set_fuselage_aero(self, func: FuselageAeroFunc) -> None:
        self._fuselage_aero = func

    def set_rotor_aero(self, func: RotorAeroFunc) -> None:
        self._rotor_aero = func

    def _compute_fuselage_aero(self, V: float, alpha_rad: float) -> Tuple[float, float, float]:
        return self._fuselage_aero(V, alpha_rad)

    def _compute_rotor_aero(self, V: float, alpha_rad: float, omega_rpm: float) -> Tuple[float, float, float]:
        return self._rotor_aero(V, alpha_rad, omega_rpm)

    def _trim_equations(self, x: np.ndarray, V: float) -> np.ndarray:
        alpha_rad, omega1, omega2 = x[0], x[1], x[2]

        D_f, L_f, M_f_y = self._compute_fuselage_aero(V, alpha_rad)
        T_p1, H_p1, M_p1_y = self._compute_rotor_aero(V, alpha_rad, omega1)
        T_p2, H_p2, M_p2_y = self._compute_rotor_aero(V, alpha_rad, omega2)

        sin_a = math.sin(alpha_rad)
        cos_a = math.cos(alpha_rad)

        F_x = 2.0 * (T_p1 + T_p2) * sin_a - 2.0 * (H_p1 + H_p2) * cos_a - D_f
        F_z = 2.0 * (T_p1 + T_p2) * cos_a + 2.0 * (H_p1 + H_p2) * sin_a - self.weight - L_f
        M_y = (2.0 * M_p1_y + 2.0 * M_p2_y - M_f_y
               + 2.0 * (T_p2 * self.l2 - T_p1 * self.l1)
               + 2.0 * (H_p1 * self.d1 + H_p2 * self.d2))

        return np.array([F_x, F_z, M_y])

    def _generate_initial_guess(self, V: float) -> np.ndarray:
        alpha_deg_init = min(5.0 + 0.5 * V, 20.0)
        alpha_rad_init = math.radians(alpha_deg_init)

        T_hover = self.weight / 4.0
        rho = 1.225
        D_prop = 0.254
        C_T_hover = 0.10
        n_hover = math.sqrt(T_hover / (C_T_hover * rho * D_prop**4))
        omega_init = n_hover * 60.0 * (1.0 + 0.02 * V)

        return np.array([alpha_rad_init, omega_init, omega_init])

    def _is_physical(self, result: TrimResult,
                     rpm_bounds: Optional[Tuple[float, float]] = None,
                     alpha_bounds_deg: Optional[Tuple[float, float]] = None,
                     ) -> bool:
        """检查配平解是否在可行域内。

        Args:
            rpm_bounds: (lo, hi) RPM 硬约束, None 则用宽松默认值 [500, 15000]
            alpha_bounds_deg: (lo, hi) 迎角硬约束(度), None 则用 [-5, 35]
        """
        a_lo, a_hi = alpha_bounds_deg or (-5.0, 35.0)
        rpm_lo, rpm_hi = rpm_bounds or (500.0, 15000.0)
        if result.alpha_deg < a_lo or result.alpha_deg > a_hi:
            return False
        if result.omega_front < rpm_lo or result.omega_front > rpm_hi:
            return False
        if result.omega_rear < rpm_lo or result.omega_rear > rpm_hi:
            return False
        return True

    def _build_initial_guesses(
        self, V: float, x0_hint: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """生成多组初始猜测 (热启动提示优先)。"""
        guesses: List[np.ndarray] = []
        if x0_hint is not None:
            guesses.append(x0_hint.copy())

        base = self._generate_initial_guess(V)
        guesses.append(base)

        for alpha_deg in [3.0, 8.0, 12.0, 18.0, 22.0]:
            g = base.copy()
            g[0] = math.radians(alpha_deg)
            guesses.append(g)

        for factor in [0.75, 1.25]:
            g = base.copy()
            g[1] *= factor
            g[2] *= factor
            guesses.append(g)

        g = base.copy()
        g[1] *= 1.1
        g[2] *= 0.9
        guesses.append(g)

        return guesses

    def solve_multi_start(
        self,
        V: float,
        x0_hint: Optional[np.ndarray] = None,
        n_starts: int = 6,
        method: str = "hybr",
        tol: float = 1e-6,
        max_iter: int = 500,
        rpm_bounds: Optional[Tuple[float, float]] = None,
        alpha_bounds_deg: Optional[Tuple[float, float]] = None,
    ) -> TrimResult:
        """多初始值配平求解。

        Args:
            rpm_bounds: 可选 (lo, hi) RPM 硬约束 (例如训练数据范围 [4000, 6500])
            alpha_bounds_deg: 可选 (lo, hi) 迎角硬约束(度)

        尝试多组初始猜测, 返回首个在约束域内的收敛解;
        若全部失败, 使用最优起点 + 放松容差重试, 最终返回残差最小结果。
        """
        if rpm_bounds is not None or alpha_bounds_deg is not None:
            guesses = self._build_bounded_guesses(
                V, x0_hint, rpm_bounds, alpha_bounds_deg,
            )
        else:
            guesses = self._build_initial_guesses(V, x0_hint)

        best_result: Optional[TrimResult] = None
        best_residual_norm = np.inf

        for x0 in guesses[:n_starts]:
            result = self.solve(V, x0=x0, method=method, tol=tol, max_iter=max_iter)
            if result.converged and self._is_physical(
                result, rpm_bounds=rpm_bounds, alpha_bounds_deg=alpha_bounds_deg,
            ):
                return result

            res_norm = float(np.linalg.norm(result.residual))
            if res_norm < best_residual_norm:
                best_residual_norm = res_norm
                best_result = result

        if best_result is not None and not best_result.converged and best_residual_norm < 100.0:
            x0_retry = np.array([
                math.radians(best_result.alpha_deg),
                best_result.omega_front,
                best_result.omega_rear,
            ])
            r = self.solve(V, x0=x0_retry, method=method,
                           tol=tol * 100, max_iter=max_iter * 2)
            if r.converged and self._is_physical(
                r, rpm_bounds=rpm_bounds, alpha_bounds_deg=alpha_bounds_deg,
            ):
                return r

        return best_result if best_result is not None else TrimResult(
            message="all starts failed",
        )

    def _build_bounded_guesses(
        self,
        V: float,
        x0_hint: Optional[np.ndarray],
        rpm_bounds: Tuple[float, float],
        alpha_bounds_deg: Optional[Tuple[float, float]],
    ) -> List[np.ndarray]:
        """在约束域内高效生成初始猜测 (优先级排序)。"""
        guesses: List[np.ndarray] = []

        rpm_lo, rpm_hi = rpm_bounds or (4000, 6500)
        a_lo = alpha_bounds_deg[0] if alpha_bounds_deg else 2.0
        a_hi = alpha_bounds_deg[1] if alpha_bounds_deg else 8.0
        rpm_mid = (rpm_lo + rpm_hi) / 2

        if x0_hint is not None:
            h = x0_hint.copy()
            h[0] = np.clip(h[0], math.radians(a_lo), math.radians(a_hi))
            h[1] = np.clip(h[1], rpm_lo, rpm_hi)
            h[2] = np.clip(h[2], rpm_lo, rpm_hi)
            guesses.append(h)

        for alpha_deg in [5.0, 3.0, 7.0, a_lo, a_hi]:
            if a_lo <= alpha_deg <= a_hi:
                guesses.append(np.array([
                    math.radians(alpha_deg), rpm_mid, rpm_mid,
                ]))

        for rpm in [rpm_mid, rpm_lo + 0.7 * (rpm_hi - rpm_lo), rpm_hi]:
            guesses.append(np.array([
                math.radians((a_lo + a_hi) / 2), rpm, rpm,
            ]))

        return guesses

    def solve(
        self,
        V: float,
        x0: Optional[np.ndarray] = None,
        method: str = "hybr",
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> TrimResult:
        if x0 is None:
            x0 = self._generate_initial_guess(V)

        sol = root(
            fun=self._trim_equations, x0=x0, args=(V,),
            method=method, tol=tol, options={"maxfev": max_iter},
        )

        alpha_rad_sol, omega1_sol, omega2_sol = sol.x

        D_f, L_f, M_f_y = self._compute_fuselage_aero(V, alpha_rad_sol)
        T_p1, H_p1, M_p1_y = self._compute_rotor_aero(V, alpha_rad_sol, omega1_sol)
        T_p2, H_p2, M_p2_y = self._compute_rotor_aero(V, alpha_rad_sol, omega2_sol)

        return TrimResult(
            alpha_deg=math.degrees(alpha_rad_sol),
            omega_front=omega1_sol,
            omega_rear=omega2_sol,
            converged=sol.success,
            residual=sol.fun,
            message=sol.message,
            aero_details={
                "V": V, "alpha_rad": alpha_rad_sol,
                "T_p1": T_p1, "H_p1": H_p1, "M_p1_y": M_p1_y,
                "T_p2": T_p2, "H_p2": H_p2, "M_p2_y": M_p2_y,
                "D_f": D_f, "L_f": L_f, "M_f_y": M_f_y,
                "total_thrust": 2.0 * (T_p1 + T_p2),
                "total_hub_force": 2.0 * (H_p1 + H_p2),
            },
        )

    def sweep_velocity(
        self, V_range: np.ndarray, method: str = "hybr", tol: float = 1e-8,
    ) -> List[TrimResult]:
        results = []
        x0 = None
        for V in V_range:
            result = self.solve(V, x0=x0, method=method, tol=tol)
            results.append(result)
            if result.converged:
                x0 = np.array([math.radians(result.alpha_deg), result.omega_front, result.omega_rear])
            else:
                x0 = None
        return results

    @staticmethod
    def print_result(result: TrimResult) -> None:
        status = "converged" if result.converged else "NOT converged"
        print("=" * 60)
        print(f"  Trim Result  [{status}]")
        print("=" * 60)
        print(f"  V       = {result.aero_details.get('V', 0):.2f} m/s")
        print(f"  alpha   = {result.alpha_deg:.4f} deg")
        print(f"  Omega_1 = {result.omega_front:.1f} RPM")
        print(f"  Omega_2 = {result.omega_rear:.1f} RPM")
        print(f"  |F_x|   = {abs(result.residual[0]):.2e} N")
        print(f"  |F_z|   = {abs(result.residual[1]):.2e} N")
        print(f"  |M_y|   = {abs(result.residual[2]):.2e} N*m")
        ad = result.aero_details
        print(f"  T_p1    = {ad.get('T_p1', 0):.4f} N")
        print(f"  T_p2    = {ad.get('T_p2', 0):.4f} N")
        print(f"  D_f     = {ad.get('D_f', 0):.4f} N")
        print("=" * 60)
