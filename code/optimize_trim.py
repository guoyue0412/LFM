"""几何优化配平求解器

在给定前飞速度下，联合优化桨叶几何外形 (控制点) 和飞行状态 (α, Ω₁, Ω₂)，
在满足力/力矩平衡约束的前提下，最小化总功率消耗。

数学模型:
    min   P = Ω₁·Q₁ + Ω₂·Q₂          (总功率)
    s.t.  F_x = 0                      (水平力平衡)
          F_z = 0                      (垂直力平衡)
          M_y = 0                      (俯仰力矩平衡)
    变量  x = [α, Ω₁, Ω₂, cp₀..cp₇]   (11 维)

用法:
    python optimize_trim.py --V 10
    python optimize_trim.py --V-min 2 --V-max 20 --V-num 10
    python optimize_trim.py --V 10 --compare-geometry
"""

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from quadcopter_trim_solver import (
    QuadcopterTrimSolver,
    TrimResult,
    dummy_fuselage_aero,
    dummy_rotor_aero,
)


# ============================================================================
# 结果数据类
# ============================================================================

@dataclass
class OptimTrimResult:
    """几何优化配平结果。"""
    V: float = 0.0
    alpha_deg: float = 0.0
    omega_front: float = 0.0
    omega_rear: float = 0.0
    geometry: np.ndarray = field(default_factory=lambda: np.array([]))
    power: float = 0.0
    converged: bool = False
    trim_residual: np.ndarray = field(default_factory=lambda: np.zeros(3))
    message: str = ""
    aero_details: Dict = field(default_factory=dict)


# ============================================================================
# 扩展气动力接口 — 同时返回配平力和扭矩
# ============================================================================

# (V, alpha_rad, omega_rpm, geometry) -> (T_p, H_p, M_p_y, Q_p)
FullRotorAeroFunc = Callable[
    [float, float, float, Optional[np.ndarray]],
    Tuple[float, float, float, float],
]


def dummy_rotor_aero_full(
    V: float, alpha_rad: float, omega_rpm: float,
    geometry: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """dummy 旋翼气动力，扩展返回扭矩 Q_p。"""
    T_p, H_p, M_p_y = dummy_rotor_aero(V, alpha_rad, omega_rpm)
    rho = 1.225
    D_prop = 0.254
    n = omega_rpm / 60.0
    C_Q = 0.008
    Q_p = C_Q * rho * abs(n) * n * D_prop**5 if abs(n) > 1e-6 else 0.0
    return T_p, H_p, M_p_y, Q_p


# ============================================================================
# 核心优化器
# ============================================================================

class GeometryTrimOptimizer:
    """几何 + 配平联合优化求解器。

    在保证力/力矩平衡 (配平约束) 的前提下，
    搜索最优的桨叶几何和飞行状态参数以最小化功率。
    """

    def __init__(
        self,
        mass: float = 3.5,
        g: float = 9.81,
        l1: float = 0.16,
        l2: float = 0.16,
        d1: float = 0.06,
        d2: float = 0.06,
        n_geometry: int = 8,
        fuselage_aero_func=None,
        rotor_aero_func: Optional[FullRotorAeroFunc] = None,
    ):
        self.mass = mass
        self.g = g
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2
        self.weight = mass * g
        self.n_geometry = n_geometry

        self._fuselage_aero = fuselage_aero_func or dummy_fuselage_aero
        self._rotor_aero_full = rotor_aero_func or dummy_rotor_aero_full

        self.alpha_bounds = (math.radians(0.01), math.radians(25.0))
        self.omega_bounds = (1000.0, 12000.0)
        self.geometry_bounds = None

    def set_geometry_bounds(self, bounds: List[Tuple[float, float]]):
        """设置每个几何参数的上下界。

        Args:
            bounds: 长度为 n_geometry 的 [(lo, hi), ...] 列表
        """
        if len(bounds) != self.n_geometry:
            raise ValueError(f"bounds 长度 {len(bounds)} != n_geometry {self.n_geometry}")
        self.geometry_bounds = bounds

    def _default_geometry_bounds(self) -> List[Tuple[float, float]]:
        """默认几何参数边界（基于训练数据的典型范围）。"""
        chord_bounds = [(0.02, 0.30)] * (self.n_geometry // 2)
        twist_bounds = [(-5.0, 30.0)] * (self.n_geometry - self.n_geometry // 2)
        return chord_bounds + twist_bounds

    def _unpack(self, x: np.ndarray):
        """从优化向量中提取各分量。"""
        alpha_rad = x[0]
        omega1 = x[1]
        omega2 = x[2]
        geo = x[3:] if self.n_geometry > 0 else None
        return alpha_rad, omega1, omega2, geo

    def _compute_aero(self, V: float, alpha_rad: float, omega: float, geo):
        """调用旋翼气动力回调。"""
        return self._rotor_aero_full(V, alpha_rad, omega, geo)

    def _objective_and_constraints(self, x: np.ndarray, V: float):
        """计算目标函数 (功率) 和约束残差。

        Returns:
            (power, [F_x, F_z, M_y])
        """
        alpha_rad, omega1, omega2, geo = self._unpack(x)

        D_f, L_f, M_f_y = self._fuselage_aero(V, alpha_rad)
        T_p1, H_p1, M_p1_y, Q_p1 = self._compute_aero(V, alpha_rad, omega1, geo)
        T_p2, H_p2, M_p2_y, Q_p2 = self._compute_aero(V, alpha_rad, omega2, geo)

        sin_a = math.sin(alpha_rad)
        cos_a = math.cos(alpha_rad)

        F_x = (2.0 * (T_p1 + T_p2) * sin_a
               - 2.0 * (H_p1 + H_p2) * cos_a
               - D_f)

        F_z = (2.0 * (T_p1 + T_p2) * cos_a
               + 2.0 * (H_p1 + H_p2) * sin_a
               - self.weight - L_f)

        M_y = (2.0 * M_p1_y + 2.0 * M_p2_y - M_f_y
               + 2.0 * (T_p2 * self.l2 - T_p1 * self.l1)
               + 2.0 * (H_p1 * self.d1 + H_p2 * self.d2))

        omega1_rad = omega1 * 2.0 * math.pi / 60.0
        omega2_rad = omega2 * 2.0 * math.pi / 60.0
        power = 2.0 * (abs(omega1_rad * Q_p1) + abs(omega2_rad * Q_p2))

        return power, np.array([F_x, F_z, M_y])

    def _generate_initial_guess(self, V: float, geo_init=None) -> np.ndarray:
        """生成初始猜测向量。"""
        alpha_init = math.radians(min(5.0 + 0.5 * V, 20.0))

        T_hover = self.weight / 4.0
        rho = 1.225
        D_prop = 0.254
        C_T_hover = 0.10
        n_hover = math.sqrt(T_hover / (C_T_hover * rho * D_prop**4))
        omega_init = n_hover * 60.0 * (1.0 + 0.02 * V)

        x0 = [alpha_init, omega_init, omega_init]

        if self.n_geometry > 0:
            if geo_init is not None:
                x0.extend(geo_init)
            else:
                bounds = self.geometry_bounds or self._default_geometry_bounds()
                for lo, hi in bounds:
                    x0.append((lo + hi) / 2.0)

        return np.array(x0)

    def solve(
        self,
        V: float,
        x0: Optional[np.ndarray] = None,
        geo_init: Optional[List[float]] = None,
        tol: float = 1e-7,
        max_iter: int = 500,
    ) -> OptimTrimResult:
        """在给定速度下求解几何优化配平。"""
        if x0 is None:
            x0 = self._generate_initial_guess(V, geo_init)

        bounds_list = [
            self.alpha_bounds,
            self.omega_bounds,
            self.omega_bounds,
        ]
        if self.n_geometry > 0:
            geo_bounds = self.geometry_bounds or self._default_geometry_bounds()
            bounds_list.extend(geo_bounds)

        constraints = {
            "type": "eq",
            "fun": lambda x: self._objective_and_constraints(x, V)[1],
        }

        sol = minimize(
            fun=lambda x: self._objective_and_constraints(x, V)[0],
            x0=x0,
            method="SLSQP",
            bounds=bounds_list,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": tol, "disp": False},
        )

        alpha_rad, omega1, omega2, geo = self._unpack(sol.x)
        power, residual = self._objective_and_constraints(sol.x, V)

        trim_ok = np.all(np.abs(residual) < 1e-3)

        alpha_rad_v, omega1_v, omega2_v, geo_v = self._unpack(sol.x)
        D_f, L_f, M_f_y = self._fuselage_aero(V, alpha_rad_v)
        T_p1, H_p1, M_p1_y, Q_p1 = self._compute_aero(V, alpha_rad_v, omega1_v, geo_v)
        T_p2, H_p2, M_p2_y, Q_p2 = self._compute_aero(V, alpha_rad_v, omega2_v, geo_v)

        return OptimTrimResult(
            V=V,
            alpha_deg=math.degrees(alpha_rad),
            omega_front=omega1,
            omega_rear=omega2,
            geometry=geo if geo is not None else np.array([]),
            power=power,
            converged=sol.success and trim_ok,
            trim_residual=residual,
            message=sol.message,
            aero_details={
                "T_p1": T_p1, "H_p1": H_p1, "M_p1_y": M_p1_y, "Q_p1": Q_p1,
                "T_p2": T_p2, "H_p2": H_p2, "M_p2_y": M_p2_y, "Q_p2": Q_p2,
                "D_f": D_f, "L_f": L_f, "M_f_y": M_f_y,
                "total_thrust": 2.0 * (T_p1 + T_p2),
                "power": power,
            },
        )

    def sweep_velocity(
        self, V_range: np.ndarray,
        geo_init: Optional[List[float]] = None,
        tol: float = 1e-7,
    ) -> List[OptimTrimResult]:
        """速度扫描，使用延续法传递上一点的解。"""
        results = []
        x0 = None

        for V in V_range:
            result = self.solve(V, x0=x0, geo_init=geo_init, tol=tol)
            results.append(result)

            if result.converged:
                x_next = [math.radians(result.alpha_deg),
                          result.omega_front, result.omega_rear]
                if len(result.geometry) > 0:
                    x_next.extend(result.geometry.tolist())
                x0 = np.array(x_next)
            else:
                x0 = None

        return results

    @staticmethod
    def print_result(r: OptimTrimResult):
        status = "converged" if r.converged else "NOT converged"
        print("=" * 70)
        print(f"  Geometry-Trim Optimization Result  [{status}]")
        print("=" * 70)
        print(f"  V       = {r.V:.2f} m/s")
        print(f"  alpha   = {r.alpha_deg:.4f} deg")
        print(f"  Omega_1 = {r.omega_front:.1f} RPM")
        print(f"  Omega_2 = {r.omega_rear:.1f} RPM")
        print(f"  Power   = {r.power:.4f} W")
        print(f"  |F_x|   = {abs(r.trim_residual[0]):.2e} N")
        print(f"  |F_z|   = {abs(r.trim_residual[1]):.2e} N")
        print(f"  |M_y|   = {abs(r.trim_residual[2]):.2e} N*m")
        if len(r.geometry) > 0:
            n_half = len(r.geometry) // 2
            print(f"  Geometry (chord): {np.array2string(r.geometry[:n_half], precision=4)}")
            print(f"  Geometry (twist): {np.array2string(r.geometry[n_half:], precision=4)}")
        print("=" * 70)


# ============================================================================
# DNN 版本的 FullRotorAeroFunc
# ============================================================================

def make_dnn_rotor_aero_full(cfg, model, scaler_X, scaler_Y, device):
    """构建 DNN 版的扩展旋翼气动力函数 (含 Torque 输出)。"""
    import torch

    out_cols = cfg.output_columns
    fx_idx = out_cols.index("Fx")
    fz_idx = out_cols.index("Fz")
    my_idx = out_cols.index("My")
    tq_idx = out_cols.index("Torque")

    def rotor_aero_full(V, alpha_rad, omega_rpm, geometry=None):
        angle_deg = 90.0 - math.degrees(alpha_rad)
        x_raw = [omega_rpm, V, angle_deg]
        if geometry is not None:
            x_raw.extend(geometry.tolist() if hasattr(geometry, "tolist") else list(geometry))

        x_scaled = scaler_X.transform([x_raw])
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_scaled = model(x_tensor).cpu().numpy()
        y = scaler_Y.inverse_transform(y_scaled)[0]

        return float(y[fx_idx]), float(y[fz_idx]), float(y[my_idx]), float(y[tq_idx])

    return rotor_aero_full


# ============================================================================
# 可视化
# ============================================================================

def plot_optim_sweep(results: List[OptimTrimResult], save_path: str = "optim_trim_sweep.png"):
    """绘制优化配平的速度扫描结果。"""
    V_arr = np.array([r.V for r in results])
    mask = np.array([r.converged for r in results])

    if not np.any(mask):
        print("[警告] 没有收敛的点，跳过绘图")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(V_arr[mask], [r.alpha_deg for r, m in zip(results, mask) if m], "o-", markersize=4)
    ax.set_ylabel("Alpha (deg)")
    ax.set_title("Angle of Attack")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(V_arr[mask], [r.omega_front for r, m in zip(results, mask) if m],
            "o-", label="Front", markersize=4)
    ax.plot(V_arr[mask], [r.omega_rear for r, m in zip(results, mask) if m],
            "s--", label="Rear", markersize=4)
    ax.set_ylabel("RPM")
    ax.set_title("Rotor Speed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(V_arr[mask], [r.power for r, m in zip(results, mask) if m], "o-", markersize=4, color="red")
    ax.set_ylabel("Power (W)")
    ax.set_title("Total Power (Optimized)")
    ax.grid(True, alpha=0.3)

    has_geo = any(len(r.geometry) > 0 for r in results)
    if has_geo:
        n_geo = len(results[0].geometry)
        n_half = n_geo // 2
        geo_converged = np.array([r.geometry for r, m in zip(results, mask) if m])
        V_conv = V_arr[mask]

        ax = axes[1, 0]
        for i in range(n_half):
            ax.plot(V_conv, geo_converged[:, i], "o-", markersize=3, label=f"chord_{i}")
        ax.set_xlabel("V (m/s)")
        ax.set_ylabel("Chord CP")
        ax.set_title("Optimized Chord Control Points")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for i in range(n_half, n_geo):
            ax.plot(V_conv, geo_converged[:, i], "o-", markersize=3, label=f"twist_{i-n_half}")
        ax.set_xlabel("V (m/s)")
        ax.set_ylabel("Twist CP")
        ax.set_title("Optimized Twist Control Points")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    ax = axes[1, 2]
    thrust_arr = [r.aero_details.get("total_thrust", 0) for r, m in zip(results, mask) if m]
    ax.plot(V_arr[mask], thrust_arr, "o-", markersize=4, color="green")
    ax.set_xlabel("V (m/s)")
    ax.set_ylabel("Total Thrust (N)")
    ax.set_title("Total Thrust")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Geometry-Trim Optimization Sweep", fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"优化扫描结果图已保存: {save_path}")


def plot_geometry_comparison(results_fixed, results_optim, save_path="geometry_comparison.png"):
    """对比固定几何配平 vs 优化几何配平。"""
    V_fixed = np.array([r.aero_details.get("V", 0) for r in results_fixed])
    V_optim = np.array([r.V for r in results_optim])
    mask_f = np.array([r.converged for r in results_fixed])
    mask_o = np.array([r.converged for r in results_optim])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(V_fixed[mask_f], [r.alpha_deg for r, m in zip(results_fixed, mask_f) if m],
            "o-", label="Fixed Geometry", markersize=4)
    ax.plot(V_optim[mask_o], [r.alpha_deg for r, m in zip(results_optim, mask_o) if m],
            "s--", label="Optimized Geometry", markersize=4)
    ax.set_xlabel("V (m/s)")
    ax.set_ylabel("Alpha (deg)")
    ax.set_title("Angle of Attack Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(V_fixed[mask_f],
            [r.omega_front for r, m in zip(results_fixed, mask_f) if m],
            "o-", label="Fixed front", markersize=4)
    ax.plot(V_optim[mask_o],
            [r.omega_front for r, m in zip(results_optim, mask_o) if m],
            "s--", label="Optim front", markersize=4)
    ax.set_xlabel("V (m/s)")
    ax.set_ylabel("RPM")
    ax.set_title("Front Rotor Speed Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    power_fixed = []
    for r, m in zip(results_fixed, mask_f):
        if m:
            ad = r.aero_details
            o1_rad = r.omega_front * 2 * math.pi / 60
            o2_rad = r.omega_rear * 2 * math.pi / 60
            rho, D = 1.225, 0.254
            n1, n2 = r.omega_front / 60, r.omega_rear / 60
            Q1 = 0.008 * rho * abs(n1) * n1 * D**5
            Q2 = 0.008 * rho * abs(n2) * n2 * D**5
            power_fixed.append(2 * (abs(o1_rad * Q1) + abs(o2_rad * Q2)))
    power_optim = [r.power for r, m in zip(results_optim, mask_o) if m]
    ax.plot(V_fixed[mask_f], power_fixed, "o-", label="Fixed Geometry", markersize=4)
    ax.plot(V_optim[mask_o], power_optim, "s--", label="Optimized Geometry", markersize=4)
    ax.set_xlabel("V (m/s)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fixed Geometry vs Optimized Geometry Trim", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"几何对比图已保存: {save_path}")


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="几何优化配平求解器")
    parser.add_argument("--dnn", action="store_true", help="使用 DNN 代理模型")
    parser.add_argument("--V", type=float, default=None, help="单点速度 (m/s)")
    parser.add_argument("--V-min", type=float, default=2.0)
    parser.add_argument("--V-max", type=float, default=20.0)
    parser.add_argument("--V-num", type=int, default=10)
    parser.add_argument("--mass", type=float, default=3.5)
    parser.add_argument("--n-geo", type=int, default=8, help="几何参数维度")
    parser.add_argument("--compare-geometry", action="store_true",
                        help="对比固定几何配平 vs 优化几何配平")
    parser.add_argument("--output", type=str, default="optim_trim_sweep.png")
    args = parser.parse_args()

    rotor_func = None
    if args.dnn:
        from config import Config
        from adapter import load_predictor
        cfg = Config()
        model, scaler_X, scaler_Y, device = load_predictor(cfg)
        rotor_func = make_dnn_rotor_aero_full(cfg, model, scaler_X, scaler_Y, device)
        args.n_geo = len(cfg.geometry_columns)
        print(f"[DNN] 已加载模型，几何维度: {args.n_geo}")

    optimizer = GeometryTrimOptimizer(
        mass=args.mass,
        n_geometry=args.n_geo,
        rotor_aero_func=rotor_func,
    )

    if args.V is not None:
        result = optimizer.solve(V=args.V)
        optimizer.print_result(result)
        return

    V_range = np.linspace(args.V_min, args.V_max, args.V_num)
    print(f"[Optim] 速度扫描: {args.V_min} ~ {args.V_max} m/s, {args.V_num} 个点")
    results_optim = optimizer.sweep_velocity(V_range)

    print(f"\n{'V':>6} {'alpha':>8} {'Omega1':>10} {'Omega2':>10} {'Power':>10} {'OK':>4}")
    print("-" * 55)
    for r in results_optim:
        ok = "Y" if r.converged else "N"
        print(f"{r.V:>6.1f} {r.alpha_deg:>8.3f} {r.omega_front:>10.1f} "
              f"{r.omega_rear:>10.1f} {r.power:>10.2f} {ok:>4}")

    plot_optim_sweep(results_optim, save_path=args.output)

    if args.compare_geometry:
        print("\n[对比] 运行固定几何配平...")
        solver_fixed = QuadcopterTrimSolver(mass=args.mass)
        results_fixed = solver_fixed.sweep_velocity(V_range)
        cmp_path = args.output.replace(".png", "_comparison.png")
        plot_geometry_comparison(results_fixed, results_optim, save_path=cmp_path)


if __name__ == "__main__":
    main()
