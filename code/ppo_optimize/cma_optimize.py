"""CMA-ES 螺旋桨气动外形优化

无梯度演化策略, 适合低维连续空间 + 含 infeasible 区域的问题。
用作 PPO 的对比基线, 共享同一套 DNN 代理模型 + 配平求解器。

用法:
    python ppo_optimize/cma_optimize.py
    python ppo_optimize/cma_optimize.py --maxiter 300 --popsize 20
    python ppo_optimize/cma_optimize.py --dummy   # 使用解析模型
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import Config
from ppo_optimize.env import (
    PropellerDesignEnv,
    cp_to_sections,
    _make_geometry_aware_dummy_funcs,
)
from quadcopter_trim_solver import QuadcopterTrimSolver


def build_objective(V: float, mass: float, use_dnn: bool, cfg: Config = None,
                    ood_penalty: float = 0.0, constrain_to_data: bool = True):
    """构建 CMA-ES 目标函数, 返回 (objective, history_list, best_fm, cp_lo, cp_hi)。

    目标: 最大化 FM = mg·V / P_total → 最小化 -FM + ood_penalty * ood_score。
    """
    from ppo_optimize.env import CP_BOUNDS_DATA, CP_BOUNDS_WIDE, DATA_RPM_BOUNDS, DATA_ALPHA_BOUNDS_DEG

    cfg = cfg or Config()
    weight = mass * 9.81
    solver = QuadcopterTrimSolver(mass=mass)

    bounds = CP_BOUNDS_DATA if constrain_to_data else CP_BOUNDS_WIDE
    cp_lo, cp_hi = bounds[:, 0], bounds[:, 1]

    rpm_bounds = DATA_RPM_BOUNDS if constrain_to_data else None
    alpha_bounds = DATA_ALPHA_BOUNDS_DEG if constrain_to_data else None

    if use_dnn:
        from adapter import (
            load_predictor,
            make_rotor_aero_func,
            make_rotor_aero_func_with_torque,
        )
        dnn_parts = load_predictor(cfg, force_cpu=True)

    train_bounds = None
    if ood_penalty > 0 and use_dnn:
        PropellerDesignEnv._ensure_train_bounds()
        train_bounds = PropellerDesignEnv._TRAIN_BOUNDS

    n_eval = [0]
    best_fm = [0.0]
    history = []
    warm_start = [None]

    def objective(cp_raw: np.ndarray) -> float:
        n_eval[0] += 1
        cp = np.clip(cp_raw, cp_lo, cp_hi)

        if use_dnn:
            sections = cp_to_sections(cp)
            geo_list = sections.tolist()
            model, scaler_X, scaler_Y, device = dnn_parts
            trim_func = make_rotor_aero_func(
                cfg, model, scaler_X, scaler_Y, device, geometry=geo_list,
            )
            torque_func = make_rotor_aero_func_with_torque(
                cfg, model, scaler_X, scaler_Y, device, geometry=geo_list,
            )
        else:
            trim_func, torque_func = _make_geometry_aware_dummy_funcs(cp)

        solver.set_rotor_aero(trim_func)
        result = solver.solve_multi_start(
            V, x0_hint=warm_start[0],
            rpm_bounds=rpm_bounds, alpha_bounds_deg=alpha_bounds,
        )

        if not result.converged:
            return 1e6

        warm_start[0] = np.array([
            math.radians(result.alpha_deg),
            result.omega_front,
            result.omega_rear,
        ])

        alpha_rad = math.radians(result.alpha_deg)
        _, _, _, Q1 = torque_func(V, alpha_rad, result.omega_front)
        _, _, _, Q2 = torque_func(V, alpha_rad, result.omega_rear)

        omega1_rad = result.omega_front * 2.0 * math.pi / 60.0
        omega2_rad = result.omega_rear * 2.0 * math.pi / 60.0
        power = 2.0 * (abs(omega1_rad * Q1) + abs(omega2_rad * Q2))
        if power < 1e-6:
            return 1e6

        fm = weight * V / power

        ood = 0.0
        if train_bounds is not None:
            sections = cp_to_sections(cp)
            b = train_bounds
            sec_range = b["sec_hi"] - b["sec_lo"] + 1e-10
            sec_viol = (np.maximum(0, sections - b["sec_hi"])
                        + np.maximum(0, b["sec_lo"] - sections))
            geo_ood = float(np.mean(sec_viol / sec_range))

            rpm_range = b["rpm_hi"] - b["rpm_lo"]
            rpm_viol = (max(0, result.omega_front - b["rpm_hi"])
                        + max(0, b["rpm_lo"] - result.omega_front)
                        + max(0, result.omega_rear - b["rpm_hi"])
                        + max(0, b["rpm_lo"] - result.omega_rear))
            rpm_ood = rpm_viol / (2.0 * rpm_range + 1e-10)

            angle = 90.0 - result.alpha_deg
            angle_range = b["angle_hi"] - b["angle_lo"]
            angle_ood = (max(0, angle - b["angle_hi"])
                         + max(0, b["angle_lo"] - angle)) / (angle_range + 1e-10)

            ood = 0.6 * geo_ood + 0.2 * rpm_ood + 0.2 * angle_ood

        if fm > best_fm[0]:
            best_fm[0] = fm

        history.append({
            "eval": n_eval[0],
            "fm": float(fm),
            "power": float(power),
            "ood_score": float(ood),
            "alpha_deg": float(result.alpha_deg),
            "omega_front": float(result.omega_front),
            "omega_rear": float(result.omega_rear),
            "geometry": cp.tolist(),
        })

        return -fm + ood_penalty * ood

    return objective, history, best_fm, cp_lo, cp_hi


def run_cma(args):
    """执行 CMA-ES 优化。"""
    try:
        import cma
    except ImportError:
        print("[错误] 请先安装 cma: pip install cma")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    constrain = getattr(args, "constrain", True)
    objective_raw, history, best_fm, cp_lo, cp_hi = build_objective(
        args.V, args.mass, not args.dummy,
        ood_penalty=args.ood_penalty, constrain_to_data=constrain,
    )
    cp_range = cp_hi - cp_lo

    def objective_norm(x_norm):
        cp = cp_lo + np.clip(x_norm, 0.0, 1.0) * cp_range
        return objective_raw(cp)

    x0 = np.full(8, 0.5)
    opts = cma.CMAOptions()
    opts["maxiter"] = args.maxiter
    opts["popsize"] = args.popsize
    opts["bounds"] = [0, 1]
    opts["seed"] = 42
    opts["verbose"] = 1
    opts["tolfun"] = 1e-6

    print("=" * 60)
    print("  CMA-ES 螺旋桨气动外形优化")
    print("=" * 60)
    print(f"  V = {args.V} m/s | mass = {args.mass} kg")
    print(f"  maxiter = {args.maxiter} | popsize = {args.popsize}")
    print(f"  mode = {'DNN' if not args.dummy else 'dummy'}")
    print("=" * 60)

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)
    es.optimize(objective_norm)
    elapsed = time.time() - t0

    x_best = es.result.xbest
    cp_best = cp_lo + np.clip(x_best, 0.0, 1.0) * cp_range
    fm_best = -es.result.fbest

    print(f"\n{'=' * 60}")
    print(f"  优化完成! 耗时 {elapsed:.1f}s")
    print(f"  最优 FM = {fm_best:.4f}")
    print(f"  最优几何 (chord): {cp_best[:4].tolist()}")
    print(f"  最优几何 (twist): {cp_best[4:].tolist()}")
    print(f"  总评估次数: {len(history)}")
    print(f"{'=' * 60}")

    result_data = {
        "fm_best": float(fm_best),
        "geometry": cp_best.tolist(),
        "n_evals": len(history),
        "elapsed_s": float(elapsed),
        "V": args.V,
        "mass": args.mass,
    }
    with open(os.path.join(args.save_dir, "cma_best.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    with open(os.path.join(args.save_dir, "cma_history.json"), "w") as f:
        json.dump(history, f)

    print(f"结果已保存至: {args.save_dir}")
    return result_data, history


def main():
    parser = argparse.ArgumentParser(description="CMA-ES 螺旋桨外形优化")
    parser.add_argument("--V", type=float, default=10.0, help="前飞速度 (m/s)")
    parser.add_argument("--mass", type=float, default=3.5, help="无人机质量 (kg)")
    parser.add_argument("--maxiter", type=int, default=300, help="最大迭代数")
    parser.add_argument("--popsize", type=int, default=20, help="种群大小")
    parser.add_argument("--sigma0", type=float, default=0.3, help="初始步长")
    parser.add_argument("--save-dir", type=str, default="./cma_results",
                        help="结果保存目录")
    parser.add_argument("--constrain", action="store_true", default=True,
                        help="硬约束到训练数据范围 (默认开启)")
    parser.add_argument("--no-constrain", dest="constrain", action="store_false",
                        help="关闭硬约束")
    parser.add_argument("--ood-penalty", type=float, default=0.0,
                        help="OOD 软惩罚权重")
    parser.add_argument("--dummy", action="store_true",
                        help="使用 dummy 解析模型 (无需 DNN)")
    args = parser.parse_args()

    run_cma(args)


if __name__ == "__main__":
    main()
