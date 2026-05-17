"""CMA-ES 优化器 — 配平约束下最小化 P_elec/V

在归一化空间 [0,1]^8 中搜索,内部映射回物理 CP 空间。
"""

import json
import os
import time

import cma
import numpy as np

from .config import OptConfig
from .geometry import normalized_to_cp, cp_to_normalized
from .objective import evaluate_design
from .trim_solver import TrimSolverV2


def run_cma_optimization(
    trim_solver: TrimSolverV2,
    cfg: OptConfig,
    x0: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """运行 CMA-ES 优化。

    Args:
        trim_solver: 配平求解器 (已绑定 MoE)
        cfg: 全局配置
        x0: 初始 CP (物理空间), None 则用边界中点

    Returns:
        dict with best_cp, best_J, best_P_elec, history, trim_result
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    if x0 is None:
        x0 = (cfg.cp_bounds[:, 0] + cfg.cp_bounds[:, 1]) / 2.0

    x0_norm = cp_to_normalized(x0, cfg.cp_bounds)

    history = {
        "J": [],
        "P_elec": [],
        "P_elec_per_V": [],
        "sigma_mean": [],
        "converged_rate": [],
        "best_J": [],
    }

    best_J = float("inf")
    best_cp = x0.copy()
    best_trim = None
    eval_count = 0
    prev_trim = None

    def objective_fn(x_norm):
        nonlocal eval_count, best_J, best_cp, best_trim, prev_trim

        cp = normalized_to_cp(np.asarray(x_norm), cfg.cp_bounds)
        result = evaluate_design(cp, trim_solver, cfg, prev_trim=prev_trim)
        eval_count += 1

        if result["trim_result"].converged:
            prev_trim = result["trim_result"]

        J = result["J"]
        if J < best_J:
            best_J = J
            best_cp = cp.copy()
            best_trim = result["trim_result"]

        history["J"].append(J)
        history["P_elec"].append(result["P_elec"] if result["P_elec"] != float("inf") else -1)
        history["P_elec_per_V"].append(result["P_elec_per_V"] if result["P_elec_per_V"] != float("inf") else -1)
        history["sigma_mean"].append(result["trim_result"].sigma_mean)
        history["best_J"].append(best_J)

        return J

    if verbose:
        print("=" * 60)
        print("  CMA-ES 配平约束螺旋桨外形优化")
        print("=" * 60)
        print(f"  V = {cfg.V_cruise} m/s | mass = {cfg.mass} kg")
        print(f"  maxiter = {cfg.cma_maxiter} | popsize = {cfg.cma_popsize}")
        print(f"  sigma0 = {cfg.cma_sigma0}")
        print(f"  目标: min P_elec/V + penalties")
        print("=" * 60)

    t0 = time.time()

    opts = {
        "maxiter": cfg.cma_maxiter,
        "popsize": cfg.cma_popsize,
        "bounds": [0.0, 1.0],
        "seed": 42,
        "verbose": -9,  # 静默
        "tolfun": 1e-8,
    }

    es = cma.CMAEvolutionStrategy(x0_norm.tolist(), cfg.cma_sigma0, opts)

    generation = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_fn(x) for x in solutions]
        es.tell(solutions, fitnesses)
        generation += 1

        converged_count = sum(1 for j in fitnesses if j < cfg.penalty_trim_fail)
        history["converged_rate"].append(converged_count / len(fitnesses))

        if verbose and generation % 10 == 0:
            elapsed = time.time() - t0
            best_gen = min(fitnesses)
            conv_rate = converged_count / len(fitnesses) * 100
            print(f"  Gen {generation:4d} | best_gen={best_gen:.4f} "
                  f"best_all={best_J:.4f} conv={conv_rate:.0f}% "
                  f"[{elapsed:.0f}s, {eval_count} evals]")

    elapsed = time.time() - t0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  优化完成: {elapsed:.1f}s, {eval_count} 次评估")
        print(f"  最优 J = {best_J:.6f}")
        if best_trim and best_trim.converged:
            from .power import compute_power, compute_range
            power = compute_power(best_trim, cfg)
            rng = compute_range(power["P_elec"], cfg.V_cruise, cfg.E_battery)
            print(f"  P_elec = {power['P_elec']:.1f} W")
            print(f"  P_elec/V = {power['P_elec_per_V']:.2f} J/m")
            print(f"  航程 = {rng/1000:.1f} km")
            print(f"  α = {best_trim.alpha_deg:.2f}°")
            print(f"  Ω_front = {best_trim.omega_front:.0f} RPM")
            print(f"  Ω_rear = {best_trim.omega_rear:.0f} RPM")
            print(f"  σ_mean = {best_trim.sigma_mean:.4f}")
        print(f"  最优 CP: {best_cp}")
        print(f"{'='*60}")

    # 保存结果
    result = {
        "best_cp": best_cp.tolist(),
        "best_J": best_J,
        "eval_count": eval_count,
        "elapsed_s": elapsed,
        "generations": generation,
        "config": {
            "V_cruise": cfg.V_cruise,
            "mass": cfg.mass,
            "eta": cfg.eta_motor_esc,
            "maxiter": cfg.cma_maxiter,
            "popsize": cfg.cma_popsize,
        },
    }

    if best_trim and best_trim.converged:
        from .power import compute_power, compute_range
        power = compute_power(best_trim, cfg)
        result["best_P_elec"] = power["P_elec"]
        result["best_P_elec_per_V"] = power["P_elec_per_V"]
        result["best_range_m"] = compute_range(power["P_elec"], cfg.V_cruise, cfg.E_battery)
        result["best_alpha_deg"] = best_trim.alpha_deg
        result["best_omega_front"] = best_trim.omega_front
        result["best_omega_rear"] = best_trim.omega_rear
        result["best_sigma_mean"] = best_trim.sigma_mean

    save_path = os.path.join(cfg.results_dir, "cma_result.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"  结果已保存: {save_path}")

    return {
        "best_cp": best_cp,
        "best_J": best_J,
        "best_trim": best_trim,
        "history": history,
        "result": result,
    }
