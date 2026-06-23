"""评估与可视化模块"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import OptConfig
from .geometry import cp_to_sections, split_sections, GEOMETRY_R
from .power import compute_power, compute_range, compute_fm
from .trim_solver import TrimResultV2


def _plot_blade_axes(ax_chord, ax_twist, sections: np.ndarray,
                     label: str, **plot_kwargs):
    """在给定的 chord/twist 轴对上绘制单个桨叶。"""
    chord, twist = split_sections(sections)
    r_mm = GEOMETRY_R * 1000

    ax_chord.plot(r_mm, chord * 1000, label=label, **plot_kwargs)
    ax_twist.plot(r_mm, twist, label=label, **plot_kwargs)


def _setup_blade_axes(ax_chord, ax_twist, title_chord: str, title_twist: str):
    ax_chord.set_xlabel("r (mm)")
    ax_chord.set_ylabel("Chord (mm)")
    ax_chord.set_title(title_chord)
    ax_chord.legend()
    ax_chord.grid(True, alpha=0.3)

    ax_twist.set_xlabel("r (mm)")
    ax_twist.set_ylabel("Twist (deg)")
    ax_twist.set_title(title_twist)
    ax_twist.legend()
    ax_twist.grid(True, alpha=0.3)


def plot_optimization_history(history: dict, save_dir: str):
    """绘制优化历史曲线。"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 目标函数
    ax = axes[0, 0]
    ax.semilogy(history["best_J"], "b-", linewidth=1.5)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best J")
    ax.set_title("Objective Function Convergence")
    ax.grid(True, alpha=0.3)

    # 电功率
    ax = axes[0, 1]
    P_valid = [p for p in history["P_elec"] if 0 < p < 1e5]
    if P_valid:
        ax.plot(P_valid, "r.", markersize=1, alpha=0.3)
        ax.axhline(min(P_valid), color="r", linestyle="--", alpha=0.7, label=f"min={min(P_valid):.1f}W")
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("P_elec (W)")
        ax.set_title("Electric Power")
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 不确定度
    ax = axes[1, 0]
    sigma_valid = [s for s in history["sigma_mean"] if s > 0]
    if sigma_valid:
        ax.plot(sigma_valid, "g.", markersize=1, alpha=0.3)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("σ_mean")
        ax.set_title("MoE Uncertainty")
    ax.grid(True, alpha=0.3)

    # 收敛率
    ax = axes[1, 1]
    if history.get("converged_rate"):
        ax.plot(history["converged_rate"], "k-", linewidth=1)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Convergence Rate")
        ax.set_title("Trim Convergence Rate per Generation")
        ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "optimization_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"优化历史图: {path}")


def plot_blade_geometry(cp: np.ndarray, save_dir: str, label: str = "optimized"):
    """绘制桨叶几何分布 (弦长 + 扭角 vs 径向位置)。"""
    os.makedirs(save_dir, exist_ok=True)
    sections = cp_to_sections(cp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_blade_axes(ax1, ax2, sections, label, marker="o", markersize=4)
    _setup_blade_axes(ax1, ax2, "Chord Distribution", "Twist Distribution")

    fig.tight_layout()
    path = os.path.join(save_dir, f"blade_geometry_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"桨叶几何图: {path}")


def plot_comparison(cp_baseline: np.ndarray, cp_optimized: np.ndarray, save_dir: str):
    """对比基准桨和优化桨的几何。"""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_blade_axes(ax1, ax2, cp_to_sections(cp_baseline), "Baseline",
                     linestyle="--", linewidth=1.5, color="b")
    _plot_blade_axes(ax1, ax2, cp_to_sections(cp_optimized), "Optimized",
                     linewidth=2, color="r")
    _setup_blade_axes(ax1, ax2,
                      "Chord Distribution Comparison",
                      "Twist Distribution Comparison")

    fig.tight_layout()
    path = os.path.join(save_dir, "geometry_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"几何对比图: {path}")


def print_result_table(results: list, cfg: OptConfig):
    """打印优化结果对比表。"""
    print(f"\n{'='*80}")
    print(f"{'设计':>8} {'P_elec(W)':>10} {'P/V(J/m)':>10} {'FM':>8} "
          f"{'α(°)':>7} {'Ω1(RPM)':>9} {'Ω2(RPM)':>9} {'σ_mean':>8} {'Range(km)':>10}")
    print("-" * 80)

    for name, result in results:
        tr = result["trim_result"]
        if tr.converged:
            power = compute_power(tr, cfg)
            fm = compute_fm(tr, cfg)
            rng = compute_range(power["P_elec"], cfg.V_cruise, cfg.E_battery)
            print(f"{name:>8} {power['P_elec']:>10.1f} {power['P_elec_per_V']:>10.2f} "
                  f"{fm:>8.4f} {tr.alpha_deg:>7.2f} {tr.omega_front:>9.0f} "
                  f"{tr.omega_rear:>9.0f} {tr.sigma_mean:>8.4f} {rng/1000:>10.1f}")
        else:
            print(f"{name:>8} {'FAILED':>10} {'-':>10} {'-':>8} "
                  f"{'-':>7} {'-':>9} {'-':>9} {'-':>8} {'-':>10}")

    print("=" * 80)


def save_final_report(best_cp: np.ndarray, best_trim: TrimResultV2,
                      cfg: OptConfig, save_dir: str):
    """保存最终优化报告。"""
    os.makedirs(save_dir, exist_ok=True)

    power = compute_power(best_trim, cfg)
    rng = compute_range(power["P_elec"], cfg.V_cruise, cfg.E_battery)
    fm = compute_fm(best_trim, cfg)

    report = {
        "design": {
            "cp_chord": best_cp[:4].tolist(),
            "cp_twist": best_cp[4:].tolist(),
        },
        "trim": {
            "alpha_deg": best_trim.alpha_deg,
            "omega_front_rpm": best_trim.omega_front,
            "omega_rear_rpm": best_trim.omega_rear,
            "residual_norm": best_trim.residual_norm,
        },
        "performance": {
            "P_aero_W": power["P_aero"],
            "P_elec_W": power["P_elec"],
            "P_elec_per_V_Jm": power["P_elec_per_V"],
            "FM": fm,
            "range_km": rng / 1000,
            "endurance_min": rng / (cfg.V_cruise * 60) if cfg.V_cruise > 0 else 0,
        },
        "uncertainty": {
            "sigma_mean": best_trim.sigma_mean,
            "sigma_max": best_trim.sigma_max,
        },
        "config": {
            "V_cruise_ms": cfg.V_cruise,
            "mass_kg": cfg.mass,
            "eta_motor_esc": cfg.eta_motor_esc,
            "E_battery_Wh": cfg.E_battery / 3600,
        },
    }

    path = os.path.join(save_dir, "final_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"最终报告: {path}")
    return report
