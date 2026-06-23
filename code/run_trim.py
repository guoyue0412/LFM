"""配平求解集成脚本

用法:
    python run_trim.py                  # 使用 dummy 气动模型运行
    python run_trim.py --dnn            # 使用 DNN 代理模型运行
    python run_trim.py --compare        # 同时运行 dummy 和 DNN 并对比
    python run_trim.py --V 10           # 指定单点速度
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadcopter_trim_solver import QuadcopterTrimSolver, TrimResult


def run_dummy(V_range: np.ndarray, **solver_kwargs):
    """使用 dummy 气动模型进行速度扫描。"""
    solver = QuadcopterTrimSolver(**solver_kwargs)
    return solver.sweep_velocity(V_range)


def run_dnn(V_range: np.ndarray, geometry=None, **solver_kwargs):
    """使用 DNN 代理模型进行速度扫描。"""
    from config import Config
    from adapter import load_predictor, make_rotor_aero_func

    cfg = Config()
    model, scaler_X, scaler_Y, device = load_predictor(cfg)

    rotor_func = make_rotor_aero_func(
        cfg, model, scaler_X, scaler_Y, device, geometry=geometry
    )

    solver = QuadcopterTrimSolver(**solver_kwargs)
    solver.set_rotor_aero(rotor_func)
    return solver.sweep_velocity(V_range)


def extract_sweep_data(results: list):
    """从结果列表中提取绘图数据。"""
    V_list, alpha_list, omega1_list, omega2_list = [], [], [], []
    thrust_list, converged_list = [], []

    for r in results:
        V_list.append(r.aero_details.get("V", 0))
        alpha_list.append(r.alpha_deg)
        omega1_list.append(r.omega_front)
        omega2_list.append(r.omega_rear)
        thrust_list.append(r.aero_details.get("total_thrust", 0))
        converged_list.append(r.converged)

    return {
        "V": np.array(V_list),
        "alpha": np.array(alpha_list),
        "omega1": np.array(omega1_list),
        "omega2": np.array(omega2_list),
        "thrust": np.array(thrust_list),
        "converged": np.array(converged_list),
    }


def plot_sweep(data_dict: dict, save_path: str = "trim_sweep.png"):
    """绘制速度扫描的 alpha-V, Omega-V, Thrust-V 曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for label, data in data_dict.items():
        V = data["V"]
        mask = data["converged"]

        ax = axes[0, 0]
        ax.plot(V[mask], data["alpha"][mask], "o-", label=label, markersize=4)
        ax.set_ylabel("Alpha (deg)")
        ax.set_title("Angle of Attack vs Velocity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(V[mask], data["omega1"][mask], "o-", label=f"{label} front", markersize=4)
        ax.plot(V[mask], data["omega2"][mask], "s--", label=f"{label} rear", markersize=4)
        ax.set_ylabel("RPM")
        ax.set_title("Rotor Speed vs Velocity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(V[mask], data["thrust"][mask], "o-", label=label, markersize=4)
        ax.set_xlabel("V (m/s)")
        ax.set_ylabel("Total Thrust (N)")
        ax.set_title("Total Thrust vs Velocity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        conv_pct = np.cumsum(mask) / np.arange(1, len(mask) + 1) * 100
        ax.plot(V, conv_pct, "o-", label=label, markersize=4)
        ax.set_xlabel("V (m/s)")
        ax.set_ylabel("Convergence Rate (%)")
        ax.set_title("Convergence Rate")
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Quadcopter Trim Sweep Results", fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"扫描结果图已保存: {save_path}")


def print_sweep_table(results: list, label: str = ""):
    """打印速度扫描结果表格。"""
    print(f"\n{'=' * 65}")
    print(f"  {label} Trim Sweep Results")
    print(f"{'=' * 65}")
    print(f"{'V (m/s)':>10} {'alpha (deg)':>12} {'Omega1 (RPM)':>14} {'Omega2 (RPM)':>14} {'OK':>4}")
    print("-" * 65)
    for r in results:
        v = r.aero_details.get("V", 0)
        ok = "Y" if r.converged else "N"
        print(f"{v:>10.1f} {r.alpha_deg:>12.4f} {r.omega_front:>14.1f} {r.omega_rear:>14.1f} {ok:>4}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="配平求解集成脚本")
    parser.add_argument("--dnn", action="store_true", help="使用 DNN 代理模型")
    parser.add_argument("--compare", action="store_true", help="对比 dummy vs DNN")
    parser.add_argument("--V", type=float, default=None, help="单点速度 (m/s)")
    parser.add_argument("--V-min", type=float, default=2.0, help="扫描最小速度")
    parser.add_argument("--V-max", type=float, default=20.0, help="扫描最大速度")
    parser.add_argument("--V-num", type=int, default=10, help="扫描速度点数")
    parser.add_argument("--mass", type=float, default=3.5, help="无人机质量 (kg)")
    parser.add_argument("--output", type=str, default="trim_sweep.png", help="输出图片路径")
    args = parser.parse_args()

    solver_kwargs = {"mass": args.mass}

    if args.V is not None:
        # ---- 单点求解 ----
        solver = QuadcopterTrimSolver(**solver_kwargs)
        if args.dnn:
            from config import Config
            from adapter import load_predictor, make_rotor_aero_func
            cfg = Config()
            model, scaler_X, scaler_Y, device = load_predictor(cfg)
            rotor_func = make_rotor_aero_func(cfg, model, scaler_X, scaler_Y, device)
            solver.set_rotor_aero(rotor_func)
            print("[DNN] 已加载 DNN 代理模型")

        result = solver.solve(V=args.V)
        solver.print_result(result)
        return

    # ---- 速度扫描 ----
    V_range = np.linspace(args.V_min, args.V_max, args.V_num)
    data_dict = {}

    if args.compare or not args.dnn:
        print("[Dummy] 使用解析气动模型扫描...")
        results_dummy = run_dummy(V_range, **solver_kwargs)
        data_dict["Dummy"] = extract_sweep_data(results_dummy)
        print_sweep_table(results_dummy, "Dummy")

    if args.compare or args.dnn:
        print("[DNN] 使用 DNN 代理模型扫描...")
        results_dnn = run_dnn(V_range, **solver_kwargs)
        data_dict["DNN"] = extract_sweep_data(results_dnn)
        print_sweep_table(results_dnn, "DNN")

    plot_sweep(data_dict, save_path=args.output)


if __name__ == "__main__":
    main()
