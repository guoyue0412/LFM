"""主入口 — 全流程 CLI

用法:
    python -m optimization_v2.run                    # 完整流程: 数据→训练→优化
    python -m optimization_v2.run --step data        # 仅生成数据
    python -m optimization_v2.run --step train       # 仅训练 MoE
    python -m optimization_v2.run --step optimize    # 仅运行优化
    python -m optimization_v2.run --step evaluate    # 仅评估最优解
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from .config import OptConfig
from .synthetic_data import generate_synthetic_dataset, save_dataset
from .train_moe import train_moe, load_trained_moe
from .trim_solver import TrimSolverV2
from .cma_optimizer import run_cma_optimization
from .active_learning import ActiveLearningManager
from .evaluate import (
    plot_optimization_history,
    plot_blade_geometry,
    plot_comparison,
    print_result_table,
    save_final_report,
)
from .objective import evaluate_design


def make_moe_predict_fn(model, scaler_X, scaler_y, device):
    """构建 MoE 预测函数 (标准化输入 → 原始物理量输出)。"""

    def predict_fn(x_raw: np.ndarray):
        """
        Args:
            x_raw: shape (N, 47) 原始物理量输入

        Returns:
            mu: shape (N, 4) 原始物理量均值
            sigma: shape (N, 4) 标准化空间的标准差
        """
        x_scaled = scaler_X.transform(x_raw)
        x_t = torch.tensor(x_scaled, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            mu_scaled, sigma_scaled, _ = model(x_t)

        mu_np = mu_scaled.cpu().numpy()
        sigma_np = sigma_scaled.cpu().numpy()

        # 均值反标准化
        mu_orig = scaler_y.inverse_transform(mu_np)

        # σ 保持在标准化空间 (用于不确定度比较)
        return mu_orig, sigma_np

    return predict_fn


def step_data(cfg: OptConfig):
    """步骤 1: 生成合成数据。"""
    print("\n" + "=" * 60)
    print("  步骤 1: 生成合成训练数据")
    print("=" * 60)

    df = generate_synthetic_dataset(cfg, n_geometries=100, noise_std=0.02)
    save_dataset(df, cfg)

    print(f"\n数据统计:")
    print(df[["T", "H", "My", "Q"]].describe())
    return df


def step_train(cfg: OptConfig):
    """步骤 2: 训练 MoE 模型。"""
    print("\n" + "=" * 60)
    print("  步骤 2: 训练 MoE 气动代理模型")
    print("=" * 60)

    model, scaler_X, scaler_y = train_moe(cfg, verbose=True)
    return model, scaler_X, scaler_y


def step_optimize(cfg: OptConfig, model=None, scaler_X=None, scaler_y=None):
    """步骤 3: CMA-ES 优化。"""
    print("\n" + "=" * 60)
    print("  步骤 3: CMA-ES 配平约束优化")
    print("=" * 60)

    if model is None:
        model, scaler_X, scaler_y, device = load_trained_moe(cfg)
    else:
        device = next(model.parameters()).device

    predict_fn = make_moe_predict_fn(model, scaler_X, scaler_y, device)
    trim_solver = TrimSolverV2(cfg, predict_fn)

    # 主动学习管理器
    al_manager = ActiveLearningManager(cfg)

    # 运行优化
    opt_result = run_cma_optimization(trim_solver, cfg, verbose=True)

    # 收集高不确定度点
    if opt_result["best_trim"] and opt_result["best_trim"].converged:
        al_manager.collect_from_trim(
            opt_result["best_cp"], opt_result["best_trim"], cfg.V_cruise
        )

    if al_manager.n_pending > 0:
        print(f"\n{al_manager.summary()}")
        al_manager.export_for_llfvw()

    return opt_result, al_manager


def step_evaluate(cfg: OptConfig, opt_result: dict = None):
    """步骤 4: 评估与可视化。"""
    print("\n" + "=" * 60)
    print("  步骤 4: 评估与可视化")
    print("=" * 60)

    if opt_result is None:
        result_path = os.path.join(cfg.results_dir, "cma_result.json")
        if not os.path.exists(result_path):
            print(f"未找到优化结果: {result_path}")
            return
        with open(result_path) as f:
            saved = json.load(f)
        best_cp = np.array(saved["best_cp"])
    else:
        best_cp = opt_result["best_cp"]

    # 绘制最优桨叶几何
    plot_blade_geometry(best_cp, cfg.results_dir, label="optimized")

    # 与基准桨对比 (边界中点作为基准)
    baseline_cp = (cfg.cp_bounds[:, 0] + cfg.cp_bounds[:, 1]) / 2.0
    plot_comparison(baseline_cp, best_cp, cfg.results_dir)

    # 绘制优化历史
    if opt_result and "history" in opt_result:
        plot_optimization_history(opt_result["history"], cfg.results_dir)

    # 加载 MoE 重新评估,生成对比表和最终报告
    try:
        model, scaler_X, scaler_y, device = load_trained_moe(cfg)
        predict_fn = make_moe_predict_fn(model, scaler_X, scaler_y, device)
        trim_solver = TrimSolverV2(cfg, predict_fn)

        baseline_result = evaluate_design(baseline_cp, trim_solver, cfg)
        optimized_result = evaluate_design(best_cp, trim_solver, cfg)

        print_result_table([
            ("Baseline", baseline_result),
            ("Optimized", optimized_result),
        ], cfg)

        if optimized_result["trim_result"].converged:
            save_final_report(best_cp, optimized_result["trim_result"],
                              cfg, cfg.results_dir)
    except Exception as e:
        print(f"评估时出错: {e}")


def run_full_pipeline(cfg: OptConfig):
    """运行完整流程。"""
    t0 = time.time()

    print("\n" + "#" * 60)
    print("#  前飞配平约束螺旋桨气动外形优化 — 全流程")
    print("#" * 60)
    print(f"#  V = {cfg.V_cruise} m/s | mass = {cfg.mass} kg | η = {cfg.eta_motor_esc}")
    print(f"#  目标: min P_elec/V (单位距离能耗)")
    print("#" * 60)

    # 1. 数据
    step_data(cfg)

    # 2. 训练
    model, scaler_X, scaler_y = step_train(cfg)

    # 3. 优化
    opt_result, al_manager = step_optimize(cfg, model, scaler_X, scaler_y)

    # 4. 评估
    step_evaluate(cfg, opt_result)

    elapsed = time.time() - t0
    print(f"\n全流程完成: {elapsed:.1f}s")

    return opt_result


def main():
    parser = argparse.ArgumentParser(
        description="前飞配平约束螺旋桨气动外形优化"
    )
    parser.add_argument("--step", type=str, default=None,
                        choices=["data", "train", "optimize", "evaluate"],
                        help="仅运行指定步骤")
    parser.add_argument("--V", type=float, default=None, help="巡航速度 m/s")
    parser.add_argument("--mass", type=float, default=None, help="无人机质量 kg")
    parser.add_argument("--maxiter", type=int, default=None, help="CMA-ES 最大迭代")
    parser.add_argument("--popsize", type=int, default=None, help="CMA-ES 种群大小")
    parser.add_argument("--moe-epochs", type=int, default=None, help="MoE 训练轮数")
    parser.add_argument("--n-experts", type=int, default=None, help="MoE expert 数")
    parser.add_argument("--eta", type=float, default=None, help="电机+ESC 效率")
    parser.add_argument("--excel", type=str, default=None,
                        help="Excel/CSV 数据文件路径 (覆盖默认合成数据)")
    parser.add_argument("--sheet", default=0, help="Excel 工作表名或索引")
    parser.add_argument("--inspect", type=str, default=None,
                        help="检视 Excel 结构 (传入文件路径)")

    args = parser.parse_args()

    if args.inspect:
        from .data_loader import inspect_excel
        inspect_excel(args.inspect, args.sheet)
        return

    cfg = OptConfig()
    if args.V:
        cfg.V_cruise = args.V
    if args.mass:
        cfg.mass = args.mass
    if args.maxiter:
        cfg.cma_maxiter = args.maxiter
    if args.popsize:
        cfg.cma_popsize = args.popsize
    if args.moe_epochs:
        cfg.moe_epochs = args.moe_epochs
    if args.n_experts:
        cfg.moe_n_experts = args.n_experts
    if args.eta:
        cfg.eta_motor_esc = args.eta
    if args.excel:
        cfg.excel_path = args.excel
        cfg.excel_sheet = args.sheet

    if args.step == "data":
        step_data(cfg)
    elif args.step == "train":
        step_train(cfg)
    elif args.step == "optimize":
        step_optimize(cfg)
    elif args.step == "evaluate":
        step_evaluate(cfg)
    else:
        run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
