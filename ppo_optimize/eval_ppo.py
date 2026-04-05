"""PPO / CMA-ES 模型评估与论文级可视化

用法:
    python ppo_optimize/eval_ppo.py
    python ppo_optimize/eval_ppo.py --model ppo_models/best_model/best_model.zip --episodes 20
    python ppo_optimize/eval_ppo.py --cma-dir cma_results   # 加入 CMA-ES 对比
"""

import argparse
import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from ppo_optimize.env import PropellerDesignEnv, cp_to_sections, _R_NORM
from plot_style import (
    apply_style, COLORS, PALETTE, savefig, make_fig, label_subplots,
    annotate_stats,
)


apply_style()

_METRIC_LABEL = r"$FM = mg\,V\,/\,P$"


# ============================================================================
#  评估
# ============================================================================

def evaluate_model(model_path: str, n_episodes: int = 10, V: float = 10.0,
                    use_dnn: bool = True):
    """运行多个 episode 并收集统计信息。"""
    env = PropellerDesignEnv(V=V, use_dnn=use_dnn)

    model = PPO.load(model_path)
    print(f"已加载模型: {model_path}")

    episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_data = {
            "rewards": [],
            "geometries": [env.cp.copy()],
            "cruise_metrics": [],
            "powers": [],
            "alphas": [],
            "omegas_front": [],
            "omegas_rear": [],
        }

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_data["rewards"].append(reward)
            ep_data["geometries"].append(env.cp.copy())

            if info.get("converged"):
                ep_data["cruise_metrics"].append(info.get("cruise_metric", 0))
                ep_data["powers"].append(info.get("power", 0))
                ep_data["alphas"].append(info.get("alpha_deg", 0))
                ep_data["omegas_front"].append(info.get("omega_front", 0))
                ep_data["omegas_rear"].append(info.get("omega_rear", 0))

        ep_data["total_reward"] = sum(ep_data["rewards"])
        ep_data["final_cruise"] = (
            ep_data["cruise_metrics"][-1] if ep_data["cruise_metrics"] else 0
        )
        ep_data["final_power"] = (
            ep_data["powers"][-1] if ep_data["powers"] else 0
        )
        ep_data["final_geometry"] = env.cp.copy()
        ep_data["steps"] = len(ep_data["rewards"])
        episodes.append(ep_data)

        print(f"  Episode {ep+1:3d}: steps={ep_data['steps']:3d}  "
              f"reward={ep_data['total_reward']:8.3f}  "
              f"FM={ep_data['final_cruise']:.4f}  "
              f"power={ep_data['final_power']:.1f}W")

    return episodes


def find_best_episode(episodes):
    """找到巡航指标最优的 episode。"""
    best_idx = max(range(len(episodes)),
                   key=lambda i: episodes[i]["final_cruise"])
    return best_idx, episodes[best_idx]


# ============================================================================
#  可视化
# ============================================================================

def plot_training_reward_curve(history_path: str, save_path: str):
    """绘制训练过程中的 FM 收敛曲线。"""
    with open(history_path, "r") as f:
        history = json.load(f)

    timesteps = [h["timestep"] for h in history]
    fm_vals = [h["cruise_metric"] for h in history]
    power_vals = [h["power"] for h in history]

    window = min(50, len(fm_vals) // 5 + 1)
    if window > 1:
        fm_smooth = np.convolve(fm_vals, np.ones(window) / window,
                                mode="valid")
        ts_smooth = timesteps[window - 1:]
    else:
        fm_smooth = fm_vals
        ts_smooth = timesteps

    fig, axes = make_fig(1, 2, scale=1.1)

    ax = axes[0]
    ax.scatter(timesteps, fm_vals, s=2, alpha=0.15,
               color=COLORS["light_gray"], rasterized=True)
    ax.plot(ts_smooth, fm_smooth, color=COLORS["primary"], linewidth=1.5)
    ax.set_xlabel("Training Timestep")
    ax.set_ylabel(_METRIC_LABEL)
    ax.set_title("Figure of Merit (PPO)")

    ax = axes[1]
    if window > 1:
        power_smooth = np.convolve(power_vals, np.ones(window) / window,
                                   mode="valid")
    else:
        power_smooth = power_vals
    ax.scatter(timesteps, power_vals, s=2, alpha=0.15,
               color=COLORS["light_gray"], rasterized=True)
    ax.plot(ts_smooth, power_smooth, color=COLORS["secondary"], linewidth=1.5)
    ax.set_xlabel("Training Timestep")
    ax.set_ylabel("Power (W)")
    ax.set_title("Total Power")

    label_subplots(axes)
    savefig(fig, save_path)
    print(f"训练曲线已保存: {save_path}")


def plot_geometry_comparison(baseline_cp, optimized_cp, save_path: str,
                             cma_cp=None):
    """对比基线 / PPO / CMA-ES 几何 (柱状图)。"""
    n_cp = len(baseline_cp)
    n_chord = n_cp // 2
    labels_chord = [f"$c_{i}$" for i in range(n_chord)]
    labels_twist = [f"$\\theta_{i}$" for i in range(n_cp - n_chord)]

    n_series = 3 if cma_cp is not None else 2
    width = 0.8 / n_series
    fig, axes = make_fig(1, 2, scale=1.1)

    for panel, (ax, labels, sl) in enumerate(zip(
        axes,
        [labels_chord, labels_twist],
        [slice(0, n_chord), slice(n_chord, None)],
    )):
        x = np.arange(len(labels))
        offsets = np.linspace(-width * (n_series - 1) / 2,
                              width * (n_series - 1) / 2, n_series)
        ax.bar(x + offsets[0], baseline_cp[sl], width, label="Baseline",
               color=COLORS["light_gray"], edgecolor="k", linewidth=0.5)
        ax.bar(x + offsets[1], optimized_cp[sl], width, label="PPO",
               color=COLORS["primary"], alpha=0.85)
        if cma_cp is not None:
            ax.bar(x + offsets[2], cma_cp[sl], width, label="CMA-ES",
                   color=COLORS["secondary"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Chord (m)" if panel == 0 else "Twist (deg)")
        ax.set_title("Chord Control Points" if panel == 0
                      else "Twist Control Points")
        ax.legend(fontsize=7)

    label_subplots(axes)
    savefig(fig, save_path)
    print(f"几何对比图已保存: {save_path}")


def plot_blade_shape(cp, save_path: str, cp_cma=None):
    """将 B-spline 控制点展开为桨叶弦长/扭转分布。"""
    r_norm = _R_NORM
    sec = cp_to_sections(cp)
    chord_ppo = sec[:22]
    twist_ppo = sec[22:]

    fig, axes = make_fig(1, 2, scale=1.1)

    ax = axes[0]
    ax.plot(r_norm, chord_ppo * 1000, "-o", markersize=3,
            color=COLORS["primary"], label="PPO Optimized")
    if cp_cma is not None:
        sec_c = cp_to_sections(np.array(cp_cma))
        ax.plot(r_norm, sec_c[:22] * 1000, "--s", markersize=3,
                color=COLORS["secondary"], label="CMA-ES")
    ax.set_xlabel("$r / R$")
    ax.set_ylabel("Chord (mm)")
    ax.set_title("Blade Chord Distribution")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(r_norm, twist_ppo, "-o", markersize=3,
            color=COLORS["primary"], label="PPO Optimized")
    if cp_cma is not None:
        ax.plot(r_norm, sec_c[22:], "--s", markersize=3,
                color=COLORS["secondary"], label="CMA-ES")
    ax.set_xlabel("$r / R$")
    ax.set_ylabel("Twist (deg)")
    ax.set_title("Blade Twist Distribution")
    ax.legend(fontsize=7)

    label_subplots(axes)
    savefig(fig, save_path)
    print(f"桨叶形状图已保存: {save_path}")


def plot_convergence_comparison(ppo_history_path: str, cma_history_path: str,
                                save_path: str):
    """PPO vs CMA-ES 收敛曲线对比。"""
    fig, ax = make_fig(1, 1, scale=0.8)
    ax = np.asarray(ax).flat[0]

    if os.path.exists(ppo_history_path):
        with open(ppo_history_path, "r") as f:
            ppo_h = json.load(f)
        ppo_fm = [h["cruise_metric"] for h in ppo_h]
        ppo_best = np.maximum.accumulate(ppo_fm)
        ax.plot(range(1, len(ppo_best) + 1), ppo_best,
                color=COLORS["primary"], linewidth=1.5, label="PPO (best so far)")

    if os.path.exists(cma_history_path):
        with open(cma_history_path, "r") as f:
            cma_h = json.load(f)
        cma_fm = [h["fm"] for h in cma_h]
        cma_best = np.maximum.accumulate(cma_fm)
        ax.plot(range(1, len(cma_best) + 1), cma_best,
                color=COLORS["secondary"], linewidth=1.5,
                label="CMA-ES (best so far)")

    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel(_METRIC_LABEL)
    ax.set_title("Optimization Convergence")
    ax.legend()
    savefig(fig, save_path)
    print(f"收敛对比图已保存: {save_path}")


def plot_optimization_trajectory(episode_data, save_path: str):
    """绘制单个 episode 内的优化轨迹。"""
    geos = np.array(episode_data["geometries"])
    steps = np.arange(len(geos))
    n_chord = geos.shape[1] // 2

    fig, axes = make_fig(2, 2, scale=1.0)

    ax = axes[0, 0]
    for i in range(n_chord):
        ax.plot(steps, geos[:, i], marker=".", markersize=3,
                color=PALETTE[i], label=f"$c_{i}$")
    ax.set_ylabel("Chord (m)")
    ax.set_title("Chord Evolution")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for i in range(n_chord, geos.shape[1]):
        ax.plot(steps, geos[:, i], marker=".", markersize=3,
                color=PALETTE[i - n_chord],
                label=f"$\\theta_{{{i - n_chord}}}$")
    ax.set_ylabel("Twist (deg)")
    ax.set_title("Twist Evolution")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    if episode_data["cruise_metrics"]:
        ax.plot(range(1, len(episode_data["cruise_metrics"]) + 1),
                episode_data["cruise_metrics"],
                color=COLORS["accent1"], marker=".", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel(_METRIC_LABEL)
    ax.set_title("Figure of Merit")

    ax = axes[1, 1]
    if episode_data["powers"]:
        ax.plot(range(1, len(episode_data["powers"]) + 1),
                episode_data["powers"],
                color=COLORS["secondary"], marker=".", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power")

    label_subplots(axes)
    savefig(fig, save_path)
    print(f"优化轨迹图已保存: {save_path}")


def plot_episode_summary(episodes, save_path: str):
    """多 episode 总结统计。"""
    total_rewards = [ep["total_reward"] for ep in episodes]
    final_cruise = [ep["final_cruise"] for ep in episodes]
    final_power = [ep["final_power"] for ep in episodes]
    steps = [ep["steps"] for ep in episodes]

    fig, axes = make_fig(2, 2, scale=1.0)
    ep_idx = np.arange(1, len(episodes) + 1)

    ax = axes[0, 0]
    ax.bar(ep_idx, total_rewards, color=COLORS["primary"], alpha=0.8)
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    annotate_stats(ax, f"mean={np.mean(total_rewards):.2f}\n"
                       f"max={np.max(total_rewards):.2f}")

    ax = axes[0, 1]
    ax.bar(ep_idx, final_cruise, color=COLORS["accent1"], alpha=0.8)
    ax.set_ylabel(_METRIC_LABEL)
    ax.set_title("Final Figure of Merit")
    annotate_stats(ax, f"mean={np.mean(final_cruise):.4f}\n"
                       f"max={np.max(final_cruise):.4f}")

    ax = axes[1, 0]
    ax.bar(ep_idx, final_power, color=COLORS["secondary"], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Power (W)")
    ax.set_title("Final Power")

    ax = axes[1, 1]
    ax.bar(ep_idx, steps, color=COLORS["accent2"], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")

    label_subplots(axes)
    savefig(fig, save_path)
    print(f"评估总结图已保存: {save_path}")


# ============================================================================
#  主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PPO/CMA-ES 评估与可视化")
    parser.add_argument("--model", type=str, default=None,
                        help="PPO 模型路径 (.zip)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="评估 episode 数")
    parser.add_argument("--V", type=float, default=10.0, help="前飞速度")
    parser.add_argument("--save-dir", type=str, default="./ppo_models",
                        help="PPO 结果目录")
    parser.add_argument("--cma-dir", type=str, default="./cma_results",
                        help="CMA-ES 结果目录")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="图表输出目录")
    parser.add_argument("--dummy", action="store_true",
                        help="使用 dummy 解析模型")
    args = parser.parse_args()

    if args.model is None:
        candidates = [
            os.path.join(args.save_dir, "best_model", "best_model.zip"),
            os.path.join(args.save_dir, "final_model.zip"),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.model = c
                break
        if args.model is None:
            print("[警告] 未找到 PPO 模型，跳过 PPO 评估")

    out_dir = args.output_dir or os.path.join(args.save_dir, "eval_plots")
    os.makedirs(out_dir, exist_ok=True)

    # --- CMA-ES 结果 ---
    cma_best_path = os.path.join(args.cma_dir, "cma_best.json")
    cma_geo = None
    if os.path.exists(cma_best_path):
        with open(cma_best_path, "r") as f:
            cma_data = json.load(f)
        cma_geo = cma_data.get("geometry")
        print(f"CMA-ES 最优 FM = {cma_data.get('fm_best', 0):.4f}")

    # --- PPO 训练曲线 ---
    history_path = os.path.join(args.save_dir, "training_history.json")
    if os.path.exists(history_path):
        plot_training_reward_curve(
            history_path,
            os.path.join(out_dir, "training_curve.png"),
        )

    # --- PPO vs CMA-ES 收敛对比 ---
    cma_hist_path = os.path.join(args.cma_dir, "cma_history.json")
    if os.path.exists(history_path) or os.path.exists(cma_hist_path):
        plot_convergence_comparison(
            history_path, cma_hist_path,
            os.path.join(out_dir, "convergence_comparison.png"),
        )

    # --- PPO 评估 ---
    episodes = None
    best_ep = None
    if args.model is not None:
        print(f"\n{'='*60}")
        print(f"  PPO 评估 — {args.episodes} episodes @ V={args.V} m/s")
        print(f"{'='*60}\n")

        episodes = evaluate_model(args.model, n_episodes=args.episodes,
                                  V=args.V, use_dnn=not args.dummy)

        best_idx, best_ep = find_best_episode(episodes)
        print(f"\n最优 Episode: #{best_idx + 1}")
        print(f"  FM:    {best_ep['final_cruise']:.4f}")
        print(f"  功率:  {best_ep['final_power']:.1f} W")
        print(f"  几何:  {best_ep['final_geometry']}")

        plot_episode_summary(episodes,
                             os.path.join(out_dir, "episode_summary.png"))
        plot_optimization_trajectory(
            best_ep, os.path.join(out_dir, "best_trajectory.png"),
        )

    # --- 几何对比 ---
    baseline_geo = (PropellerDesignEnv.CP_BOUNDS[:, 0]
                    + PropellerDesignEnv.CP_BOUNDS[:, 1]) / 2.0

    best_design_path = os.path.join(args.save_dir, "best_design.json")
    ppo_geo = None
    if os.path.exists(best_design_path):
        with open(best_design_path, "r") as f:
            ppo_info = json.load(f)
        if "geometry" in ppo_info:
            ppo_geo = np.array(ppo_info["geometry"])

    if ppo_geo is not None:
        plot_geometry_comparison(
            baseline_geo, ppo_geo,
            os.path.join(out_dir, "geometry_comparison.png"),
            cma_cp=np.array(cma_geo) if cma_geo else None,
        )
        plot_blade_shape(
            ppo_geo,
            os.path.join(out_dir, "blade_shape.png"),
            cp_cma=cma_geo,
        )
    elif cma_geo is not None:
        plot_blade_shape(
            np.array(cma_geo),
            os.path.join(out_dir, "blade_shape.png"),
        )

    # --- 保存评估汇总 ---
    summary = {}
    if episodes is not None:
        summary["ppo"] = {
            "n_episodes": len(episodes),
            "mean_reward": float(np.mean([e["total_reward"] for e in episodes])),
            "max_fm": float(np.max([e["final_cruise"] for e in episodes])),
            "mean_fm": float(np.mean([e["final_cruise"] for e in episodes])),
            "best_geometry": best_ep["final_geometry"].tolist(),
        }
    if cma_geo is not None:
        summary["cma_es"] = {
            "fm_best": cma_data.get("fm_best", 0),
            "geometry": cma_geo,
            "n_evals": cma_data.get("n_evals", 0),
        }

    summary_path = os.path.join(out_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()
