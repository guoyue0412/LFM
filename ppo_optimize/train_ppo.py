"""PPO 训练脚本 — 螺旋桨气动外形优化

用法:
    python ppo_optimize/train_ppo.py
    python ppo_optimize/train_ppo.py --total-timesteps 200000 --quick
    python ppo_optimize/train_ppo.py --n-envs 4 --ood-penalty 2.0   # 并行 + OOD 约束
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from ppo_optimize.env import PropellerDesignEnv
from ppo_optimize.callbacks import BestDesignCallback, MetricsLoggerCallback


def make_env_fn(args, seed=0):
    """返回创建环境的工厂函数 (用于 SubprocVecEnv)。"""
    def _init():
        env = PropellerDesignEnv(
            V=args.V,
            max_steps=args.max_steps,
            no_improve_limit=args.no_improve_limit,
            delta_scale=args.delta_scale,
            mass=args.mass,
            ood_penalty=args.ood_penalty,
            constrain_to_data=args.constrain,
            use_dnn=not args.dummy,
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def train(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("  PPO 螺旋桨气动外形优化训练")
    print("=" * 60)
    print(f"  V = {args.V} m/s | mass = {args.mass} kg")
    print(f"  total_timesteps = {args.total_timesteps}")
    print(f"  max_steps/episode = {args.max_steps}")
    print(f"  delta_scale = {args.delta_scale}")
    print(f"  n_envs = {args.n_envs}")
    print(f"  constrain_to_data = {args.constrain}")
    print(f"  ood_penalty = {args.ood_penalty}")
    print(f"  save_dir = {save_dir}")
    print("=" * 60)

    if args.n_envs > 1:
        train_env = SubprocVecEnv(
            [make_env_fn(args, seed=42 + i) for i in range(args.n_envs)],
        )
    else:
        train_env = DummyVecEnv([make_env_fn(args, seed=42)])

    eval_env = DummyVecEnv([make_env_fn(args, seed=99)])

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    n_steps_per_env = max(args.n_steps // args.n_envs, 64)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=n_steps_per_env,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        device="cpu",
        seed=42,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=log_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 100),
        n_eval_episodes=5,
        deterministic=True,
    )

    design_callback = BestDesignCallback(save_dir=save_dir, verbose=1)
    metrics_callback = MetricsLoggerCallback(verbose=0)

    callbacks = CallbackList([eval_callback, design_callback, metrics_callback])

    print("\n开始训练...\n")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\n训练完成! 模型已保存至: {final_path}")

    if design_callback.best_geometry is not None:
        print(f"\n最优巡航指标: {design_callback.best_cruise_metric:.4f}")
        geo = design_callback.best_geometry
        print(f"最优几何 (chord): {geo[:4]}")
        print(f"最优几何 (twist): {geo[4:]}")

    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="PPO 螺旋桨外形优化训练")

    parser.add_argument("--V", type=float, default=10.0, help="前飞速度 (m/s)")
    parser.add_argument("--mass", type=float, default=3.5, help="无人机质量 (kg)")
    parser.add_argument("--max-steps", type=int, default=50, help="每 episode 最大步数")
    parser.add_argument("--no-improve-limit", type=int, default=10, help="连续无改善终止阈值")
    parser.add_argument("--delta-scale", type=float, default=0.10, help="每步最大变化比例")

    parser.add_argument("--total-timesteps", type=int, default=500_000, help="总训练步数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n-steps", type=int, default=2048, help="每次更新的采样步数 (总)")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch 大小")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epoch 数")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="熵系数")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="评估频率 (总步数)")

    parser.add_argument("--n-envs", type=int, default=1,
                        help="并行环境数 (>1 使用 SubprocVecEnv)")
    parser.add_argument("--constrain", action="store_true", default=True,
                        help="硬约束到训练数据范围 (默认开启)")
    parser.add_argument("--no-constrain", dest="constrain", action="store_false",
                        help="关闭训练数据硬约束")
    parser.add_argument("--ood-penalty", type=float, default=0.0,
                        help="OOD 软惩罚权重 (0=不惩罚)")

    parser.add_argument("--save-dir", type=str, default="./ppo_models", help="保存目录")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--dummy", action="store_true",
                        help="使用 dummy 解析模型 (无需 DNN, 用于验证流程)")

    args = parser.parse_args()

    if args.quick:
        args.total_timesteps = min(args.total_timesteps, 20_000)
        args.eval_freq = min(args.eval_freq, 5_000)
        args.n_steps = min(args.n_steps, 512)

    train(args)


if __name__ == "__main__":
    main()
