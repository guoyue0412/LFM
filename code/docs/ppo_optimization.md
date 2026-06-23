# 基于 PPO 的螺旋桨气动外形优化技术方案

## 1 问题描述

在固定前飞速度 V = 10 m/s 条件下，通过强化学习优化四旋翼无人机的螺旋桨几何外形（8 个 B 样条控制点），使其在满足力/力矩配平约束的前提下，最大化巡航距离代理指标。

### 1.1 优化变量

| 变量 | 维度 | 范围 | 说明 |
|------|------|------|------|
| chord 控制点 | 4 | [0.002, 0.05] m | B 样条弦长控制点 |
| twist 控制点 | 4 | [5, 80] deg | B 样条扭转角控制点 |

### 1.2 约束（由配平求解器自动满足）

$$F_x = 0, \quad F_z = 0, \quad M_y = 0$$

其中迎角 $\alpha$ 和前后旋翼转速 $\Omega_1, \Omega_2$ 由 `scipy.optimize.root` 求解。

### 1.3 优化目标

最大化巡航距离代理指标：

$$R_{cruise} = \eta_{prop} \cdot \frac{L}{D}$$

其中：
- $\eta_{prop} = \frac{T_{total} \cdot V}{P_{total}}$ 为推进效率
- $L/D$ 为升力阻力比（来自配平解的机身气动力）
- $P_{total} = 2(\Omega_1 Q_1 + \Omega_2 Q_2)$ 为总功率

---

## 2 MDP 建模

### 2.1 状态空间 (13 维)

$$s = [cp_0, \ldots, cp_7, \alpha_{deg}, \Omega_1, \Omega_2, P, t/T]$$

所有分量归一化到 $[-1, 1]$。

| 分量 | 维度 | 归一化依据 |
|------|------|-----------|
| 几何控制点 | 8 | 各自的上下界 |
| 迎角 | 1 | [0, 25] deg |
| 前/后旋翼转速 | 2 | [1000, 12000] RPM |
| 功率 | 1 | [0, P_max_estimated] |
| 步数比例 | 1 | [0, 1] |

### 2.2 动作空间 (8 维连续)

$$a = [\delta cp_0, \ldots, \delta cp_7] \in [-1, 1]^8$$

映射到实际几何增量：$cp_{new} = \text{clip}(cp_{old} + a \cdot \Delta_{max}, lb, ub)$

其中 $\Delta_{max}$ 为每步最大变化量（边界范围的 10%）。

### 2.3 奖励函数

$$r_t = \begin{cases}
R_{cruise}(s_{t+1}) - R_{cruise}(s_t) + \beta \cdot R_{cruise}(s_{t+1}) & \text{trim converged} \\
-10 & \text{trim failed}
\end{cases}$$

- 增量奖励鼓励持续改进
- 绝对值项 ($\beta = 0.1$) 引导整体方向
- 配平失败给予大负奖励

### 2.4 终止条件

- 最大步数 $T = 50$（truncated）
- 连续 10 步无改善（terminated）

---

## 3 算法选择: PPO

使用 Proximal Policy Optimization (PPO, Schulman et al. 2017)：

- 适合连续动作空间
- 稳定性好，超参数不敏感
- 库: stable-baselines3

### 3.1 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| policy | MlpPolicy [256, 256] | 两层全连接 |
| n_steps | 2048 | 每次更新的采样步数 |
| batch_size | 64 | mini-batch 大小 |
| n_epochs | 10 | 每次更新的 epoch 数 |
| learning_rate | 3e-4 | 学习率 |
| gamma | 0.99 | 折扣因子 |
| clip_range | 0.2 | PPO 裁剪范围 |
| ent_coef | 0.01 | 熵系数（鼓励探索） |
| total_timesteps | 500,000 ~ 1,000,000 | 总训练步数 |

---

## 4 系统架构

```
PPO Agent                      PropellerDesignEnv
  |                                |
  |-- action: delta_cp[8] -------->|
  |                                |-- 1. 更新几何 cp
  |                                |-- 2. adapter → DNN 预测
  |                                |-- 3. TrimSolver.solve(V=10)
  |                                |-- 4. 计算 P, eta, L/D, reward
  |<-- (obs, reward, done) --------|
```

### 4.1 数据流

1. PPO 输出 8 维动作 (归一化增量)
2. 环境将增量映射为新几何控制点
3. `adapter.make_rotor_aero_func_with_torque(geometry=cp)` 生成气动回调
4. `QuadcopterTrimSolver.solve(V=10)` 求解配平
5. 从 DNN 输出获取 Torque，计算功率和效率
6. 组装状态向量和奖励返回给 PPO

---

## 5 实验设计

### 5.1 训练

```bash
python ppo_optimize/train_ppo.py --total-timesteps 500000
```

### 5.2 评估

```bash
python ppo_optimize/eval_ppo.py --model ppo_models/best_model.zip
```

### 5.3 基线对比

| 方法 | 工具 | 说明 |
|------|------|------|
| SLSQP | optimize_trim.py | 基于梯度的局部优化 |
| PPO | ppo_optimize/ | 强化学习全局搜索 |
| 固定几何 | run_trim.py | 不优化几何，仅配平 |

### 5.4 评估指标

- 巡航距离代理指标 $R_{cruise}$
- 总功率 $P_{total}$
- 推进效率 $\eta_{prop}$
- 收敛率（配平成功率）
- 训练奖励曲线

---

## 6 文件结构

```
LFM/
├── ppo_optimize/
│   ├── env.py            # Gymnasium 环境
│   ├── train_ppo.py      # PPO 训练脚本
│   ├── eval_ppo.py       # 评估 + 论文级可视化
│   └── callbacks.py      # 训练回调
├── ppo_models/           # 训练产物
└── docs/
    └── ppo_optimization.md   # 本文档
```
