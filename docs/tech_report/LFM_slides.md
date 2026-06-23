---
title: "LFM：基于深度代理模型与主动学习的螺旋桨气动优化全栈系统"
author: "LFM 项目组"
date: "2026-06-03"
theme: metropolis
colortheme: seahorse
---

# 1. 研究背景与目标

## 问题陈述

::: columns

:::: column
**电动无人机续航瓶颈：**

- 单位距离能耗 $P_{\text{elec}}/V$ 取决于螺旋桨气动效率
- 传统设计依赖经验公式 + 单点 CFD
- 设计空间高维 (44 截面 chord/twist)
::::

:::: column
**本文目标：**

构建一套**全栈闭环**系统：

1. 高保真仿真数据生成 (QBlade LLFVW)
2. 不确定度感知代理模型 (MoE)
3. 配平约束优化 (CMA-ES)
4. 主动学习闭环
::::

:::

# 2. 系统总览

## LFM 四阶段管线

```
[数据生成] → [代理模型] → [优化求解] → [验证迭代]
  QBlade       DNN/MoE      CMA-ES/PPO     主动学习
   ↑                                            │
   └────────────────  σ 触发  ─────────────────┘
```

- **阶段①**：1000 几何 × 42 工况 = 42,000 样本
- **阶段②**：V1 MLP → V2 Mixture-of-Experts (UQ)
- **阶段③**：min $P_{\text{elec}}/V$ s.t. 配平 + 几何约束
- **阶段④**：QBlade 回验，高 σ 区域补样

# 3. 几何参数化

## 8 控制点 B 样条降维

::: columns

:::: column
**降维**：44 维 → 8 维

$$
\text{chord}(r) = \sum_{i=0}^{3} N_{i,3}(\bar r) \cdot c_i
$$

- degree=3 B 样条
- 4 chord CP + 4 twist CP
- 节点向量针对桨尖优化分辨率
::::

:::: column
**Knot 向量：**

- chord: $[0,0,0, 0.40, 0.90, 1,1,1]$
- twist: $[0,0,0, 0.20, 0.90, 1,1,1]$

**优势：**

- 几何光顺，可制造
- 设计空间显著降维
- CMA-ES 可在 8 维高效搜索
::::

:::

# 4. 数据生成 (LHS + QBlade)

## 采样策略

- **LHS**：scipy.stats.qmc.LatinHypercube, d=8, seed=42
- **范围**：baseline ±30%
- **过滤**：chord∈[2,50]mm, twist∈[5°,70°]
- **有效率**：982/1000 + 1 baseline

## QBlade LLFVW 仿真

| 参数 | 取值 |
|------|------|
| 求解器 | Lifting-Line Free Vortex Wake |
| 步数 | 1200 步 |
| 尾迹长度 | ZONE1=2, ZONE2=4, ZONE3=6 (~12.5 圈) |
| 数据窗 | 最后 120 步均值 |
| 工况数 | 6 RPM × 7 ANGLE = 42 / 几何 |

# 5. 多进程并行架构

## 隔离方案

**问题**：QBlade .so 有进程级全局状态，同进程多次实例化导致几何修改失效

**解决方案**：multiprocessing.fork + tmpfs 工作树

```
每 worker 进程：
  /tmp/lfm_w<pid>/
    ├── code/           深拷贝
    ├── QBlade_data/    深拷贝 (写入隔离)
    └── QBladeCE/       软链接 (只读共享)
  os.chdir → 独立工作目录
```

## 性能基准

| 配置 | Workers × OMP | 总线程 | 单几何 (200步) | 单几何 (1200步) |
|------|---------------|--------|----------------|-----------------|
| A | 1 × 16 | 16 | 2046 s | ~12,276 s |
| **D** | **6 × 4** | **24** | **~340 s** | **~2,040 s** |

**生产配置 D：1000 几何 ≈ 57 天 (壁钟)**

# 6. 代理模型 V1：MLP DNN

## 架构

- **输入**：47 维 = 3 工况 + 44 几何
- **网络**：`[64, 128, 64]` MLP
- **激活**：BatchNorm + ReLU
- **正则**：Dropout (第 3 层后)
- **输出**：4 维 $[F_x, F_y, F_z, Q]$

## V1 测试集表现

| 输出 | MAPE | $R^2$ |
|------|------|-------|
| $F_x$ | 4.2% | 0.972 |
| $F_z$ | 4.8% | 0.965 |
| $Q$ | 3.9% | 0.978 |
| $F_y$ | 32.1% | 0.41 (舍弃) |

# 7. 代理模型 V2：Mixture-of-Experts

## MoE 架构

```
x ──[SharedEncoder 47→64]──→ h
                              │
                  ┌───────────┼───────────┐
                  ↓           ↓           ↓
              Gating π    Expert_1    Expert_K
                          [μ, log σ]  [μ, log σ]
                              │           │
                              └───→ 混合 ←─┘
```

## 异方差预测

混合方差公式：

$$
\sigma^2 = \sum_{k=1}^{K} \pi_k (\sigma_k^2 + \mu_k^2) - \mu^2
$$

**损失**：异方差 NLL + 门控熵 + 专家均衡

# 8. 优化求解器

## 前飞配平约束

$$
\begin{aligned}
F_x &= T \sin\alpha + H \cos\alpha + D_{\text{fuse}} = 0 \\
F_z &= T \cos\alpha - H \sin\alpha - mg - L_{\text{fuse}} = 0 \\
M_y &= (M_{p,y,1} - M_{p,y,2}) + M_{\text{fuse}} = 0
\end{aligned}
$$

未知量 $[\alpha, \Omega_1, \Omega_2]$，scipy.optimize.root 多初值收敛率 100%

## CMA-ES 目标

$$
\boxed{
J(\mathbf{cp}) = \frac{P_{\text{elec}}}{V} + \lambda_R \|R\|^2 + \lambda_U \sigma + \lambda_G C_{\text{geo}}
}
$$

- $P_{\text{elec}} = 2(Q_1\Omega_1 + Q_2\Omega_2) / \eta_{\text{motor·esc}}$
- 归一化空间 $[0,1]^8$，popsize=16, maxiter=200

# 9. 数据驱动硬约束

## 外推风险

V1-V3 宽约束实验：

- CMA-ES 收敛至 **FM=1.004 (虚假最优)**
- 22/22 几何 OOD
- 物理不可行

## V4 解决方案

数据驱动约束 + 5% 边距：

$$
\text{CP}_i \in [\text{CP}_{i,\min}^{\text{data}} \cdot 0.95,\ \text{CP}_{i,\max}^{\text{data}} \cdot 1.05]
$$

**收敛结果：FM=0.947, OOD=0/22**

# 10. 主动学习闭环

## 触发机制

```
            ┌──────────────────────┐
            │  CMA-ES 搜索 cp*    │
            └──────────┬───────────┘
                       │
                  MoE(cp*) → (μ, σ)
                       │
              σ > σ_max? ──Y─→ 候选采样
                       │       │
                       N       ▼
                       │   导出 CSV
                       │       │
                       ▼       ▼
                 接受最优解   QBlade 仿真
                            │
                            ▼
                       新数据 → 重训 MoE
```

# 11. 关键性能指标

::: columns

:::: column
**数据层**：

- 1,000 几何
- 42 工况/几何
- 42,000 样本
- 1200 步仿真

**计算层**：

- 6 workers × 4 OMP
- 24 线程并行
- ~4.1 step/s (CPU)
::::

:::: column
**模型层**：

- MAPE < 5% (主输出)
- $R^2$ > 0.96
- σ 不确定度量化

**优化层**：

- FM: 0.852 → **0.947** (+11.1%)
- OOD: 22/22 → **0/22**
- 配平收敛率: **100%**
::::

:::

# 12. 技术创新点

1. **异方差 MoE 代理模型**：同时输出预测均值与不确定度，支持主动学习
2. **B 样条降维**：44 → 8 维设计空间，保证几何光顺与可制造性
3. **数据驱动硬约束**：CP_BOUNDS_DATA 消除 DNN 外推虚假最优
4. **多进程 QBlade 隔离**：规避 .so 全局状态污染
5. **前飞配平约束**：以 min $P_{\text{elec}}/V$ 替代纯 FM 最大化，符合实际续航任务

# 13. 局限与未来工作

## 当前局限

- 单风速 (Wind=10 m/s)
- 1000 几何生产耗时 ~57 天
- MoE 需 100,000+ 样本验证扩展性

## 未来方向

- **多速度联合优化**：V ∈ [6, 12] m/s
- **GPU 加速 QBlade**：解决 .so OpenCL 死锁
- **物理增强 MoE**：嵌入 momentum/blade-element 先验
- **多目标优化**：续航 + 噪声 + 制造成本

# 14. 结论

::: columns

:::: column
**贡献：**

- 端到端可复现框架
- MoE + UQ 代理模型
- 数据驱动硬约束
- 主动学习闭环
::::

:::: column
**成果：**

- FM 提升 **+11.1%**
- OOD 率 **0/22**
- 配平收敛率 **100%**
- 1000 几何数据集 (生产中)
::::

:::

**LFM 为电动无人机续航优化提供系统化方法论与工具链**

---

# 致谢

QBlade 开源团队 | PyTorch | stable-baselines3 | scipy | CMA-ES 社区

---

# 参考文献

1. Marten, D. *QBlade: HAWT/VAWT Simulation*, 2020.
2. Lakshminarayanan, B. *Deep Ensembles for UQ*, NeurIPS 2017.
3. Hansen, N. *CMA Evolution Strategy*, arXiv:1604.00772.
4. Schulman, J. *PPO Algorithms*, arXiv:1707.06347.
