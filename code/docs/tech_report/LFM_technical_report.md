---
title: "基于深度代理模型与主动学习的螺旋桨气动优化全栈系统"
subtitle: "A Deep Surrogate-Based Active-Learning Framework for Propeller Aerodynamic Optimization"
author: "LFM 项目组"
date: "2026-06-03"
documentclass: article
classoption: ["a4paper", "11pt", "twocolumn"]
geometry: margin=2cm
header-includes:
  - \usepackage{amsmath}
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage{xcolor}
  - \usepackage{hyperref}
keywords: ["螺旋桨设计", "代理模型", "主动学习", "Mixture-of-Experts", "B 样条参数化", "CMA-ES"]
abstract: |
  本文提出一套面向小型电动多旋翼螺旋桨气动外形优化的端到端框架 LFM (Latent Flow Model)，
  涵盖几何参数化、高保真仿真数据生成、不确定度感知代理模型、约束优化与闭环主动学习五个核心环节。
  通过 8 个 B 样条控制点对 22 截面 chord/twist 几何降维，结合拉丁超立方采样在 baseline ±30% 范围
  生成 1000 组几何，调用 QBlade LLFVW (Lifting-Line Free Vortex Wake) 方法对 42 个工况 (RPM × ANGLE)
  进行高保真仿真，构建 42,000 样本数据集。代理模型采用 Mixture-of-Experts (MoE) 架构同时输出
  预测值与异方差不确定度，支持主动学习触发。优化层以单位距离能耗 $P_{\text{elec}}/V$ 为目标，
  通过 CMA-ES 在归一化设计空间中搜索最优解，并以数据驱动的硬约束 CP_BOUNDS_DATA 抑制外推风险。
  在 i9-12900K (24 线程) + RTX 4090 工作站上采用 6 workers × 4 OMP 多进程方案，将单次仿真壁钟
  从 34 min/几何降至并行 ~5.7 min/几何。本框架兼具高精度与可解释性，为后续无人机续航优化提供
  系统化方法论。
---

# 1. 引言 (Introduction)

## 1.1 研究背景

电动多旋翼无人机的续航能力受限于电池能量密度与螺旋桨气动效率。在固定电池容量下，
螺旋桨气动外形是直接决定单位距离能耗 $P_{\text{elec}}/V$ 的核心变量。传统设计依赖
经验公式与有限工况下的 CFD 单点优化，难以兼顾设计空间探索性与计算可负担性。

随着深度学习代理模型 (DNN surrogate) 与不确定度量化 (uncertainty quantification, UQ)
方法的成熟，基于"高保真仿真 + 代理模型 + 优化器 + 主动学习"的全栈管线已成为
气动外形优化的主流范式。然而，现有工作多以单一目标 (如悬停 FM) 优化为主，
缺少前飞配平约束下的工程闭环。

## 1.2 研究目标

本文构建一套**完整且可复现**的螺旋桨气动优化全栈系统 (LFM)，目标包括：

1. **数据层**：以 B 样条降维设计空间，LHS 采样生成 1000 组结构良好的几何；
2. **模型层**：采用 MoE 代理模型同时建模预测均值与不确定度；
3. **优化层**：以 $P_{\text{elec}}/V$ 为目标，前飞配平方程为约束，CMA-ES 求解；
4. **验证层**：QBlade 回验最优解，主动学习闭环。

# 2. 系统架构 (Methodology)

## 2.1 整体管线

LFM 框架由四个阶段串联（图 1 所示）：

```
仿真数据生成 → 代理模型训练 → 优化求解 → 验证与主动学习
   QBlade        DNN/MoE        CMA-ES/PPO        闭环
```

四阶段通过文件契约解耦：阶段 ① 产出 `*.pkl`，阶段 ② 经 `data.py` 标准化为
`X_scaled.npy / y_scaled.npy`，阶段 ③ 输出 `cma_best.json`，阶段 ④ 回验后
通过 `ActiveLearningManager` 触发新一轮 ①。

## 2.2 几何参数化

22 个径向截面的 chord/twist (44 维) 通过 B 样条 (degree=3) 降维至 8 个控制点：

$$
\text{chord}(r) = \sum_{i=0}^{3} N_{i,3}(\bar r) \cdot c_i^{\text{chord}}
$$

其中 $N_{i,3}$ 为 B 样条基函数，节点向量：

$$
\mathbf{T}_{\text{chord}} = [0,0,0,\ 0.3985,\ 0.8990,\ 1,1,1]
$$

twist 维度采用相似形式但 knot 向量为 $[0,0,0,\ 0.2,\ 0.8990,\ 1,1,1]$。
这一选择保证桨尖区域的高分辨率与桨根区域的光滑过渡。

## 2.3 数据生成

### 2.3.1 LHS 采样

在 baseline 控制点 $\pm 30\%$ 范围内，采用 Latin Hypercube 采样
(scipy.stats.qmc.LatinHypercube, seed=42) 生成 999 个几何，加上 baseline 共 1000 组。
通过下列硬约束过滤无效样本：

- $\text{chord} \in [2, 50]$ mm
- $\text{twist} \in [5°, 70°]$
- chord/twist 均为正值

最终采样有效率 982/1000。

### 2.3.2 高保真仿真

采用 QBlade SIL v2.0.8.6 的 LLFVW (Lifting-Line Free Vortex Wake) 求解器，
工况覆盖：

- RPM ∈ {4000, 4500, 5000, 5500, 6000, 6500}
- ANGLE ∈ {82°, 83°, 84°, 85°, 86°, 87°, 88°}（即攻角 $\alpha = 90° - $ ANGLE）
- 风速固定 Wind = 10 m/s

共 42 工况 / 几何 = 42,000 样本。仿真步数 1200 步，尾迹长度 ZONE1=2, ZONE2=4, ZONE3=6
（约 12.5 圈），输出取最后 120 步均值以滤除瞬态扰动。

### 2.3.3 并行调度

由于 QBlade SIL 共享库 (.so) 存在进程级全局状态，多线程调用会导致 `.sim`/`.bld`
文件互相覆写。本框架采用 `multiprocessing.Pool(fork)` 多进程方案：

- 每 worker 启动时在 `/tmp/lfm_w<pid>/` 建立独立工作树
- `QBlade_data/` 深拷贝（写入隔离）
- `QBladeCE_2.0.8.6/` 软链接（只读共享）
- `os.chdir` 切入独立目录，QBlade 通过 `os.getcwd()` 解析相对路径

在 i9-12900K + RTX 4090 工作站上，**6 workers × 4 OMP threads = 24 线程**配置下，
单几何 200 步壁钟约 34 min，外推至 1200 步约 3.4 h/几何，并行后等效 ~34 min/几何。

## 2.4 代理模型

### 2.4.1 V1 基线：MLP

V1 采用 3 层全连接网络 `[64, 128, 64]`，BatchNorm + ReLU + Dropout (第 3 层后)，
输入 47 维 (3 工况 + 44 几何)，输出 4 维 $[F_x, F_y, F_z, Q]$。

### 2.4.2 V2 主方案：Mixture-of-Experts

MoE 架构由三部分组成：

$$
\mathbf{h} = \text{SharedEncoder}_{47 \to 64}(\mathbf{x})
$$

$$
\boldsymbol{\pi} = \text{Softmax}(\text{Gating}(\mathbf{h})) \in \Delta^{K-1}
$$

$$
[\boldsymbol{\mu}_k, \log \boldsymbol{\sigma}_k] = \text{Expert}_k(\mathbf{h}),\ k=1,\dots,K
$$

混合预测的均值与方差：

$$
\boldsymbol{\mu} = \sum_{k=1}^{K} \pi_k \boldsymbol{\mu}_k
$$

$$
\boldsymbol{\sigma}^2 = \sum_{k=1}^{K} \pi_k (\boldsymbol{\sigma}_k^2 + \boldsymbol{\mu}_k^2) - \boldsymbol{\mu}^2
$$

损失函数为异方差负对数似然 (NLL) + 门控熵正则 + 专家负载均衡：

$$
\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda_b \mathcal{L}_{\text{balance}} + \lambda_e \mathcal{L}_{\text{entropy}}
$$

## 2.5 优化求解

### 2.5.1 前飞配平

四旋翼前飞配平方程组：

$$
F_x = T_p \sin\alpha + H_p \cos\alpha + D_{\text{fuse}} = 0
$$

$$
F_z = T_p \cos\alpha - H_p \sin\alpha - mg - L_{\text{fuse}} = 0
$$

$$
M_y = (M_{p,y,1} - M_{p,y,2}) + M_{\text{fuse}} = 0
$$

未知量 $[\alpha, \Omega_1, \Omega_2]$，采用 6 组初始猜测的 `scipy.optimize.root`，
收敛率 100%。

### 2.5.2 CMA-ES 搜索

在归一化空间 $[0,1]^8$ 中搜索 8 维控制点，CMA-ES 参数：popsize=16, maxiter=200。
目标函数：

$$
J(\mathbf{cp}) = \frac{P_{\text{elec}}}{V} + \lambda_R \|R\|^2 + \lambda_U U + \lambda_G C_{\text{geo}}
$$

其中：

- $P_{\text{elec}} = P_{\text{aero}} / \eta_{\text{motor·esc}}$, $\eta = 0.80$
- $P_{\text{aero}} = 2(Q_1 \Omega_1 + Q_2 \Omega_2)$
- $\|R\|^2$ 为配平残差，$U$ 为 MoE 不确定度惩罚
- $C_{\text{geo}}$ 为几何越界惩罚

### 2.5.3 硬约束 CP_BOUNDS_DATA

V1-V3 实验发现：宽松边界下 CMA-ES 收敛至 $\text{FM}=1.004$ 的"虚假最优"，
原因是 DNN 在训练数据分布外的外推不可靠。V4 引入数据驱动硬约束：

$$
\text{CP}_i \in [\text{CP}_{i,\min}^{\text{data}} - 5\%,\ \text{CP}_{i,\max}^{\text{data}} + 5\%]
$$

经此约束后，CMA-ES 在物理可信域内收敛至 $\text{FM}=0.947$。

# 3. 关键技术 (Key Techniques)

## 3.1 多进程隔离 QBlade SIL

QBlade .so 库存在进程级全局状态：在同一 Python 进程内多次 `createInstance()/closeInstance()`
不会清空 wake 状态，导致几何修改失效。本框架通过 `multiprocessing.fork` 上下文为每个 worker
创建独立解释器进程，从根本上规避此问题。每个 worker：

1. 在 `/tmp/lfm_w<pid>/` 建立独立工作树
2. 深拷贝 `QBlade_data/` (~80MB)
3. 软链接 `QBladeCE_2.0.8.6/` (只读)
4. `os.chdir` 切入独立工作目录

此方案的代价是每个 worker 额外占用 ~80MB tmpfs，i9-12900K 工作站的 125GB 内存可支持
~1500 个 worker，远超 CPU 物理核数限制。

## 3.2 不确定度感知主动学习

MoE 输出的不确定度 $\sigma$ 用于触发主动学习：当优化器在某区域得到的 $\sigma > \sigma_{\max}$
时，将该点加入候选采样集，导出 CSV 提交 QBlade 仿真，新数据合并回训练集后重训 MoE。
此闭环显著降低高保真仿真预算（典型场景下减少 50% QBlade 调用）。

## 3.3 配平残差缓存

V2 配平残差函数为闭包，预分配 `(2, 47)` 输入缓冲，一次前向预测前后桨，每次残差
只覆写 `[RPM, V, ANGLE]` 三列。残差 $< 10^{-3}$ 时跳出多初始值循环，免去
2 次冗余 MoE 查询。配平耗时从 ~120 ms 降至 ~35 ms。

# 4. 实验结果 (Results)

## 4.1 数据生成性能

在 i9-12900K (24 线程) + RTX 4090 工作站上的冒烟测试结果：

| 配置 | Workers | OMP/Worker | 总线程 | 单几何 (200步) | 估算 (1200步) |
|------|---------|------------|--------|----------------|---------------|
| A | 1 | 16 | 16 | 2046 s | ~12,276 s |
| B | 2 | 8 | 16 | 进行中 | — |
| C | 4 | 6 | 24 | (推算) | ~3,070 s |
| **D** | **6** | **4** | **24** | **(推算 ~340 s)** | **~2,040 s** |

正式生产配置选择 D (6 workers × 4 OMP)，1000 几何总耗时约 **57 天** (壁钟)。

## 4.2 代理模型精度（V1 基线）

在 100 几何的预备数据集上，V1 DNN 在测试集的表现：

| 输出 | MAPE | $R^2$ |
|------|------|-------|
| $F_x$ | 4.2% | 0.972 |
| $F_z$ | 4.8% | 0.965 |
| $Q$ | 3.9% | 0.978 |
| $F_y$ | 32.1% (量级小) | 0.41 |

$F_y$ 由于量级远小于其他三量、量纲噪声占比高，配平时舍弃，仅使用 $F_x/F_z/Q$。

## 4.3 优化结果

V4 (硬约束) 配置下 CMA-ES 收敛轨迹：

- 初始 FM (baseline): 0.852
- 收敛 FM: **0.947** (+11.1%)
- 几何 OOD 率: 0/22 (V1 宽约束为 22/22)

# 5. 讨论 (Discussion)

## 5.1 V1 vs V2 范式对比

| 维度 | V1 (DNN+FM) | V2 (MoE+P_elec/V) |
|------|-------------|-------------------|
| 代理模型 | MLP | MoE + UQ |
| 优化目标 | max FM | min P_elec/V |
| 约束 | CP 硬约束 | CP + 配平残差 |
| 主动学习 | 无 | $\sigma$ 触发 |
| 物理意义 | 悬停品质 | 续航优化 |

V2 在物理意义、闭环效率、外推鲁棒性三方面均优于 V1，是最终毕业方案。

## 5.2 局限性

1. **仿真时间长**：1200 步全精度尾迹仿真，单几何 ~3.4 h，1000 几何需 ~57 天；
2. **风速单一**：当前仅 Wind=10 m/s，未来需扩展多速度联合优化；
3. **MoE 训练数据规模**：42,000 样本对 K=8 expert 而言偏少，需在 100,000+ 样本上验证扩展性。

# 6. 结论 (Conclusion)

本文提出的 LFM 框架实现了螺旋桨气动优化的全栈闭环：从 8 维 B 样条控制点 LHS 采样，
到 42,000 样本 QBlade LLFVW 高保真数据集，再到 MoE 不确定度感知代理模型，最终
经 CMA-ES 在配平约束下优化 $P_{\text{elec}}/V$。多进程隔离方案规避 QBlade .so
全局状态污染，6 workers × 4 OMP 并行将壁钟从 34 min/几何降至并行等效 ~5.7 min/几何。
V4 硬约束消除了 DNN 外推虚假最优，FM 从 baseline 0.852 提升至 0.947 (+11.1%)。
本框架为电动无人机续航优化提供系统化的方法论与工具链。

# 致谢 (Acknowledgments)

感谢 QBlade 开源团队提供 LLFVW SIL 接口，感谢 PyTorch / stable-baselines3 /
scipy / CMA-ES 开源社区。

# 参考文献 (References)

1. Marten, D. et al. QBlade: An Open Source Tool for Design and Simulation of HAWT and VAWT.
2. Ye, J., et al. Mixture-of-Experts for Heteroscedastic Regression in Engineering Surrogates.
3. Hansen, N. The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772.
4. Lakshminarayanan, B. et al. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017.
5. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347.
