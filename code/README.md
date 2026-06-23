# LFM — 螺旋桨气动优化全栈系统

> **L**ifting-line **F**ree-wake **M**ethod 数据驱动的螺旋桨气动力代理模型 + 四旋翼配平求解 + RL/进化优化 + QBlade 并行仿真

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![QBlade](https://img.shields.io/badge/QBlade-2.0.9-green)](https://qblade.org)

---

## 系统架构

```
                        ┌──────────────────────────────────────────────────────┐
                        │              LFM 全栈流程                             │
                        └──────────────────────────────────────────────────────┘

  ①  QBlade 仿真          ②  DNN 代理模型        ③  优化求解              ④  验证
  ──────────────          ──────────────        ──────────              ──────
  Propeller_project       data.py               ppo_optimize/          QBlade 回验
  ├── batch.py            train.py              ├── env.py (Gym)
  ├── runner.py           sweep.py              ├── train_ppo.py
  └── geometry.py         adapter.py            ├── cma_optimize.py
       │                      │                 └── eval_ppo.py
       │                      │                      │
       ▼                      ▼                      ▼
  raw_data.csv  ────→  DNN (47→4)  ────→  FM*=0.947 最优桨叶
  (7200 条)            MAPE<1.5%           (8 个 B 样条 CP)
```

### 核心能力

| 能力 | 模块 | 说明 |
|------|------|------|
| **QBlade 批量仿真** | `Propeller_project-main/` | GPU/CPU 并行, ProcessPoolExecutor |
| **DNN 代理模型** | `train.py` / `adapter.py` | MLP [64,128,64], R²>0.99, 论文级报告自动导出 |
| **四旋翼配平** | `quadcopter_trim_solver.py` | 多初始值 + 热启动 + 硬约束 |
| **PPO 优化** | `ppo_optimize/train_ppo.py` | stable-baselines3, SubprocVecEnv 并行 |
| **CMA-ES 优化** | `ppo_optimize/cma_optimize.py` | 进化策略基线对照 |
| **OOD 检测** | `ppo_optimize/env.py` | 训练域外惩罚 + 硬约束 |

---

## 目录结构

```
LFM/
│
├── config.py                     # 全局配置 (数据、模型、训练超参)
├── data.py                       # 数据加载: pkl → CSV → 标准化
├── train.py                      # DNN 训练 + TensorBoard + 论文级报告
├── eval.py                       # 模型评估 & 单点预测
├── sweep.py                      # 超参数网格/随机扫描
├── adapter.py                    # DNN → 配平回调适配层
├── quadcopter_trim_solver.py     # 四旋翼纵向配平求解器
├── uav_model.py                  # 无人机参数化气动模型
├── run_trim.py                   # 配平求解集成脚本
├── optimize_trim.py              # 几何 + 状态联合优化
├── plot_style.py                 # 统一绘图风格
│
├── ppo_optimize/                 # 强化学习 & 进化优化
│   ├── env.py                    #   Gymnasium 环境 (v4 硬约束)
│   ├── train_ppo.py              #   PPO 训练 (支持并行)
│   ├── cma_optimize.py           #   CMA-ES 优化
│   ├── callbacks.py              #   训练回调
│   └── eval_ppo.py               #   评估 & 可视化
│
├── Propeller_project-main/       # QBlade 仿真管线
│   ├── code/class_sim/           #   重构后的仿真模块
│   │   ├── config.py             #     SimConfig / SimPaths 数据类
│   │   ├── sim_file.py           #     .sim 模板读写
│   │   ├── geometry.py           #     B-spline + .bld 修改
│   │   ├── runner.py             #     QBladeRunner (context-manager)
│   │   └── batch.py              #     并行批量调度器 (GPU/CPU/Hybrid)
│   ├── get_data.py               #   几何级并行数据采集 (CLI)
│   ├── project_config.py         #   仿真路径 & 基线几何
│   ├── Qblade_data/              #   .sim/.bld/.afl/.plr 模板
│   └── QBladeCE_2.0.9.2/         #   QBlade SIL 接口 & DLL
│
├── data_for_train/               # 训练数据
│   └── geometry_run_results_100/ #   raw_data.csv (7200 条)
├── trained_models/               # DNN 权重 & 报告
├── sweep_results/                # 超参扫描结果
├── ppo_models_v4_constrained/    # PPO 优化产物
├── cma_results_constrained/      # CMA-ES 优化产物
│
├── docs/                         # 文档
│   ├── ppo_optimization.md       #   PPO 技术方案
│   └── uav_model_references.md   #   机身模型参考文献
├── plot/                         # 可视化 notebook
│
├── RESEARCH_LOG.md               # 研究日志 (实验记录)
├── requirements.txt              # Python 依赖
└── .gitignore                    # Git 忽略规则
```

---

## 快速开始

### 环境安装

```bash
# 创建 conda 环境
conda create -n LFM python=3.10 -y
conda activate LFM

# 安装依赖
pip install -r requirements.txt

# 验证 PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Linux 部署（含 QBlade 子模块）

`Propeller_project-main/` 是 git submodule，指向 `guoyue0412/Propeller_project_linux_version.git` 的 `guoyue0412-Linux_version` 分支，含 18MB 的 `libQBladeCE_2.0.8.6.so.1.0.0` Linux 共享库。Mac 上仅做开发与 git 推送，仿真数据生成在远端 Linux 节点（如 `3090_node1`）跑：

```bash
# 在 Linux 节点上：
git clone --recursive git@github.com:guoyue0412/LFM.git
cd LFM
bash scripts/install_linux.sh         # 校验 submodule + .so + ldd 系统依赖 + 建 LFM conda env
bash scripts/smoke_test.sh            # 1 几何 × 全工况 → data.py → 校验 raw_data.csv
bash scripts/run_data_gen.sh --help   # 完整数据生成 CLI
```

已有本地 clone 想拉取 submodule：`git submodule update --init --recursive`

### 最简三步 (DNN 训练)

```bash
python data.py       # 1. 处理数据
python train.py      # 2. 训练模型
python eval.py       # 3. 评估效果
```

### 优化流程 (PPO + CMA-ES)

```bash
# CMA-ES 基线 (硬约束, ~5 分钟)
python ppo_optimize/cma_optimize.py --maxiter 200 --popsize 16 --constrain

# PPO 训练 (4 并行环境, ~30 分钟)
python ppo_optimize/train_ppo.py --total-timesteps 200000 --n-envs 4 --constrain

# 评估对比
python ppo_optimize/eval_ppo.py
```

### QBlade 数据生成（Linux）

通过 `scripts/run_data_gen.py` 包装子模块的 `SIMULATION` 类，输出对齐 `data.py` 期待的 `{geometry_idx: {"geometry": ..., "RPM*_Wind*_Angle*": DataFrame, ...}}` 格式 pkl，直接落到 `data_for_train/data/`：

```bash
# 仅 baseline 几何 × 全工况（冒烟）
bash scripts/run_data_gen.sh --device CPU --tag smoke

# 批量多几何（先把 (N, 22, 3) 几何打成 npy）
bash scripts/run_data_gen.sh \
    --geometry-npy ./geometries.npy \
    --device CPU --num-timesteps 1000 --tag run_$(date +%Y%m%d)

# 紧接着 data.py 即可消费
python data.py
```

工况组合（RPM × WIND × ANGLE）由子模块内 `code/Simulation_QBlade/simulation_parameters/Parameters.xlsx` 定义。

---

## 模块详解

### 1. DNN 代理模型

**架构**: `PropellerPredictor` — MLP [64, 128, 64]

```
Input(47) → [Linear → BatchNorm → ReLU → Dropout] × 3 → Linear(4)
```

| 输入 (47 维) | 输出 (4 维) |
|-------------|------------|
| RPM, WIND, ANGLE | Fx (轴向力) |
| chord_0 ~ chord_21 | Fy (侧向力) |
| twist_0 ~ twist_21 | Fz (法向力) |
| | Torque (扭矩) |

**训练数据**: 100 种几何 × 72 工况 = 7200 条 QBlade 仿真数据

**精度** (测试集):

| 输出 | MAPE | R² |
|------|:----:|:--:|
| Fx | 1.03% | 0.999 |
| Fz | 1.50% | 0.998 |
| Torque | 0.99% | 0.999 |

### 2. 配平求解器

解 3 个非线性方程 (Fx=0, Fz=0, My=0), 未知量为 [alpha, Omega1, Omega2]:

```
                T_p1 ↑                    ↑ T_p2
              H_p1 →  ┌────────────────┐  ← H_p2
                      │    机身 (mg)    │
                      └────────────────┘
                      ├── l₁ → CG ← l₂ ─┤
                               ↓ D_f
                       来流 →→→ V
```

- **多初始值**: 6 组初始猜测 + 热启动 → 收敛率 100%
- **硬约束**: RPM ∈ [4000, 6500], alpha ∈ [2°, 8°]

### 3. PPO 优化环境

| 项目 | 值 |
|------|---|
| 状态空间 | 13 维: 8 CP + alpha + Omega1,2 + power + step_ratio |
| 动作空间 | 8 维连续 [-1, 1] → CP 增量 |
| 奖励 | FM 改善 + beta·FM - OOD 惩罚 |
| 目标 | max FM = mg·V / P_total |

**约束演进**:

| 版本 | 策略 | FM | 可信度 |
|------|------|:--:|:-----:|
| V1 无约束 | 宽松边界 | 1.004 | ★ (外推) |
| V3 OOD 惩罚 | 软惩罚 | 0.941 | ★★ |
| **V4 硬约束** | **数据驱动 CP 边界** | **0.947** | **★★★** |

### 4. QBlade 并行仿真

```
BatchRunner
├── gpu     → 顺序执行, GPU OpenCL 加速
├── cpu     → ProcessPoolExecutor N 进程并行
└── hybrid  → 1 GPU + (N-1) CPU 同时执行
```

每个 worker 进程独立加载 QBlade DLL, 无共享状态。

---

## 全流程工作流

```
流程 A: 代理模型训练
  QBlade .pkl → data.py → sweep.py → train.py → eval.py

流程 B: DNN 驱动配平
  train.py → adapter.py → run_trim.py --dnn

流程 C: 几何优化
  train.py → ppo_optimize/train_ppo.py (PPO)
           → ppo_optimize/cma_optimize.py (CMA-ES)
           → ppo_optimize/eval_ppo.py

流程 D: QBlade 验证
  最优 CP → Propeller_project-main/batch.py → QBlade 仿真 → 对比 DNN 预测
```

---

## 命令行参考

### DNN 训练

```bash
python data.py                                     # 数据处理
python sweep.py --quick --max-trials 4             # 快速超参扫描
python train.py                                     # 正式训练
tensorboard --logdir=./runs                         # 实时监控
python eval.py --rpm 5000 --wind 9 --angle 85      # 单点预测
```

### 配平求解

```bash
python run_trim.py --dnn --V 10                    # DNN 单点配平
python run_trim.py --compare                       # dummy vs DNN 对比
python optimize_trim.py --V 10                     # 几何优化配平
```

### PPO / CMA-ES 优化

| 参数 | 说明 | PPO | CMA |
|------|------|:---:|:---:|
| `--constrain` | 硬约束 (默认开) | ✓ | ✓ |
| `--no-constrain` | 关闭硬约束 | ✓ | ✓ |
| `--n-envs N` | 并行环境数 | ✓ | — |
| `--ood-penalty X` | OOD 软惩罚权重 | ✓ | ✓ |
| `--total-timesteps` | 总训练步数 | ✓ | — |
| `--maxiter` | 最大迭代 | — | ✓ |
| `--popsize` | 种群大小 | — | ✓ |

---

## 配平原理

### 纵向平面模型

基于 Ye et al. (2021) "Propulsion optimization of a quadcopter in forward state":

**方程 1 — 水平力平衡 (Fx=0)**

$$2(T_{p1} + T_{p2})\sin\alpha - 2(H_{p1} + H_{p2})\cos\alpha - D_f = 0$$

**方程 2 — 垂直力平衡 (Fz=0)**

$$2(T_{p1} + T_{p2})\cos\alpha + 2(H_{p1} + H_{p2})\sin\alpha - mg - L_f = 0$$

**方程 3 — 俯仰力矩平衡 (My=0)**

$$2M_{p1}^y + 2M_{p2}^y - M_f^y + 2(T_{p2}l_2 - T_{p1}l_1) + 2(H_{p1}d_1 + H_{p2}d_2) = 0$$

---

## 论文数据导出

| 产物 | 路径 | 用途 |
|------|------|------|
| 模型精度表 | `trained_models/report/test_metrics.csv` | 表格数据源 |
| LaTeX 表格 | `trained_models/report/report.tex` | 直接粘贴论文 |
| 预测散点图 | `trained_models/report/pred_vs_true.png` | 论文图 (300dpi) |
| 误差分布图 | `trained_models/report/error_distribution.png` | 论文图 |
| 训练曲线 | `trained_models/report/training_curves.png` | 收敛性分析 |
| 超参对比 | `sweep_results/sweep_*.tex` | Top-10 架构 LaTeX |
| 优化结果 | `cma_results_constrained/cma_best.json` | 最优几何 |

---

## 注意事项

1. **角度约定**: QBlade 中 `VERTANGLE = 90° - alpha`, 其中 90° 对应悬停
2. **外推风险**: 硬约束模式 (`--constrain`) 确保查询在 DNN 训练域内, 强烈推荐开启
3. **QBlade DLL**: 需自行下载 (许可证限制), 放置于 `Propeller_project-main/QBladeCE_2.0.9.2/`
4. **Windows 编码**: 建议 `set PYTHONIOENCODING=utf-8`
5. **GPU 并行**: QBlade GPU 实例在单进程内互斥, 大批量仿真优先使用 CPU 多进程模式

---

## 致谢

- [QBlade](https://qblade.org/) — 开源风力机 / 螺旋桨仿真平台
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO 实现
- [CMA-ES (pycma)](https://github.com/CMA-ES/pycma) — 进化策略

## 作者

Guo Yue — [github.com/guoyue0412](https://github.com/guoyue0412)
