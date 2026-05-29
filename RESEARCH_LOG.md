# 螺旋桨气动外形优化 — 研究日志

> 最后更新: 2026-03-28

## 1. 问题定义

**目标**: 基于 DNN 代理模型，对四旋翼无人机螺旋桨叶片几何进行优化，最大化巡航品质因数 FM = mg·V / P_total。

**设计变量**: 8 个 B 样条控制点 (4 chord + 4 twist) → B 样条插值 → 22×2 截面 (chord, twist)

**约束**:
- 配平方程: Fx, Fz, My 三轴力矩平衡
- 无人机参数: m=3.5kg, V=10m/s, 450mm 轴距, 10" 螺旋桨

**评价指标**: FM = mg·V / P_total（品质因数，正比于 Breguet 航程公式）

---

## 2. 数据基础

| 项目 | 数值 |
|------|------|
| 数据来源 | QBlade 仿真 |
| 唯一几何数 | 100 |
| 工况组合 | 72 (RPM×WIND×ANGLE) |
| 总样本数 | 7200 |
| RPM 范围 | [4000, 6500] |
| WIND 范围 | [6, 15] m/s |
| ANGLE 范围 | [82°, 88°] |

**DNN 代理模型**: 3 层全连接网络 [64, 128, 64]，输入 47 维 (3 工况 + 44 截面)，输出 4 维 (Fx, Fy, Fz, Torque)。

**模型精度** (训练数据 MAPE):

| 输出 | MAPE |
|------|------|
| Fx | 1.03% |
| Fy | 62.42% (量级极小, 不影响配平) |
| Fz | 1.50% |
| Torque | 0.99% |

---

## 3. 优化方法

### 3.1 PPO (Proximal Policy Optimization)

- **环境**: 自定义 Gymnasium 环境 `PropellerDesignEnv`
- **状态空间**: 13 维 [8 CP + alpha + Omega1 + Omega2 + power + step_ratio]
- **动作空间**: 8 维连续 [-1, 1]，映射到 CP 增量
- **奖励**: FM 改善量 + β·FM (β=0.1)
- **并行训练**: SubprocVecEnv (4 并行环境, FPS 10→20)

### 3.2 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

- **编码**: 8 CP 归一化到 [0, 1]
- **目标**: 最小化 -FM
- **种群**: 16-20, 最大迭代 100-300

---

## 4. 关键发现与迭代

### 4.1 V1: 基础实现

- 配平收敛率仅 ~3%
- 原因: scipy.optimize.root 对初始猜测敏感

### 4.2 V2: 配平鲁棒性改进

- **多初始值配平** (solve_multi_start): 6 组初始猜测 + 热启动
- **软失败惩罚**: 失败时 reward=-1 (非 -10), 允许 3 次连续失败
- **效果**: 配平收敛率 3% → 100%

### 4.3 V3: OOD 检测 — 发现核心问题

**差异度分析结论**: 优化结果严重外推！

| 检测项 | CMA-ES (无约束) | 训练数据范围 |
|--------|:--------------:|:----------:|
| twist OOD 截面 | **22/22 (100%)** | — |
| chord OOD 截面 | 15/22 (68%) | — |
| RPM | 7034-7421 | [4000, 6500] |
| ANGLE | 80.0° | [82°, 88°] |
| FM | 1.004 | **不可信** |

**根因**: 优化器找到了 DNN 外推区域中的"虚假最优"。模型在训练分布内精度很高 (MAPE 1%)，但对分布外数据的预测完全不可靠。

### 4.4 V4: 硬约束 — 限制在 DNN 可信域内 (当前版本)

**CP 边界**: 从训练数据 100 个几何反推 (最小二乘 B 样条拟合 + 5% 边距)

| 控制点 | V1-V3 宽松边界 | V4 数据驱动边界 |
|--------|:------------:|:-------------:|
| chord_0 | [0.010, 0.030] | [0.0136, 0.0263] |
| chord_1 | [0.018, 0.050] | [0.0222, 0.0438] |
| chord_2 | [0.008, 0.024] | [0.0101, 0.0198] |
| chord_3 | [0.003, 0.012] | [0.0048, 0.0092] |
| twist_0 | [30.0, 80.0] | [37.4, 72.7] |
| twist_1 | [10.0, 32.0] | [13.4, 26.5] |
| twist_2 | [8.0, 26.0] | [11.3, 22.1] |
| twist_3 | [6.0, 20.0] | [8.4, 16.6] |

**配平约束**: RPM ∈ [4000, 6500], α ∈ [2°, 8°] (ANGLE 82-88°)

**效果**: 所有 DNN 查询保证在训练数据分布内，优化结果可信。

---

## 5. 优化结果汇总

### 各版本 FM 对比

| 方法 | FM | 功率(W) | 几何 OOD | RPM OOD | ANGLE OOD | 可信度 |
|------|:---:|:------:|:-------:|:-------:|:---------:|:-----:|
| CMA-ES v1 (无约束) | 1.004 | 342 | 37/44 | +1455 | 超出 | ★ |
| CMA-ES v3 (OOD惩罚) | 0.941 | 365 | 15/44 | +1144 | 范围内 | ★★ |
| **CMA-ES v4 (硬约束)** | **0.947** | **363** | **0/44** | **范围内** | **范围内** | **★★★** |
| PPO v4 (硬约束, 训练中) | 0.899 | 382 | 0/44 | 范围内 | 范围内 | ★★★ |

### 硬约束 CMA-ES 最优几何 (v4)

```
FM = 0.9469 | Power = 362.7 W
chord CP: [0.01360, 0.04379, 0.01981, 0.00478]
twist CP: [37.43, 13.45, 11.26, 8.45]
保存路径: cma_results_constrained/cma_best.json
```

> **注意**: 所有 8 个 CP 均卡在约束边界上（chord 交替触碰上/下限，twist 全部在下限），
> 说明 DNN 可信域内的最优实际上是一个边界解。
> 真正的全局最优需要扩展训练数据的覆盖范围。

### 硬约束 PPO (v4, 训练中)

```
状态: 运行中 (4 并行环境, FPS≈7)
当前最优 FM: 0.899
保存路径: ppo_models_v4_constrained/
```

---

## 6. 文件结构

```
LFM/
├── config.py                  # 全局配置
├── data.py                    # 数据加载/处理
├── train.py                   # DNN 模型训练
├── adapter.py                 # DNN → 配平回调适配器
├── quadcopter_trim_solver.py  # 四旋翼配平求解器 (支持约束)
├── ppo_optimize/
│   ├── env.py                 # Gymnasium 环境 (v4, 硬约束)
│   ├── train_ppo.py           # PPO 训练 (并行接口)
│   ├── cma_optimize.py        # CMA-ES 优化 (并行接口)
│   ├── callbacks.py           # 训练回调
│   └── eval_ppo.py            # 评估与可视化
├── data_for_train/
│   └── geometry_run_results_100/
│       ├── raw_data.csv       # 7200 条 QBlade 数据
│       ├── train_data.csv     # 80% 训练集
│       └── test_data.csv      # 20% 测试集
├── trained_models/            # DNN 模型权重
├── analysis_gap/              # OOD 分析结果
│   ├── gap_analysis.json
│   ├── design_comparison.png
│   └── ood_heatmap_comparison.png
└── RESEARCH_LOG.md            # 本文档
```

---

## 7. 使用命令

### 训练 PPO (硬约束 + 并行)

```bash
python ppo_optimize/train_ppo.py \
    --total-timesteps 200000 \
    --n-envs 4 \
    --constrain \
    --save-dir ./ppo_models_v4_constrained
```

### 运行 CMA-ES (硬约束)

```bash
python ppo_optimize/cma_optimize.py \
    --maxiter 200 \
    --popsize 16 \
    --constrain \
    --save-dir ./cma_results_constrained
```

### 关闭约束 (允许外推, 不推荐)

```bash
python ppo_optimize/train_ppo.py --no-constrain ...
python ppo_optimize/cma_optimize.py --no-constrain ...
```

---

## 8. 并行优化接口

### 命令行参数

| 参数 | 说明 | PPO | CMA-ES |
|------|------|:---:|:------:|
| `--constrain` | 硬约束到训练数据范围 (默认开) | ✓ | ✓ |
| `--no-constrain` | 关闭硬约束 | ✓ | ✓ |
| `--n-envs N` | 并行环境数 (SubprocVecEnv) | ✓ | — |
| `--ood-penalty X` | OOD 软惩罚权重 | ✓ | ✓ |
| `--total-timesteps` | 总训练步数 | ✓ | — |
| `--maxiter` | 最大迭代数 | — | ✓ |
| `--popsize` | 种群大小 | — | ✓ |

### 约束体系 (constrain_to_data=True)

1. **CP 边界**: 由训练数据 100 个几何经最小二乘 B 样条拟合反推，+ 5% 边距
2. **RPM 约束**: 配平求解限制在 [4000, 6500] RPM
3. **ANGLE 约束**: 迎角限制在 [2°, 8°]（对应 ANGLE 82-88°）
4. **初始猜测**: 在约束域内密集生成，优先使用热启动

---

## 9. QBlade 仿真脚本重构 (2026-03-28)

将 `Propeller_project-main/code/class_sim/simulation.py` (577 行巨型类) 重构为高聚合低耦合的模块化架构:

### 重构前问题

| 问题 | 严重度 |
|------|:------:|
| SIMULATION 类承担全部职责 (God Class) | 🔴 |
| .sim 文件修改逻辑重复 3 次 | 🔴 |
| `run_all_simulation` 无并行, 纯 for 循环 | 🔴 |
| 每次仿真重新加载/卸载 DLL | 🟡 |
| `setOmpNumThreads` 从未调用 | 🟡 |
| `run_all_simulation` 模板名过滤 Bug | 🟡 |

### 重构后架构

```
code/class_sim/
├── __init__.py     # 公开 API 导出
├── config.py       # SimConfig / SimPaths / SimCondition 数据类
├── sim_file.py     # .sim 模板读写 (统一正则替换, 消除重复)
├── geometry.py     # B-spline 插值 + .bld 文件修改
├── runner.py       # QBladeRunner (进程安全, context-manager)
├── batch.py        # BatchRunner 并行调度器
└── simulation.py   # 旧 SIMULATION 类 → 薄封装 (向后兼容)
```

### 并行策略 (batch.py)

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `gpu` | GPU OpenCL 顺序执行 | 少量工况, 单 GPU |
| `cpu` | N 进程并行 (ProcessPoolExecutor) | 大批量, 多核 CPU |
| `hybrid` | 1 GPU + (N-1) CPU 并行 | 有 GPU + 多核 |

- 每个 worker 进程独立加载 DLL 实例, 无共享状态
- OMP 线程数自动按 worker 数分配
- progress_callback 实时报告进度

### 使用示例

```python
from class_sim import (
    SimPaths, SimConfig, SimCondition,
    BatchRunner, run_batch_cpu, load_conditions_from_excel,
    generate_sim_files,
)

paths = SimPaths(dll_file=..., base_sim=..., ...)
config = SimConfig(device="CPU", num_timesteps=1800)
conditions = load_conditions_from_excel("conditions.xlsx")

# CPU 4 进程并行
results = run_batch_cpu(paths, conditions, config, max_workers=4)
for r in results:
    if r.ok:
        print(f"{r.condition.name}: THRUST={r.data['THRUST'].iloc[0]:.2f}")
```

---

## 10. Linux 部署管线接入 (2026-05-29)

### 背景
LFM 顶层（DNN 训练、PPO/CMA-ES 优化）跨平台无忧，但数据生成依赖 QBlade，仓库内原 `Propeller_project-main/` 目录是 Windows 版（`QBladeCE_2.0.9.2.dll` + `get_data.py` CLI），无法在 Linux 节点跑。本节记录从"嵌入 Windows 目录"到"指向独立 Linux submodule + LFM 顶层 wrapper"的改造。

### 改动

| 项 | 旧 | 新 |
|----|----|----|
| `Propeller_project-main/` | LFM 仓库内嵌入的 Windows 目录（79 文件） | git submodule，URL=`guoyue0412/Propeller_project_linux_version.git`，branch=`guoyue0412-Linux_version`，HEAD=`08ab7703` |
| 数据生成入口 | `python Propeller_project-main/get_data.py --device CPU --workers 4` | `bash scripts/run_data_gen.sh --device CPU --tag <tag>` |
| QBlade 共享库 | `QBladeCE_2.0.9.2/QBladeCE_2.0.9.2.dll`（Win PE32+） | `QBladeCE_2.0.8.6/libQBladeCE_2.0.8.6.so.1.0.0`（ELF x86-64, 18MB） |
| 仿真接口 | 函数式 CLI + ProcessPoolExecutor | OOP `SIMULATION` 类（`run_one_simulation` / `run_all_simulation` / `change_propeller_geometry`） |
| pkl 格式 | `sample_XXXX.pkl`（扁平） | `{geometry_idx: {"geometry": np.ndarray, "RPM*_Wind*_Angle*": DataFrame}}`（data.py 格式 B，已原生支持） |

### Mac 端归档
旧 Windows 版以 `git rm --cached -r Propeller_project-main` 从 index 移除，磁盘上改名为 `Propeller_project-main.win.bak/` 作离线回滚源，已加入 `.gitignore`。如需彻底清理：`rm -rf Propeller_project-main.win.bak`（验证 Linux 链路无误后再做）。

### 新增脚本

```
scripts/
├── install_linux.sh     # conda + pip + submodule + .so + ldd 一键校验
├── run_data_gen.py      # chdir 到子模块 → import SIMULATION → 循环几何 → 聚合 → 落 pkl
├── run_data_gen.sh      # bash 包装，自动激活 LFM env
└── smoke_test.sh        # baseline 几何 × 全工况 → data.py → 校验 raw_data.csv
```

### 已知约束
- `code/Simulation_QBlade/config.py` 用 `os.getcwd()` 解析路径，wrapper 必须先 `chdir` 再 import。
- submodule 目录名含 `-`，不能作 Python 包；通过 `sys.path` 注入解决。
- `ctypes.CDLL` 需要 `libgomp` / `libstdc++` / `libgfortran` 等系统库；`install_linux.sh` 用 `ldd` 校验，缺失时**仅提示**，不自动 `sudo apt`。
- 子模块 clone 时默认 detached HEAD；`install_linux.sh` 会自动 `git checkout` 到分支以防意外丢提交。

### 验证（在 3090_node1 上）
1. `git clone --recursive ...`
2. `bash scripts/install_linux.sh`
3. `bash scripts/smoke_test.sh` → 应产出 `data_for_train/data/data_smoke_*.pkl` + `data_for_train/geometry_run_results_100/raw_data.csv`
4. 训练链路依旧 `python data.py && python train.py`，无需改动。

---

## 11. 后续计划

1. ~~等待 V4 硬约束优化收敛~~ ✓ CMA-ES FM=0.947 (PPO 仍在训练)
2. **QBlade 验证**: 用 V4 最优几何在 QBlade 中仿真，验证 DNN 预测精度
3. **数据扩展**: 补充 QBlade 仿真 (高 RPM 6500-8000 + 低 twist 几何)，重训 DNN
4. **多速度优化**: 在 V=6,8,10,12 m/s 多个速度点联合优化
5. **主动学习**: 在当前最优解附近采样新几何 → QBlade 仿真 → 扩展 DNN 训练集 → 再优化
6. **QBlade 批量仿真**: 利用新的 BatchRunner 并行接口加速数据扩展
