# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

LFM 是一个**螺旋桨气动优化全栈系统**,围绕"DNN 代理模型 + 四旋翼配平 + RL/进化优化 + QBlade 验证"四个阶段构建。设计变量是 8 个 B 样条控制点(4 chord + 4 twist),目标是最大化巡航品质因数 FM = mg·V / P_total。所有终端输出和文档使用中文。

**两套并行管线**:
- **V1 管线**(根目录:`config.py` / `train.py` / `adapter.py` / `quadcopter_trim_solver.py` / `ppo_optimize/`):基于单 MLP DNN 代理模型 + FM 最大化目标。
- **V2 管线**(`optimization_v2/`):**最终毕业方案** — LLFVW-MoE 代理模型 + 前飞配平约束 + 单位距离能耗最小化 (`min P_elec/V`)。直接使用 MoE 输出的 `[T, H, My, Q] + [σ_T, σ_H, σ_My, σ_Q]`,不再用解析估算 M_p_y。

## 常用命令

环境:
```bash
conda create -n LFM python=3.10 -y && conda activate LFM
pip install -r requirements.txt
```

DNN 训练流程(三步):
```bash
python data.py                                  # pkl → 标准化 CSV (写到 data_for_train/geometry_run_results_100/)
python train.py                                 # 训练 + TensorBoard + 论文级报告
python eval.py                                  # 测试集评估
python eval.py --rpm 5000 --wind 9 --angle 85   # 单点预测
tensorboard --logdir=./runs
```

超参扫描:
```bash
python sweep.py --quick --max-trials 4          # 快速模式
python sweep.py --max-trials 20 --epochs 1000   # 正式扫描
```

配平求解:
```bash
python run_trim.py --dnn --V 10                 # DNN 单点配平
python run_trim.py --compare                    # dummy vs DNN 速度扫描对比
python optimize_trim.py --V 10                  # 几何 + 状态联合优化
```

PPO / CMA-ES 优化(默认开启硬约束 `--constrain`):
```bash
python ppo_optimize/cma_optimize.py --maxiter 200 --popsize 16 --constrain --save-dir ./cma_results_constrained
python ppo_optimize/train_ppo.py --total-timesteps 200000 --n-envs 4 --constrain --save-dir ./ppo_models_v4_constrained
python ppo_optimize/eval_ppo.py
```

V2 管线(LLFVW-MoE + 前飞配平 + min P_elec/V):
```bash
python -m optimization_v2.run                       # 完整流程: data → train → optimize → evaluate
python -m optimization_v2.run --step data           # 仅生成合成数据 (6000 条)
python -m optimization_v2.run --step train --moe-epochs 500
python -m optimization_v2.run --step optimize --maxiter 30 --popsize 8
python -m optimization_v2.run --step evaluate
python -m optimization_v2.run --V 12 --mass 4.0 --eta 0.85   # 自定义工况

# Excel 真实数据驱动 (替代合成数据)
python -m optimization_v2.run --inspect data.xlsx                       # 先检视结构
python -m optimization_v2.run --excel data.xlsx --step train            # Excel → MoE 训练
python -m optimization_v2.run --excel data.xlsx --sheet "sheet1"        # 指定工作表
```

QBlade 数据采集(Linux 节点,`Propeller_project-main/` 是 submodule):
```bash
bash scripts/install_linux.sh                    # 首次部署:校验 submodule + .so + 建 LFM env
bash scripts/run_data_gen.sh --device CPU        # 默认 baseline 几何,全工况
bash scripts/run_data_gen.sh --geometry-npy <(N,22,3).npy --device CPU --tag run1   # 批量多几何
bash scripts/smoke_test.sh                       # 端到端冒烟(到 data.py 为止,不跑 train.py)
```
**注**:旧 `Propeller_project-main/get_data.py` CLI 已随 Windows 版归档(`Propeller_project-main.win.bak/`),Linux 版用 OOP `SIMULATION` 类,wrapper 在 `scripts/run_data_gen.py`,见 RESEARCH_LOG §10。

## 高层架构

### 流水线四阶段

```
QBlade 仿真          DNN 代理模型         优化求解              QBlade 验证
─────────────       ─────────────       ──────────────       ─────────────
get_data.py    →    data.py        →    ppo_optimize/   →    BatchRunner
batch.py            train.py            cma_optimize.py      回验最优 CP
                    sweep.py            train_ppo.py
                    eval.py
```

### DNN 代理模型(`config.py` / `data.py` / `train.py` / `adapter.py`)

- 模型 `PropellerPredictor`(`train.py:40`):MLP `[64, 128, 64]`,带 BatchNorm + ReLU,前两层无 Dropout,**第三层及之后**才施加 Dropout(`train.py:55-56`,`if i >= 2`)。
- 输入维度由 `Config.geometry_mode` 决定:`"none"` → 3,`"control_points"` → 11,`"sections"` → 47(默认,`config.py:34`)。改 mode 后 `__post_init__` 会自动重算 `input_columns` / `input_dim`。
- 输出 4 维 `[Fx, Fy, Fz, Torque]`,但 `Fy` 量级极小、MAPE 极大,配平时仅使用 Fx / Fz / Torque(见 RESEARCH_LOG)。
- 数据流:`data.py` 递归扫描 `data_for_train/data` 下的 `data*.pkl` / `sample_*.pkl` → 抽取工况(RPM/WIND/ANGLE)+ 几何(22 截面 chord/twist)→ `StandardScaler` → 写 `processed_data_dir` 下的 `X_*_scaled` / `y_*_scaled` 和 `scaler_X.pkl` / `scaler_Y.pkl`。
- `adapter.py` 是关键耦合层:把 DNN 包装为 `(V, alpha_rad, omega_rpm) → (T_p, H_p, M_p_y[, Q_p])`。**坐标约定** `ANGLE = 90° − degrees(alpha)`(QBlade 90° 对应悬停),`Fx → T_p`、`Fz → H_p`、`Torque → Q_p`,`M_p_y` 由桨叶元解析公式 `k_m · T_p · R · μ` 估算(DNN 不直接预测)。

### 配平求解器(`quadcopter_trim_solver.py` / `uav_model.py`)

- 解 3 个非线性方程 `Fx=0, Fz=0, My=0`,未知量 `[alpha, Omega1, Omega2]`,`scipy.optimize.root`。
- **多初始值 + 热启动**(`solve_multi_start`):6 组初始猜测,收敛率从 ~3% 提升到 100%。环境/优化代码失败时不要降低 `solve_multi_start` 的鲁棒性。
- `dummy_fuselage_aero` / `dummy_rotor_aero` 是解析基线;`uav_model.py` 是参数化机身模型;DNN 通过 `solver.set_rotor_aero(rotor_func)` 注入。

### PPO / CMA-ES 优化(`ppo_optimize/`)

- `env.PropellerDesignEnv`:Gymnasium 环境,状态 13 维 `[8 CP + alpha + Omega1 + Omega2 + power + step_ratio]`,动作 8 维 `[-1, 1]` → CP 增量。奖励 = FM 改善 + β·FM − OOD 惩罚。
- **关键约束系统**(V4 版本默认开启,见 RESEARCH_LOG §4.4):
  - `CP_BOUNDS_DATA`(`env.py:116`):由训练数据 100 个几何反推的硬边界,加 5% 边距。
  - `CP_BOUNDS_WIDE`:V1-V3 的宽松边界,仅向后兼容,**不要用作默认**。
  - 配平 RPM 限制 `[4000, 6500]`、ANGLE 限制 `[82°, 88°]`(对应 alpha `[2°, 8°]`)。
  - **设计原则**:`--no-constrain` 会让 DNN 外推产生"虚假最优"(V1 FM=1.004,几何 OOD 22/22),除非显式调试外推行为,否则不要默认关闭。
- `cp_to_sections`(`env.py:49`):用预定义的 `_GEOMETRY_R` / `_CHORD_KNOTS` / `_TWIST_KNOTS` B 样条插值,8 CP → 22 chord + 22 twist。这些常量与训练数据几何分辨率绑定,改动会破坏 DNN 输入分布。
- `train_ppo.py` 用 stable-baselines3 + `SubprocVecEnv`(`--n-envs N`)并行;每个子进程独立加载模型避免 GPU 争抢。

### QBlade 仿真管线(`Propeller_project-main/code/class_sim/`)

⚠️ **2026-05-29 起 `Propeller_project-main/` 已切换为 Linux submodule**(`guoyue0412/Propeller_project_linux_version.git`),目录结构与本节描述的 Windows 重构版不同。Linux 版只暴露 `code/Simulation_QBlade/class_sim/simulation.py` 中的 OOP `SIMULATION` 类(`run_one_simulation` / `run_all_simulation` / `change_propeller_geometry`),输出 `geometry_simulation_dict.pkl` 兼容 `data.py` 格式 B。完整迁移说明见 RESEARCH_LOG §10,数据生成入口已迁移到 `scripts/run_data_gen.py`。

以下 Windows 重构版架构留作历史参考(对应 `Propeller_project-main.win.bak/`):

模块化重构后的架构(2026-03-28,见 RESEARCH_LOG §9):

- `config.py`:`SimPaths`(DLL/模板路径)、`SimConfig`(device/timesteps)、`SimCondition`(单工况)。
- `sim_file.py`:`.sim` 模板正则替换(消除原 `simulation.py` 中的 3 处重复)。
- `geometry.py`:B-spline 插值 + `.bld` 文件修改。
- `runner.py`:`QBladeRunner` context-manager,负责 DLL 加载/卸载。
- `batch.py`:`BatchRunner` 三种调度策略 — `gpu`(顺序,OpenCL 互斥)、`cpu`(`ProcessPoolExecutor` N 进程)、`hybrid`(1 GPU + (N-1) CPU)。每个 worker 独立 DLL,无共享状态。
- 公开 API 全部从 `class_sim/__init__.py` 导出,**不要直接 import 子模块内部符号**。
- `get_data.py` 是 CLI 入口,使用 `clone_sim_tree` 给每个 worker 复制独立的 `.sim`/`.bld` 模板树。

### V2 管线(`optimization_v2/` — 最终毕业方案)

LLFVW-MoE 代理模型 + 前飞配平约束 + min P_elec/V 目标。模块化、与 V1 完全独立:

- `config.OptConfig`:全局配置数据类(工况、CP 边界、MoE 超参、CMA-ES 超参、电气参数、`uncertainty_weights` 元组)。
- `geometry.py`:**V2 内部唯一的列 schema 真相源**。导出 `INPUT_COLS`(47)、`OUTPUT_COLS`(`["T","H","My","Q"]`)、`CHORD_COLS`、`TWIST_COLS`、`ALL_COLS`、`split_sections()` helper、`N_SECTIONS=22`、`TWIST_NORM_SCALE=10.0`。`train_moe`/`data_loader`/`active_learning`/`synthetic_data`/`evaluate` 全部 import 这些常量,改 schema 时只改一处。`cp_to_sections` 与 V1 的 `ppo_optimize/env.py` 共享同一套 knot 向量和径向站位,但 V2 重新定义而非 import 以维持模块隔离。
- `synthetic_data.py`:基于物理的合成数据生成器(在真实 LLFVW 数据就绪前使用)。生成 100 几何 × 60 工况 = 6000 条样本,格式与真实 LLFVW 输出对齐(`T, H, My, Q`)。
- `data_loader.py`:**Excel/CSV 真实数据加载器**,自动识别两种几何格式:`sections` 模式(`chord_0..21 + twist_0..21`)和 `cp` 模式(`cp_0..7` 或 `chord_cp_0..3 + twist_cp_0..3`,自动 B-Spline 展开)。可配置列名映射(`ColumnMapping`),支持别名表(`RPM/rpm/rotor_speed`、`T/Thrust/Fx`、`Q/Torque/Mz` 等)。CLI: `python -m optimization_v2.run --inspect file.xlsx` 检视结构,`--excel file.xlsx` 加载使用。
- `moe_model.MoEPredictor`:Mixture of Experts 架构 — `SharedEncoder(47→64) → GatingNetwork(K) + K×Expert(64→128→8)`,每个 expert 输出 `[μ_T, μ_H, μ_My, μ_Q, log_σ_T, log_σ_H, log_σ_My, log_σ_Q]`。混合方差公式 `σ² = Σ π_k(σ_k² + μ_k²) - μ²`。
- `moe_model.MoELoss`:异方差 NLL + balance 损失 + 门控熵损失。
- `train_moe.py`:训练脚本,使用 `StandardScaler` 标准化输入输出,保存 `moe_best.pth` + `scaler_X.pkl` + `scaler_y.pkl`(注意 V1 是 `scaler_Y.pkl` 大写 Y)。
- `fuselage.py`:cross-flow 半经验机身气动模型(`FuselageParams` 数据类)。**与 V1 `dummy_fuselage_aero` 物理参数相同但故意不复用**(全新管线 isolation 决策)。
- `trim_solver.TrimSolverV2`:**与 V1 关键区别** — `M_p_y` 和 `Q` **直接来自 MoE**,不再用解析估算;不确定度 `σ` 随配平结果一起返回(`TrimResultV2.sigma_mean`/`sigma_max`)。残差函数为闭包,通过预分配 `(2, 47)` 输入缓冲一次前向预测前后桨,每次残差只覆写 `[RPM, V, ANGLE]` 三列。`_make_residuals_fn` 内部 `last` 字典缓存最后一次 mu/sigma,配平收敛后免去 2 次冗余 MoE 查询。残差 `< 1e-3` 时跳出多初始值循环。
- `power.py`:`P_aero = 2(Q1·Ω1 + Q2·Ω2)` → `P_elec = P_aero / η_motor_esc` (η=0.80) → `Range = V·E_b / P_elec`。
- `objective.evaluate_design`:单点评估 `J = P_elec/V + λ_R·|R|² + λ_U·U + λ_G·C_geo`。配平失败返回 `cfg.penalty_trim_fail = 1e6`。`uncertainty_penalty` 用 `cfg.uncertainty_weights = (1.0, 0.5, 0.5, 1.0)`(T、Q 入功率,权重最高)。
- `cma_optimizer.run_cma_optimization`:在归一化空间 `[0,1]^8` 中运行 CMA-ES,内部映射回物理 CP。
- `active_learning.ActiveLearningManager`:基于 `σ > σ_max` 触发,导出待补点 CSV 供 LLFVW 仿真,新数据合并回主数据库。
- `evaluate.py`:可视化(几何对比、优化历史、报告表),`_plot_blade_axes` helper 复用 chord/twist 双子图。
- `run.py`:CLI 主入口,支持 `--step {data,train,optimize,evaluate}` 分步运行或全流程。
- 产物:`optimization_v2/data/`(数据集 CSV)、`models/`(MoE 权重 + scalers + train_history.json)、`results/`(`cma_result.json`、`final_report.json`、`*.png`)。

## 重要约定

- **角度**:配平用弧度;DNN/QBlade 用 `ANGLE = 90° − α(deg)`;切勿混用。
- **CP 边界**:任何新优化器、约束、初始猜测必须使用 `CP_BOUNDS_DATA`(数据驱动),除非是对 V1-V3 行为的回归测试。
- **几何 sections**:`n_sections=22`、`n_control_points=8` 与训练数据几何采样点严格对齐;改动需同时更新 `data.py` 提取逻辑、`env.cp_to_sections` 的 knot/R 数组、训练数据。
- **V2 列 schema 改动**:任何新增/重命名输入或输出列(如 `T → Thrust`)只改 `optimization_v2/geometry.py` 顶部常量;勿在 `data_loader`/`train_moe`/`active_learning`/`synthetic_data` 中重复定义。`OUTPUT_COLS = ["T","H","My","Q"]` 与 V1 的 `["Fx","Fy","Fz","Torque"]` 描述同一物理量但命名故意不同(V2 更接近 Ye 文献符号)。
- **V2 真实数据**:用户数据为 Excel 格式,每行一个样本,几何列可能是 8 CP 或 44 sections;`data_loader.load_excel_dataset` 自动识别两种格式并经 B-Spline 展开为统一的 47 维 MoE 输入。真实 LLFVW 数据生成中(2026-05-17),开发先用 `synthetic_data.py` 跑通闭环。
- **绘图**:统一通过 `plot_style.apply_style()` + `savefig`(`plot_style.py`),保证论文级一致性;新脚本要 `import matplotlib; matplotlib.use("Agg")` 避免无显示环境报错。
- **QBlade DLL**:许可证限制,`*.dll` / `Propeller_project-main/QBladeCE_*/Binaries/` 已在 `.gitignore`。本地缺失 DLL 时仿真路径不可运行,但 DNN 训练/优化路径不依赖。
- **大文件**:`*.pkl` 训练数据、`*.pth` 权重、`runs/` TensorBoard 日志、`*_detail.json` / `*_curves.npz` 扫描详情都不入库;只提交脚本和 `report/` 摘要。

## 输出产物路径

| 产物 | 路径 |
|------|------|
| DNN 权重 | `trained_models/propeller_predictor.pth` |
| 标准化器 | `data_for_train/geometry_run_results_100/scaler_{X,Y}.pkl` |
| 论文级报告 | `trained_models/report/{test_metrics.csv, report.tex, *.png}` |
| 超参扫描 | `sweep_results/sweep_*.{csv,tex,png}`(`*_detail.json` 不入库) |
| PPO 模型 | `ppo_models_v4_constrained/` |
| CMA-ES 最优几何 | `cma_results_constrained/cma_best.json` |
| OOD 分析 | `analysis_gap/{gap_analysis.json, *.png}` |

## 参考文档

- `README.md`:对外完整说明 + 命令参考表。
- `RESEARCH_LOG.md`:实验记录、版本演进(V1→V4)、各方法 FM 对比、QBlade 重构细节。**做架构变更前先读 §4-§5 理解为何 V4 选择硬约束**。
- `docs/ppo_optimization.md`:PPO 技术方案细节。
- `docs/uav_model_references.md`:机身气动模型的文献来源。
