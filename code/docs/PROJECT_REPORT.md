# LFM 项目综合交付文档

> **L**ifting-line **F**ree-wake **M**ethod 数据驱动的螺旋桨气动力代理模型 + 四旋翼配平求解 + RL/进化优化 + QBlade 并行仿真
>
> 最后更新：2026-06-01 · 仓库：`guoyue0412/LFM` @ `version_3`

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 仓库结构](#2-仓库结构)
- [3. 核心模块详解](#3-核心模块详解)
- [4. 数据集](#4-数据集)
- [5. 部署架构](#5-部署架构)
- [6. 优化历程（v3 会话）](#6-优化历程v3-会话)
- [7. 当前运行状态](#7-当前运行状态)
- [8. 已知问题与限制](#8-已知问题与限制)
- [9. 后续工作](#9-后续工作)

---

## 1. 项目概述

### 1.1 总体目标
为四旋翼无人机螺旋桨叶片几何寻找最优设计，最大化巡航品质因数：

```
FM = m·g·V / P_total
```

无人机参数：m=3.5 kg, V=10 m/s, 450 mm 轴距, 10" 螺旋桨（半径 R=0.127 m, 2 叶）。

### 1.2 设计变量
8 个 B 样条控制点：
- 前 4 个 → chord（弦长）
- 后 4 个 → twist（扭转角）
- B 样条插值 → 22 截面 × (r, chord, twist)

### 1.3 工作流（端到端）

```
①  QBlade 仿真         ②  DNN 代理模型      ③  优化求解          ④  验证
   │                       │                    │                    │
   ├─ 1000 几何 ×          ├─ 47→4 MLP          ├─ PPO (SAC env)    ├─ QBlade
   │  42 工况 ×            │  [64,128,64]        ├─ CMA-ES           │  回算
   │  1200 步              │                    └─ trim_solver       │
   │                       ├─ R² > 0.99         (scipy.optimize)    │
   ▼                       ▼                    ▼                    ▼
 ~42000 个                 MAPE < 1.5%          FM* = 0.95         ✓
 仿真样本                                       8 个最优 CP
```

### 1.4 当前阶段
- ✅ 数据生成管线已完整（QBlade SIL + 多进程并行 + 字段验证）
- ⏳ 100 几何减规模数据采集运行中（node1，预估 16 天）
- ⏸ 训练管线已就绪（train.py, sweep.py），等数据到位
- ⏸ 优化求解管线已就绪（ppo_optimize/, trim_solver），等模型到位

---

## 2. 仓库结构

```
LFM/                                                  4745 lines Python
├── README.md                       项目主页（架构 + 快速开始）
├── CLAUDE.md                       开发准则（中文规范 + 验证机制）
├── RESEARCH_LOG.md                 研究日志（优化结果 + 历史数据集）
│
├── docs/                           文档
│   ├── PROJECT_REPORT.md           ⭐ 本文件
│   ├── qblade_data_generation.html ⭐ QBlade 数据生成详解（含 wake 截断）
│   ├── ppo_optimization.md         PPO 优化笔记
│   ├── uav_model_references.md     无人机模型参考
│   └── 当前工作说明与技术路线图.md 旧版状态（2026-05-08）
│
├── config.py                       (73)    全局配置（GeometryMode, DataConfig, ModelConfig 等）
├── data.py                         (458)   pkl 加载 + 标准化 + 三种几何模式 (none/raw/derived)
├── uav_model.py                    (535)   PyTorch MLP 模型 + 训练/评估
├── train.py                        (757)   训练入口（含跨种子、绘图、报告）
├── sweep.py                        (681)   超参网格 / 随机搜索（[64,128,64] 等架构对比）
├── eval.py                         (160)   模型评估 + 推理工具
├── adapter.py                      (216)   DNN → 配平求解器适配（输入归一化、字段映射）
├── plot_style.py                   (304)   matplotlib 风格（论文级 figure）
│
├── quadcopter_trim_solver.py       (398)   ⭐ 四旋翼配平求解（scipy.optimize.root + 多初始值 + 热启动）
├── optimize_trim.py                (548)   贝叶斯优化 + 配平求解组合（B 样条 CP 优化）
├── run_trim.py                     (180)   配平求解 CLI 入口
│
├── ppo_optimize/                   PPO + CMA-ES 优化栈
│   ├── env.py                      (440)   ⭐ Gymnasium 环境（v4：硬约束 + OOD 惩罚）
│   ├── train_ppo.py                (174)   PPO 训练（stable-baselines3, SubprocVecEnv 并行）
│   ├── cma_optimize.py             (252)   CMA-ES 优化（pycma, 进化策略基线）
│   ├── eval_ppo.py                 (488)   PPO 评估 + 决策可视化
│   └── callbacks.py                (101)   训练回调（best model save 等）
│
├── scripts/                        数据生成入口（本次重构）
│   ├── run_data_gen.py             (126)   单进程入口（chdir + import SIMULATION）
│   ├── run_data_gen.sh                     bash 包装（自动激活 LFM env）
│   ├── run_data_gen_parallel.py    (309)   ⭐ 多进程并行入口（multiprocessing.Pool）
│   ├── smoke_test.sh                       端到端 smoke（→ data.py → CSV）
│   └── install_linux.sh                    Linux 部署 + ctypes 校验
│
├── data_for_train/                 数据目录（不进 git）
│   └── data/                       原始 pkl → 训练 CSV
│
├── checkpoints/                    模型 checkpoint（不进 git）
│
└── Propeller_project-main/         ⭐ git submodule
    │                               (https://github.com/guoyue0412/Propeller_project_linux_version
    │                                @ branch: guoyue0412-Linux_version)
    ├── code/Simulation_QBlade/
    │   ├── config.py               file_path 解析（os.getcwd() 相对路径）
    │   ├── class_sim/simulation.py SIMULATION class（含 #3/#4 优化）
    │   └── simulation_parameters/
    │       ├── APC107E_geometry.xlsx    baseline 几何
    │       └── Parameters.xlsx     ⭐ 42 工况表（RPM × WIND × ANGLE）
    ├── QBlade_data/
    │   ├── QBR_file/               .qpr 项目模板
    │   └── QBlade_sim/
    │       ├── Base_simulation.sim 模板
    │       └── Baseline_Blade_Turb/
    │           ├── *.trb           ⭐ 含 wake 截断参数
    │           └── Aero/
    │               ├── Baseline_Blade.bld
    │               ├── *.plr (5 个极坐标)
    │               └── Airfoils/*.afl (5 个翼型)
    └── QBladeCE_2.0.8.6/
        ├── libQBladeCE_2.0.8.6.so.1.0.0   (18 MB Linux .so)
        ├── Libraries/                       (fortran/OpenCL/Qt/...)
        └── SIL_Interface/                   (QBladeLibrary.py + 文档)
```

---

## 3. 核心模块详解

### 3.1 数据生成（`scripts/run_data_gen_parallel.py`）

**输入**：
- `--geometry-npy`：(N, 22, 3) 几何 numpy 文件
- `--num-timesteps`：仿真步数（默认 1000，实测 1200 步稳态最小）
- `--workers`：并行 worker 数
- `--omp-threads`：每 worker OMP/MKL/Qt 线程数
- `--batch-size`：每 N 几何落一个 pkl
- `--start-idx` / `--end-idx`：断点续传

**核心设计**：
1. Pool worker 用 `fork` 创建
2. 每 worker 启动时 `setup_worker_workdir`：
   - `/tmp/lfm_w<pid>/code/` ← 深拷贝（不能 symlink，否则 chdir 跟随回 source）
   - `/tmp/lfm_w<pid>/QBlade_data/` ← 深拷贝
   - `/tmp/lfm_w<pid>/QBladeCE_2.0.8.6/` ← symlink（read-only）
3. `worker_init` 设 OMP/MKL/BLAS/Qt 线程数环境变量
4. `chdir` 到独立 cwd → `config.py` 用 `os.getcwd()` 解析正确路径
5. `imap_unordered` 动态分发几何，每完成 N 个落盘一次

**输出格式**（每个 pkl）：
```python
{
    "geometry_0": {
        "geometry": np.ndarray (22, 3),
        "RPM5000_Wind10_Angle82": pd.DataFrame (120 行 × 28 列),
        "RPM5000_Wind10_Angle83": pd.DataFrame,
        ... (42 工况)
    },
    "geometry_1": {...},
    ...
}
```

DataFrame 列（28 列）：
- 时间序列：`Time, Thrust, Power, Torque, Thrust_y, Thrust_z, Mx, My, Mz`
- 标识：`RPM, WIND_SPEED, ANGLE`
- 聚合标量：`THRUST, POWER, TORQUE, THRUST_Y, THRUST_Z, FX, FY, FZ, MX, MY, MZ, Ct, Cp, eta`

详细的 QBlade SIL 原理 + Free Vortex Wake (FVW) 算法 + 6 个优化里程碑见 `docs/qblade_data_generation.html`。

### 3.2 数据加载（`data.py`）

**支持 3 种 pkl 格式**：
- A：`{"case_results": {RPM*_Wind*_Angle*: DataFrame}}`（旧版 sample_XXXX.pkl）
- **B：`{"geometry_<idx>": {"geometry": ndarray, RPM*_Wind*_Angle*: DataFrame}}`（本次重构格式 ⭐）**
- C：顶层 RPM*_Wind*_Angle* 键（无几何）

**几何特征模式**（`config.GeometryMode`）：
| 模式 | 输入维度 | 用途 |
|---|---|---|
| `none` | 3（RPM/WIND/ANGLE） | baseline，单几何 |
| `raw` | 3 + 22×2 = 47 | 直接传 chord/twist 截面 |
| `derived` | 3 + N（B 样条 CP） | 降维输入 |

**输出列**（DNN 拟合目标）：默认 `[THRUST, POWER, FY, FZ]`（4 维），可配置加 `Mx/My/Mz`。

### 3.3 DNN 模型（`uav_model.py` + `train.py`）

**架构**：MLP `[64, 128, 64]`，ReLU，dropout 0.1
**训练**：Adam, MSE loss, early stopping, 跨种子 ensemble
**评估**：MAPE per output, R², residual plot
**报告**：`sweep_results/<run>/` 含 LaTeX/Markdown 自动导出

历史数据集（旧版）训练精度（RESEARCH_LOG.md §2）：
| Output | MAPE |
|---|---|
| Fx | 1.03% |
| Fy | 62.42%（量级极小，不影响配平） |
| Fz | 1.50% |
| Torque | 0.99% |

### 3.4 配平求解（`quadcopter_trim_solver.py`）

scipy.optimize.root 求解 4 个未知数：
- `δ_pitch` (姿态俯仰角)
- `δ_motor_front`, `δ_motor_rear`（前/后电机 RPM）
- 1 个冗余度

满足 3 个方程：
- **Fx 平衡**（前进推力 = 阻力）
- **Fz 平衡**（升力 = 重力）
- **My 平衡**（俯仰力矩 = 0）

**鲁棒性技术**（v2 → v3 → v4）：
- 多初始值（6 组）+ 热启动 → 收敛率 3% → 100%
- 软失败惩罚（reward=-1，而非 -10）
- 训练域内边界约束（CP 范围 [+5% 边距]）

### 3.5 优化栈

#### PPO（`ppo_optimize/train_ppo.py`）
- stable-baselines3 PPO
- 8 路 SubprocVecEnv 并行
- 环境：`env.py` v4（DNN 代理 + 硬 CP 边界 + 训练域 OOD 惩罚）
- reward = `FM - failure_penalty`

#### CMA-ES（`ppo_optimize/cma_optimize.py`）
- pycma 实现
- 8 维搜索空间（B 样条 CP）
- 与 PPO 对照基线

---

## 4. 数据集

### 4.1 历史数据集（参考）
- 几何数：**100** → 现扩展为 **1000**（`geometry_data.npy` (1000, 22, 3)）
- 工况组合：原 72（RPM × WIND × ANGLE）→ 现 **42**（WIND=10 固定）
- 总样本数：原 7200 → 现 42000（预期）
- 范围：
  - RPM ∈ [4000, 6500]，6 档（步长 500）
  - WIND = **10 m/s 固定**（用户钦定）
  - ANGLE ∈ [82°, 88°]，7 档（步长 1°）

### 4.2 当前数据生成参数
```
geometry_data.npy   : (1000, 22, 3)  来源 Propeller_project-main.win.bak/geometry_generated/
Parameters.xlsx     : 42 工况  (6 RPM × 7 Angle, Wind=10)
NUMTIMESTEPS        : 1200 步  (实测 800/400 步 Fy/My 误差 90%+，1200 步是稳态最小)
TIMESTEP            : 0.000083 s  (单步 0.083 ms，1200 步 = 0.1 s 物理时间)
末 120 步取均值     : data_num = 1200 - 120
```

---

## 5. 部署架构

### 5.1 节点资源

| 节点 | CPU | RAM | GPU | 用途 |
|---|---|---|---|---|
| **node1 (3090_node1)** | 80 核 | 251 GB | 8 × 3090 | 主数据生成（当前运行中） |
| **node6 (ksyun_sh_node6_ngc_container)** | 160 核 | 1 TiB | 8 × A800 80GB | 备用 / mini 验证 |
| Mac (本地) | 12 核 | 36 GB | — | 代码编辑 + scp 中转 |

### 5.2 部署流程（node1 / node6）

1. `git clone --recursive https://github.com/guoyue0412/LFM.git`
2. `bash scripts/install_linux.sh --skip-conda`（依赖检查）
3. `pip install -r requirements.txt`（含 torch / sb3 / cma）
4. node6 离线场景：从 Mac scp Qt5/libGLU/libiconv 依赖到 `/root/propeller/extra_libs/`
5. 设置 `LD_LIBRARY_PATH=$LIB_DIR:$CONDA_LIB`
6. 运行 `scripts/run_data_gen_parallel.py`

### 5.3 worker 隔离架构

```
node1
└─ python run_data_gen_parallel.py --workers 10 --omp-threads 8
    ├─ main process (Pool dispatch)
    └─ worker 0..9 (fork)
        ├─ /tmp/lfm_w<pid>/
        │   ├─ code/                 ← 深拷贝 (~164KB)
        │   ├─ QBladeCE_2.0.8.6/     → symlink
        │   └─ QBlade_data/          ← 深拷贝 (~80MB)
        ├─ chdir(/tmp/lfm_w<pid>/code/Simulation_QBlade)
        ├─ os.environ["OMP_NUM_THREADS"] = "8"
        ├─ os.environ["QT_THREAD_POOL_MAX_THREAD_COUNT"] = "8"
        └─ QBladeLibrary createInstance(0, 32) → loadSim → advance
```

---

## 6. 优化历程（v3 会话）

本次会话完成的 **9 个里程碑**（详细 commit 见 git log）：

### 🔴 Fix 类（数据正确性）

#### #1 Linux 适配 + submodule 化（commit a705683）
- 旧版 Propeller_project-main 是 Windows DLL，不可用
- 接入 [guoyue0412/Propeller_project_linux_version](https://github.com/guoyue0412/Propeller_project_linux_version) 作 git submodule
- 写 `scripts/install_linux.sh` 自动校验依赖 + ctypes dry-load

#### #2 修复 `.bld` 列错位（commit 265afa4）
- Linux 分支历史误把 chord=55m / twist=0.02° 写入（应该是 chord=0.018m / twist=45°）
- 导致 wake 计算从 step 0 NaN abort
- 修复：从 main 分支同步正确 QBlade_data/

#### #3 NaN abort 静默写零（commit f249608）
- 子模块 `advanceTurbineSimulation` 没检查返回值
- NaN 后继续 `getCustomData` 拿 0 值，混入"无效数据"
- 修复：参考 QBlade SIL `sampleScript.py` 加 success 检查 + 双重 THRUST 兜底

#### #4 FX 字段错误（commit ad8b9bb）
- 之前用 `b"Aerodynamic Force in Hub X_g Direction"` 不存在
- `getCustomData_at_num` 找不到字段返回 -1，× -1 → FX 恒为 1
- 修复：删除 raw Fx 采集，FX 作为 THRUST 别名（X 方向 = 旋转轴向）

### 🟢 Perf 类（性能优化）

#### #5 跳过 storeProject（commit 70fe8e8）
- 每工况末尾 `storeProject` 写 13MB .qpr 项目文件 ≈ 60s IO
- 训练数据全在 pkl 中，.qpr 仅 GUI 调试用
- **收益**：全规模 ~28 天 IO 节省

#### #6 createInstance 跨工况复用（commit f249608）
- 原本每工况 `createInstance` + `closeInstance`（OpenCL 枚举 ~4s）
- 独立测试验证 `loadSimDefinition` 第二次调用能完全 reset wake/state（**Thrust 误差 0.0000%**）
- 重构 SIMULATION 类：`self.QBLADE` lazy 创建 + 跨工况复用
- **收益**：单工况 init 4.5s → 2.4s

#### #7 Wake 截断（commit bbaff14）
- 修改 `.trb` 文件 FVW 参数：
  - `MAXWAKEDIST` 100 → 10（10 倍螺旋桨直径）
  - `ZONE1LENGTH` 2 → **0.5** 转
  - `ZONE2LENGTH` 4 → **1** 转
  - `ZONE3LENGTH` 6 → **1** 转
  - `ZONE3FACTOR` 2 → 8（远场粗化）
  - `MAXWAKESIZE` 200000 → 30000
- **实测**：1200 步 wall clock 几乎无变化（wake 截断在末段才生效），但数据 **0.0% 误差**（小数点 4 位）
- 详见 `docs/qblade_data_generation.html`

### 🔵 Feat 类（功能扩展）

#### #8 数据采集字段扩展（commit 03bdfb7）
- 新增 `Mx, My, Mz` 三轴力矩采集
- 聚合输出 `FX/FY/FZ/MX/MY/MZ`（向后兼容旧字段）
- 工况表扩到 **42**（之前只有 2 个 RPM×Angle）

#### #9 多进程并行（commit 800f594）
- `scripts/run_data_gen_parallel.py`：`multiprocessing.Pool` + 工作目录隔离
- 加 `--start-idx/--end-idx/--batch-size` 支持断点续传
- 加 OMP/MKL/Qt 线程数限流（avoid over-subscription）

---

## 7. 当前运行状态

### 7.1 节点 node1（task #17）
```
进程：     main PID 500283 (在 node1 后台跑)
启动时间： 2026-06-01 11:30 UTC
配置：     workers=10, omp=8, timesteps=1200, batch_size=16
规模：     100 几何 × 42 工况 × 1200 步
输出：     /root/LFM/data_for_train/data/data_prod100_node1_*_b*.pkl (预期 7 个 pkl)
日志：     /root/prod100_node1.log
```

**性能实测（5.6 小时后）**：
- worker 完成 ~700 / 1200 步（第 1 工况）
- 实测平均单步 **~28 s/step**（vs 单 worker mini 1.1 s/step）
- **预估全规模 wall clock：~16-30 天**（取决于末段 wake 加速）

### 7.2 监控命令

```bash
# 进度
ssh 3090_node1 "tail -c 1500 /root/prod100_node1.log | grep -E '进度|落盘'"

# pkl 数
ssh 3090_node1 "ls /root/LFM/data_for_train/data/data_prod100*.pkl | wc -l"

# 进程存活
ssh 3090_node1 "pgrep -af run_data_gen_parallel | head"

# 断点续传（崩了或自愿停）
ssh 3090_node1 "cd /root/LFM && export LD_LIBRARY_PATH=... && \
  nohup python scripts/run_data_gen_parallel.py \
    --workers 10 --omp-threads 8 --num-timesteps 1200 \
    --geometry-npy /root/geometry_data.npy --batch-size 16 \
    --start-idx <last_done> --end-idx 100 \
    --tag prod100_resume > /root/resume.log 2>&1 & disown"
```

---

## 8. 已知问题与限制

### 8.1 已解决（v3 会话）
- ✅ Windows-only 仓库不可用 → submodule 化
- ✅ .bld 列错位 → 从 main 同步
- ✅ NaN 静默写零 → 加 success 检查
- ✅ FX 恒为 1 → 删除字段，作为 THRUST 别名
- ✅ Linux 部署 + ctypes 加载 → install_linux.sh
- ✅ node6 离线 → 通过 Mac 中转 Qt5/GLU/iconv 依赖

### 8.2 部分缓解
- ⚠️ **多进程性能退化**：单 worker 22.5 min/工况，但 10 worker 并发实测 28 s/step（比单 worker 慢 26 倍平均）
  - 假说：80 MB QBlade_data × 10 副本触发 L3 cache thrashing
  - 假说：10 worker × 18 NLWP = 180 thread 过度调度
  - 缓解尝试：OMP_NUM_THREADS=8 限流（已加），但未根本解决
  - 待测：`--workers 4 --omp-threads 16` 减并发度

### 8.3 不可解决（QBlade SIL 固有）
- ❌ **GPU 路径不可用**：QBlade 2.0.8.6 OpenCL kernel 与 propeller wake 不兼容
  - 单 GPU 可跑（200 步 169s）但 1200 步精度未验证
  - 多 GPU 并发触发 `CL_INVALID_DEVICE(-33)` core dump
  - CUDA_VISIBLE_DEVICES 隔离改 -33 为 hang
  - Docker 不能解决（OpenCL ICD 在驱动层全局共享）
- ❌ **400/800 步精度不足**：Fy/My 误差 90-115%（量级小被噪声放大）
  - 1200 步为稳态最小要求

### 8.4 限制
- 100 几何（10% 总规模）受 wall clock 制约
- LFM 训练后需要扩展到 1000 几何以满足 RESEARCH_LOG 中提到的 OOD 检测需求
- 若 PPO 优化阶段发现 OOD 问题严重，需补 GPU + 1000 几何 + 1200 步 ≈ ~160 天（不现实）

---

## 9. 后续工作

### 9.1 短期（数据到位后）
- [ ] 跑 `python data.py` 整合 7 个 pkl → `raw_data.csv`（4200 样本）
- [ ] 跑 `python train.py` 训出 DNN baseline（[64,128,64]）
- [ ] 跑 `python sweep.py --quick` 找架构最优
- [ ] 跑 `python run_trim.py` 验证配平求解器

### 9.2 中期（优化栈）
- [ ] PPO 训练（ppo_optimize/train_ppo.py）
- [ ] CMA-ES 对照基线
- [ ] DNN OOD 检测（v3 → v4 演化）

### 9.3 长期（数据扩展）
- [ ] 解决多进程退化（试 `--workers 4 --omp-threads 16`）
- [ ] 或换更大节点（如 256+ 核 EPYC）
- [ ] 把 100 几何扩到 500-1000

### 9.4 文档维护
- [ ] 数据到位后更新 `RESEARCH_LOG.md` 表格（v3 实际 MAPE）
- [ ] 把本文档转 HTML（与 `qblade_data_generation.html` 同风格）

---

## 附录：关键 commit 与文档

| Commit | 类型 | 内容 |
|---|---|---|
| `a705683` | Linux 部署 | submodule 化 + install_linux.sh |
| `265afa4` | Fix | .bld 列错位修复 |
| `f249608` | Perf | createInstance 复用 + NaN 兜底 |
| `70fe8e8` | Perf | 跳过 storeProject |
| `03bdfb7` | Feat | 数据字段扩展 + 42 工况 |
| `ad8b9bb` | Fix | FX 字段错误 |
| `bbaff14` | Perf | wake 截断（激进版） |
| `b7a2a7f` | Docs | qblade_data_generation.html |

### 子模块仓库
- 主：[guoyue0412/LFM](https://github.com/guoyue0412/LFM) @ `version_3`
- 子：[guoyue0412/Propeller_project_linux_version](https://github.com/guoyue0412/Propeller_project_linux_version) @ `guoyue0412-Linux_version`

### 文档清单
- 本文件：`docs/PROJECT_REPORT.md`（综合交付）
- 数据生成：`docs/qblade_data_generation.html`（QBlade 原理 + wake 截断详解）
- 优化笔记：`docs/ppo_optimization.md`
- 模型参考：`docs/uav_model_references.md`
- 研究日志：`RESEARCH_LOG.md`（历史结果）
- 开发准则：`CLAUDE.md`
- 主页：`README.md`
