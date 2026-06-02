# LFM 技术参考文档

> 代码级 API 参考 + 算法推导 + 数据流详解
>
> 配套阅读：[`PROJECT_REPORT.md`](./PROJECT_REPORT.md) (项目综述) · [`qblade_data_generation.html`](./qblade_data_generation.html) (QBlade 细节)
>
> 最后更新：2026-06-01 · 适用版本：`version_3` (commit `d32f2b9`+)

---

## 目录

1. [系统数据流](#1-系统数据流)
2. [Propeller_project-main 子模块（仿真层）](#2-propeller_project-main-子模块仿真层)
3. [scripts/ 数据生成层](#3-scripts-数据生成层)
4. [config.py 全局配置](#4-configpy-全局配置)
5. [data.py 数据加载与标准化](#5-datapy-数据加载与标准化)
6. [uav_model.py 无人机物理模型](#6-uav_modelpy-无人机物理模型)
7. [train.py 与 uav_model 训练栈](#7-trainpy-与-uav_model-训练栈)
8. [adapter.py 代理模型适配器](#8-adapterpy-代理模型适配器)
9. [quadcopter_trim_solver.py 配平求解](#9-quadcopter_trim_solverpy-配平求解)
10. [ppo_optimize/ PPO + CMA-ES 优化栈](#10-ppo_optimize-ppo--cma-es-优化栈)
11. [关键算法推导](#11-关键算法推导)
12. [完整 API 使用示例](#12-完整-api-使用示例)
13. [关键设计决策](#13-关键设计决策)
14. [配置参考表](#14-配置参考表)

---

## 1. 系统数据流

```
                        ┌─────────────────────────────────────────────────────────────┐
                        │                    LFM 端到端数据流                          │
                        └─────────────────────────────────────────────────────────────┘

   ① 几何输入                ② QBlade 仿真             ③ 数据整合             ④ DNN 训练
   ─────────                ───────────────           ──────────             ─────────
   geometry_data.npy        SIMULATION class          data.py                train.py
   (1000, 22, 3)            │                         │                      │
        │                    ├─ generating_sim_file   ├─ load_raw_pkl()      ├─ PropellerPredictor
        │                    │  └─ 写 42 个 .sim       │  ├─ 扫描 *.pkl       │  (MLP [64,128,64])
        │                    ├─ createInstance(0,32)  │  ├─ 识别格式 A/B/C   │
        │                    ├─ loadSimDefinition     │  ├─ 提取 geometry    ├─ DataLoader
        ▼                    ├─ initializeSimulation  │  ├─ 提取工况 DF      ├─ Adam + early stop
   geometry_idx              │                         │  └─ 合并成 DF         │
        │                    └─ 循环 1200 step        ├─ process_and_save() ├─ TensorBoard log
        │                       ├─ advanceTurbineSim  │  ├─ split train/test │
        │                       └─ 末 120 步取均值    │  └─ StandardScaler    ▼
        ▼                                              ▼                      .pth + report.tex
   42 个 sim 文件             scripts/                 raw_data.csv          │
        │                    run_data_gen_parallel.py X/y_train/test_scaled  ▼
        │                    │ (multiprocessing.Pool) (pickle)               adapter.py
        │                    │ + worker_init()                                │
        │                    │ + setup_worker_workdir                         ▼
        ▼                    │                                              ⑤ 配平求解
   .sim → QBlade .so         pkl                                            ─────────
                              │                                              quadcopter_trim_solver
   /tmp/lfm_w<pid>/           │                                              │  ├─ scipy.optimize.root
    └─ chdir 隔离             └─ data_*_b<batch>.pkl                          │  ├─ 多初始值 + 热启动
                                                                              │  └─ 软失败惩罚
                                                                              │
                                                                              ▼
                                                                            ⑥ 优化
                                                                            ─────
                                                                            ppo_optimize/
                                                                            ├─ env.py (Gym)
                                                                            ├─ train_ppo.py (SAC env)
                                                                            └─ cma_optimize.py (基线)
                                                                              │
                                                                              ▼
                                                                          8 个最优 CP
                                                                          → B 样条插值
                                                                          → QBlade 回验
```

### 数据格式演化（关键中间件）

| 阶段 | 类型 | 形状/字段 | 存储 |
|---|---|---|---|
| 几何输入 | `np.ndarray` | (1000, 22, 3) = (geom_idx, section, [r,c,t]) | `geometry_data.npy` |
| QBlade 单工况输出 | `pd.DataFrame` | (120, 28) = 末 120 步 × 28 列 | `pkl` 内 dict 值 |
| 单几何 pkl 单元 | `dict` | `{geometry: ndarray, RPM*_Wind*_Angle*: DataFrame, ...}` | pickle |
| 整批 pkl | `dict` | `{geometry_<idx>: <单几何 dict>, ...}` | `data_*_b<i>.pkl` |
| 整合后 raw | `pd.DataFrame` | (~42000, ~50) | `raw_data.csv` |
| 训练 tensor | `torch.Tensor` | (N_train, input_dim) / (N_train, output_dim) | `X/y_train_scaled` |

---

## 2. Propeller_project-main 子模块（仿真层）

### 2.1 文件结构

```
Propeller_project-main/         git submodule
├── QBladeCE_2.0.8.6/
│   ├── libQBladeCE_2.0.8.6.so.1.0.0   ← 18 MB Linux 动态库
│   ├── Libraries/                       ← 依赖 .so (fortran/openCL/Qt/...)
│   └── SIL_Interface/
│       ├── QBladeLibrary.py             ← Python ctypes wrapper（自带）
│       ├── QBladeLibInclude.h           ← C 函数声明
│       └── sampleScript.py              ← 官方调用示例（success 检查参考）
└── code/Simulation_QBlade/
    ├── config.py                        ← os.getcwd() 解析路径
    ├── simulation_parameters/
    │   ├── APC107E_geometry.xlsx        ← baseline 几何（22 截面）
    │   └── Parameters.xlsx              ← 42 工况表 (RPM × WIND × ANGLE)
    └── class_sim/simulation.py          ← SIMULATION 类（含 #3/#4/#7 优化）
```

### 2.2 `class_sim/simulation.py` — SIMULATION 类

#### 关键属性
```python
class SIMULATION:
    self.file_path: dict[str, str]               # 路径字典（dll/sim/qpr/bld）
    self.simulation_parameters: list[list[tuple]]  # 从 Parameters.xlsx 解析
    self.one_simulation_data: pd.DataFrame       # 当前工况结果
    self.all_simulation_data: dict               # {label: DataFrame, "geometry": ndarray}
    self.number_of_timesteps: int = 1200         # 仿真步数
    self.geometry_baseline: np.ndarray           # (22, 3) baseline 几何
    self.current_propeller: np.ndarray           # 当前几何
    self.airDensity: float = 1.225               # kg/m³
    self.R: float = 0.127                        # m 螺旋桨半径
    self.device: int                             # 0=CPU, 1=GPU OpenCL
    self.QBLADE = None                           # ⭐ 优化 #4: lazy 创建跨工况复用
```

#### 公共方法

##### `__init__(file_path, geometry_baseline, device_type, number_of_timesteps)`
1. 解析路径字典
2. 调 `generating_simulation_parameters_tuple()` 从 `Parameters.xlsx` 读 42 工况
3. 调 `generating_sim_file()` 写 42 个 `.sim` 文件到 `QBlade_data/QBlade_sim/`
4. 设置 `self.device`（GPU=1, CPU=0）
5. 设置 `self.QBLADE = None`（lazy）

##### `run_one_simulation(RPM=None, WIND_SPEED=None, ANGLE=None, SIM_file_path=None)`
两种调用方式：
- `RPM/WIND_SPEED/ANGLE`：在线修改 `Base_simulation.sim` 中的参数
- `SIM_file_path`：直接加载已有 `.sim` 文件

核心循环（优化 #3 + #4 后）：
```python
# 优化 #4: lazy createInstance，不每次新建
if self.QBLADE is None:
    self.QBLADE = QBladeLibrary(self.file_path["dll_file"])
    self.QBLADE.createInstance(self.device, 32)
QBLADE = self.QBLADE

QBLADE.loadSimDefinition(path.encode())
QBLADE.initializeSimulation()

# 数据采集字段（9 个时间序列）
data_keys = ["Time", "Thrust", "Power", "Torque",
             "Thrust_y", "Thrust_z",        # Hub global Y/Z 方向力
             "Mx", "My", "Mz"]              # Hub global 三轴力矩
data_values = {k: [] for k in data_keys}

# 优化 #2: NaN abort 检测
sim_aborted_at = -1
for i in tqdm(range(1200)):
    success = QBLADE.advanceTurbineSimulation()
    if not success:
        sim_aborted_at = i
        break
    if i >= 1080:  # 末 120 步
        gd = QBLADE.getCustomData_at_num
        data_values["Time"].append(float(gd(b"Time [s]", 0, 0)))
        data_values["Thrust"].append(float(gd(b"Aerodynamic Thrust [N]", 0, 0)))
        # ... 8 个字段 ...

# NaN 兜底
if sim_aborted_at >= 0 or len(data_values["Time"]) == 0:
    return pd.DataFrame()  # 不污染 all_simulation_data

# Post-process: 末 120 步均值 × -1
df["THRUST"] = df["Thrust"].mean() * -1
# ...
df["FX"] = df["THRUST"]    # X 方向力 = 主推力（QBlade 不暴露 Hub X_g Force）
df["FY"] = df["THRUST_Y"]  # 别名
df["FZ"] = df["THRUST_Z"]  # 别名
df["MX"] = df["Mx"].mean() * -1
df["MY"] = df["My"].mean() * -1
df["MZ"] = df["Mz"].mean() * -1

# 兜底校验
if not np.isfinite(df["THRUST"].iloc[0]) or abs(df["THRUST"].iloc[0]) < 1e-6:
    return pd.DataFrame()

# 优化 #3: 不再调 storeProject (省 60s IO)
# 优化 #4: 不再 closeInstance (跨工况复用)
return self.one_simulation_data
```

##### `run_all_simulation()`
1. `os.walk(SIM_folder)` 扫描所有非 Base 的 `.sim`
2. 循环调 `run_one_simulation(SIM_file_path=)`
3. 收尾 `self.close()`（关 QBlade 实例）

##### `change_propeller_geometry(section_data=None, control_point=None)`
- `section_data`：直接传 (22, 3) 截面数据
- `control_point`：8 维 CP → B 样条插值 → 22 截面
- 修改 `.bld` 文件中 22 行截面数据

##### `close()` ⭐ 优化 #4
显式关闭 QBlade 实例（`run_all_simulation` 末尾自动调）。

##### `__del__()` 兜底释放（worker 异常退出时）。

### 2.3 `simulation_parameters/Parameters.xlsx` 工况表

42 行（本次扩展）：
```
RPM   | windSpeed | windAngle
4000  | 10        | 82
4000  | 10        | 83
...
6500  | 10        | 88
```

修改：见 [PROJECT_REPORT §4](./PROJECT_REPORT.md#4-数据集) 工况网格设计。

### 2.4 `simulation_parameters/APC107E_geometry.xlsx` baseline 几何

22 截面 × 3 列 = `[r/R, c/R, twist_angle]`，乘以 R=0.127 后得到 (r[m], chord[m], twist[°])。
config.py 自动读取并归一化。

### 2.5 `Baseline_Blade_Turb.trb` Wake 截断参数 ⭐ 优化 #7

```
WAKETYPE          = 0       # 0 = FVW
MAXWAKESIZE       = 30000   # 涡环硬上限 (原 200000)
MAXWAKEDIST       = 10      # 截断距离 (原 100, 单位 = 直径)
WAKELENGTHTYPE    = 0       # 0 = 转数
NEARWAKELENGTH    = 0.50
ZONE1LENGTH       = 0.50    # 原 2.00 (← #A 激进版)
ZONE2LENGTH       = 1.00    # 原 4.00
ZONE3LENGTH       = 1.00    # 原 6.00
ZONE1/2FACTOR     = 2
ZONE3FACTOR       = 8       # 远场粗化 (原 2)
WAKECORERADIUS    = 0.05
MAXSTRAIN         = 50
```

详细物理含义见 [qblade_data_generation.html §6](./qblade_data_generation.html#wake-detail)。

---

## 3. `scripts/` 数据生成层

### 3.1 `run_data_gen.py` — 单进程入口

124 行的简单版本。核心：
```python
def setup_submodule_import() -> tuple:
    if not SUB_PY_ROOT.is_dir():
        sys.exit(f"✖ 子模块未初始化")
    os.chdir(SUB_PY_ROOT)               # ⭐ 必须先 chdir 再 import
    sys.path.insert(0, str(SUB_PY_ROOT))
    import config
    from class_sim.simulation import SIMULATION
    return config, SIMULATION

def run_one_geometry(config_module, SIMULATION_cls, geom, args) -> dict:
    sim = SIMULATION_cls(
        file_path=config_module.file_path,
        geometry_baseline=config_module.geometry_baseline,
        device_type=args.device,
        number_of_timesteps=args.num_timesteps or config_module.number_of_timesteps,
    )
    if not np.array_equal(geom, config_module.geometry_baseline):
        sim.change_propeller_geometry(section_data=geom)
    sim.run_all_simulation()
    return sim.all_simulation_data
```

**CLI 参数**：
| 参数 | 默认 | 含义 |
|---|---|---|
| `--geometry-npy` | None | (N,22,3) 文件路径，省略只跑 baseline |
| `--device` | CPU | `CPU` 或 `GPU` |
| `--num-timesteps` | 1000 | 仿真步数 |
| `--out-dir` | `data_for_train/data` | pkl 输出目录 |
| `--tag` | linux | 输出文件名 tag |
| `--first-n-geoms` | None | 只跑前 N 个几何 |

### 3.2 `run_data_gen_parallel.py` ⭐ — 多进程入口

309 行，本次会话的核心成果。

#### 全局常量
```python
LFM_ROOT       = Path(__file__).resolve().parent.parent
SUBMODULE_ROOT = LFM_ROOT / "Propeller_project-main"
SUB_CODE_ROOT  = SUBMODULE_ROOT / "code" / "Simulation_QBlade"
SUB_DATA_ROOT  = SUBMODULE_ROOT / "QBlade_data"
SUB_LIB_ROOT   = SUBMODULE_ROOT / "QBladeCE_2.0.8.6"
```

#### CLI 参数
| 参数 | 默认 | 含义 |
|---|---|---|
| `--workers` | 16 | 并行 worker 数 |
| `--omp-threads` | 8 | 每 worker OMP/MKL/BLAS/Qt 线程数 |
| `--device` | CPU | 强制 CPU（GPU 路径在 propeller 工况死锁） |
| `--geometry-npy` | None | (N,22,3) 几何文件 |
| `--first-n-geoms` | None | 只跑前 N（覆盖 npy 总数） |
| `--num-timesteps` | None | 仿真步数（默认从 config 读 1000） |
| `--out-dir` | `data_for_train/data` | pkl 输出目录 |
| `--tag` | parallel | 文件名 tag |
| `--worktree-root` | `/tmp` | worker 工作目录父级 |
| `--keep-worktree` | False | 调试用，跑完不删 `/tmp/lfm_w<pid>` |
| **`--start-idx`** | 0 | 几何起始（含），断点续传 |
| **`--end-idx`** | None | 几何结束（不含），断点续传 |
| **`--batch-size`** | 0 | 每 N 几何落一个 pkl（0 = 一次性） |

#### 关键函数

##### `setup_worker_workdir(worktree_root: str) -> Path`
为每个 worker 创建独立工作目录：
```
/tmp/lfm_w<pid>/
  ├── code/                    ← shutil.copytree (~164KB)  ⭐ 必须深拷贝
  ├── QBladeCE_2.0.8.6/        → symlink (read-only)
  └── QBlade_data/             ← shutil.copytree (~80MB)
```

**关键设计**：`code/` 不能 symlink，否则 `chdir(symlink)` 后 `os.getcwd()` 跟随到 source 路径，config.py 的 `os.getcwd()` 解析全部指向 submodule 原文件 → 多 worker 写 .sim/.bld 互相覆盖。

##### `worker_init(worktree_root: str, omp_threads: int) -> None`
Pool worker 启动钩子（每个 fork 子进程仅一次）：
1. 设置 6 个环境变量限制线程数：
   - `OMP_NUM_THREADS = OMP_THREAD_LIMIT = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = NUMEXPR_NUM_THREADS = QT_THREAD_POOL_MAX_THREAD_COUNT = "<omp_threads>"`
2. 调 `setup_worker_workdir` 建独立目录
3. `os.chdir(work_cwd)` + `sys.path.insert(0, work_cwd)`
4. `importlib.reload(sys.modules["config"])` 让 config 重新读 `os.getcwd()`
5. import SIMULATION，存到 worker 全局变量 `_config`, `_SIMULATION`

##### `worker_run(task: tuple) -> tuple`
处理一个几何 task = `(geom_idx, geom_array, num_timesteps, device)`：
```python
sim = _SIMULATION(
    file_path=_config.file_path,
    geometry_baseline=_config.geometry_baseline,
    device_type=device,
    number_of_timesteps=num_timesteps or _config.number_of_timesteps,
)
if not np.array_equal(geom_array, _config.geometry_baseline):
    sim.change_propeller_geometry(section_data=geom_array)
sim.run_all_simulation()
return (geom_idx, sim.all_simulation_data, None)
# 异常时 return (geom_idx, None, error_str)
```

##### `main()` 主循环
```python
ctx = mp.get_context("fork")
with ctx.Pool(processes=n_workers,
              initializer=worker_init,
              initargs=(args.worktree_root, args.omp_threads)) as pool:
    aggregated = {}  # batch buffer
    for (geom_idx, result, err) in pool.imap_unordered(worker_run, tasks):
        if result is not None:
            aggregated[f"geometry_{geom_idx}"] = result
        # 分批落盘
        if args.batch_size > 0 and len(aggregated) >= args.batch_size:
            flush_batch()  # 写 pkl + 清空 buffer

flush_batch()  # 收尾剩余
cleanup_worktrees(args.worktree_root, args.keep_worktree)
```

#### 输出 pkl 格式
```python
# 一个 batch 文件
{
    "geometry_0":  {"geometry": np.ndarray, "RPM4000_Wind10_Angle82": pd.DataFrame, ...},
    "geometry_1":  {...},
    ...
    "geometry_15": {...},
}
# 文件名: data_<tag>_<stamp>_b0000.pkl, data_<tag>_<stamp>_b0001.pkl, ...
```

---

## 4. `config.py` 全局配置

```python
@dataclass
class Config:
    # 路径
    raw_data_dir: str = "./data_for_train/data"
    processed_data_dir: str = "./data_for_train/geometry_run_results_100"
    model_dir: str = "./trained_models"
    log_dir: str = "./runs"

    # 工况/输出列
    condition_columns: List[str] = ["RPM", "WIND", "ANGLE"]
    output_columns: List[str] = ["Fx", "Fy", "Fz", "Torque"]
    test_size: float = 0.2
    random_state: int = 42

    # 几何模式
    geometry_mode: str = "sections"   # "none" | "control_points" | "sections"
    n_control_points: int = 8         # 4 chord + 4 twist
    n_sections: int = 22

    # 模型 (sweep 最优 trial #21)
    hidden_dims: List[int] = [64, 128, 64]
    dropout: float = 0.0

    # 训练
    batch_size: int = 64
    epochs: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 200
    min_lr: float = 1e-6
    early_stop_patience: int = 500
    grad_clip: float = 1.0

    # 自动计算
    geometry_columns: List[str]  # 根据 geometry_mode 推导
    input_columns: List[str]     # condition + geometry
    input_dim: int               # 3 / 11 / 47
    output_dim: int              # 4 (默认 Fx/Fy/Fz/Torque)
```

#### `geometry_mode` 三种模式

| 模式 | input_dim | geometry_columns |
|---|---|---|
| `"none"` | 3 | `[]` |
| `"control_points"` | 3 + 8 = 11 | `["cp_0", ..., "cp_7"]` |
| `"sections"` | 3 + 44 = 47 | `["chord_0..21", "twist_0..21"]` |

---

## 5. `data.py` 数据加载与标准化

### 5.1 主要函数

##### `_extract_geometry(data: dict, cfg: Config) -> dict`
从 pkl 字典提取几何信息并转为列字典：
```python
# 输入: {"geometry": np.ndarray (22, 3), ...}
# 输出: {"chord_0": v0, "chord_1": v1, ..., "twist_21": v21}  # 44 列
```
根据 `cfg.geometry_mode` 决定输出列数。

##### `_extract_from_case_results(case_dict, pattern, records, skipped_log, source, geometry) -> int`
扫描工况字典，对每个工况 DataFrame 提取一行汇总记录：
```python
# 输入 case_dict: {"RPM5000_Wind10_Angle85": DataFrame(120, 28), ...}
# 输出: 每个工况 1 条 record dict，append 到 records 列表
record = {
    "RPM": 5000, "WIND": 10, "ANGLE": 85,
    "Fx": -df["THRUST"].iloc[0], "Fy": df["THRUST_Y"].iloc[0], ...,
    "chord_0": ..., "twist_21": ...,
    "source": "data_xxx_b0000.pkl/geometry_0",
}
```

##### `load_raw_pkl(data_dir: str, cfg: Config) -> pd.DataFrame`
**入口函数**。
1. `os.walk(data_dir)` 找所有 `data*.pkl`/`sample_*.pkl`
2. 对每个 pkl 用 `case_pattern = re.compile(r"RPM([\d.]+)_Wind([\d.]+)_Angle([\d.]+)")` 匹配工况名
3. 自动识别 3 种格式：
   - **A**: 含 `"case_results"` 键 → 旧版 sample_XXXX.pkl
   - **B**: 顶层有 `"geometry_<idx>"` 键 → 本次重构格式 ⭐
   - **C**: 顶层直接是工况键 → 无几何
4. 返回 `pd.DataFrame(records)`（每行 = 一个工况）
5. 写 `skipped_cases_log.csv` 记录跳过的工况

##### `process_and_save(cfg: Config, raw_df: pd.DataFrame = None) -> None`
1. 调 `load_raw_pkl` 拿到 raw_df
2. 写 `raw_data.csv` 供检查
3. `train_test_split`（80/20）
4. `StandardScaler` 标准化 X 和 y
5. 落盘到 `cfg.processed_data_dir`：
   - `X_train_scaled.pkl`, `X_test_scaled.pkl`
   - `y_train_scaled.pkl`, `y_test_scaled.pkl`
   - `scaler_X.pkl`, `scaler_y.pkl`
6. 调 `plot_data_distribution()` 生成分布图

##### `plot_data_distribution(cfg, X_train, X_test, y_train, y_test) -> None`
画 4 子图：
- 工况分布（RPM/WIND/ANGLE 直方图）
- 几何分布（chord/twist 范围）
- 输出分布（Fx/Fy/Fz/Torque）
- train vs test 散点

---

## 6. `uav_model.py` 无人机物理模型

### 6.1 `QuadcopterConfig` 物理参数（dataclass）

```python
@dataclass
class QuadcopterConfig:
    # 质量
    mass: float = 3.5            # kg
    g: float = 9.81
    I_xx: float = 0.040          # kg·m² 横滚
    I_yy: float = 0.045          # kg·m² 俯仰
    I_zz: float = 0.070          # kg·m² 偏航

    # 几何（X 构型 450 mm 轴距）
    arm_length: float = 0.225
    l1: float = 0.16             # CG 到前旋翼对纵向距离
    l2: float = 0.16             # CG 到后旋翼对纵向距离
    d1: float = 0.06             # 前桨毂高于 CG
    d2: float = 0.06             # 后桨毂高于 CG

    # 螺旋桨
    n_rotors: int = 4
    prop_diameter: float = 0.254  # m (10 inch)
    prop_radius: float = 0.127

    # 机身气动
    l_body: float = 0.35
    S_front: float = 0.035       # m² 正面迎风
    S_top: float = 0.080
    S_side: float = 0.025

    # 阻力系数（公开文献参考）
    Cd_front_at_zero_alpha: float = 1.5
    Cd_max_at_90_deg: float = 1.8
    Cl_max: float = 0.4
    ...
```

### 6.2 `fuselage_aero_model(cfg)`
返回闭包函数 `aero_func(V, alpha_rad) -> (D, L, M_y)`：

**阻力**:
```
D = 0.5 · ρ · V² · S_eff(α) · Cd(α)
其中:
  S_eff(α) = S_front · cos²(α) + S_top · sin²(α)
  Cd(α)    = Cd_front + (Cd_max - Cd_front) · sin²(α)
```

**升力**:
```
L = 0.5 · ρ · V² · S_top · Cl(α)
其中:
  Cl(α)    = Cl_max · sin(2α)  # 平板模型
```

**俯仰力矩**:
```
M_y = 0.5 · ρ · V² · S_top · l_body · Cm(α)
```

参考文献见 `docs/uav_model_references.md`。

---

## 7. `train.py` 与 uav_model 训练栈

### 7.1 `PropellerPredictor(nn.Module)` MLP 模型

```python
class PropellerPredictor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        dims = [cfg.input_dim] + cfg.hidden_dims  # [47, 64, 128, 64]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            if i >= 2:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(dims[-1], cfg.output_dim))  # 输出 4 维
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

**参数量**：47 → 64 → 128 → 64 → 4 ≈ **20k 参数**（sweep #21 选定）。

### 7.2 关键函数

##### `load_tensors(cfg) -> X_train, X_test, y_train, y_test`
从 `cfg.processed_data_dir` 读 4 个 pickle 文件，转 `torch.float32`。

##### `compute_metrics(y_true, y_pred) -> dict`
整体指标：`MSE, MAE, R², MAPE`。

##### `compute_per_output_metrics(y_true, y_pred, names) -> dict`
逐输出指标：`{Fx: {R², MAPE}, Fy: {...}, ...}`。

##### `train(cfg) -> model`
**主训练循环**：
```python
model = PropellerPredictor(cfg)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
writer = SummaryWriter(cfg.log_dir)

best_test_loss = float("inf")
no_improve = 0
for epoch in range(cfg.epochs):
    # train
    model.train()
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)
        loss = F.mse_loss(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    # eval
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = F.mse_loss(y_pred, y_test).item()
        metrics = compute_metrics(y_test.numpy(), y_pred.numpy())

    scheduler.step(test_loss)

    # early stop
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), os.path.join(cfg.model_dir, "best.pth"))
        no_improve = 0
    else:
        no_improve += 1
        if no_improve > cfg.early_stop_patience:
            break

    # TensorBoard
    log_scalars(writer, epoch, train_loss, test_loss, metrics, optimizer.param_groups[0]["lr"])
    if epoch % cfg.plot_every == 0:
        log_figures(writer, epoch, y_test, y_pred, scaler_y, cfg.output_columns)

return model
```

##### `save_training_report(...)` ⭐
生成完整训练报告：
- `report.tex` LaTeX 表格
- `pred_vs_true.png` 预测散点
- `error_distribution.png` 残差分布
- `training_curves.png` train/test loss + R² 曲线

---

## 8. `adapter.py` 代理模型适配器

封装 DNN 推理 + 配平求解的数据流。

### 主要函数

##### `make_dnn_aero_func(model, scaler_X, scaler_y, geom_features, cfg) -> Callable`
返回一个 `(V, alpha_rad, omega_rpm) -> (T, H, M_y)` 函数，给 `QuadcopterTrimSolver` 使用：
```python
def dnn_aero(V, alpha_rad, omega_rpm):
    # 1. 把 (V, alpha_rad, omega_rpm) → DNN 输入 (RPM, WIND, ANGLE, geom...)
    RPM   = omega_rpm
    WIND  = V
    ANGLE = math.degrees(alpha_rad) + 90  # alpha 0 → ANGLE 90 (水平)
    x = np.concatenate([[RPM, WIND, ANGLE], geom_features])  # (47,)

    # 2. 标准化
    x_scaled = scaler_X.transform(x.reshape(1, -1))

    # 3. DNN 推理
    with torch.no_grad():
        y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
    y = scaler_y.inverse_transform(y_scaled).flatten()  # (4,)

    # 4. 反推回旋翼坐标
    Fx, Fy, Fz, Torque = y
    T = -Fx  # 主推力（轴向）
    H = math.hypot(Fy, Fz)  # 侧向力合成
    M_y = Torque  # 俯仰力矩

    return T, H, M_y
```

##### `load_trained_model(cfg, ckpt_path)`
加载 `.pth` + 还原 scaler，返回 `(model, scaler_X, scaler_y)`。

---

## 9. `quadcopter_trim_solver.py` 配平求解

### 9.1 `TrimResult(dataclass)` 求解结果

```python
@dataclass
class TrimResult:
    success: bool
    V: float                    # m/s 飞行速度
    alpha_deg: float            # ° 俯仰角
    omega1_rpm: float           # 前旋翼 RPM
    omega2_rpm: float           # 后旋翼 RPM
    P_total: float              # W 总功率
    F_x: float                  # N x 残差
    F_z: float                  # N z 残差
    M_y: float                  # Nm y 残差
    iterations: int
    failure_reason: str = ""
```

### 9.2 `QuadcopterTrimSolver` 类

#### 物理模型（3 方程 3 未知数）

**未知数** `x = [α, Ω₁, Ω₂]`：
- `α` 俯仰角（rad）
- `Ω₁` 前旋翼 RPM
- `Ω₂` 后旋翼 RPM

**方程**（NED 坐标系）：
```
F_x = 2·(T_p1 + T_p2)·sin(α) - 2·(H_p1 + H_p2)·cos(α) - D_f  = 0
F_z = 2·(T_p1 + T_p2)·cos(α) + 2·(H_p1 + H_p2)·sin(α) - mg - L_f = 0
M_y = 2·(M_p1_y + M_p2_y) - M_f_y
       + 2·(T_p2·l₂ - T_p1·l₁)
       + 2·(H_p1·d₁ + H_p2·d₂) = 0
```

其中：
- `T_pi, H_pi, M_pi_y` = 旋翼 i 推力/侧向力/力矩（来自 DNN 代理）
- `D_f, L_f, M_f_y` = 机身阻力/升力/力矩（来自 `uav_model.fuselage_aero_model`）
- `l₁, l₂, d₁, d₂` = 几何力臂

#### 主要方法

##### `_trim_equations(x, V) -> np.ndarray (3,)`
计算 3 个方程残差（用于 `scipy.optimize.root`）。

##### `_generate_initial_guess(V) -> np.ndarray`
基于飞行速度生成初始猜测：
```python
α_init = min(5 + 0.5·V, 20)  # 简单经验
T_hover = mg / 4
n_hover = sqrt(T_hover / (C_T · ρ · D⁴))   # 悬停 RPM
return [α_init, n_hover_RPM, n_hover_RPM]
```

##### `solve(V, x0=None, method="hybr", tol=1e-6) -> TrimResult`
**单次求解**。

##### `solve_multi_start(V, n_initial_guesses=6) -> TrimResult` ⭐ 鲁棒版
生成 6 组扰动初始值，全部尝试，选 F_x²+F_z²+M_y² 最小的为最优解：
```python
guesses = [
    self._generate_initial_guess(V),
    # + 5 组扰动: α ± 30%, Ω ± 20% 等
]
best = None
for x0 in guesses:
    result = self.solve(V, x0=x0)
    if result.success:
        if best is None or result.residual < best.residual:
            best = result
return best or TrimResult(success=False, ...)
```

##### `solve_with_warm_start(V, prev_result) -> TrimResult` ⭐ 热启动
用上一次解作为初始值（PPO/CMA-ES 多次配平时大幅提速）。

---

## 10. `ppo_optimize/` PPO + CMA-ES 优化栈

### 10.1 `env.py` Gymnasium 环境

#### 类常量
```python
class PropellerDesignEnv(gym.Env):
    # 观测归一化范围
    ALPHA_RANGE = (0.0, 25.0)        # deg
    OMEGA_RANGE = (1000.0, 12000.0)  # RPM
    POWER_RANGE = (0.0, 500.0)       # W

    # 配平失败容忍
    MAX_CONSECUTIVE_FAIL = 3

    # CP 边界（训练域内）
    _TRAIN_BOUNDS = None  # 懒加载
```

#### State (13-dim)
```
state = [
    cp_0, cp_1, ..., cp_7,    # 8 维 control points
    alpha_deg,                # 配平结果俯仰角
    Omega1_RPM,               # 前旋翼 RPM
    Omega2_RPM,               # 后旋翼 RPM
    power,                    # 总功率
    step_ratio,               # current_step / max_steps
]
```

#### Action (8-dim)
`delta_cp ∈ [-1, 1]`，乘以 `delta_scale * cp_range` 映射到几何边界。

#### Reward
```python
reward = FM = m·g·V / P_total
if not trim_result.success:
    reward = -1.0  # 软失败惩罚（v2: -10 → -1，避免过度惩罚）
```

#### 关键函数

##### `cp_to_sections(cp: np.ndarray) -> np.ndarray`
B 样条插值：8 维 CP → (22, 2) 截面 (chord, twist)。
```python
from scipy.interpolate import BSpline
degree = 3
knots = np.concatenate([[0]*degree, [0.4, 0.9], [1]*degree])
chord_bspline = BSpline(knots, cp[:4], degree)
twist_bspline = BSpline(twist_knots, cp[4:], degree)
x_norm = baseline_geometry[:, 0] / 0.127  # r/R
return np.column_stack([chord_bspline(x_norm), twist_bspline(x_norm)])
```

##### `_make_geometry_aware_dummy_funcs(cp) -> tuple(fuselage_func, rotor_func)`
返回 `(fuselage_aero_func, rotor_aero_func)`。如果 `use_dnn=False`，用解析公式做兜底测试。

##### `_load_dnn_model() -> None`
从 `cfg.model_dir/best.pth` 加载 DNN + scaler，缓存到 `self.model, self.scaler_X, self.scaler_y`。

##### `_ensure_train_bounds() -> None`
扫描训练 raw_data.csv，统计 `cp_i`, `RPM`, `ANGLE` 的真实范围 + 5% 边距，存到类变量 `_TRAIN_BOUNDS`。

##### `reset(seed=None) -> (obs, info)`
- 随机初始化 CP（在边界内）
- 单次配平（获取初始 state）
- 重置 step_count, no_improve_counter

##### `step(action) -> (obs, reward, terminated, truncated, info)`
1. 解码 action → `delta_cp`
2. `new_cp = clip(self.cp + delta_cp · delta_scale · cp_range, cp_lo, cp_hi)`
3. `sections = cp_to_sections(new_cp)`
4. 调 DNN + Trim Solver `solver.solve_multi_start(V)`
5. 计算 reward
6. 检查终止条件（步数 / 连续失败 / no_improve）

### 10.2 `train_ppo.py` PPO 训练入口

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank, seed):
    def _init():
        env = PropellerDesignEnv(V=10.0, max_steps=50, constrain_to_data=True)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    n_envs = 8
    env = SubprocVecEnv([make_env(i, seed=42) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_logs/",
    )
    model.learn(total_timesteps=1_000_000, callback=[best_model_callback])
    model.save("ppo_propeller_final")
```

### 10.3 `cma_optimize.py` CMA-ES 基线

```python
import cma
es = cma.CMAEvolutionStrategy(
    initial_cp,         # (8,) 中心
    sigma=0.1,          # 初始标准差
    {"bounds": [cp_lo, cp_hi], "popsize": 16, "maxiter": 100},
)
while not es.stop():
    solutions = es.ask()  # (16, 8)
    fitness = [-evaluate_fitness(cp) for cp in solutions]  # 负号转最大化为最小化
    es.tell(solutions, fitness)
    es.logger.add()
```

### 10.4 `eval_ppo.py` 评估 + 可视化
- 加载训练好的 PPO，跑 N 个 episode 收集决策
- 画 reward 曲线、CP 演化、最优桨叶 chord/twist 分布
- 与 CMA-ES 对照

---

## 11. 关键算法推导

### 11.1 Free Vortex Wake (QBlade)

**离散化**：每个叶片截面在每个时间步释放一个涡环 Γᵢ，强度与升力变化成正比：
```
Γᵢ(t) = (L(t) - L(t-Δt)) / (ρ · V_eff)
```

**Biot-Savart 诱导速度**：每个涡环对场点 p 的诱导速度：
```
v_ind(p) = Γᵢ / (4π) · ∫ dl × r / |r|³
```

**总诱导速度**：叶片截面 j 受所有 N 个涡环影响：
```
v_total(j) = Σᵢ v_ind(j, vortex_i)   # O(N) per blade element
```

**涡-涡相互作用**：所有涡环也彼此漂移：
```
v_drift(i) = Σ_{k≠i} v_ind(vortex_i, vortex_k)   # O(N²) total
```

1200 步累积涡环数：`N = 2 叶片 × 22 截面 × 1200 步 = 52800`
涡-涡作用 ≈ **2.8 × 10⁹** 浮点运算/步。这就是性能瓶颈。

### 11.2 配平方程推导（QuadcopterTrimSolver）

**坐标系**：机体系，x 前、y 右、z 下（NED）。

**单旋翼力分解**（俯仰角 α）：
```
T_axial : 沿旋转轴（机体 z 方向，受 α 影响）
H_lateral : 旋翼平面内水平力
```

**机体合力**（4 旋翼，假设左右对称）：
```
ΣF_x = 2(T₁ + T₂)sin(α) - 2(H₁ + H₂)cos(α) - D_f = 0  (前后向)
ΣF_z = 2(T₁ + T₂)cos(α) + 2(H₁ + H₂)sin(α) - mg - L_f = 0  (上下向)
```

**俯仰力矩平衡**（绕 CG，机体 y 轴）：
```
ΣM_y = M_rotors + M_torque_arm + M_lateral_arm - M_f_y = 0

其中：
  M_rotors = 2 M_p1_y + 2 M_p2_y                 # 旋翼自身扭矩
  M_torque_arm = 2 (T₂·l₂ - T₁·l₁)               # 推力 × 力臂
  M_lateral_arm = 2 (H₁·d₁ + H₂·d₂)              # 侧向力 × 垂直距离
```

求解器寻找 `(α, Ω₁, Ω₂)` 使 3 方程为 0。

### 11.3 B 样条几何参数化（cp_to_sections）

**控制点 → 截面**：
```
chord_cp = [cp_0, cp_1, cp_2, cp_3]   # 4 个控制点
twist_cp = [cp_4, cp_5, cp_6, cp_7]

knots_chord = [0, 0, 0, 0.4, 0.9, 1, 1, 1]    # cubic 节点
knots_twist = [0, 0, 0, 0.2, 0.9, 1, 1, 1]

chord(r/R) = BSpline(knots_chord, chord_cp, degree=3)(r/R)
twist(r/R) = BSpline(knots_twist, twist_cp, degree=3)(r/R)

# 22 截面采样
x_norm = baseline_geometry[:, 0] / R   # r/R 数组
sections = stack([chord(x_norm), twist(x_norm)])  # (22, 2)
```

参数化好处：8 维搜索空间（vs 直接 44 维截面），平滑保证，优化收敛快。

---

## 12. 完整 API 使用示例

### 12.1 从零开始训练新模型

```bash
# 1. 生成数据（占 ~16 天，100 几何）
ssh 3090_node1 "cd /root/LFM && export LD_LIBRARY_PATH=... && \
  python scripts/run_data_gen_parallel.py \
    --workers 10 --omp-threads 8 --num-timesteps 1200 \
    --geometry-npy /root/geometry_data.npy \
    --batch-size 16 --tag prod_v1"

# 2. 整合 pkl → CSV
cd /Users/guoyue/gy_2026/graduation/LFM
scp 3090_node1:/root/LFM/data_for_train/data/*.pkl ./data_for_train/data/
python data.py    # 写 raw_data.csv + X/y_*_scaled.pkl

# 3. 训练 DNN
python train.py   # 用 config.py 默认参数，跑 5000 epoch 或 early stop

# 4. 超参 sweep（可选）
python sweep.py --max-trials 50 --quick

# 5. 配平验证
python run_trim.py   # 用 baseline 几何 + DNN 求解，画图

# 6. PPO 优化
python ppo_optimize/train_ppo.py --total-timesteps 1000000
python ppo_optimize/eval_ppo.py
```

### 12.2 Python API（单元测试 / Jupyter）

```python
# 加载数据
from config import Config
import data
cfg = Config()
data.process_and_save(cfg)   # 重新整合 + 标准化

# 训练
from train import train, PropellerPredictor
model = train(cfg)

# 评估
from train import compute_metrics, compute_per_output_metrics
y_pred = model(X_test).detach().numpy()
metrics = compute_metrics(y_test.numpy(), y_pred)
per_output = compute_per_output_metrics(y_test.numpy(), y_pred, cfg.output_columns)

# 适配器
from adapter import load_trained_model, make_dnn_aero_func
model, scaler_X, scaler_y = load_trained_model(cfg, "trained_models/best.pth")

# 给定几何
import numpy as np
cp = np.array([0.020, 0.030, 0.015, 0.007, 45, 20, 17, 12])  # 8 维
from ppo_optimize.env import cp_to_sections
sections = cp_to_sections(cp)  # (22, 2)
geom_features = sections.flatten()  # 44 维

# 配平
from quadcopter_trim_solver import QuadcopterTrimSolver
rotor_func = make_dnn_aero_func(model, scaler_X, scaler_y, geom_features, cfg)
solver = QuadcopterTrimSolver(mass=3.5)
solver.set_rotor_aero(rotor_func)
result = solver.solve_multi_start(V=10.0)
print(f"α = {result.alpha_deg:.2f}° Ω = {result.omega1_rpm:.0f} RPM, P = {result.P_total:.1f} W")
print(f"FM = {3.5 * 9.81 * 10 / result.P_total:.4f}")
```

### 12.3 PPO 单独调用

```python
from stable_baselines3 import PPO
from ppo_optimize.env import PropellerDesignEnv

env = PropellerDesignEnv(V=10.0, max_steps=50,
                         constrain_to_data=True,
                         delta_scale=0.10,
                         reward_beta=0.1,
                         ood_penalty=0.5)

model = PPO("MlpPolicy", env, learning_rate=3e-4)
model.learn(total_timesteps=100_000)

# 推理
obs, _ = env.reset()
for i in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(f"best CP: {info['best_cp']}")
print(f"best FM: {info['best_reward']}")
```

---

## 13. 关键设计决策

### 13.1 为什么 `code/` 深拷贝而非 symlink？
- Python `os.chdir(symlink)` 会让 `os.getcwd()` 返回**符号链接的目标路径**（真实路径）
- 子模块 config.py 用 `os.getcwd()` 解析 `bld_file_path = os.path.join(grandparent, "QBlade_data/...")`
- 若 worker cwd 是 symlink → grandparent 是 submodule 真实位置 → 多 worker 写同一文件冲突
- 修复：深拷贝 code/（仅 164KB，零成本）→ 真实独立路径

### 13.2 为什么 OMP_NUM_THREADS=8？
- QBlade SIL 内部 OpenMP 默认 32 thread
- 实测 4 worker × 32 thread + Qt helper ≈ 900 NLWP → load 200+ → wall clock 退化 8×
- 限到 8 thread/worker 后 → 16 worker × 8 = 128 核 ≤ 160 → load 健康

### 13.3 为什么跳过 `storeProject`？
- 每工况 13 MB `.qpr` 文件 IO ≈ 60s
- 大规模 1000 几何 × 42 工况 ≈ 700 小时纯 IO
- pkl 已含全部物理量，`.qpr` 仅 GUI 调试用

### 13.4 为什么 `createInstance` 复用安全？
- 独立测试 `test_createinstance_reuse.py`：
  - 第 1 次 createInstance + sim_A + close + 第 2 次 createInstance + sim_B = baseline
  - 第 1 次 createInstance + sim_A + sim_B + close = reuse
  - 两者 Thrust 误差 **0.0000%**（精度无损）
- `loadSimDefinition` 第二次调用能完全 reset 内部 wake/state
- 每工况 init 4.5s → 2.4s（省 OpenCL device 枚举 + OpenMP 池初始化）

### 13.5 为什么 wake 截断没加速？
- MAXWAKEDIST 截断从未触发（1200 步 wake 漂移仅 1 米 << 2.5 米阈值）
- ZONE 截断只在最后 22% 时间生效（step 940-1200）
- 真正瓶颈在累积期前 940 步的 O(N²) 涡-涡相互作用，无法绕开

### 13.6 为什么 1200 步是稳态最小？
- 实测对比：
  | 步数 | Wall clock | THRUST 误差 | Fy 误差 | My 误差 |
  |---|---|---|---|---|
  | 400 | 9.9 min | 2.66% | **105%** | **115%** |
  | 800 | 22.1 min | 2.0% | **92%** | **111%** |
  | 1200 | 44.9 min | 基线 | 基线 | 基线 |
- Fy/My 量级小（0.01-0.06），少于 1200 步时末 120 步均值落在振荡半周期

### 13.7 为什么 GPU 不可用？
- node1 单卡 GPU + wake 截断：跑通但只比 CPU 快 25%
- node1/node6 多卡并发：
  - 直接 ladder → 3 个 `CL_INVALID_DEVICE(-33)` core dump + 5 个 hang
  - CUDA_VISIBLE_DEVICES 隔离 → 全 hang
- 是 NVIDIA OpenCL 驱动层 ICD 竞争，用户态隔离无法解决
- Docker 也救不了（容器共享宿主驱动）

---

## 14. 配置参考表

### 14.1 数据生成（命令行）

```bash
python scripts/run_data_gen_parallel.py \
  --workers 10 \                # 并行进程数（node1: 10, node6: 24）
  --omp-threads 8 \             # 每 worker OMP 线程
  --num-timesteps 1200 \        # 稳态最小
  --device CPU \                # GPU 不可用
  --geometry-npy /root/geometry_data.npy \  # (1000, 22, 3)
  --first-n-geoms 100 \         # 100 几何减规模
  --batch-size 16 \             # 每 16 几何落盘
  --tag prod_v1 \               # 输出文件名 tag
  --start-idx 0 --end-idx 100  # 几何切片（断点续传）
```

### 14.2 训练（config.py 字段）

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `geometry_mode` | str | `"sections"` | 输入特征模式 |
| `condition_columns` | list | `["RPM","WIND","ANGLE"]` | 工况列名 |
| `output_columns` | list | `["Fx","Fy","Fz","Torque"]` | DNN 输出 |
| `hidden_dims` | list | `[64,128,64]` | MLP 隐藏层 |
| `dropout` | float | `0.0` | Dropout 比例 |
| `batch_size` | int | `64` | 训练 batch |
| `epochs` | int | `5000` | 最大 epoch |
| `lr` | float | `1e-3` | Adam 学习率 |
| `weight_decay` | float | `1e-3` | L2 正则 |
| `early_stop_patience` | int | `500` | early stop |
| `grad_clip` | float | `1.0` | 梯度裁剪 |
| `random_state` | int | `42` | train/test split 种子 |
| `test_size` | float | `0.2` | 测试集占比 |

### 14.3 PPO 环境（env.py 参数）

| 参数 | 默认 | 含义 |
|---|---|---|
| `V` | 10.0 | 飞行速度 m/s |
| `max_steps` | 50 | 单 episode 最大步数 |
| `no_improve_limit` | 10 | 无改善早停步数 |
| `delta_scale` | 0.10 | action → CP delta 缩放 |
| `mass` | 3.5 | 无人机质量 kg |
| `reward_beta` | 0.1 | reward 平滑系数 |
| `ood_penalty` | 0.0 | OOD 惩罚强度 |
| `constrain_to_data` | True | 硬约束到训练域 |
| `use_dnn` | True | 用 DNN 代理 vs dummy |

### 14.4 物理常数（uav_model.py）

| 常数 | 值 | 来源 |
|---|---|---|
| `m` | 3.5 kg | RotorPy 同类机型 |
| `V_cruise` | 10 m/s | 设计巡航速度 |
| `prop_R` | 0.127 m | 10" 螺旋桨半径 |
| `arm` | 0.225 m | 450 mm 轴距 / 2 |
| `Cd_front` | 1.5 | Hoerner Fluid Dynamic Drag |
| `Cd_max` | 1.8 | 钝体（90° 攻角） |
| `Cl_max` | 0.4 | 平板模型 |
| `ρ` | 1.225 kg/m³ | 海平面标准大气 |

---

## 附录

### A. 关键 commit 列表

| Commit | 文件 | 说明 |
|---|---|---|
| `a705683` | scripts/install_linux.sh | Linux 部署 |
| `265afa4` | submodule .bld | 列错位修复 |
| `f249608` | submodule simulation.py | createInstance 复用 + NaN 兜底 |
| `70fe8e8` | submodule simulation.py | 跳过 storeProject |
| `03bdfb7` | submodule simulation.py | 字段扩展 Mx/My/Mz |
| `ad8b9bb` | submodule simulation.py | FX 字段修正 |
| `bbaff14` | submodule .trb | wake 截断激进版 |
| `800f594` | scripts/run_data_gen_parallel.py | 多进程并行 |
| `b7a2a7f` | docs/qblade_data_generation.html | 数据生成详解 |
| `d32f2b9` | docs/PROJECT_REPORT.md | 项目综述 |

### B. 参考文献

- **配平方程**：Ye et al. (2021) "Aerospace Sci. Tech." [R1]
- **多旋翼建模**：Pounds et al. (2010) "Control Eng. Practice" [R2]
- **旋翼气动**：Leishman (2006) "Principles of Helicopter Aerodynamics" [R3]
- **开源参数**：Folk et al. (2023) "RotorPy" arXiv:2306.04485 [R4]
- **FVW 算法**：[QBlade Theory: LLT Free Vortex Wake](https://docs.qblade.org/src/theory/aerodynamics/lifting_line/lifting_line.html)
- **Wake 参数**：[QBlade Turbine Definition](https://docs.qblade.org/src/user/turbine/turbineexport.html)

### C. 文档清单

| 文件 | 角色 |
|---|---|
| `docs/PROJECT_REPORT.md` | 项目综合交付（架构 + 历程 + 状态） |
| `docs/TECHNICAL_REFERENCE.md` | ⭐ 本文件（代码 + 算法 + API） |
| `docs/qblade_data_generation.html` | QBlade 数据生成详解（wake 截断深挖） |
| `docs/ppo_optimization.md` | PPO 优化笔记 |
| `docs/uav_model_references.md` | 物理建模文献参考 |
| `docs/当前工作说明与技术路线图.md` | 旧版状态（2026-05-08） |
| `RESEARCH_LOG.md` | 实验日志（旧版结果 + OOD 分析） |
| `README.md` | 项目主页（快速开始） |
| `CLAUDE.md` | 开发准则（中文规范） |
