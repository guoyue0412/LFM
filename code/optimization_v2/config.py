"""全局配置 — optimization_v2"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptConfig:
    # ==================== 飞行工况 ====================
    V_cruise: float = 10.0          # 巡航速度 m/s
    mass: float = 3.5               # 无人机质量 kg
    rho: float = 1.225              # 空气密度 kg/m³
    g: float = 9.81                 # 重力加速度

    # ==================== 螺旋桨几何 ====================
    n_cp_chord: int = 4             # 弦长控制点数
    n_cp_twist: int = 4             # 扭角控制点数
    n_sections: int = 22            # 径向截面数
    R_prop: float = 0.127           # 螺旋桨半径 m
    D_prop: float = 0.254           # 螺旋桨直径 m

    # ==================== 机体参数 ====================
    l1: float = 0.16                # 前桨到重心距离 m
    l2: float = 0.16                # 后桨到重心距离 m
    d1: float = 0.05                # 前桨高于重心距离 m
    d2: float = 0.05                # 后桨高于重心距离 m

    # ==================== 配平约束 ====================
    omega_min: float = 4000.0       # 最小转速 RPM
    omega_max: float = 6500.0       # 最大转速 RPM
    alpha_min_deg: float = 2.0      # 最小迎角 deg
    alpha_max_deg: float = 8.0      # 最大迎角 deg

    # ==================== CP 边界 (数据驱动, +5% 边距) ====================
    cp_bounds: np.ndarray = field(default_factory=lambda: np.array([
        [0.013599, 0.026293],   # chord cp 0
        [0.022239, 0.043821],   # chord cp 1
        [0.010052, 0.019808],   # chord cp 2
        [0.004783, 0.009199],   # chord cp 3
        [37.4, 72.7],           # twist cp 0 (deg)
        [13.4, 26.5],           # twist cp 1
        [11.3, 22.1],           # twist cp 2
        [8.4, 16.6],            # twist cp 3
    ], dtype=np.float64))

    # ==================== 电气系统 ====================
    eta_motor_esc: float = 0.80     # 电机+ESC 综合效率
    E_battery: float = 150.0 * 3600.0  # 电池能量 J (150 Wh)

    # ==================== MoE 模型 ====================
    moe_n_experts: int = 4
    moe_hidden_dim: int = 128
    moe_shared_dim: int = 64
    moe_input_dim: int = 47         # 3 工况 + 44 截面
    moe_output_dim: int = 4         # T, H, My, Q

    # ==================== MoE 训练 ====================
    moe_epochs: int = 3000
    moe_batch_size: int = 64
    moe_lr: float = 1e-3
    moe_weight_decay: float = 1e-3
    moe_balance_weight: float = 0.1
    moe_smooth_weight: float = 0.01

    # ==================== 优化目标权重 ====================
    lambda_residual: float = 1000.0     # 配平残差惩罚
    lambda_uncertainty: float = 50.0    # 不确定度惩罚
    lambda_geometry: float = 10.0       # 几何平滑惩罚
    penalty_trim_fail: float = 1e6      # 配平失败惩罚值

    # 不确定度按 [T, H, My, Q] 加权 (T 和 Q 直接进功率,权重最高)
    uncertainty_weights: tuple = (1.0, 0.5, 0.5, 1.0)

    # ==================== 主动补点 ====================
    sigma_max: float = 0.1          # 不确定度阈值 (归一化)
    active_learning_batch: int = 10 # 每轮补点数

    # ==================== CMA-ES ====================
    cma_maxiter: int = 200
    cma_popsize: int = 16
    cma_sigma0: float = 0.3

    # ==================== 数据路径 ====================
    data_dir: str = "./optimization_v2/data"
    model_dir: str = "./optimization_v2/models"
    results_dir: str = "./optimization_v2/results"

    # ==================== 数据源 (Excel / CSV) ====================
    # 优先使用 Excel; 若为 None 则回退到 synthetic_aero_data.csv
    excel_path: str = None              # 例: "./data_for_train/aero_data.xlsx"
    excel_sheet: int = 0                # 工作表名或索引

    # 列名显式映射 (None 时使用 data_loader 中的别名表自动匹配)
    rpm_col: str = None
    wind_col: str = None
    angle_col: str = None
    T_col: str = None
    H_col: str = None
    My_col: str = None
    Q_col: str = None
    cp_prefix: str = "cp_"
    chord_prefix: str = "chord_"
    twist_prefix: str = "twist_"
    chord_cp_prefix: str = "chord_cp_"
    twist_cp_prefix: str = "twist_cp_"

    # ==================== 派生属性 ====================
    @property
    def n_cp(self) -> int:
        return self.n_cp_chord + self.n_cp_twist

    @property
    def alpha_min_rad(self) -> float:
        return np.radians(self.alpha_min_deg)

    @property
    def alpha_max_rad(self) -> float:
        return np.radians(self.alpha_max_deg)

    @property
    def weight(self) -> float:
        return self.mass * self.g
