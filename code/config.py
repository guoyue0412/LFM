"""螺旋桨气动力预测模型 — 配置文件

几何模式:
  "none"           → 仅工况输入 (RPM, WIND, ANGLE)          → input_dim = 3
  "control_points" → 工况 + 8个B样条控制点 (4 chord + 4 twist) → input_dim = 11
  "sections"       → 工况 + 22截面 chord + 22截面 twist       → input_dim = 47
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ==================== 路径 ====================
    raw_data_dir: str = "./data_for_train/data"
    processed_data_dir: str = "./data_for_train/geometry_run_results_100"
    model_dir: str = "./trained_models"
    log_dir: str = "./runs"

    # ==================== 数据 — 工况 ====================
    condition_columns: List[str] = field(default_factory=lambda: ["RPM", "WIND", "ANGLE"])
    output_columns: List[str] = field(default_factory=lambda: ["Fx", "Fy", "Fz", "Torque"])
    test_size: float = 0.2
    random_state: int = 42

    # ==================== 数据 — 几何 ====================
    geometry_mode: str = "sections"   # "none" | "control_points" | "sections"
    n_control_points: int = 8               # 4 chord + 4 twist
    n_sections: int = 22

    # ==================== 模型 ====================
    # sweep 最优: trial #21 (sweep_20260401), R²=0.988, 20k params
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 64])
    dropout: float = 0.0

    # ==================== 训练 ====================
    batch_size: int = 64
    epochs: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 200
    min_lr: float = 1e-6
    early_stop_patience: int = 500
    grad_clip: float = 1.0
    print_every: int = 500
    plot_every: int = 200

    # ==================== 自动计算（勿手动设置） ====================
    geometry_columns: List[str] = field(default_factory=list, init=False, repr=False)
    input_columns: List[str] = field(default_factory=list, init=False, repr=False)
    input_dim: int = field(default=0, init=False)
    output_dim: int = field(default=0, init=False)

    def __post_init__(self):
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        if self.geometry_mode == "control_points":
            self.geometry_columns = [f"cp_{i}" for i in range(self.n_control_points)]
        elif self.geometry_mode == "sections":
            self.geometry_columns = (
                [f"chord_{i}" for i in range(self.n_sections)]
                + [f"twist_{i}" for i in range(self.n_sections)]
            )
        else:
            self.geometry_columns = []

        self.input_columns = self.condition_columns + self.geometry_columns
        self.input_dim = len(self.input_columns)
        self.output_dim = len(self.output_columns)
