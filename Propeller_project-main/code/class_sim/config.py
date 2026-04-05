"""仿真配置 — 集中管理路径、常量和运行参数"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimPaths:
    """QBlade 仿真所需的所有文件路径。"""
    dll_file: str
    base_sim: str
    sim_folder: str
    bld_file_path: str
    qbr_file_folder: str
    simulation_parameter_path: str

    def validate(self) -> None:
        for name, path in [
            ("dll_file", self.dll_file),
            ("base_sim", self.base_sim),
            ("bld_file_path", self.bld_file_path),
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"SimPaths.{name} 不存在: {path}")
        for name, folder in [
            ("sim_folder", self.sim_folder),
            ("qbr_file_folder", self.qbr_file_folder),
        ]:
            os.makedirs(folder, exist_ok=True)

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "SimPaths":
        """从旧版 file_path 字典构造 (向后兼容)。"""
        key_map = {
            "dll_file": "dll_file",
            "base_sim": "base_sim",
            "sim_folder": "SIM_folder",
            "bld_file_path": "bld_file_path",
            "qbr_file_folder": "QBR_file_folder",
            "simulation_parameter_path": "simulation_parameter_path",
        }
        return cls(**{new: d[old] for new, old in key_map.items()})


@dataclass
class SimCondition:
    """单个仿真工况。"""
    rpm: float
    wind_speed: float
    angle: float

    @property
    def name(self) -> str:
        return f"RPM{self.rpm}_Wind{self.wind_speed}_Angle{self.angle}"

    def to_sim_params(self) -> List[Tuple[str, str]]:
        return [
            ("OBJECTNAME", self.name),
            ("RPMPRESCRIBED", str(self.rpm)),
            ("MEANINF", str(self.wind_speed)),
            ("VERTANGLE", str(self.angle)),
        ]


@dataclass
class SimConfig:
    """仿真运行配置。"""
    device: str = "GPU"
    num_timesteps: int = 1800
    warmup_steps: int = 1200
    omp_threads: int = 4
    air_density: float = 1.225
    propeller_radius: float = 0.127

    @property
    def device_id(self) -> int:
        return 1 if self.device.upper() == "GPU" else 0

    @property
    def record_start(self) -> int:
        return self.num_timesteps - self.warmup_steps

    # QBlade createInstance 的 groupSize 参数
    group_size: int = 32


# .sim 模板中需要保留的文件 (不删除)
SIM_TEMPLATE_FILES = frozenset({
    "Baseline_Simulation.sim",
    "Baseline_Blade_Turb",
    "UAV_urban_pro.bts",
    "Base_simulation.sim",
})

# QBlade 结果列名
QBLADE_DATA_KEYS = (
    "Time [s]",
    "Aerodynamic Thrust [N]",
    "Aerodynamic Power [W]",
    "Aerodynamic Torque [Nm]",
    "Aerodynamic Force in Hub Y_g Direction [N]",
    "Aerodynamic Force in Hub Z_g Direction [N]",
)

RESULT_COLUMNS = ("Time", "Thrust", "Power", "Torque", "Thrust_y", "Thrust_z")


def load_conditions_from_excel(path: str) -> List[SimCondition]:
    """从 Excel 文件加载仿真工况列表。"""
    df = pd.read_excel(path)
    conditions = []
    for _, row in df.iterrows():
        conditions.append(SimCondition(
            rpm=float(row["RPM"]),
            wind_speed=float(row["windSpeed"]),
            angle=float(row["windAngle"]),
        ))
    return conditions
