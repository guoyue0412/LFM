# @Time    : 2025/03/04 (original)  2026/03 (refactored)
# @Author  : github.com/guoyue0412
# @Refactor: 高聚合低耦合重构 — 向后兼容薄封装
#
# 原始 SIMULATION 类保留全部公开方法签名,
# 内部委托给 config / sim_file / geometry / runner / batch 子模块。
# 新代码应直接使用子模块 API。

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import (
    SimCondition,
    SimConfig,
    SimPaths,
    load_conditions_from_excel,
)
from .sim_file import generate_sim_files, collect_sim_files, modify_sim_inplace
from .geometry import control_points_to_sections, modify_bld_file
from .runner import QBladeRunner
from .batch import BatchRunner, BatchResult


class SIMULATION:
    """QBlade 螺旋桨仿真管理器 (向后兼容封装)。

    ⚠️ 推荐新代码直接使用 ``class_sim`` 子模块:
        - ``QBladeRunner``  — 单次仿真
        - ``BatchRunner``   — 批量并行仿真
        - ``geometry.*``    — 几何管理
        - ``sim_file.*``    — .sim 文件操作

    保留旧 API 以兼容现有调用方。
    """

    def __init__(
        self,
        file_path: dict,
        geometry_baseline: np.ndarray,
        device_type: str = "GPU",
        number_of_timesteps: int = 1800,
    ):
        # ─── 路径 ───
        self.file_path = file_path
        self.paths = SimPaths.from_dict(file_path)

        # ─── 配置 ───
        self.config = SimConfig(
            device=device_type,
            num_timesteps=number_of_timesteps,
        )

        # ─── 状态 ───
        self.simulation_parameters: List[List[tuple]] = []
        self.one_simulation_data: pd.DataFrame = pd.DataFrame()
        self.all_simulation_data: Dict[str, Union[pd.DataFrame, np.ndarray]] = {}
        self.number_of_timesteps = number_of_timesteps
        self.geometry_baseline = geometry_baseline
        self.current_propeller = pd.DataFrame(geometry_baseline)
        self.all_simulation_data["geometry"] = geometry_baseline
        self.segment_load_data = pd.DataFrame()
        self.geometry_id: int = 1

        # 旧属性 (兼容)
        self.airDensity = self.config.air_density
        self.R = self.config.propeller_radius
        self.device = self.config.device_id

        # 自动初始化
        self._conditions = load_conditions_from_excel(
            self.paths.simulation_parameter_path
        )
        self._fill_legacy_params()
        self._sim_files = generate_sim_files(self.paths, self._conditions)

    # ───────────────────────────────────────
    #  旧 API — 保持签名兼容
    # ───────────────────────────────────────

    def str_to_byte(self, file_path: str):
        import ctypes
        return ctypes.create_string_buffer(file_path.encode("utf-8"))

    def generating_simulation_parameters_tuple(self, path: str) -> None:
        self._conditions = load_conditions_from_excel(path)
        self._fill_legacy_params()

    def generating_sim_file(self) -> None:
        self._sim_files = generate_sim_files(self.paths, self._conditions)

    def run_one_simulation(
        self,
        RPM: Optional[float] = None,
        WIND_SPEED: Optional[float] = None,
        ANGLE: Optional[float] = None,
        SIM_file_path: Optional[str] = None,
        IS_Turbulence: bool = False,
    ) -> pd.DataFrame:
        """运行单次仿真 (兼容旧签名)。"""
        if SIM_file_path is not None:
            cond = self._parse_condition_from_path(SIM_file_path)
            if cond is None:
                cond = SimCondition(rpm=RPM or 0, wind_speed=WIND_SPEED or 0, angle=ANGLE or 0)
            sim_path = SIM_file_path
        else:
            cond = SimCondition(
                rpm=RPM or 0,
                wind_speed=WIND_SPEED or 0,
                angle=ANGLE or 0,
            )
            if IS_Turbulence:
                params = [
                    ("OBJECTNAME", cond.name),
                    ("RPMPRESCRIBED", str(cond.rpm)),
                    ("VERTANGLE", str(cond.angle)),
                ]
            else:
                params = cond.to_sim_params()

            modify_sim_inplace(self.paths.base_sim, params)
            sim_path = self.paths.base_sim

        with QBladeRunner(self.paths, self.config, geometry_id=self.geometry_id) as runner:
            df = runner.run(cond, sim_file=sim_path)

        self.one_simulation_data = df
        self.all_simulation_data[os.path.splitext(sim_path)[0]] = df
        print(f"{cond.name}")
        return df

    def run_all_simulation(self) -> None:
        """批量运行所有 .sim (顺序, 兼容旧行为)。"""
        sim_files = collect_sim_files(self.paths.sim_folder)
        for sf in sim_files:
            print(os.path.splitext(sf)[0])
            self.run_one_simulation(SIM_file_path=sf)

    def run_all_parallel(
        self,
        max_workers: int = 4,
        device_strategy: str = "cpu",
        progress_callback=None,
    ) -> List[BatchResult]:
        """✨ 新增: 并行批量仿真。

        Args:
            max_workers: CPU worker 数量
            device_strategy: 'gpu' | 'cpu' | 'hybrid'
            progress_callback: fn(done, total, result)

        Returns:
            BatchResult 列表
        """
        sim_files = collect_sim_files(self.paths.sim_folder)
        conditions = [
            self._parse_condition_from_path(sf)
            for sf in sim_files
        ]
        conditions = [c if c else SimCondition(0, 0, 0) for c in conditions]

        batch = BatchRunner(
            self.paths,
            self.config,
            max_workers=max_workers,
            device_strategy=device_strategy,
            geometry_id=self.geometry_id,
        )
        results = batch.run_batch(conditions, sim_files, progress_callback=progress_callback)

        for r in results:
            if r.ok:
                self.all_simulation_data[r.condition.name] = r.data

        return results

    def change_propeller_geometry(
        self,
        section_data: Optional[np.ndarray] = None,
        control_point: Optional[np.ndarray] = None,
    ) -> None:
        """修改螺旋桨几何。"""
        if section_data is None and control_point is None:
            self.current_propeller = self.geometry_baseline
        elif section_data is not None:
            self.current_propeller = section_data
        elif control_point is not None:
            radial = self.geometry_baseline[:, 0]
            self.current_propeller = control_points_to_sections(
                control_point, radial, self.R
            )

        modify_bld_file(self.paths.bld_file_path, np.asarray(self.current_propeller))
        self._save_geometry_dict()
        self.all_simulation_data.clear()
        self.all_simulation_data["geometry"] = self.current_propeller

    # ─── private helpers ───

    def _fill_legacy_params(self) -> None:
        self.simulation_parameters = [c.to_sim_params() for c in self._conditions]

    @staticmethod
    def _parse_condition_from_path(path: str) -> Optional[SimCondition]:
        import re
        m = re.search(r"RPM([\d.]+)_Wind([\d.]+)_Angle([\d.]+)", path)
        if m:
            return SimCondition(
                rpm=float(m.group(1)),
                wind_speed=float(m.group(2)),
                angle=float(m.group(3)),
            )
        return None

    def _save_geometry_dict(self) -> None:
        gfile = "geometry_simulation_dict.pkl"
        if os.path.exists(gfile):
            with open(gfile, "rb") as f:
                try:
                    gdict = pickle.load(f)
                except EOFError:
                    gdict = {}
        else:
            gdict = {}

        if gdict:
            last_num = int(next(reversed(gdict)).split("_")[-1])
            num = last_num + 1
        else:
            num = 0

        gdict[f"geometry_{num}"] = self.all_simulation_data

        with open(gfile, "wb") as f:
            pickle.dump(gdict, f)

        with open("propeller_simulation_data_for_one.pkl", "wb") as f:
            pickle.dump(self.all_simulation_data, f)
