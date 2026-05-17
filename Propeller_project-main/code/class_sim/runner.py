"""QBlade 单次仿真运行器 — 支持 context-manager 和进程安全"""

from __future__ import annotations

import ctypes
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    QBLADE_DATA_KEYS,
    RESULT_COLUMNS,
    SimCondition,
    SimConfig,
    SimPaths,
)


def _to_bytes(path: str) -> ctypes.Array:
    return ctypes.create_string_buffer(path.encode("utf-8"))


class QBladeRunner:
    """进程安全的单次仿真执行器。

    设计为 context-manager, 确保 DLL 资源始终释放::

        with QBladeRunner(paths, config) as runner:
            df = runner.run(condition)
    """

    def __init__(
        self,
        paths: SimPaths,
        config: SimConfig,
        *,
        geometry_id: int = 1,
    ):
        self.paths = paths
        self.config = config
        self.geometry_id = geometry_id
        self._lib = None

    # ─── context manager ───
    def __enter__(self) -> "QBladeRunner":
        self._load_dll()
        return self

    def __exit__(self, *exc) -> None:
        self._unload_dll()

    # ─── public API ───
    def run(
        self,
        condition: SimCondition,
        sim_file: Optional[str] = None,
        *,
        export_results: bool = True,
        store_project: bool = True,
    ) -> pd.DataFrame:
        """执行单次仿真并返回处理后的 DataFrame。

        Args:
            condition: 仿真工况 (RPM, WIND, ANGLE)
            sim_file: 指定 .sim 路径; None 则使用 base_sim
            export_results: 是否导出 QBlade 结果文件
            store_project: 是否保存 .qpr
        """
        path = sim_file or self.paths.base_sim
        self._ensure_dll()

        self._lib.loadSimDefinition(_to_bytes(path))
        self._lib.initializeSimulation()

        raw = self._run_timesteps()
        df = self._postprocess(raw, condition)

        if export_results:
            self._export(path)
        if store_project:
            self._store_qpr(path)

        return df

    def run_full(
        self,
        condition: SimCondition,
        sim_file: Optional[str] = None,
    ) -> bool:
        """使用 DLL 的 runFullSimulation 一次性跑完 (无中间数据)。

        适用于只需最终结果的批量场景; 返回 DLL 的 bool 状态。
        """
        path = sim_file or self.paths.base_sim
        self._ensure_dll()
        self._lib.loadSimDefinition(_to_bytes(path))
        self._lib.initializeSimulation()
        return bool(self._lib.runFullSimulation())

    # ─── internal ───
    def _load_dll(self) -> None:
        from QBladeLibrary import QBladeLibrary  # type: ignore

        self._lib = QBladeLibrary(self.paths.dll_file)
        self._lib.createInstance(self.config.device_id, self.config.group_size)

        if self.config.omp_threads > 0:
            self._lib.setOmpNumThreads(self.config.omp_threads)

    def _unload_dll(self) -> None:
        if self._lib is not None:
            try:
                self._lib.unload()
            except Exception:
                pass
            self._lib = None

    def _ensure_dll(self) -> None:
        if self._lib is None:
            self._load_dll()

    def _run_timesteps(self) -> Dict[str, List[float]]:
        n = self.config.num_timesteps
        start = self.config.record_start
        keys = list(RESULT_COLUMNS)
        data_keys = list(QBLADE_DATA_KEYS)
        buf: Dict[str, List[float]] = {k: [] for k in keys}

        for step in range(n):
            self._lib.advanceTurbineSimulation()
            if step >= start:
                for col, qkey in zip(keys, data_keys):
                    buf[col].append(
                        float(self._lib.getCustomData_at_num(qkey.encode(), 0, 0))
                    )
        return buf

    def _postprocess(
        self,
        raw: Dict[str, List[float]],
        cond: SimCondition,
    ) -> pd.DataFrame:
        df = pd.DataFrame(raw)
        df["RPM"] = cond.rpm
        df["WIND_SPEED"] = cond.wind_speed
        df["ANGLE"] = cond.angle

        rho = self.config.air_density
        R = self.config.propeller_radius
        D = 2 * R
        n_rps = cond.rpm / 60.0

        df["THRUST"] = df["Thrust"].mean() * -1
        df["POWER"] = df["Power"].mean() * -1
        df["TORQUE"] = df["Torque"].mean() * -1
        df["THRUST_Y"] = df["Thrust_y"].mean() * -1
        df["THRUST_Z"] = df["Thrust_z"].mean() * -1

        df["Ct"] = df["THRUST"] / (rho * n_rps ** 2 * D ** 4)
        df["Cp"] = df["POWER"] * 1000 / (rho * n_rps ** 3 * D ** 5)

        if cond.wind_speed != 0:
            J = cond.wind_speed / (n_rps * D)
            df["eta"] = J * (df["Ct"] / df["Cp"])
        else:
            df["eta"] = 0.0

        return df

    def _export(self, sim_path: str) -> None:
        name = f"./simulation_data/geometrical_{self.geometry_id}_results"
        self._lib.exportResults(0, b"./", name.encode(), b"")

    def _store_qpr(self, sim_path: str) -> None:
        stem = os.path.splitext(os.path.basename(sim_path))[0]
        qpr = os.path.join(self.paths.qbr_file_folder, f"{stem}.qpr")
        self._lib.storeProject(_to_bytes(qpr))
