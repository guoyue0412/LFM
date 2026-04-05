"""并行批量仿真调度器 — 支持 GPU 单机 / CPU 多进程 / 混合模式

架构说明
--------
QBlade DLL (基于 OpenCL) 在单进程内只能持有一个 GPU 实例,
因此并行策略基于 **多进程**:

- GPU 模式: 顺序执行, 利用 GPU OpenCL 加速单次仿真
- CPU 模式: N 个 worker 进程并行, 每个持有独立 DLL 实例
- 混合模式: 1 个 GPU worker + (N-1) 个 CPU worker

进程间通信使用 `multiprocessing.Queue`, 避免共享状态。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .config import SimCondition, SimConfig, SimPaths
from .runner import QBladeRunner

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """单条仿真结果封装。"""
    condition: SimCondition
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    elapsed_sec: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None


# ───────────────────────────────────────────────
#  Worker 函数 (在子进程中执行, 必须为模块级 top-level)
# ───────────────────────────────────────────────

def _worker_run_sim(
    paths_dict: Dict[str, str],
    config_dict: Dict,
    condition_dict: Dict,
    sim_file: Optional[str],
    geometry_id: int,
    bld_source: Optional[str],
) -> Dict:
    """子进程 worker: 初始化 DLL → 执行仿真 → 返回序列化结果。

    所有参数均使用 plain dict/str 以确保 pickle 可序列化。
    """
    from .config import SimCondition, SimConfig, SimPaths

    paths = SimPaths(**paths_dict)
    config = SimConfig(**config_dict)
    cond = SimCondition(**condition_dict)

    if bld_source and bld_source != paths.bld_file_path:
        shutil.copy2(bld_source, paths.bld_file_path)

    t0 = time.perf_counter()
    try:
        with QBladeRunner(paths, config, geometry_id=geometry_id) as runner:
            df = runner.run(cond, sim_file=sim_file,
                            export_results=False, store_project=False)
        return {
            "condition": condition_dict,
            "data": df.to_dict("list"),
            "error": None,
            "elapsed": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "condition": condition_dict,
            "data": None,
            "error": str(e),
            "elapsed": time.perf_counter() - t0,
        }


class BatchRunner:
    """批量仿真并行调度器。

    Usage::

        runner = BatchRunner(paths, config, max_workers=4, device_strategy="cpu")
        results = runner.run_batch(conditions, sim_files)

        for r in results:
            if r.ok:
                print(r.condition.name, r.data["THRUST"].iloc[0])
    """

    def __init__(
        self,
        paths: SimPaths,
        config: SimConfig,
        *,
        max_workers: int = 1,
        device_strategy: str = "gpu",
        geometry_id: int = 1,
    ):
        """
        Args:
            paths: 仿真文件路径
            config: 仿真运行配置
            max_workers: 最大 worker 进程数 (GPU 模式忽略)
            device_strategy: 'gpu' | 'cpu' | 'hybrid'
            geometry_id: 几何编号
        """
        self.paths = paths
        self.config = config
        self.max_workers = max_workers
        self.device_strategy = device_strategy.lower()
        self.geometry_id = geometry_id

    def run_batch(
        self,
        conditions: List[SimCondition],
        sim_files: Optional[List[str]] = None,
        *,
        progress_callback: Optional[Callable[[int, int, BatchResult], None]] = None,
    ) -> List[BatchResult]:
        """批量执行仿真。

        Args:
            conditions: 工况列表
            sim_files: 对应 .sim 文件路径列表; None 则全部使用 base_sim
            progress_callback: fn(done, total, result) 每完成一个仿真回调

        Returns:
            与 conditions 顺序对应的结果列表
        """
        n = len(conditions)
        if sim_files is None:
            sim_files = [None] * n
        assert len(sim_files) == n

        if self.device_strategy == "gpu" or self.max_workers <= 1:
            return self._run_sequential(conditions, sim_files, progress_callback)
        elif self.device_strategy == "cpu":
            return self._run_parallel_cpu(conditions, sim_files, progress_callback)
        elif self.device_strategy == "hybrid":
            return self._run_hybrid(conditions, sim_files, progress_callback)
        else:
            raise ValueError(f"未知 device_strategy: {self.device_strategy}")

    # ─── GPU 顺序模式 ───
    def _run_sequential(
        self,
        conditions: List[SimCondition],
        sim_files: List[Optional[str]],
        callback,
    ) -> List[BatchResult]:
        results: List[BatchResult] = []
        with QBladeRunner(self.paths, self.config, geometry_id=self.geometry_id) as runner:
            for i, (cond, sf) in enumerate(zip(conditions, sim_files)):
                t0 = time.perf_counter()
                try:
                    df = runner.run(cond, sim_file=sf)
                    br = BatchResult(cond, df, elapsed_sec=time.perf_counter() - t0)
                except Exception as e:
                    br = BatchResult(cond, error=str(e), elapsed_sec=time.perf_counter() - t0)
                results.append(br)
                if callback:
                    callback(i + 1, len(conditions), br)
        return results

    # ─── CPU 多进程模式 ───
    def _run_parallel_cpu(
        self,
        conditions: List[SimCondition],
        sim_files: List[Optional[str]],
        callback,
    ) -> List[BatchResult]:
        cpu_config = SimConfig(
            device="CPU",
            num_timesteps=self.config.num_timesteps,
            warmup_steps=self.config.warmup_steps,
            omp_threads=max(1, self.config.omp_threads // self.max_workers),
            air_density=self.config.air_density,
            propeller_radius=self.config.propeller_radius,
            group_size=self.config.group_size,
        )
        return self._submit_to_pool(
            conditions, sim_files, cpu_config, self.max_workers, callback
        )

    # ─── 混合模式: 1 GPU + (N-1) CPU ───
    def _run_hybrid(
        self,
        conditions: List[SimCondition],
        sim_files: List[Optional[str]],
        callback,
    ) -> List[BatchResult]:
        if len(conditions) <= 1:
            return self._run_sequential(conditions, sim_files, callback)

        gpu_cond, gpu_sf = conditions[0], sim_files[0]
        cpu_conds, cpu_sfs = conditions[1:], sim_files[1:]

        results = [None] * len(conditions)

        cpu_config = SimConfig(
            device="CPU",
            num_timesteps=self.config.num_timesteps,
            warmup_steps=self.config.warmup_steps,
            omp_threads=max(1, self.config.omp_threads // self.max_workers),
            air_density=self.config.air_density,
            propeller_radius=self.config.propeller_radius,
            group_size=self.config.group_size,
        )

        gpu_result = self._run_sequential([gpu_cond], [gpu_sf], None)[0]
        results[0] = gpu_result
        if callback:
            callback(1, len(conditions), gpu_result)

        cpu_results = self._submit_to_pool(
            cpu_conds, cpu_sfs, cpu_config, max(1, self.max_workers - 1), callback,
            done_offset=1, total_override=len(conditions),
        )
        for i, r in enumerate(cpu_results):
            results[i + 1] = r

        return results

    # ─── 进程池提交 ───
    def _submit_to_pool(
        self,
        conditions: List[SimCondition],
        sim_files: List[Optional[str]],
        config: SimConfig,
        n_workers: int,
        callback,
        done_offset: int = 0,
        total_override: Optional[int] = None,
    ) -> List[BatchResult]:
        total = total_override or len(conditions)
        paths_d = {
            "dll_file": self.paths.dll_file,
            "base_sim": self.paths.base_sim,
            "sim_folder": self.paths.sim_folder,
            "bld_file_path": self.paths.bld_file_path,
            "qbr_file_folder": self.paths.qbr_file_folder,
            "simulation_parameter_path": self.paths.simulation_parameter_path,
        }
        config_d = {
            "device": config.device,
            "num_timesteps": config.num_timesteps,
            "warmup_steps": config.warmup_steps,
            "omp_threads": config.omp_threads,
            "air_density": config.air_density,
            "propeller_radius": config.propeller_radius,
            "group_size": config.group_size,
        }

        results: List[Optional[BatchResult]] = [None] * len(conditions)
        done = done_offset

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {}
            for i, (cond, sf) in enumerate(zip(conditions, sim_files)):
                fut = pool.submit(
                    _worker_run_sim,
                    paths_d, config_d,
                    {"rpm": cond.rpm, "wind_speed": cond.wind_speed, "angle": cond.angle},
                    sf, self.geometry_id, None,
                )
                future_to_idx[fut] = i

            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                cond = conditions[idx]
                try:
                    raw = fut.result()
                    if raw["data"] is not None:
                        df = pd.DataFrame(raw["data"])
                        br = BatchResult(cond, df, elapsed_sec=raw["elapsed"])
                    else:
                        br = BatchResult(cond, error=raw["error"], elapsed_sec=raw["elapsed"])
                except Exception as e:
                    br = BatchResult(cond, error=str(e))
                results[idx] = br
                done += 1
                if callback:
                    callback(done, total, br)

        return results


# ─── 便捷函数 ───

def run_batch_gpu(
    paths: SimPaths,
    conditions: List[SimCondition],
    config: Optional[SimConfig] = None,
    **kw,
) -> List[BatchResult]:
    """GPU 顺序批量仿真 (利用 OpenCL 加速单次)。"""
    cfg = config or SimConfig(device="GPU")
    return BatchRunner(paths, cfg, device_strategy="gpu", **kw).run_batch(conditions)


def run_batch_cpu(
    paths: SimPaths,
    conditions: List[SimCondition],
    config: Optional[SimConfig] = None,
    max_workers: int = 4,
    **kw,
) -> List[BatchResult]:
    """CPU 多进程并行批量仿真。"""
    cfg = config or SimConfig(device="CPU")
    return BatchRunner(paths, cfg, max_workers=max_workers,
                       device_strategy="cpu", **kw).run_batch(conditions)


def run_batch_hybrid(
    paths: SimPaths,
    conditions: List[SimCondition],
    config: Optional[SimConfig] = None,
    max_workers: int = 4,
    **kw,
) -> List[BatchResult]:
    """混合模式: 1 GPU + (N-1) CPU 并行。"""
    cfg = config or SimConfig(device="GPU")
    return BatchRunner(paths, cfg, max_workers=max_workers,
                       device_strategy="hybrid", **kw).run_batch(conditions)
