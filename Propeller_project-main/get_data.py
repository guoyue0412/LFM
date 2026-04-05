"""几何级并行 QBlade 数据采集

架构:
    主进程                       Worker 进程 (×N)
    ────────                    ────────────────
    加载 geometry_list           clone_sim_tree → 独立副本
    分发几何索引 ──────────→     modify_bld → run_sim → 返回 Dict
    收集结果 ←──────────────     释放 DLL
    增量保存 pkl/csv

并行策略:
  - GPU (OpenCL): 单进程, max_workers=1
  - CPU: 每个 worker 独立 DLL + 文件树, 天然可并行

用法:
    python get_data.py --device GPU
    python get_data.py --device CPU --workers 4
    python get_data.py --device CPU --workers 4 --start 100
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "QBladeCE_2.0.9.2" / "SIL_Interface"))


# ═══════════════════════════════════════════════
#  Worker 隔离: 为每个进程创建独立的仿真文件树
# ═══════════════════════════════════════════════

def clone_sim_tree(src_sim_folder: str, worker_id: int, tmp_root: str) -> str:
    """为 worker 克隆完整的仿真目录树。

    对大文件 (.bts/.afl/.plr) 使用硬链接节省空间。
    """
    dst = os.path.join(tmp_root, f"worker_{worker_id}")
    if os.path.exists(dst):
        shutil.rmtree(dst)

    def _copy_fn(src, dst_path):
        if os.path.splitext(src)[1].lower() in (".bts", ".afl", ".plr"):
            try:
                os.link(src, dst_path)
                return
            except OSError:
                pass
        shutil.copy2(src, dst_path)

    shutil.copytree(src_sim_folder, dst, copy_function=_copy_fn)
    return dst


# ═══════════════════════════════════════════════
#  单个几何的仿真 (在 worker 进程中执行)
# ═══════════════════════════════════════════════

_BLD_LINE_RE = re.compile(
    r"^([+-]?\d*\.\d*(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d*\.\d*(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d*\.\d*(?:[eE][+-]?\d+)?)\s+"
    r"(.*)$"
)
_SIM_LINE_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(-\s+.+)")


def _modify_bld(bld_path: str, geometry: np.ndarray) -> None:
    """修改 .bld 文件中的 chord/twist 值。"""
    with open(bld_path, "r") as f:
        lines = f.readlines()

    pos_lookup = {float(row[0]): (abs(row[1]), abs(row[2])) for row in geometry}
    W = 19
    modified = []
    for line in lines:
        m = _BLD_LINE_RE.match(line)
        if not m:
            modified.append(line)
            continue
        pos_val = float(m.group(1))
        remaining = m.group(4)
        matched = next((k for k in pos_lookup if abs(pos_val - k) <= 1e-4), None)
        if matched is not None:
            c, t = pos_lookup[matched]
            modified.append(f"{pos_val:<{W}.5f} {c:<{W}.5f} {t:<{W}.5f} {remaining:<200}\n")
        else:
            modified.append(line)

    with open(bld_path, "w") as f:
        f.writelines(modified)


def _modify_sim(
    sim_path: str, rpm: float, wind_speed: float, angle: float, num_timesteps: int
) -> None:
    """修改 .sim 文件中的 RPM/WIND/ANGLE/NUMTIMESTEPS。"""
    with open(sim_path, "r") as f:
        lines = f.readlines()

    params = {
        "OBJECTNAME": f"RPM{rpm}_Wind{wind_speed}_Angle{angle}",
        "RPMPRESCRIBED": str(rpm),
        "MEANINF": str(wind_speed),
        "VERTANGLE": str(angle),
        "NUMTIMESTEPS": str(num_timesteps),
    }

    modified = []
    for line in lines:
        m = _SIM_LINE_RE.match(line)
        if not m:
            modified.append(line)
            continue
        first_col, keyword, comment = m.groups()
        if keyword in params:
            indent = "    " if keyword == "RPMPRESCRIBED" else ""
            w = 40 - len(indent)
            modified.append(f"{indent}{params[keyword]:<{w}} {keyword:<18} {comment:<50}\n")
        else:
            modified.append(line)

    with open(sim_path, "w") as f:
        f.writelines(modified)


def _run_one_geometry(
    worker_sim_folder: str,
    dll_file: str,
    geometry_data: np.ndarray,
    geometry_idx: int,
    rpm: float,
    wind_speed: float,
    angle: float,
    num_timesteps: int,
    device_id: int,
    omp_threads: int,
) -> Dict:
    """Worker 函数: 修改 .bld → 运行仿真 → 返回结果字典。"""
    import ctypes

    bld_path = os.path.join(
        worker_sim_folder, "Baseline_Blade_Turb", "Aero", "Baseline_Blade.bld"
    )
    sim_path = os.path.join(worker_sim_folder, "Baseline_Simulation.sim")

    _modify_bld(bld_path, geometry_data)
    _modify_sim(sim_path, rpm, wind_speed, angle, num_timesteps)

    t0 = time.perf_counter()
    orig_cwd = os.getcwd()
    try:
        os.chdir(worker_sim_folder)

        dll_dir = os.path.dirname(os.path.abspath(dll_file))
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

        from QBladeLibrary import QBladeLibrary

        lib = QBladeLibrary(dll_file)
        lib.createInstance(device_id, 32)
        if omp_threads > 0:
            lib.setOmpNumThreads(omp_threads)

        sim_basename = os.path.basename(sim_path)
        lib.loadSimDefinition(ctypes.create_string_buffer(sim_basename.encode("utf-8")))
        lib.initializeSimulation()

        warmup = max(0, num_timesteps - 600)
        data_keys = [
            b"Time [s]",
            b"Aerodynamic Thrust [N]",
            b"Aerodynamic Power [W]",
            b"Aerodynamic Torque [Nm]",
            b"Aerodynamic Force in Hub Y_g Direction [N]",
            b"Aerodynamic Force in Hub Z_g Direction [N]",
        ]
        col_names = ["Time", "Thrust", "Power", "Torque", "Thrust_y", "Thrust_z"]
        buf = {k: [] for k in col_names}

        for step in range(num_timesteps):
            lib.advanceTurbineSimulation()
            if step >= warmup:
                for col, qkey in zip(col_names, data_keys):
                    buf[col].append(float(lib.getCustomData_at_num(qkey, 0, 0)))

        lib.unload()

        df = pd.DataFrame(buf)
        rho, R = 1.225, 0.127
        D, n_rps = 2 * R, rpm / 60.0

        result = {
            "geometry_idx": geometry_idx,
            "RPM": rpm,
            "WIND_SPEED": wind_speed,
            "ANGLE": angle,
            "THRUST": float(df["Thrust"].mean() * -1),
            "POWER": float(df["Power"].mean() * -1),
            "TORQUE": float(df["Torque"].mean() * -1),
            "THRUST_Y": float(df["Thrust_y"].mean() * -1),
            "THRUST_Z": float(df["Thrust_z"].mean() * -1),
            "elapsed_sec": time.perf_counter() - t0,
            "error": None,
        }
        result["Ct"] = result["THRUST"] / (rho * n_rps ** 2 * D ** 4)
        result["Cp"] = result["POWER"] * 1000 / (rho * n_rps ** 3 * D ** 5)
        if wind_speed != 0:
            J = wind_speed / (n_rps * D)
            result["eta"] = J * (result["Ct"] / result["Cp"])
        else:
            result["eta"] = 0.0

        return result

    except Exception as e:
        return {
            "geometry_idx": geometry_idx,
            "RPM": rpm,
            "WIND_SPEED": wind_speed,
            "ANGLE": angle,
            "elapsed_sec": time.perf_counter() - t0,
            "error": str(e),
        }
    finally:
        os.chdir(orig_cwd)


# ═══════════════════════════════════════════════
#  主调度器
# ═══════════════════════════════════════════════

_RESULT_COLS = [
    "geometry_idx", "RPM", "WIND_SPEED", "ANGLE",
    "THRUST", "POWER", "TORQUE", "THRUST_Y", "THRUST_Z",
    "Ct", "Cp", "eta", "elapsed_sec",
]


def run_geometry_batch(
    geometry_list: np.ndarray,
    *,
    dll_file: str,
    sim_folder: str,
    rpm: float = 6000,
    wind_speed: float = 10,
    angle: float = 85,
    num_timesteps: int = 1800,
    device: str = "GPU",
    max_workers: int = 1,
    omp_threads: int = 4,
    start_idx: int = 0,
    output_pkl: str = "geometry_results.pkl",
    output_csv: str = "geometry_results.csv",
    save_every: int = 1,
) -> List[Dict]:
    """批量采集多几何仿真数据 (自动断点续传 + 去重)。"""
    device_id = 1 if device.upper() == "GPU" else 0
    if device.upper() == "GPU":
        max_workers = 1
        log.info("GPU 模式: 强制 max_workers=1 (OpenCL 单实例)")

    all_results = _load_checkpoint(output_pkl)
    completed = {r["geometry_idx"] for r in all_results if r.get("error") is None}
    pending = [i for i in range(start_idx, len(geometry_list)) if i not in completed]

    if not pending:
        log.info("所有几何已完成!")
        return all_results

    log.info(f"总: {len(geometry_list)}, 已完成: {len(completed)}, 待执行: {len(pending)}, workers: {max_workers}")

    if max_workers <= 1:
        results = _run_sequential(
            geometry_list, pending, dll_file, sim_folder,
            rpm, wind_speed, angle, num_timesteps, device_id, omp_threads,
            all_results, output_pkl, output_csv, save_every,
        )
    else:
        results = _run_parallel(
            geometry_list, pending, dll_file, sim_folder,
            rpm, wind_speed, angle, num_timesteps, device_id, omp_threads,
            max_workers, all_results, output_pkl, output_csv, save_every,
        )
    return results


def _run_sequential(
    geometry_list, pending, dll_file, sim_folder,
    rpm, wind_speed, angle, num_timesteps, device_id, omp_threads,
    all_results, output_pkl, output_csv, save_every,
):
    done, total = 0, len(pending)
    t_start = time.perf_counter()

    for geo_idx in pending:
        _modify_bld(
            os.path.join(sim_folder, "Baseline_Blade_Turb", "Aero", "Baseline_Blade.bld"),
            geometry_list[geo_idx],
        )
        result = _run_one_geometry(
            sim_folder, dll_file, geometry_list[geo_idx], geo_idx,
            rpm, wind_speed, angle, num_timesteps, device_id, omp_threads,
        )
        all_results.append(result)
        done += 1
        _log_progress(done, total, geo_idx, result, t_start)
        if done % save_every == 0:
            _save_results(all_results, output_pkl, output_csv)

    _save_results(all_results, output_pkl, output_csv)
    return all_results


def _run_parallel(
    geometry_list, pending, dll_file, sim_folder,
    rpm, wind_speed, angle, num_timesteps, device_id, omp_threads,
    max_workers, all_results, output_pkl, output_csv, save_every,
):
    tmp_root = tempfile.mkdtemp(prefix="qblade_parallel_")
    log.info(f"临时工作目录: {tmp_root}")

    per_worker_omp = max(1, omp_threads // max_workers)
    worker_dirs = {w: clone_sim_tree(sim_folder, w, tmp_root) for w in range(max_workers)}
    log.info(f"已创建 {max_workers} 个 worker 文件树副本")

    done, total = 0, len(pending)
    t_start = time.perf_counter()

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}
            for i, geo_idx in enumerate(pending):
                fut = pool.submit(
                    _run_one_geometry,
                    worker_dirs[i % max_workers], dll_file,
                    geometry_list[geo_idx], geo_idx,
                    rpm, wind_speed, angle, num_timesteps,
                    device_id, per_worker_omp,
                )
                future_map[fut] = geo_idx

            for fut in as_completed(future_map):
                geo_idx = future_map[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"geometry_idx": geo_idx, "error": str(e)}

                all_results.append(result)
                done += 1
                _log_progress(done, total, geo_idx, result, t_start)
                if done % save_every == 0:
                    _save_results(all_results, output_pkl, output_csv)
    finally:
        _save_results(all_results, output_pkl, output_csv)
        log.info(f"清理临时目录: {tmp_root}")
        shutil.rmtree(tmp_root, ignore_errors=True)

    return all_results


def _log_progress(done: int, total: int, geo_idx: int, result: Dict, t_start: float):
    elapsed = time.perf_counter() - t_start
    eta = elapsed / done * (total - done) if done else 0
    status = "OK" if result.get("error") is None else "ERR"
    log.info(
        f"[{done}/{total}] geo_{geo_idx} "
        f"{result.get('elapsed_sec', 0):.0f}s | {status} | ETA: {eta / 60:.0f}min"
    )


# ═══════════════════════════════════════════════
#  结果 I/O
# ═══════════════════════════════════════════════

def _load_checkpoint(pkl_path: str) -> List[Dict]:
    if not os.path.exists(pkl_path):
        return []
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            log.info(f"已加载 {len(data)} 条已有结果 from {pkl_path}")
            return data
    except Exception as e:
        log.warning(f"加载 {pkl_path} 失败: {e}")
    return []


def _save_results(results: List[Dict], pkl_path: str, csv_path: str) -> None:
    seen: Dict[int, Dict] = {}
    for r in results:
        idx = r.get("geometry_idx")
        if idx is not None:
            seen[idx] = r
    deduped = list(seen.values())

    with open(pkl_path, "wb") as f:
        pickle.dump(deduped, f)

    ok = [r for r in deduped if r.get("error") is None]
    if ok:
        df = pd.DataFrame(ok)
        cols = [c for c in _RESULT_COLS if c in df.columns]
        df[cols].sort_values("geometry_idx").to_csv(csv_path, index=False)


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

def main():
    from project_config import (
        DLL_FILE, SIM_FOLDER, GEOMETRY_NPY,
        DEFAULT_RPM, DEFAULT_WIND, DEFAULT_ANGLE, NUM_TIMESTEPS,
    )

    parser = argparse.ArgumentParser(description="QBlade 几何级并行数据采集")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--workers", type=int, default=1, help="CPU 并行 worker 数")
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--start", type=int, default=0, help="起始几何索引")
    parser.add_argument("--end", type=int, default=None, help="结束几何索引 (不含)")
    parser.add_argument("--rpm", type=float, default=DEFAULT_RPM)
    parser.add_argument("--wind", type=float, default=DEFAULT_WIND)
    parser.add_argument("--angle", type=float, default=DEFAULT_ANGLE)
    parser.add_argument("--timesteps", type=int, default=NUM_TIMESTEPS)
    parser.add_argument("--output", default="geometry_results.pkl")
    parser.add_argument("--geometry-npy", default=GEOMETRY_NPY)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    geometry_list = np.load(args.geometry_npy)
    if args.end:
        geometry_list = geometry_list[:args.end]

    log.info(f"加载 {len(geometry_list)} 个几何 from {args.geometry_npy}")
    log.info(f"工况: RPM={args.rpm}, WIND={args.wind}, ANGLE={args.angle}")
    log.info(f"设备: {args.device}, Workers: {args.workers}")

    run_geometry_batch(
        geometry_list,
        dll_file=DLL_FILE,
        sim_folder=SIM_FOLDER,
        rpm=args.rpm,
        wind_speed=args.wind,
        angle=args.angle,
        num_timesteps=args.timesteps,
        device=args.device,
        max_workers=args.workers,
        omp_threads=args.omp_threads,
        start_idx=args.start,
        output_pkl=args.output,
        output_csv=args.output.replace(".pkl", ".csv"),
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
