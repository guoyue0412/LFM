"""多进程并行数据生成：用 multiprocessing.Pool(N) 跑 SIMULATION，每进程独立工作目录。

设计要点
========
- QBlade SIL 用 cwd 解析 .sim/.bld/.qpr 相对路径，并发跑同一目录会互相覆写。
- 每个 worker 进程启动时：
    1. 用 PID 在 /tmp/lfm_w<pid>/ 建独立工作树
    2. QBlade_data/ 深拷贝（写操作集中在这里）
    3. code/、QBladeCE_2.0.8.6/ 符号链接共享（只读）
    4. chdir 到 <work>/code/Simulation_QBlade（config.py 用 os.getcwd() 解析）
- 任务粒度：1 个几何 = 1 个 task；几何数 > workers 时排队。
- QBlade GPU 路径在 propeller 工况下死锁/崩溃（见 RESEARCH_LOG #11），强制走 CPU(d0)。

输出格式与单进程版完全一致：
    {f'geometry_{idx}': {'geometry': np.ndarray, 'RPM*_Wind*_Angle*': pd.DataFrame, ...}}

典型用法
========
    # baseline × 16 worker（最小冒烟）
    python scripts/run_data_gen_parallel.py --workers 16 --tag para_smoke

    # 多几何批量生成
    python scripts/run_data_gen_parallel.py --workers 16 --geometry-npy <path> \
        --tag run_$(date +%Y%m%d) --first-n-geoms 32

注意
====
- 每 worker 占用：QBlade SIL 32 thread + 80MB 工作目录（RAM tmpfs OK）
- node6 推荐：--workers 16 → 占 16×32=512 thread > 160 核（OK，超额订阅 QBlade 内部 OpenMP 会自适应）
            实际产能 = 16 个几何 同时跑，每个仍然 ~25 min / 几何
"""

import argparse
import importlib
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

LFM_ROOT = Path(__file__).resolve().parent.parent
SUBMODULE_ROOT = LFM_ROOT / "Propeller_project-main"
SUB_CODE_ROOT = SUBMODULE_ROOT / "code" / "Simulation_QBlade"
SUB_DATA_ROOT = SUBMODULE_ROOT / "QBlade_data"
SUB_LIB_ROOT = SUBMODULE_ROOT / "QBladeCE_2.0.8.6"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workers", type=int, default=16,
                   help="并行进程数；node6 默认 16；node1 推荐 8")
    p.add_argument("--omp-threads", type=int, default=8,
                   help="每 worker 内部 OMP/MKL/BLAS 线程数；默认 8。"
                        "workers × omp-threads 应 ≤ 物理核心数；"
                        "node6 (160 核) 推荐 16×8=128，留 32 核给系统/QBlade Qt helper")
    p.add_argument("--device", choices=["CPU"], default="CPU",
                   help="强制 CPU 模式（GPU 路径在 propeller 工况下 hang，详见 RESEARCH_LOG #11）")
    p.add_argument("--geometry-npy", type=str, default=None,
                   help="形状 (N, 22, 3) 的几何 numpy 文件；省略则只跑 baseline 1 个几何")
    p.add_argument("--first-n-geoms", type=int, default=None,
                   help="只跑前 N 个几何，用于产能基线测试")
    p.add_argument("--num-timesteps", type=int, default=None,
                   help="仿真步数，默认取 config.number_of_timesteps（1000）")
    p.add_argument("--out-dir", type=str, default=str(LFM_ROOT / "data_for_train" / "data"),
                   help="输出 pkl 目录；data.py 默认扫这里")
    p.add_argument("--tag", type=str, default="parallel",
                   help="输出 pkl 文件名 tag")
    p.add_argument("--worktree-root", type=str, default="/tmp",
                   help="worker 工作目录的父目录；node6 上 /tmp 是 tmpfs RAM-backed")
    p.add_argument("--keep-worktree", action="store_true",
                   help="跑完不删 /tmp/lfm_w<pid>，便于调试")
    return p.parse_args()


def setup_worker_workdir(worktree_root: str) -> Path:
    """初始化当前 worker 进程的工作目录，返回 cwd 应当切到的位置。

    工作目录结构：
        <worktree_root>/lfm_w<pid>/
            ├── code/                   ← 深拷贝（仅 ~164KB；worker 要 chdir 进去）
            ├── QBladeCE_2.0.8.6/       → 符号链接（read-only，QBlade SIL 入口）
            └── QBlade_data/            ← 深拷贝（每 worker 独立写 .sim/.qpr）

    重要：code/ 必须深拷贝（不能 symlink），否则 chdir(symlink) 后
    os.getcwd() 会跟随到真实路径（submodule 原位），config.py 用 os.getcwd()
    解析的所有 file_path 都会指回 source 目录，worker 之间互相冲突。
    """
    pid = os.getpid()
    work_root = Path(worktree_root) / f"lfm_w{pid}"
    work_root.mkdir(parents=True, exist_ok=True)

    # 共享只读：QBladeCE_2.0.8.6/（worker 不会 chdir 进去，symlink 安全）
    lib_link = work_root / "QBladeCE_2.0.8.6"
    if not lib_link.exists():
        lib_link.symlink_to(SUB_LIB_ROOT)

    # 必须独立：code/（worker 要 chdir 到 code/Simulation_QBlade）
    code_dir = work_root / "code"
    if not code_dir.exists():
        shutil.copytree(SUBMODULE_ROOT / "code", code_dir,
                        symlinks=False, ignore=shutil.ignore_patterns("__pycache__"))

    # 必须独立：QBlade_data/（.sim/.bld/.qpr 写入隔离）
    data_dir = work_root / "QBlade_data"
    if not data_dir.exists():
        shutil.copytree(SUB_DATA_ROOT, data_dir)

    return work_root / "code" / "Simulation_QBlade"


def worker_init(worktree_root: str, omp_threads: int) -> None:
    """Pool worker 启动钩子：建工作目录、chdir、注入 sys.path、import 子模块。

    在 fork 子进程中执行（每个 worker 进程仅一次）。

    限制每 worker 内部并行度（OMP/BLAS/MKL），避免多 worker × 多线程
    在 N 核机器上 over-subscription：实测 4 worker × 32 OMP 线程
    + ~200 Qt/OpenCL helper 线程 = 900+ runnable thread → load avg
    ~200, 单几何 wall clock 从 5min 拖到 20+min。
    """
    # 必须在 import QBlade SIL 之前设置，否则 .so 启动时已读取环境
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["OMP_THREAD_LIMIT"] = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(omp_threads)
    os.environ["QT_THREAD_POOL_MAX_THREAD_COUNT"] = str(omp_threads)

    work_cwd = setup_worker_workdir(worktree_root)
    os.chdir(work_cwd)
    sys.path.insert(0, str(work_cwd))

    # 子模块 import 必须在 chdir 之后，因为 config.py 用 os.getcwd() 解析路径
    global _config, _SIMULATION
    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config as _cfg  # noqa: E402
    from class_sim.simulation import SIMULATION as _SIM  # noqa: E402
    _config = _cfg
    _SIMULATION = _SIM
    print(f"[worker {os.getpid()}] ready, cwd={os.getcwd()}", flush=True)


def worker_run(task: tuple) -> tuple:
    """单 worker 处理一个几何：返回 (geom_idx, label, all_simulation_data 或 None)。

    Pool.map 会自动分发 task 到空闲 worker。
    """
    geom_idx, geom_array, num_timesteps, device = task
    pid = os.getpid()
    t0 = time.time()
    try:
        sim = _SIMULATION(
            file_path=_config.file_path,
            geometry_baseline=_config.geometry_baseline,
            device_type=device,
            number_of_timesteps=num_timesteps or _config.number_of_timesteps,
        )
        if not np.array_equal(geom_array, _config.geometry_baseline):
            sim.change_propeller_geometry(section_data=geom_array)
        sim.run_all_simulation()
        elapsed = time.time() - t0
        print(f"[worker {pid}] geom {geom_idx} 完成 ({elapsed:.1f}s, "
              f"{len(sim.all_simulation_data)-1} 工况)", flush=True)
        return (geom_idx, sim.all_simulation_data, None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[worker {pid}] geom {geom_idx} FAIL: {err}", flush=True)
        return (geom_idx, None, err)


def load_geometries(npy_path: str | None, baseline: np.ndarray,
                    first_n: int | None) -> list[np.ndarray]:
    """与单进程版相同的几何加载逻辑。"""
    if npy_path is None:
        return [baseline]
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        sys.exit(f"✖ 几何 npy 形状不对：{arr.shape}，期望 (N, 22, 3)")
    if first_n is not None:
        arr = arr[:first_n]
    return [arr[i] for i in range(arr.shape[0])]


def cleanup_worktrees(worktree_root: str, keep: bool) -> None:
    """跑完后清理所有 /tmp/lfm_w<pid>/，避免 tmpfs 占用堆积。"""
    if keep:
        return
    root = Path(worktree_root)
    for d in root.glob("lfm_w*"):
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f"[warn] 清理 {d} 失败：{e}", flush=True)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 在主进程加载 baseline geometry（仅用于 task 列表的几何数据）
    # 临时 chdir 一次，读完恢复
    saved_cwd = os.getcwd()
    os.chdir(SUB_CODE_ROOT)
    sys.path.insert(0, str(SUB_CODE_ROOT))
    import config as _main_cfg  # noqa: E402
    baseline = _main_cfg.geometry_baseline
    os.chdir(saved_cwd)

    geometries = load_geometries(args.geometry_npy, baseline, args.first_n_geoms)
    n_geoms = len(geometries)
    n_workers = min(args.workers, n_geoms) if n_geoms > 0 else 1

    print(f"▸ 几何数: {n_geoms}, workers: {n_workers}, device: {args.device}, "
          f"timesteps: {args.num_timesteps or '<default>'}")
    print(f"▸ 工作目录父级: {args.worktree_root}/lfm_w<pid>/")
    print(f"▸ submodule: {SUBMODULE_ROOT}")

    tasks = [(idx, geom, args.num_timesteps, args.device)
             for idx, geom in enumerate(geometries)]

    # 用 fork 上下文（子进程继承 LD_LIBRARY_PATH 与 sys.path）
    # spawn 也可以但启动慢、需要重新 import；fork 在 Linux 上更高效
    ctx = mp.get_context("fork")

    t_start = time.time()
    aggregated: dict[str, dict] = {}
    failed: list[tuple] = []

    try:
        with ctx.Pool(processes=n_workers,
                      initializer=worker_init,
                      initargs=(args.worktree_root, args.omp_threads)) as pool:
            for (geom_idx, result, err) in pool.imap_unordered(worker_run, tasks):
                if result is not None:
                    aggregated[f"geometry_{geom_idx}"] = result
                else:
                    failed.append((geom_idx, err))
                done = len(aggregated) + len(failed)
                print(f"  进度: {done}/{n_geoms} "
                      f"(成功 {len(aggregated)}, 失败 {len(failed)})", flush=True)
    finally:
        cleanup_worktrees(args.worktree_root, args.keep_worktree)

    total = time.time() - t_start
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_pkl = out_dir / f"data_{args.tag}_{stamp}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(aggregated, f)

    print()
    print(f"✅ 总用时 {total:.1f}s（壁钟）={total/60:.1f}min")
    print(f"   平均每几何 {total/max(n_geoms,1):.1f}s（壁钟）")
    print(f"   理论加速比 ≈ {n_workers}× vs 单进程")
    print(f"   输出 pkl: {out_pkl} ({len(aggregated)} 个几何)")
    if failed:
        print(f"   ⚠ {len(failed)} 个几何失败：{[i for i,_ in failed]}")
    print(f"   下一步: cd {LFM_ROOT} && python data.py")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
