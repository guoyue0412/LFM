"""数据生成 wrapper：调用 Propeller_project-main 子模块的 SIMULATION 类批量产出 pkl。

设计要点
========
- 子模块 config.py 用 os.getcwd() 解析路径，因此必须先 chdir 到子模块目录再 import。
- 子模块顶层目录名含连字符（Propeller_project-main），无法直接作 Python 包；通过 sys.path 注入解决。
- 输出 pkl 结构对齐 data.py 的"格式 B"：
      {geometry_idx: {"geometry": np.ndarray(22,3),
                      "RPM{rpm}_Wind{wind}_Angle{angle}": pd.DataFrame, ...}}
  这样 LFM 顶层 python data.py 可以直接消费。
- 几何来源：默认 baseline（仅 1 个）；--geometry-npy 指定 (N, 22, 3) 文件做批量。

典型用法
========
    # 仅 baseline 几何 × 全工况（最小冒烟）
    python scripts/run_data_gen.py --device CPU --tag smoke

    # 批量多几何
    python scripts/run_data_gen.py \
        --geometry-npy Propeller_project-main.win.bak/geometry_generated/geometry_data.npy \
        --device CPU --num-timesteps 1000 --tag run_$(date +%Y%m%d)
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

LFM_ROOT = Path(__file__).resolve().parent.parent
SUB_PY_ROOT = LFM_ROOT / "Propeller_project-main" / "code" / "Simulation_QBlade"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometry-npy", type=str, default=None,
                   help="形状 (N, 22, 3) 的 numpy 文件；省略则只跑 baseline")
    p.add_argument("--device", choices=["CPU", "GPU"], default="CPU",
                   help="GPU 走 OpenCL（device_id=1，串行），CPU 串行 device_id=0")
    p.add_argument("--num-timesteps", type=int, default=None,
                   help="仿真时间步数；默认取子模块 config.number_of_timesteps")
    p.add_argument("--out-dir", type=str, default=str(LFM_ROOT / "data_for_train" / "data"),
                   help="输出 pkl 目录；data.py 默认扫描 data_for_train/data/")
    p.add_argument("--tag", type=str, default="linux",
                   help="输出文件名包含的 tag，便于区分批次")
    p.add_argument("--first-n-geoms", type=int, default=None,
                   help="只跑前 N 个几何（调试用）")
    return p.parse_args()


def setup_submodule_import() -> tuple:
    """切到子模块工作目录并 import 关键符号。

    返回 (config_module, SIMULATION_cls)。这里必须先 chdir 再 import，否则
    config.py 中 os.getcwd() 解析出的路径全错（指向 LFM 根）。
    """
    if not SUB_PY_ROOT.is_dir():
        sys.exit(f"✖ 子模块未初始化：{SUB_PY_ROOT}\n  先跑：git submodule update --init --recursive")
    os.chdir(SUB_PY_ROOT)
    sys.path.insert(0, str(SUB_PY_ROOT))
    import config  # noqa: E402
    from class_sim.simulation import SIMULATION  # noqa: E402
    return config, SIMULATION


def load_geometries(npy_path: str | None, baseline: np.ndarray, first_n: int | None) -> list[np.ndarray]:
    """返回几何列表；省略 npy 时只含 baseline。"""
    if npy_path is None:
        return [baseline]
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        sys.exit(f"✖ 几何 npy 形状不对：{arr.shape}，期望 (N, 22, 3)")
    if first_n is not None:
        arr = arr[:first_n]
    return [arr[i] for i in range(arr.shape[0])]


def run_one_geometry(config_module, SIMULATION_cls, geom: np.ndarray, args) -> dict:
    """对单个几何跑全工况，返回 all_simulation_data dict。"""
    sim = SIMULATION_cls(
        file_path=config_module.file_path,
        geometry_baseline=config_module.geometry_baseline,
        device_type=args.device,
        number_of_timesteps=args.num_timesteps or config_module.number_of_timesteps,
    )
    # 仅当几何与 baseline 有差异时才调用 change_propeller_geometry
    if not np.array_equal(geom, config_module.geometry_baseline):
        sim.change_propeller_geometry(section_data=geom)
    sim.run_all_simulation()
    return sim.all_simulation_data


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_module, SIMULATION_cls = setup_submodule_import()
    geometries = load_geometries(args.geometry_npy, config_module.geometry_baseline, args.first_n_geoms)
    print(f"▸ 几何数：{len(geometries)}，device={args.device}，"
          f"num_timesteps={args.num_timesteps or config_module.number_of_timesteps}")

    aggregated: dict[int, dict] = {}
    t_start = time.time()
    for idx, geom in enumerate(geometries):
        t_geo = time.time()
        print(f"  [{idx + 1}/{len(geometries)}] 几何 #{idx} 开跑 ...")
        aggregated[idx] = run_one_geometry(config_module, SIMULATION_cls, geom, args)
        print(f"  [{idx + 1}/{len(geometries)}] 完成，用时 {time.time() - t_geo:.1f}s")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_pkl = out_dir / f"data_{args.tag}_{stamp}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(aggregated, f)
    print(f"\n✅ 总用时 {time.time() - t_start:.1f}s，输出：{out_pkl}")
    print(f"   下一步：cd {LFM_ROOT} && python data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
