"""class_sim — QBlade 螺旋桨仿真工具包

模块结构:
    config      : SimConfig, SimPaths, SimCondition 数据类
    sim_file    : .sim 模板读写
    geometry    : B-spline 插值 + .bld 文件修改
    runner      : 单次仿真执行器 (QBladeRunner, context-manager)
    batch       : 并行批量调度器 (GPU / CPU / Hybrid)
    simulation  : 向后兼容的 SIMULATION 类 (薄封装)
"""

from .config import SimCondition, SimConfig, SimPaths, load_conditions_from_excel
from .geometry import control_points_to_sections, modify_bld_file
from .runner import QBladeRunner
from .batch import BatchRunner, run_batch_cpu, run_batch_gpu, run_batch_hybrid
from .sim_file import generate_sim_files, collect_sim_files

__all__ = [
    "SimCondition",
    "SimConfig",
    "SimPaths",
    "load_conditions_from_excel",
    "control_points_to_sections",
    "modify_bld_file",
    "QBladeRunner",
    "BatchRunner",
    "run_batch_cpu",
    "run_batch_gpu",
    "run_batch_hybrid",
    "generate_sim_files",
    "collect_sim_files",
]
