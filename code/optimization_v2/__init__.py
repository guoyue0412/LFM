"""optimization_v2 — 前飞配平约束螺旋桨气动外形优化管线

基于 LLFVW-MoE 代理模型的四旋翼无人机前飞配平约束螺旋桨气动外形优化。
目标: min P_elec(x_g, V) / V

模块:
    config          : 全局配置
    geometry        : B-Spline 参数化 (8 CP → 44 sections)
    synthetic_data  : 合成数据生成 (模拟 LLFVW)
    moe_model       : MoE 气动代理模型
    train_moe       : MoE 训练
    fuselage        : 机身半经验气动模型
    trim_solver     : 前飞配平求解器
    power           : 功率与航程计算
    objective       : 优化目标函数
    cma_optimizer   : CMA-ES 优化器
    active_learning : 不确定度驱动主动补点
    evaluate        : 评估与可视化
    run             : 主入口
"""

from .config import OptConfig
from .geometry import cp_to_sections, validate_geometry
from .data_loader import load_excel_dataset, ColumnMapping, inspect_excel
from .moe_model import MoEPredictor
from .trim_solver import TrimSolverV2, TrimResultV2
from .power import compute_power, compute_range
from .objective import evaluate_design

__all__ = [
    "OptConfig",
    "cp_to_sections",
    "validate_geometry",
    "load_excel_dataset",
    "ColumnMapping",
    "inspect_excel",
    "MoEPredictor",
    "TrimSolverV2",
    "TrimResultV2",
    "compute_power",
    "compute_range",
    "evaluate_design",
]
