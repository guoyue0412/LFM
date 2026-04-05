"""项目级配置 — QBlade 仿真路径、基线几何、工况参数

使用方法:
    from project_config import DLL_FILE, SIM_FOLDER, GEOMETRY_BASELINE
"""

from __future__ import annotations

import os
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── QBlade DLL (2.0.9.2) ───
DLL_FILE = os.path.join(_ROOT, "QBladeCE_2.0.9.2", "QBladeCE_2.0.9.2.dll")

# ─── 仿真文件路径 ───
QBLADE_DATA = os.path.join(_ROOT, "Qblade_data")
SIM_FOLDER = os.path.join(QBLADE_DATA, "QBlade_sim")
BASE_SIM = os.path.join(SIM_FOLDER, "Baseline_Simulation.sim")
BLD_FILE = os.path.join(SIM_FOLDER, "Baseline_Blade_Turb", "Aero", "Baseline_Blade.bld")
QBR_FOLDER = os.path.join(QBLADE_DATA, "QBR_file")
PARAM_EXCEL = os.path.join(_ROOT, "code", "simulation_parameters", "Parameters.xlsx")

# ─── 几何数据 ───
GEOMETRY_NPY = os.path.join(_ROOT, "geometry_generated", "geometry_data.npy")
CONTROLPOINTS_NPY = os.path.join(_ROOT, "geometry_generated", "controlpoints_data.npy")

# 基线几何 (22 截面 × 3: [r, chord, twist])
GEOMETRY_BASELINE = np.array([
    [0.02013, 0.01660, 39.73000],
    [0.02521, 0.01792, 45.92000],
    [0.03156, 0.01931, 33.64134],
    [0.03791, 0.02043, 30.02153],
    [0.04426, 0.02127, 27.01212],
    [0.05061, 0.02182, 24.54728],
    [0.05696, 0.02209, 22.56118],
    [0.06331, 0.02207, 20.98798],
    [0.06966, 0.02176, 19.76186],
    [0.07601, 0.02116, 18.81699],
    [0.08236, 0.02026, 18.08746],
    [0.08871, 0.01905, 17.50708],
    [0.09508, 0.01755, 17.01010],
    [0.10144, 0.01573, 16.53066],
    [0.10781, 0.01361, 16.00307],
    [0.11418, 0.01118, 15.35930],
    [0.11738, 0.00983, 14.97223],
    [0.12059, 0.00841, 14.52908],
    [0.12382, 0.00689, 14.01946],
    [0.12540, 0.00612, 13.73489],
    [0.12604, 0.00586, 13.62725],
    [0.12700, 0.00550, 13.46483],
])

# ─── 默认工况 ───
NUM_TIMESTEPS = 1800
DEFAULT_RPM = 6000
DEFAULT_WIND = 10
DEFAULT_ANGLE = 85
