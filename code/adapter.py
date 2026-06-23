"""螺旋桨气动力预测模型 — 配平求解器适配层

将 PropellerPredictor (DNN) 包装为 QuadcopterTrimSolver 所需的
rotor_aero_func 回调签名:
    (V, alpha_rad, omega_rpm) -> (T_p, H_p, M_p_y)

坐标映射:
    配平求解器            QBlade/LFM
    ──────────           ──────────
    V  (m/s)        →    WIND
    alpha_rad (rad) →    ANGLE = 90° - degrees(alpha)
    omega_rpm (RPM) →    RPM

输出映射:
    LFM 输出             配平变量
    ──────────           ──────────
    Fx  (推力)       →    T_p
    Fz  (面内力)     →    H_p
    Torque           →    Q_p

    M_p_y (俯仰力矩) 由 DNN 不直接预测,
    采用经典桨叶元分析估算: M_p_y ≈ k_m · T_p · R · μ,
    其中 μ = V·cos(α) / (Ω·R) 为前进比。
"""

import math
import os
import warnings
from typing import Callable, List, Optional, Tuple

import joblib
import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from config import Config
from train import PropellerPredictor


def load_predictor(cfg: Config, model_path: str = None, force_cpu: bool = False):
    """加载训练好的模型和标准化器。

    Args:
        force_cpu: 为 True 时强制使用 CPU (单点推理场景下避免 GPU 传输开销)。

    Returns:
        (model, scaler_X, scaler_Y, device)
    """
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PropellerPredictor(cfg).to(device)

    if model_path is None:
        model_path = os.path.join(cfg.model_dir, "propeller_predictor.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    d = cfg.processed_data_dir
    scaler_X = joblib.load(os.path.join(d, "scaler_X.pkl"))
    scaler_Y = joblib.load(os.path.join(d, "scaler_Y.pkl"))

    return model, scaler_X, scaler_Y, device


def _estimate_mpy(T_p: float, V: float, alpha_rad: float,
                   omega_rpm: float, R: float = 0.127,
                   k_m: float = 0.02) -> float:
    """基于桨叶元理论估算单旋翼俯仰力矩 M_p_y。

    M_p_y ≈ k_m · T_p · R · μ
    其中 μ = V·cos(α) / (Ω·R) 为前进比 (advance ratio)。
    k_m ≈ 0.02 来源于典型 10" 螺旋桨风洞实测 (Gill & D'Andrea 2019)。
    """
    omega_rad = omega_rpm * 2.0 * math.pi / 60.0
    denom = abs(omega_rad * R)
    if denom < 1e-6:
        return 0.0
    mu = V * math.cos(alpha_rad) / denom
    return k_m * T_p * R * mu


def make_rotor_aero_func(
    cfg: Config,
    model: torch.nn.Module,
    scaler_X,
    scaler_Y,
    device: torch.device,
    geometry: Optional[List[float]] = None,
) -> Callable[[float, float, float], Tuple[float, float, float]]:
    """构建配平求解器所需的旋翼气动力回调函数。

    DNN 预测 Fx(推力) 和 Fz(面内力), M_p_y 由解析模型估算。

    Returns:
        rotor_aero(V, alpha_rad, omega_rpm) -> (T_p, H_p, M_p_y)
    """
    out_cols = cfg.output_columns
    fx_idx = out_cols.index("Fx")
    fz_idx = out_cols.index("Fz")
    geo_arr = np.asarray(geometry, dtype=np.float64) if geometry is not None else None
    n_features = scaler_X.n_features_in_

    def rotor_aero(V: float, alpha_rad: float, omega_rpm: float) -> Tuple[float, float, float]:
        angle_deg = 90.0 - math.degrees(alpha_rad)

        x = np.empty((1, n_features), dtype=np.float64)
        x[0, 0] = omega_rpm
        x[0, 1] = V
        x[0, 2] = angle_deg
        if geo_arr is not None:
            x[0, 3:] = geo_arr

        x_scaled = scaler_X.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            y_scaled = model(x_tensor).cpu().numpy()

        y = scaler_Y.inverse_transform(y_scaled)[0]

        T_p = float(y[fx_idx])
        H_p = float(y[fz_idx])
        M_p_y = _estimate_mpy(T_p, V, alpha_rad, omega_rpm)

        return T_p, H_p, M_p_y

    return rotor_aero


def make_rotor_aero_func_with_torque(
    cfg: Config,
    model: torch.nn.Module,
    scaler_X,
    scaler_Y,
    device: torch.device,
    geometry: Optional[List[float]] = None,
) -> Callable[[float, float, float], Tuple[float, float, float, float]]:
    """构建含扭矩输出的旋翼气动力回调，用于功率计算。

    DNN 预测 Fx, Fz, Torque; M_p_y 由解析模型估算。

    Returns:
        rotor_aero(V, alpha_rad, omega_rpm) -> (T_p, H_p, M_p_y, Q_p)
    """
    out_cols = cfg.output_columns
    fx_idx = out_cols.index("Fx")
    fz_idx = out_cols.index("Fz")
    tq_idx = out_cols.index("Torque")
    geo_arr_t = np.asarray(geometry, dtype=np.float64) if geometry is not None else None
    n_feat = scaler_X.n_features_in_

    def rotor_aero(V: float, alpha_rad: float, omega_rpm: float
                   ) -> Tuple[float, float, float, float]:
        angle_deg = 90.0 - math.degrees(alpha_rad)

        x = np.empty((1, n_feat), dtype=np.float64)
        x[0, 0] = omega_rpm
        x[0, 1] = V
        x[0, 2] = angle_deg
        if geo_arr_t is not None:
            x[0, 3:] = geo_arr_t

        x_scaled = scaler_X.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_scaled = model(x_tensor).cpu().numpy()
        y = scaler_Y.inverse_transform(y_scaled)[0]

        T_p = float(y[fx_idx])
        M_p_y = _estimate_mpy(T_p, V, alpha_rad, omega_rpm)
        return T_p, float(y[fz_idx]), M_p_y, float(y[tq_idx])

    return rotor_aero


def make_batch_predictor(
    cfg: Config,
    model: torch.nn.Module,
    scaler_X,
    scaler_Y,
    device: torch.device,
    geometry: Optional[List[float]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """构建批量预测函数，用于速度扫描后的分析。

    Args:
        同 make_rotor_aero_func

    Returns:
        predict(conditions) -> predictions
        conditions: shape (N, 3) — [RPM, WIND, ANGLE] per row
        predictions: shape (N, output_dim) — 原始物理量
    """
    def predict_batch(conditions: np.ndarray) -> np.ndarray:
        if geometry is not None:
            geo_arr = np.tile(geometry, (len(conditions), 1))
            x_raw = np.hstack([conditions, geo_arr])
        else:
            x_raw = conditions

        x_scaled = scaler_X.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            y_scaled = model(x_tensor).cpu().numpy()

        return scaler_Y.inverse_transform(y_scaled)

    return predict_batch
