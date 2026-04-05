"""螺旋桨气动外形优化 Gymnasium 环境 (v4 — 硬约束)

PPO 智能体通过调整 8 个 B 样条控制点优化桨叶几何，
配平求解器自动求解迎角和转速以满足力矩平衡，
目标是最大化巡航品质因数 FM = mg·V / P_total。

v4 关键改进 (硬约束模式):
    - CP_BOUNDS 由训练数据反推 (最小二乘拟合 + 5% 边距)
    - 配平 RPM 限制在训练范围 [4000, 6500]
    - 配平 ANGLE 限制在训练范围 [82°, 88°] → alpha ∈ [2°, 8°]
    - 保留 OOD 软惩罚接口 (可选, 默认关闭)
    - 支持 constrain_to_data=True 硬约束模式
"""

import math
import sys
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from quadcopter_trim_solver import (
    QuadcopterTrimSolver,
    dummy_fuselage_aero,
    dummy_rotor_aero,
)


# B 样条插值: 8 控制点 → 22 chord + 22 twist (44 sections)
_GEOMETRY_R = np.array([
    0.0201254, 0.0252054, 0.0315579, 0.0379079, 0.0442579, 0.0506079,
    0.0569579, 0.0633079, 0.0696579, 0.0760079, 0.0823586, 0.0887148,
    0.0950769, 0.1014424, 0.107804,  0.1141792, 0.1173761, 0.1205864,
    0.1238169, 0.1255,    0.12625,   0.127,
])
_R_NORM = _GEOMETRY_R / 0.127
_CHORD_KNOTS = np.concatenate(([0] * 3, [0.3984874, 0.89904882], [1] * 3))
_TWIST_KNOTS = np.concatenate(([0] * 3, [0.2, 0.89904882], [1] * 3))


def cp_to_sections(cp: np.ndarray) -> np.ndarray:
    """将 8 个 B 样条控制点展开为 44 个 section 值 (22 chord + 22 twist)。"""
    chord_cp = cp[:4]
    twist_cp = cp[4:8]
    chord_vals = BSpline(_CHORD_KNOTS, chord_cp, 3)(_R_NORM)
    twist_vals = BSpline(_TWIST_KNOTS, twist_cp, 3)(_R_NORM)
    return np.concatenate([chord_vals, twist_vals])


def _make_geometry_aware_dummy_funcs(cp: np.ndarray):
    """根据几何控制点生成 dummy 旋翼气动力函数。"""
    chord_mean = np.mean(cp[:4])
    twist_mean = np.mean(cp[4:8])
    twist_std = np.std(cp[4:8])

    chord_ref = 0.026
    solidity_ratio = chord_mean / chord_ref
    C_T0 = 0.12 * np.clip(solidity_ratio, 0.3, 2.5)
    twist_opt = 35.0
    twist_penalty = 1.0 - 0.3 * ((twist_mean - twist_opt) / 40.0) ** 2
    twist_uniformity = 1.0 / (1.0 + 0.5 * twist_std / 20.0)
    k_J = 0.15 * (1.0 / twist_uniformity)
    C_Q = 0.008 * solidity_ratio * (1.0 + 0.2 * (1.0 - twist_penalty))

    rho = 1.225
    D_prop = 0.254
    R_prop = D_prop / 2

    def rotor_aero(V, alpha_rad, omega_rpm):
        n = omega_rpm / 60.0
        if abs(n) < 1e-6:
            return 0.0, 0.0, 0.0
        V_axial = V * math.sin(alpha_rad) if abs(alpha_rad) > 1e-6 else 0.0
        V_inplane = V * math.cos(alpha_rad)
        J = V_axial / (n * D_prop) if abs(n * D_prop) > 1e-6 else 0.0
        C_T = max(C_T0 * twist_penalty - k_J * J ** 2, 0.01)
        T_p = C_T * rho * n ** 2 * D_prop ** 4
        omega_rad = omega_rpm * 2 * math.pi / 60.0
        mu = V_inplane / (omega_rad * R_prop) if abs(omega_rad * R_prop) > 1e-6 else 0.0
        H_p = 0.05 * T_p * mu
        M_p_y = 0.02 * T_p * R_prop * mu
        return T_p, H_p, M_p_y

    def rotor_aero_with_torque(V, alpha_rad, omega_rpm):
        T_p, H_p, M_p_y = rotor_aero(V, alpha_rad, omega_rpm)
        n = omega_rpm / 60.0
        Q_p = C_Q * rho * abs(n) * n * D_prop ** 5 if abs(n) > 1e-6 else 0.0
        return T_p, H_p, M_p_y, Q_p

    return rotor_aero, rotor_aero_with_torque


# ============================================================================
# 两套 CP 边界: 宽松 (向后兼容) / 数据驱动 (硬约束)
# ============================================================================

CP_BOUNDS_WIDE = np.array([
    [0.010, 0.030],   # chord cp 0
    [0.018, 0.050],   # chord cp 1
    [0.008, 0.024],   # chord cp 2
    [0.003, 0.012],   # chord cp 3
    [30.0,  80.0],    # twist cp 0
    [10.0,  32.0],    # twist cp 1
    [8.0,   26.0],    # twist cp 2
    [6.0,   20.0],    # twist cp 3
], dtype=np.float64)

CP_BOUNDS_DATA = np.array([
    [0.013599, 0.026293],   # chord cp 0  (训练数据 + 5% 边距)
    [0.022239, 0.043821],   # chord cp 1
    [0.010052, 0.019808],   # chord cp 2
    [0.004783, 0.009199],   # chord cp 3
    [37.4285,  72.6715],    # twist cp 0
    [13.4458,  26.4942],    # twist cp 1
    [11.2559,  22.0639],    # twist cp 2
    [8.4469,   16.5578],    # twist cp 3
], dtype=np.float64)

# 训练数据工况范围
DATA_RPM_BOUNDS = (4000.0, 6500.0)
DATA_ALPHA_BOUNDS_DEG = (2.0, 8.0)    # ANGLE 82-88° → alpha 2-8°


class PropellerDesignEnv(gym.Env):
    """螺旋桨几何优化 RL 环境 (v4)。

    State (13-dim):  [cp_0..cp_7, alpha_deg, Omega1, Omega2, power, step_ratio]
    Action (8-dim):  delta control points in [-1, 1], mapped to geometry bounds
    Reward:          FM = mg·V / P_total

    constrain_to_data=True (推荐):
        CP 边界来自训练数据, RPM 和 ANGLE 硬约束到训练范围,
        确保所有 DNN 查询都在可信域内。
    constrain_to_data=False:
        宽松边界, 兼容旧版行为。
    """

    metadata = {"render_modes": []}

    ALPHA_RANGE = (0.0, 25.0)        # deg (观测归一化用)
    OMEGA_RANGE = (1000.0, 12000.0)  # RPM (观测归一化用)
    POWER_RANGE = (0.0, 500.0)       # W

    MAX_CONSECUTIVE_FAIL = 3

    _TRAIN_BOUNDS = None

    def __init__(
        self,
        V: float = 10.0,
        max_steps: int = 50,
        no_improve_limit: int = 10,
        delta_scale: float = 0.10,
        mass: float = 3.5,
        reward_beta: float = 0.1,
        ood_penalty: float = 0.0,
        constrain_to_data: bool = True,
        cfg: Config = None,
        use_dnn: bool = True,
    ):
        super().__init__()

        self.V = V
        self.max_steps = max_steps
        self.no_improve_limit = no_improve_limit
        self.delta_scale = delta_scale
        self.mass = mass
        self.weight = mass * 9.81
        self.reward_beta = reward_beta
        self.ood_penalty = ood_penalty
        self.constrain_to_data = constrain_to_data
        self.use_dnn = use_dnn

        self.cfg = cfg or Config()

        if use_dnn:
            self._load_dnn_model()

        if (ood_penalty > 0 or constrain_to_data) and use_dnn:
            self._ensure_train_bounds()

        self.solver = QuadcopterTrimSolver(mass=mass)

        # 选择 CP 边界
        self.CP_BOUNDS = CP_BOUNDS_DATA if constrain_to_data else CP_BOUNDS_WIDE
        self.n_cp = len(self.CP_BOUNDS)
        self.cp_lo = self.CP_BOUNDS[:, 0]
        self.cp_hi = self.CP_BOUNDS[:, 1]
        self.cp_range = self.cp_hi - self.cp_lo
        self.max_delta = self.cp_range * self.delta_scale

        # 配平约束
        if constrain_to_data:
            self._trim_rpm_bounds = DATA_RPM_BOUNDS
            self._trim_alpha_bounds = DATA_ALPHA_BOUNDS_DEG
        else:
            self._trim_rpm_bounds = None
            self._trim_alpha_bounds = None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_cp,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        self.cp = None
        self.step_count = 0
        self.best_reward_in_episode = -np.inf
        self.no_improve_count = 0
        self.last_cruise_metric = 0.0

        self._warm_start = None
        self._consecutive_fail = 0
        self._last_trim_state = (5.0, 5000.0, 5000.0, 200.0)

    @classmethod
    def _ensure_train_bounds(cls):
        """从训练 CSV 加载各特征的 min/max (进程级缓存)。"""
        if cls._TRAIN_BOUNDS is not None:
            return
        cfg = Config()
        csv_path = os.path.join(cfg.processed_data_dir, "raw_data.csv")
        df = pd.read_csv(csv_path)
        sec_cols = [f"chord_{i}" for i in range(22)] + [f"twist_{i}" for i in range(22)]
        cls._TRAIN_BOUNDS = {
            "sec_lo": df[sec_cols].min().values.astype(np.float64),
            "sec_hi": df[sec_cols].max().values.astype(np.float64),
            "rpm_lo": float(df["RPM"].min()),
            "rpm_hi": float(df["RPM"].max()),
            "angle_lo": float(df["ANGLE"].min()),
            "angle_hi": float(df["ANGLE"].max()),
        }

    def _compute_ood_score(self, sections: np.ndarray,
                           alpha_deg: float, omega1: float, omega2: float) -> float:
        """计算 OOD 分数 [0, ∞), 0 = 完全在分布内。"""
        if self._TRAIN_BOUNDS is None or self.ood_penalty <= 0:
            return 0.0
        b = self._TRAIN_BOUNDS

        sec_range = b["sec_hi"] - b["sec_lo"] + 1e-10
        sec_violation = (np.maximum(0, sections - b["sec_hi"])
                         + np.maximum(0, b["sec_lo"] - sections))
        geo_ood = float(np.mean(sec_violation / sec_range))

        rpm_range = b["rpm_hi"] - b["rpm_lo"]
        rpm_viol = (max(0, omega1 - b["rpm_hi"]) + max(0, b["rpm_lo"] - omega1)
                    + max(0, omega2 - b["rpm_hi"]) + max(0, b["rpm_lo"] - omega2))
        rpm_ood = rpm_viol / (2.0 * rpm_range + 1e-10)

        angle = 90.0 - alpha_deg
        angle_range = b["angle_hi"] - b["angle_lo"]
        angle_ood = (max(0, angle - b["angle_hi"])
                     + max(0, b["angle_lo"] - angle)) / (angle_range + 1e-10)

        return 0.6 * geo_ood + 0.2 * rpm_ood + 0.2 * angle_ood

    def _load_dnn_model(self):
        """加载 DNN 代理模型 (单点推理用 CPU)。"""
        from adapter import (
            load_predictor,
            make_rotor_aero_func,
            make_rotor_aero_func_with_torque,
        )
        self._dnn_parts = load_predictor(self.cfg, force_cpu=True)
        self._make_trim = make_rotor_aero_func
        self._make_torque = make_rotor_aero_func_with_torque

    def _build_aero_funcs(self, cp: np.ndarray):
        """根据当前几何构建气动力回调。"""
        if not self.use_dnn:
            return _make_geometry_aware_dummy_funcs(cp)

        sections = cp_to_sections(cp)
        model, scaler_X, scaler_Y, device = self._dnn_parts
        geo_list = sections.tolist()
        trim_func = self._make_trim(
            self.cfg, model, scaler_X, scaler_Y, device, geometry=geo_list,
        )
        torque_func = self._make_torque(
            self.cfg, model, scaler_X, scaler_Y, device, geometry=geo_list,
        )
        return trim_func, torque_func

    def _run_trim(self, cp: np.ndarray):
        """执行多初始值配平 (带约束) 并返回结果。"""
        trim_func, torque_func = self._build_aero_funcs(cp)
        self.solver.set_rotor_aero(trim_func)
        result = self.solver.solve_multi_start(
            self.V,
            x0_hint=self._warm_start,
            rpm_bounds=self._trim_rpm_bounds,
            alpha_bounds_deg=self._trim_alpha_bounds,
        )

        if not result.converged:
            return result, None, None, None

        self._warm_start = np.array([
            math.radians(result.alpha_deg),
            result.omega_front,
            result.omega_rear,
        ])

        alpha_rad = math.radians(result.alpha_deg)
        _, _, _, Q1 = torque_func(self.V, alpha_rad, result.omega_front)
        _, _, _, Q2 = torque_func(self.V, alpha_rad, result.omega_rear)

        omega1_rad = result.omega_front * 2.0 * math.pi / 60.0
        omega2_rad = result.omega_rear * 2.0 * math.pi / 60.0
        power = 2.0 * (abs(omega1_rad * Q1) + abs(omega2_rad * Q2))

        ad = result.aero_details
        T_total = ad.get("total_thrust", 0.0)
        D_f = ad.get("D_f", 1e-8)
        L_f = ad.get("L_f", 0.0)

        eta_prop = (T_total * self.V) / max(power, 1e-8)
        fm = self.weight * self.V / max(power, 1e-8)

        sections = cp_to_sections(cp) if self.use_dnn else np.zeros(44)
        ood = self._compute_ood_score(
            sections, result.alpha_deg, result.omega_front, result.omega_rear,
        )

        details = {
            "power": power, "eta_prop": eta_prop, "fm": fm,
            "cruise_metric": fm,
            "ood_score": ood,
            "T_total": T_total, "D_f": D_f, "L_f": L_f,
            "Q1": Q1, "Q2": Q2,
        }

        self._last_trim_state = (
            result.alpha_deg, result.omega_front,
            result.omega_rear, power,
        )
        return result, power, fm, details

    def _normalize_obs(self, cp, alpha_deg, omega1, omega2, power):
        """将原始物理量归一化到 [-1, 1]。"""
        cp_norm = 2.0 * (cp - self.cp_lo) / self.cp_range - 1.0
        a_norm = 2.0 * (alpha_deg - self.ALPHA_RANGE[0]) / (self.ALPHA_RANGE[1] - self.ALPHA_RANGE[0]) - 1.0
        o1_norm = 2.0 * (omega1 - self.OMEGA_RANGE[0]) / (self.OMEGA_RANGE[1] - self.OMEGA_RANGE[0]) - 1.0
        o2_norm = 2.0 * (omega2 - self.OMEGA_RANGE[0]) / (self.OMEGA_RANGE[1] - self.OMEGA_RANGE[0]) - 1.0
        p_norm = 2.0 * power / self.POWER_RANGE[1] - 1.0
        step_norm = 2.0 * self.step_count / self.max_steps - 1.0

        obs = np.concatenate([
            cp_norm,
            [a_norm, o1_norm, o2_norm, p_norm, step_norm],
        ])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.best_reward_in_episode = -np.inf
        self.no_improve_count = 0
        self.last_cruise_metric = 0.0
        self._warm_start = None
        self._consecutive_fail = 0
        self._last_trim_state = (5.0, 5000.0, 5000.0, 200.0)

        for _attempt in range(20):
            self.cp = self.np_random.uniform(self.cp_lo, self.cp_hi)
            result, power, cruise_metric, details = self._run_trim(self.cp)
            if result.converged and power is not None:
                self.last_cruise_metric = cruise_metric
                obs = self._normalize_obs(
                    self.cp, result.alpha_deg, result.omega_front,
                    result.omega_rear, power,
                )
                info = {"converged": True}
                if details:
                    info.update(details)
                return obs, info

        self.last_cruise_metric = 0.0
        obs = self._normalize_obs(self.cp, *self._last_trim_state)
        return obs, {"converged": False}

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        delta = action * self.max_delta
        self.cp = np.clip(self.cp + delta, self.cp_lo, self.cp_hi)

        result, power, cruise_metric, details = self._run_trim(self.cp)

        if not result.converged or power is None:
            self._consecutive_fail += 1
            reward = -1.0
            obs = self._normalize_obs(self.cp, *self._last_trim_state)
            terminated = self._consecutive_fail >= self.MAX_CONSECUTIVE_FAIL
            truncated = self.step_count >= self.max_steps
            info = {"converged": False, "cruise_metric": 0.0}
            return obs, reward, terminated, truncated, info

        self._consecutive_fail = 0
        improvement = cruise_metric - self.last_cruise_metric
        ood = details.get("ood_score", 0.0) if details else 0.0
        reward = improvement + self.reward_beta * cruise_metric - self.ood_penalty * ood
        self.last_cruise_metric = cruise_metric

        if reward > self.best_reward_in_episode:
            self.best_reward_in_episode = reward
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        terminated = self.no_improve_count >= self.no_improve_limit
        truncated = self.step_count >= self.max_steps

        obs = self._normalize_obs(
            self.cp, result.alpha_deg, result.omega_front,
            result.omega_rear, power,
        )
        info = {
            "converged": True,
            "step": self.step_count,
            "geometry": self.cp.copy(),
            "alpha_deg": result.alpha_deg,
            "omega_front": result.omega_front,
            "omega_rear": result.omega_rear,
        }
        if details:
            info.update(details)

        return obs, reward, terminated, truncated, info
