"""四旋翼无人机物理建模

基于公开文献的实验数据建立完整无人机参数模型，包括:
  1. 机体结构/质量特性
  2. 力臂几何 (前后旋翼纵向/垂向偏移)
  3. 机身外壳气动力模型 (阻力、升力、俯仰力矩)

主要参考 (详见 docs/uav_model_references.md):
  [R1]  Ye et al. (2021) Aerospace Sci. Tech. — 配平方程组来源
  [R2]  Pounds et al. (2010) Control Eng. Practice — X-4 Flyer 4 kg 四旋翼建模
  [R3]  Leishman (2006) Principles of Helicopter Aerodynamics — 旋翼力理论
  [R4]  Folk et al. (2023) RotorPy, arXiv:2306.04485 — 开源参数参考
  [R5]  Bouabdallah (2007) EPFL Ph.D. — 四旋翼设计建模
  [R8]  Russell et al. (2016) NASA/TM-2016-219310 — 多旋翼风洞实验
  [R9]  Theys & De Schutter (2020) Int. J. MAV — 四旋翼阻力实测
  [R10] Prudden et al. (2023) CEAS Aero. J. — 多旋翼自由落体阻力
  [R13] Bangura (2017) ANU Ph.D. — 四旋翼气动力学与控制
  [R14] Hoerner (1965) Fluid Dynamic Drag — 钝体阻力/升力经典数据

用法:
    from uav_model import QuadcopterConfig, fuselage_aero_model

    cfg = QuadcopterConfig()                   # 默认 3.5 kg 四旋翼
    aero_func = fuselage_aero_model(cfg)       # 获取气动力回调
    D, L, M = aero_func(V=10.0, alpha_rad=0.1) # 计算

    # 可视化气动特性
    python uav_model.py                         # 输出气动力曲线图
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


# ============================================================================
#  无人机物理参数配置
# ============================================================================

@dataclass
class QuadcopterConfig:
    r"""四旋翼无人机完整物理参数。

    坐标系定义 (机体系, NED 惯例):
      x → 前方 (机头)
      y → 右侧
      z → 下方

    侧视图 (纵向平面):
    ::

              Rotor 1,2 (front)            Rotor 3,4 (rear)
                  ─┼─          arm          ─┼─
                   |    ←── l₁ ──→ CG ←── l₂ ──→   |
                   |          ┌──────┐          |
              d₁ ↕|          │ Body │          |↕ d₂
                             │  ●CG │
                             └──────┘
                            ← l_body →

    力臂约定:
      l₁ : CG 到前旋翼纵向距离 (正值)
      l₂ : CG 到后旋翼纵向距离 (正值)
      d₁ : 前旋翼桨毂高于 CG 的垂直距离 (正值 = 桨在上)
      d₂ : 后旋翼桨毂高于 CG 的垂直距离 (正值 = 桨在上)
    """

    # ===================== 质量特性 =====================
    mass: float = 3.5       # kg — 含电池、相机等有效载荷
    g: float = 9.81         # m/s²
    I_xx: float = 0.040     # kg·m² — 横滚转动惯量
    I_yy: float = 0.045     # kg·m² — 俯仰转动惯量 (主配平轴)
    I_zz: float = 0.070     # kg·m² — 偏航转动惯量

    # ===================== 机架几何 =====================
    frame_type: str = "X"       # "X" 或 "+"
    arm_length: float = 0.225   # m — CG 到电机轴距离
    #   X 构型: l = arm × cos(45°) ≈ 0.159 m
    #   + 构型: l = arm = 0.225 m
    l1: float = 0.16            # m — CG 到前旋翼对纵向距离 [1]
    l2: float = 0.16            # m — CG 到后旋翼对纵向距离 [1]
    d1: float = 0.06            # m — 前桨毂高于 CG [1][3]
    d2: float = 0.06            # m — 后桨毂高于 CG [1][3]

    # ===================== 螺旋桨 =====================
    n_rotors: int = 4
    prop_diameter: float = 0.254    # m (10 inch)
    prop_radius: float = 0.127      # m

    # ===================== 机身几何 (外壳) =====================
    l_body: float = 0.35        # m — 机身纵向长度 (力矩参考长度)
    w_body: float = 0.20        # m — 机身横向宽度
    h_body: float = 0.12        # m — 机身高度 (含起落架)
    S_front: float = 0.035      # m² — 正面迎风参考面积 (≈ w × h + 电机)
    S_top: float = 0.080        # m² — 俯视投影面积 (含机臂)
    S_side: float = 0.025       # m² — 侧面投影面积

    # ===================== 机身气动系数 =====================
    # --- 阻力 ---
    #   D(α) = q × [f_front × cos²α + f_top × sin²α + f_arms]
    #   其中 f = Cd × S 为等效平板面积 (flat plate area)
    #
    #   参考: [2] NASA 风洞测试 multicopter f ≈ 0.02-0.05 m²
    #         [4] Theys (2020) 实测 Cd_front ≈ 0.5-0.9
    #         [3] RotorPy Hummingbird c_Dx = 0.005 → 缩放至 3.5 kg
    Cd_front: float = 0.65      # 正面体阻力系数 (基于 S_front) [4]
    Cd_top: float = 1.28        # 底面/顶面阻力系数 (平板垂直流) [2]
    f_arms: float = 0.005       # m² — 机臂+电机座等效平板面积 (各向同性)

    # --- 升力 ---
    #   L(α) = q × S_top × CL_alpha × sin(2α)
    #   机身为钝体，升力很小; CL_alpha ≈ 0.10-0.20 [5]
    CL_alpha: float = 0.12      # 机身升力线斜率 (1/rad, 基于 S_top)

    # --- 俯仰力矩 ---
    #   M(α) = q × S_front × l_body × [Cm_0 + Cm_alpha × α]
    #   Cm_alpha < 0 为静稳定 (气动中心在 CG 前方)
    Cm_0: float = 0.0           # 零迎角俯仰力矩系数
    Cm_alpha: float = -0.03     # 俯仰力矩斜率 (1/rad) — 弱静稳定

    @property
    def weight(self) -> float:
        return self.mass * self.g

    @property
    def f_front(self) -> float:
        """正面等效平板面积 m²。"""
        return self.Cd_front * self.S_front

    @property
    def f_top(self) -> float:
        """顶面/底面等效平板面积 m²。"""
        return self.Cd_top * self.S_top

    def flat_plate_area(self, alpha_rad: float) -> float:
        """给定迎角下的总等效平板面积 m²。"""
        sa = math.sin(alpha_rad)
        ca = math.cos(alpha_rad)
        return self.f_front * ca**2 + self.f_top * sa**2 + self.f_arms

    def summary(self) -> str:
        """打印配置摘要。"""
        w = self.weight
        lines = [
            "=" * 60,
            "  四旋翼无人机物理参数",
            "=" * 60,
            f"  质量        = {self.mass:.2f} kg  (重力 {w:.2f} N)",
            f"  转动惯量    = Ixx={self.I_xx:.4f}  Iyy={self.I_yy:.4f}  Izz={self.I_zz:.4f} kg·m²",
            f"  机架        = {self.frame_type} 构型, 臂长 {self.arm_length*1e3:.0f} mm",
            f"  力臂 (纵向) = l₁={self.l1*1e3:.0f} mm, l₂={self.l2*1e3:.0f} mm",
            f"  力臂 (垂向) = d₁={self.d1*1e3:.0f} mm, d₂={self.d2*1e3:.0f} mm",
            f"  螺旋桨      = {self.n_rotors}× {self.prop_diameter*1e3:.0f}mm ({self.prop_diameter/0.0254:.0f}\")",
            f"  机身尺寸    = {self.l_body*1e3:.0f}×{self.w_body*1e3:.0f}×{self.h_body*1e3:.0f} mm",
            "",
            "  气动参考面积:",
            f"    S_front   = {self.S_front*1e4:.1f} cm²",
            f"    S_top     = {self.S_top*1e4:.1f} cm²",
            f"    f_arms    = {self.f_arms*1e4:.1f} cm²",
            "",
            "  气动系数:",
            f"    Cd_front  = {self.Cd_front:.2f}  →  f_front = {self.f_front*1e4:.1f} cm²",
            f"    Cd_top    = {self.Cd_top:.2f}  →  f_top   = {self.f_top*1e4:.1f} cm²",
            f"    CL_alpha  = {self.CL_alpha:.2f} /rad",
            f"    Cm_alpha  = {self.Cm_alpha:.3f} /rad  ({'静稳定' if self.Cm_alpha < 0 else '静不稳定'})",
            "",
            "  V=10 m/s, α=5° 时:",
        ]
        alpha = math.radians(5.0)
        D, L, M = _fuselage_aero_impl(10.0, alpha, self)
        lines.append(f"    D_f = {D:.3f} N  ({D/w*100:.1f}% of W)")
        lines.append(f"    L_f = {L:.3f} N  ({L/w*100:.1f}% of W)")
        lines.append(f"    M_f = {M:.4f} N·m")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
#  预设无人机配置
# ============================================================================

def config_3kg_10inch() -> QuadcopterConfig:
    """3.5 kg 级四旋翼, 10 英寸螺旋桨 (默认)。"""
    return QuadcopterConfig()


def config_1kg_8inch() -> QuadcopterConfig:
    """1.0 kg 级小型四旋翼, 8 英寸螺旋桨。

    参考: AscTec Hummingbird 缩放 [3]
    """
    return QuadcopterConfig(
        mass=1.0,
        I_xx=0.008,   I_yy=0.009,   I_zz=0.015,
        arm_length=0.17,
        l1=0.12,      l2=0.12,
        d1=0.04,      d2=0.04,
        prop_diameter=0.203,  prop_radius=0.1015,
        l_body=0.25,  w_body=0.15,  h_body=0.08,
        S_front=0.015, S_top=0.035,  S_side=0.012,
        Cd_front=0.55, Cd_top=1.20,  f_arms=0.002,
        CL_alpha=0.10, Cm_alpha=-0.02,
    )


def config_7kg_15inch() -> QuadcopterConfig:
    """7 kg 级大型测绘四旋翼, 15 英寸螺旋桨。

    参考: DJI Matrice 210 风洞数据缩放 [6]
    """
    return QuadcopterConfig(
        mass=7.0,
        I_xx=0.12,    I_yy=0.13,    I_zz=0.20,
        arm_length=0.33,
        l1=0.23,      l2=0.23,
        d1=0.08,      d2=0.08,
        prop_diameter=0.381,  prop_radius=0.1905,
        l_body=0.50,  w_body=0.30,  h_body=0.18,
        S_front=0.065, S_top=0.16,   S_side=0.045,
        Cd_front=0.70, Cd_top=1.30,  f_arms=0.010,
        CL_alpha=0.15, Cm_alpha=-0.04,
    )


# ============================================================================
#  机身气动力模型
# ============================================================================

def _fuselage_aero_impl(V: float, alpha_rad: float,
                         cfg: QuadcopterConfig) -> Tuple[float, float, float]:
    r"""计算机身气动力。

    物理模型:
      动压:  q = 0.5 × ρ × V²

      阻力 (cross-flow 分解):
        D_f = q × [f_front × cos²α + f_top × sin²α + f_arms]
        其中 f = Cd × S 为等效平板面积

      升力 (钝体, sin(2α) 模型):
        L_f = q × S_top × CL_alpha × sin(2α)

      俯仰力矩 (线性模型):
        M_f = q × S_front × l_body × [Cm_0 + Cm_alpha × α]
        正值 = 机头上仰

    Args:
        V: 空速 (m/s)
        alpha_rad: 机身迎角 (rad), 正值 = 机头上仰
        cfg: 无人机配置

    Returns:
        (D_f, L_f, M_f_y) — 阻力 N, 升力 N, 俯仰力矩 N·m
    """
    rho = 1.225
    q = 0.5 * rho * V * V

    if q < 1e-12:
        return 0.0, 0.0, 0.0

    sa = math.sin(alpha_rad)
    ca = math.cos(alpha_rad)

    # --- 阻力: cross-flow 分解 ---
    f_total = cfg.f_front * ca**2 + cfg.f_top * sa**2 + cfg.f_arms
    D_f = q * f_total

    # --- 升力: sin(2α) 钝体升力 ---
    L_f = q * cfg.S_top * cfg.CL_alpha * math.sin(2.0 * alpha_rad)

    # --- 俯仰力矩 ---
    M_f_y = q * cfg.S_front * cfg.l_body * (cfg.Cm_0 + cfg.Cm_alpha * alpha_rad)

    return D_f, L_f, M_f_y


def fuselage_aero_model(
    cfg: QuadcopterConfig = None,
) -> Callable[[float, float], Tuple[float, float, float]]:
    """创建机身气动力回调函数, 签名兼容 QuadcopterTrimSolver。

    Returns:
        func(V, alpha_rad) -> (D_f, L_f, M_f_y)
    """
    if cfg is None:
        cfg = QuadcopterConfig()

    def aero_func(V: float, alpha_rad: float) -> Tuple[float, float, float]:
        return _fuselage_aero_impl(V, alpha_rad, cfg)

    return aero_func


# ============================================================================
#  气动系数查看 / 对比工具
# ============================================================================

def compute_aero_table(cfg: QuadcopterConfig,
                       V_values=None, alpha_values=None):
    """计算气动力表格, 用于分析和校验。

    Returns:
        dict 包含 V, alpha(deg), D, L, M, f(等效平板面积), Cd_eff, CL_eff 数组
    """
    if V_values is None:
        V_values = [5.0, 10.0, 15.0, 20.0]
    if alpha_values is None:
        alpha_values = np.linspace(-5, 30, 36)

    records = []
    for V in V_values:
        q = 0.5 * 1.225 * V**2
        for alpha_deg in alpha_values:
            alpha_rad = math.radians(alpha_deg)
            D, L, M = _fuselage_aero_impl(V, alpha_rad, cfg)
            f_eff = D / q if q > 1e-12 else 0.0
            records.append({
                "V": V, "alpha_deg": alpha_deg, "alpha_rad": alpha_rad,
                "q": q, "D_f": D, "L_f": L, "M_f_y": M,
                "f_eff_m2": f_eff,
                "D_over_W_pct": D / cfg.weight * 100,
                "L_over_W_pct": L / cfg.weight * 100,
            })
    return records


def print_aero_table(cfg: QuadcopterConfig, V: float = 10.0):
    """打印给定速度下的气动力表。"""
    print(f"\n机身气动力表 @ V = {V:.1f} m/s  (W = {cfg.weight:.1f} N)")
    print(f"{'α(°)':>6} {'D(N)':>8} {'D/W%':>7} {'L(N)':>8} {'L/W%':>7} "
          f"{'M(Nm)':>8} {'f(cm²)':>8}")
    print("-" * 60)
    q = 0.5 * 1.225 * V**2
    for alpha_deg in np.arange(-5, 31, 5):
        alpha_rad = math.radians(alpha_deg)
        D, L, M = _fuselage_aero_impl(V, alpha_rad, cfg)
        f_eff = D / q * 1e4
        print(f"{alpha_deg:>6.0f} {D:>8.3f} {D/cfg.weight*100:>6.1f}% "
              f"{L:>8.3f} {L/cfg.weight*100:>6.1f}% {M:>8.4f} {f_eff:>8.1f}")


# ============================================================================
#  可视化
# ============================================================================

def plot_aero_characteristics(cfg: QuadcopterConfig = None,
                               save_path: str = "uav_aero_characteristics.png"):
    """绘制机身气动特性曲线 (论文级)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from plot_style import apply_style, COLORS, savefig, make_fig, label_subplots
        apply_style()
    except ImportError:
        COLORS = {"primary": "#2563EB", "secondary": "#DC2626",
                  "accent1": "#059669", "accent2": "#D97706"}
        savefig = None

    if cfg is None:
        cfg = QuadcopterConfig()

    alpha_arr = np.linspace(-5, 30, 100)
    V_list = [5, 10, 15, 20]
    rho = 1.225

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # (a) 等效平板面积 vs α
    ax = axes[0, 0]
    f_arr = np.array([cfg.flat_plate_area(math.radians(a)) for a in alpha_arr])
    ax.plot(alpha_arr, f_arr * 1e4, color=COLORS["primary"], linewidth=1.5)
    ax.set_ylabel("$f$ (cm²)")
    ax.set_title("Equivalent Flat Plate Area")
    ax.axhline(cfg.f_front * 1e4, ls="--", color=COLORS.get("light_gray", "#999"),
               linewidth=0.8, label=f"$f_{{front}}$={cfg.f_front*1e4:.1f}")
    ax.legend(fontsize=7)

    # (b) 阻力 vs α (多速度)
    ax = axes[0, 1]
    colors_v = [COLORS["primary"], COLORS["secondary"],
                COLORS["accent1"], COLORS["accent2"]]
    for i, V in enumerate(V_list):
        D_arr = [_fuselage_aero_impl(V, math.radians(a), cfg)[0] for a in alpha_arr]
        ax.plot(alpha_arr, D_arr, color=colors_v[i], label=f"V={V} m/s")
    ax.set_ylabel("$D_f$ (N)")
    ax.set_title("Fuselage Drag")
    ax.legend(fontsize=7)

    # (c) 阻力占比 vs V
    ax = axes[0, 2]
    V_sweep = np.linspace(2, 25, 50)
    for alpha_deg, ls in [(5, "-"), (10, "--"), (15, ":")]:
        D_pct = [_fuselage_aero_impl(V, math.radians(alpha_deg), cfg)[0]
                 / cfg.weight * 100 for V in V_sweep]
        ax.plot(V_sweep, D_pct, ls, color=COLORS["primary"],
                label=f"α={alpha_deg}°")
    ax.set_xlabel("V (m/s)")
    ax.set_ylabel("$D_f / W$ (%)")
    ax.set_title("Drag-to-Weight Ratio")
    ax.legend(fontsize=7)

    # (d) 升力 vs α
    ax = axes[1, 0]
    for i, V in enumerate(V_list):
        L_arr = [_fuselage_aero_impl(V, math.radians(a), cfg)[1] for a in alpha_arr]
        ax.plot(alpha_arr, L_arr, color=colors_v[i], label=f"V={V} m/s")
    ax.set_xlabel("α (°)")
    ax.set_ylabel("$L_f$ (N)")
    ax.set_title("Fuselage Lift")
    ax.legend(fontsize=7)

    # (e) 俯仰力矩 vs α
    ax = axes[1, 1]
    for i, V in enumerate(V_list):
        M_arr = [_fuselage_aero_impl(V, math.radians(a), cfg)[2] for a in alpha_arr]
        ax.plot(alpha_arr, M_arr, color=colors_v[i], label=f"V={V} m/s")
    ax.set_xlabel("α (°)")
    ax.set_ylabel("$M_{f,y}$ (N·m)")
    ax.set_title("Fuselage Pitching Moment")
    ax.axhline(0, ls="-", color="gray", linewidth=0.5)
    ax.legend(fontsize=7)

    # (f) 力臂示意图
    ax = axes[1, 2]
    _draw_schematic(ax, cfg)

    fig.suptitle(f"Quadcopter Fuselage Aerodynamics — "
                 f"{cfg.mass:.1f} kg, {cfg.prop_diameter/0.0254:.0f}\" props",
                 fontsize=13)
    fig.tight_layout()

    if savefig is not None:
        savefig(fig, save_path)
    else:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"气动特性图已保存: {save_path}")


def _draw_schematic(ax, cfg: QuadcopterConfig):
    """在 axes 上绘制简化的力臂示意图。"""
    from matplotlib.patches import Rectangle

    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-0.15, 0.20)
    ax.set_aspect("equal")
    ax.set_title("Force / Moment Arm Layout")
    ax.set_xlabel("Longitudinal (m)")
    ax.set_ylabel("Vertical (m)")

    bw = cfg.l_body / 2
    bh = cfg.h_body / 2
    body = Rectangle((-bw, -bh), cfg.l_body, cfg.h_body,
                     edgecolor="black", facecolor="#E5E7EB", linewidth=1.2)
    ax.add_patch(body)
    ax.plot(0, 0, "ko", markersize=5)
    ax.annotate("CG", (0, 0), (0.01, -0.03), fontsize=7)

    for sign, l_val, d_val, label in [
        (1, cfg.l1, cfg.d1, "Front"),
        (-1, cfg.l2, cfg.d2, "Rear"),
    ]:
        x_rotor = sign * l_val
        y_rotor = d_val
        ax.plot(x_rotor, y_rotor, "^", color="#DC2626", markersize=8)
        ax.annotate(label, (x_rotor, y_rotor),
                    (x_rotor, y_rotor + 0.025), fontsize=6, ha="center")

        ax.annotate("", xy=(x_rotor, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=1))
        mid_x = x_rotor / 2
        lbl = f"$l_1$={cfg.l1*1e3:.0f}" if sign > 0 else f"$l_2$={cfg.l2*1e3:.0f}"
        ax.text(mid_x, -0.02, lbl, fontsize=6, ha="center", color="#2563EB")

        ax.annotate("", xy=(x_rotor, y_rotor), xytext=(x_rotor, 0),
                    arrowprops=dict(arrowstyle="<->", color="#059669", lw=1))
        d_lbl = f"$d$={d_val*1e3:.0f}" if sign > 0 else ""
        if d_lbl:
            ax.text(x_rotor + 0.025, y_rotor / 2, d_lbl,
                    fontsize=6, color="#059669")

    ax.grid(True, alpha=0.2)


# ============================================================================
#  配平求解器集成接口
# ============================================================================

def create_trim_solver(cfg: QuadcopterConfig = None):
    """基于 UAV 配置创建配置好的 QuadcopterTrimSolver。

    自动设置质量、力臂和机身气动力回调。
    """
    from quadcopter_trim_solver import QuadcopterTrimSolver

    if cfg is None:
        cfg = QuadcopterConfig()

    solver = QuadcopterTrimSolver(
        mass=cfg.mass,
        g=cfg.g,
        l1=cfg.l1,
        l2=cfg.l2,
        d1=cfg.d1,
        d2=cfg.d2,
        fuselage_aero_func=fuselage_aero_model(cfg),
    )
    return solver


# ============================================================================
#  CLI 入口
# ============================================================================

def main():
    """打印配置摘要并生成气动特性图。"""
    import argparse
    parser = argparse.ArgumentParser(description="四旋翼无人机物理模型")
    parser.add_argument("--preset", choices=["3kg", "1kg", "7kg"], default="3kg",
                        help="预设配置")
    parser.add_argument("--output", type=str, default="uav_aero_characteristics.png")
    args = parser.parse_args()

    presets = {"3kg": config_3kg_10inch, "1kg": config_1kg_8inch, "7kg": config_7kg_15inch}
    cfg = presets[args.preset]()

    print(cfg.summary())
    print_aero_table(cfg, V=10.0)
    plot_aero_characteristics(cfg, save_path=args.output)


if __name__ == "__main__":
    main()
