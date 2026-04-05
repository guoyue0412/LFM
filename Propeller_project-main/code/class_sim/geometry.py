"""叶片几何管理 — B 样条插值 + .bld 文件修改"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
from scipy.interpolate import BSpline


# B 样条参数 (与 LFM 优化管线共享同一组 knots)
_DEGREE = 3
_CHORD_KNOTS = np.concatenate(([0] * _DEGREE, [0.3984874, 0.89904882], [1] * _DEGREE))
_TWIST_KNOTS = np.concatenate(([0] * _DEGREE, [0.2, 0.89904882], [1] * _DEGREE))
_BLD_LINE_PATTERN = re.compile(
    r"^([+-]?\d*\.\d*(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d*\.\d*(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d*\.\d*(?:[eE][+-]?\d+)?)\s+"
    r"(.*)$"
)


def control_points_to_sections(
    control_points: np.ndarray,
    radial_positions: np.ndarray,
    radius: float = 0.127,
) -> np.ndarray:
    """将 8 个控制点 (4 chord + 4 twist) 展开为截面数据。

    Returns:
        (n_sections, 3) 数组: [r, chord, twist]
    """
    chord_cp = control_points[:4]
    twist_cp = control_points[4:8]
    x_norm = radial_positions / radius

    chord_vals = BSpline(_CHORD_KNOTS, chord_cp, _DEGREE)(x_norm)
    twist_vals = BSpline(_TWIST_KNOTS, twist_cp, _DEGREE)(x_norm)

    return np.column_stack([radial_positions, chord_vals, twist_vals])


def modify_bld_file(
    bld_path: str,
    section_data: np.ndarray,
    output_path: Optional[str] = None,
) -> str:
    """修改 .bld 文件中的 chord/twist 值。

    Args:
        bld_path: 原始 .bld 文件路径
        section_data: (n, 3) 数组 [r, chord, twist]
        output_path: 输出路径, None 则原地覆盖

    Returns:
        输出文件路径
    """
    output_path = output_path or bld_path

    with open(bld_path, "r") as f:
        lines = f.readlines()

    pos_lookup = {float(row[0]): (abs(row[1]), abs(row[2])) for row in section_data}
    col_w = 19

    modified = []
    for line in lines:
        m = _BLD_LINE_PATTERN.match(line)
        if not m:
            modified.append(line)
            continue

        pos_str, _, _, remaining = m.groups()
        pos_val = float(pos_str)

        matched_key = None
        for key in pos_lookup:
            if abs(pos_val - key) <= 1e-4:
                matched_key = key
                break

        if matched_key is not None:
            chord, twist = pos_lookup[matched_key]
            modified.append(
                f"{pos_val:<{col_w}.5f} {chord:<{col_w}.5f} "
                f"{twist:<{col_w}.5f} {remaining:<200}\n"
            )
        else:
            modified.append(line)

    with open(output_path, "w") as f:
        f.writelines(modified)

    return output_path
