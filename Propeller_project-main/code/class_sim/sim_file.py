"""仿真文件操作 — .sim 模板读写 (消除原 3 处重复逻辑)"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from .config import SimCondition, SimPaths, SIM_TEMPLATE_FILES


_SIM_LINE_PATTERN = re.compile(r"^\s*(\S+)\s+(\S+)\s+(-\s+.+)")
_COL_WIDTHS = (40, 18, 50)


def _apply_params_to_template(
    template_lines: List[str],
    params: List[Tuple[str, str]],
) -> List[str]:
    """在 .sim 模板行列表上应用参数替换, 返回新行列表。

    这是核心逻辑, 原代码中重复 3 次的正则替换统一到此处。
    """
    first_w, second_w, third_w = _COL_WIDTHS
    result: List[str] = []
    param_dict = {k: v for k, v in params}

    for line in template_lines:
        match = _SIM_LINE_PATTERN.match(line)
        if not match:
            result.append(line)
            continue

        first_col, keyword, comment = match.groups()
        if keyword in param_dict:
            new_val = param_dict[keyword]
            indent = "    " if keyword == "RPMPRESCRIBED" else ""
            w = first_w - len(indent)
            result.append(
                f"{indent}{new_val:<{w}} {keyword:<{second_w}} {comment:<{third_w}}\n"
            )
        else:
            result.append(line)
    return result


def read_template(sim_path: str) -> List[str]:
    """读取 .sim 模板文件。"""
    with open(sim_path, "r") as f:
        return f.readlines()


def write_sim_file(
    template_lines: List[str],
    params: List[Tuple[str, str]],
    output_path: str,
) -> str:
    """基于模板 + 参数生成新 .sim 文件, 返回输出路径。"""
    modified = _apply_params_to_template(template_lines, params)
    with open(output_path, "w") as f:
        f.writelines(modified)
    return output_path


def modify_sim_inplace(
    sim_path: str,
    params: List[Tuple[str, str]],
) -> None:
    """就地修改 .sim 文件 (用于 run_one_simulation 动态模式)。"""
    lines = read_template(sim_path)
    modified = _apply_params_to_template(lines, params)
    with open(sim_path, "w") as f:
        f.writelines(modified)


def generate_sim_files(
    paths: SimPaths,
    conditions: List[SimCondition],
) -> List[str]:
    """批量生成 .sim 文件, 返回生成的文件路径列表。"""
    _clean_sim_folder(paths.sim_folder)
    template = read_template(paths.base_sim)

    generated: List[str] = []
    for cond in conditions:
        out = os.path.join(paths.sim_folder, f"{cond.name}.sim")
        write_sim_file(template, cond.to_sim_params(), out)
        generated.append(out)
    return generated


def _clean_sim_folder(folder: str) -> None:
    """清理 SIM 目录中的旧文件 (保留模板)。"""
    for name in os.listdir(folder):
        if name in SIM_TEMPLATE_FILES:
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            if not os.access(path, os.W_OK):
                os.chmod(path, 0o777)
            os.remove(path)
        except OSError:
            pass


def collect_sim_files(folder: str) -> List[str]:
    """收集目录下所有 .sim 文件 (排除模板)。"""
    result = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.endswith(".sim") and f not in SIM_TEMPLATE_FILES:
                result.append(os.path.join(root, f))
    return result
