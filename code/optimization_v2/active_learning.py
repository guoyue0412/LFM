"""主动学习补点模块 — 不确定度驱动的 LLFVW 数据扩展

当 MoE 对某个查询点的不确定度 σ 超过阈值时,标记该点需要 LLFVW 重新仿真。
收集高不确定度样本 → 调用 LLFVW → 加入数据库 → 微调 MoE。
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import OptConfig
from .geometry import (
    cp_to_sections, split_sections, CHORD_COLS, TWIST_COLS,
)


@dataclass
class QueryPoint:
    """一个待补点的查询。"""
    cp: np.ndarray                  # shape (8,)
    rpm: float
    wind: float
    angle: float
    sigma: np.ndarray               # shape (4,) — [σ_T, σ_H, σ_My, σ_Q]
    sigma_max: float
    source: str = "optimization"    # 来源标记


class ActiveLearningManager:
    """主动学习管理器。

    职责:
        1. 收集优化过程中高不确定度的查询点
        2. 去重和优先级排序
        3. 导出为 LLFVW 仿真任务
        4. 将新数据合并回训练集
    """

    def __init__(self, cfg: OptConfig):
        self.cfg = cfg
        self.pending_queries: List[QueryPoint] = []
        self.completed_queries: List[QueryPoint] = []
        self._seen_hashes = set()

    def check_and_collect(
        self,
        cp: np.ndarray,
        rpm: float,
        wind: float,
        angle: float,
        sigma: np.ndarray,
    ) -> bool:
        """检查不确定度,若超阈值则收集该点。

        Returns:
            True if point was collected (high uncertainty)
        """
        sigma_max = float(np.max(np.abs(sigma)))
        if sigma_max <= self.cfg.sigma_max:
            return False

        # 去重: 基于 CP + 工况的粗略哈希
        h = hash((
            tuple(np.round(cp, 5)),
            round(rpm, -1),
            round(wind, 0),
            round(angle, 0),
        ))
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)

        self.pending_queries.append(QueryPoint(
            cp=cp.copy(),
            rpm=rpm,
            wind=wind,
            angle=angle,
            sigma=sigma.copy(),
            sigma_max=sigma_max,
        ))
        return True

    def collect_from_trim(self, cp: np.ndarray, trim_result, V: float):
        """从配平结果中收集高不确定度点。"""
        if not trim_result.converged:
            return

        alpha_deg = trim_result.alpha_deg
        angle = 90.0 - alpha_deg

        # 前桨
        self.check_and_collect(
            cp, trim_result.omega_front, V, angle, trim_result.sigma_front
        )
        # 后桨
        self.check_and_collect(
            cp, trim_result.omega_rear, V, angle, trim_result.sigma_rear
        )

    @property
    def n_pending(self) -> int:
        return len(self.pending_queries)

    def get_top_queries(self, n: int = None) -> List[QueryPoint]:
        """获取优先级最高的 n 个待补点 (按 σ_max 降序)。"""
        if n is None:
            n = self.cfg.active_learning_batch
        sorted_q = sorted(self.pending_queries, key=lambda q: -q.sigma_max)
        return sorted_q[:n]

    def export_for_llfvw(self, output_path: str = None) -> str:
        """导出待仿真点为 CSV,供 LLFVW/QBlade 批量仿真。

        CSV 格式: RPM, WIND, ANGLE, chord_0..21, twist_0..21
        """
        if output_path is None:
            output_path = os.path.join(self.cfg.data_dir, "active_learning_queries.csv")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        queries = self.get_top_queries()
        if not queries:
            print("无待补点")
            return output_path

        records = []
        for q in queries:
            sections = cp_to_sections(q.cp)
            chord, twist = split_sections(sections)
            row = {"RPM": q.rpm, "WIND": q.wind, "ANGLE": q.angle}
            row.update(dict(zip(CHORD_COLS, chord)))
            row.update(dict(zip(TWIST_COLS, twist)))
            row["sigma_max"] = q.sigma_max
            records.append(row)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"已导出 {len(df)} 个待仿真点: {output_path}")
        return output_path

    def merge_new_data(self, new_data_path: str):
        """将 LLFVW 仿真结果合并回主数据集。

        Args:
            new_data_path: 新数据 CSV 路径 (含 T, H, My, Q 列)
        """
        main_path = os.path.join(self.cfg.data_dir, "synthetic_aero_data.csv")

        new_df = pd.read_csv(new_data_path)
        required_cols = ["RPM", "WIND", "ANGLE", "T", "H", "My", "Q"]
        missing = [c for c in required_cols if c not in new_df.columns]
        if missing:
            raise ValueError(f"新数据缺少列: {missing}")

        if os.path.exists(main_path):
            main_df = pd.read_csv(main_path)
            merged = pd.concat([main_df, new_df], ignore_index=True)
        else:
            merged = new_df

        merged.to_csv(main_path, index=False)
        print(f"数据已合并: {len(new_df)} 条新数据 → 总计 {len(merged)} 条")

        # 移动已处理的查询
        processed = self.get_top_queries(len(new_df))
        self.completed_queries.extend(processed)
        self.pending_queries = [q for q in self.pending_queries if q not in processed]

    def summary(self) -> str:
        """返回当前状态摘要。"""
        lines = [
            f"主动学习状态:",
            f"  待补点: {self.n_pending}",
            f"  已完成: {len(self.completed_queries)}",
        ]
        if self.pending_queries:
            sigmas = [q.sigma_max for q in self.pending_queries]
            lines.append(f"  σ_max 范围: [{min(sigmas):.4f}, {max(sigmas):.4f}]")
            lines.append(f"  阈值: {self.cfg.sigma_max}")
        return "\n".join(lines)
