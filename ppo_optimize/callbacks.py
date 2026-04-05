"""PPO 训练回调

- BestDesignCallback: 记录训练过程中发现的最优几何设计
- TensorBoardCallback: 记录额外的自定义指标
"""

import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class BestDesignCallback(BaseCallback):
    """跟踪训练过程中的最优巡航距离指标和对应几何设计。"""

    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.best_cruise_metric = -np.inf
        self.best_geometry = None
        self.best_info = {}
        self.history = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            cm = info.get("cruise_metric", None)
            if cm is not None and info.get("converged", False):
                self.history.append({
                    "timestep": self.num_timesteps,
                    "cruise_metric": float(cm),
                    "power": float(info.get("power", 0)),
                    "eta_prop": float(info.get("eta_prop", 0)),
                })

                if cm > self.best_cruise_metric:
                    self.best_cruise_metric = cm
                    self.best_geometry = info.get("geometry", None)
                    self.best_info = {
                        k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in info.items()
                        if k != "geometry" and not isinstance(v, np.ndarray)
                    }
                    if self.best_geometry is not None:
                        self.best_info["geometry"] = self.best_geometry.tolist()
                    self.best_info["timestep"] = self.num_timesteps

                    if self.verbose > 0:
                        print(f"[Best] step={self.num_timesteps} "
                              f"cruise={cm:.4f} power={info.get('power', 0):.2f}W")

        if self.num_timesteps % 2048 == 0:
            self._save_intermediate()
        return True

    def _save_intermediate(self) -> None:
        """定期保存中间结果, 避免训练中断导致数据丢失。"""
        if self.best_info:
            path = os.path.join(self.save_dir, "best_design.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.best_info, f, indent=2, ensure_ascii=False)
        if self.history:
            path = os.path.join(self.save_dir, "training_history.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False)

    def _on_training_end(self) -> None:
        self._save_intermediate()


class MetricsLoggerCallback(BaseCallback):
    """将自定义指标写入 TensorBoard。"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._recent_fm = []
        self._recent_power = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("converged", False):
                fm = info.get("cruise_metric", None)
                power = info.get("power", None)
                if fm is not None:
                    self._recent_fm.append(float(fm))
                if power is not None:
                    self._recent_power.append(float(power))

        if len(self._recent_fm) >= 50:
            tail = self._recent_fm[-50:]
            self.logger.record("custom/fm_mean", np.mean(tail))
            self.logger.record("custom/fm_max", np.max(tail))

        if len(self._recent_power) >= 50:
            tail = self._recent_power[-50:]
            self.logger.record("custom/power_mean", np.mean(tail))

        return True
