"""螺旋桨气动力预测模型 — 评估脚本

用法:
    python eval.py                        # 在测试集上评估
    python eval.py --rpm 5000 --wind 9 --angle 85   # 单次预测
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

from config import Config
from train import PropellerPredictor


# ================================================================
#  加载模型
# ================================================================

def load_model(cfg: Config, model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PropellerPredictor(cfg).to(device)

    if model_path is None:
        model_path = os.path.join(cfg.model_dir, "propeller_predictor.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


# ================================================================
#  测试集评估
# ================================================================

def evaluate(cfg: Config, model_path: str = None):
    model, device = load_model(cfg, model_path)
    d = cfg.processed_data_dir

    X_test_df = pd.read_pickle(os.path.join(d, "X_test_scaled"))
    y_test_df = pd.read_pickle(os.path.join(d, "y_test_scaled"))
    scaler_Y = joblib.load(os.path.join(d, "scaler_Y.pkl"))

    X_tensor = torch.tensor(X_test_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_s = model(X_tensor).cpu().numpy()

    y_pred = scaler_Y.inverse_transform(y_pred_s)
    y_true = scaler_Y.inverse_transform(y_test_df.values)

    # ---------- 逐输出打印指标 ----------
    print("=" * 60)
    print(f"{'输出':>8s} | {'R²':>8s} | {'MAE':>12s} | {'平均相对误差':>10s}")
    print("-" * 60)

    all_rel = []
    for i, name in enumerate(cfg.output_columns):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rel = np.where(
            np.abs(y_true[:, i]) > 1e-6,
            np.abs(y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i]) * 100,
            0.0,
        )
        all_rel.append(rel)
        print(f"{name:>8s} | {r2:>8.4f} | {mae:>12.6f} | {np.mean(rel):>8.2f}%")

    print("=" * 60)

    # ---------- 保存结果 ----------
    pred_cols = [f"{c}_pred" for c in cfg.output_columns]
    true_cols = [f"{c}_true" for c in cfg.output_columns]
    err_cols = [f"{c}_err%" for c in cfg.output_columns]

    df_all = pd.concat([
        pd.DataFrame(y_true, columns=true_cols),
        pd.DataFrame(y_pred, columns=pred_cols),
        pd.DataFrame(np.column_stack(all_rel), columns=err_cols),
    ], axis=1)

    out_path = os.path.join(d, "predict_vs_true.xlsx")
    df_all.to_excel(out_path, index=False)
    print(f"详细结果已保存至 {out_path}")


# ================================================================
#  单次预测
# ================================================================

def predict(cfg: Config, rpm: float, wind: float, angle: float,
            geometry: list = None, model_path: str = None) -> dict:
    """给定工况和几何参数，返回 {Fx, Fy, Fz, Torque} 的预测值。

    Args:
        rpm, wind, angle: 工况参数
        geometry: 几何参数列表，长度需与 cfg.geometry_columns 一致:
            - control_points 模式: [cp_0, cp_1, ..., cp_7]  (8个值)
            - sections 模式: [chord_0..chord_21, twist_0..twist_21] (44个值)
            - none 模式: 不需要传入
    """
    model, device = load_model(cfg, model_path)
    d = cfg.processed_data_dir

    scaler_X = joblib.load(os.path.join(d, "scaler_X.pkl"))
    scaler_Y = joblib.load(os.path.join(d, "scaler_Y.pkl"))

    x_raw = [rpm, wind, angle]
    if cfg.geometry_mode != "none":
        if geometry is None:
            raise ValueError(
                f"当前几何模式为 '{cfg.geometry_mode}'，"
                f"需要传入 {len(cfg.geometry_columns)} 个几何参数")
        if len(geometry) != len(cfg.geometry_columns):
            raise ValueError(
                f"geometry 长度 {len(geometry)} != "
                f"期望 {len(cfg.geometry_columns)} ({cfg.geometry_mode})")
        x_raw.extend(geometry)

    x_scaled = scaler_X.transform([x_raw])
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_scaled = model(x_tensor).cpu().numpy()

    y = scaler_Y.inverse_transform(y_scaled)[0]
    result = dict(zip(cfg.output_columns, y))

    print(f"输入: RPM={rpm}, WIND={wind}, ANGLE={angle}, "
          f"geometry={cfg.geometry_mode}({len(cfg.geometry_columns)}维)")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}")
    return result


# ================================================================
#  CLI 入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="螺旋桨气动力预测 — 评估/预测")
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--wind", type=float, default=None)
    parser.add_argument("--angle", type=float, default=None)
    parser.add_argument("--model", type=str, default=None, help="模型文件路径")
    args = parser.parse_args()

    cfg = Config()

    if args.rpm is not None and args.wind is not None and args.angle is not None:
        predict(cfg, args.rpm, args.wind, args.angle, model_path=args.model)
    else:
        evaluate(cfg, model_path=args.model)


if __name__ == "__main__":
    main()
