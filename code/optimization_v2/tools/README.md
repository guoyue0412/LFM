# MoE 训练工具链

3 个独立脚本,串联起 `pkl → 训练 → 可视化` 的完整路径,与 `optimization_v2/run.py` 解耦。

## 流水线

```
data_for_train/data_lhs/*.pkl
        │  (1) convert_pkl_to_dataset.py
        ▼
data_for_train/processed_lhs/dataset_v2_YYYYMMDD.csv  (+ .xlsx)
        │  (2) analyze_and_normalize.py  --plot
        ▼
processed_lhs/norm/
  ├── normalizer.pkl          ← 双向变换器 (StandardScaler / QuantileTransformer / log1p)
  ├── decisions.json          ← 每列采用的方法 + 统计量
  ├── data_normalized.npz     ← 归一化后的 (N, 51) 训练矩阵
  ├── distribution_grid.png   ← 全列 raw vs normalized 直方图
  └── qq_outputs.png          ← 输出列归一化后 Q-Q 图
        │  (3) sweep_moe.py
        ▼
sweep_results/moe_v2/
  ├── results.csv             ← 每个 trial 的指标
  ├── histories.npz           ← 每 trial 的 epoch 级 train/test/mae
  ├── sweep_overview.png      ← 10-面板综合图
  └── best_trial/
        ├── model.pth
        ├── config.json
        ├── diagnostics.png   ← 对角图 / 残差 / σ 校准
        └── gate_assignment.png
```

## 一键串联 (示例)

```bash
cd ~/gy_2026/graduation/LFM
export LD_LIBRARY_PATH="$PWD/Propeller_project-main/QBladeCE_2.0.8.6/Libraries:$LD_LIBRARY_PATH"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate LFM

# (1) 转换
python -m optimization_v2.tools.convert_pkl_to_dataset \
    --src ./data_for_train/data_lhs \
    --dst ./data_for_train/processed_lhs \
    --formats csv xlsx --last-n 120 --tag v2_$(date +%Y%m%d)

# (2) 归一化分析
python -m optimization_v2.tools.analyze_and_normalize \
    --csv ./data_for_train/processed_lhs/dataset_v2_$(date +%Y%m%d).csv \
    --out ./data_for_train/processed_lhs/norm --plot

# (3) Sweep (32 个 trial × 1500 epoch ≈ A4000 上 1-2 小时)
python -m optimization_v2.tools.sweep_moe \
    --data ./data_for_train/processed_lhs/norm/data_normalized.npz \
    --geom-idx-csv ./data_for_train/processed_lhs/dataset_v2_$(date +%Y%m%d).csv \
    --out ./sweep_results/moe_v2_$(date +%Y%m%d) \
    --epochs 1500 --n-trials 32 --device cuda
```

## 归一化决策规则速查

| 规则 (优先级 1 → 5) | 条件 | 方法 | 典型适用列 |
|---|---|---|---|
| 1 | std < 1e-12 | **skip** (常量) | — |
| 2 | skew > 2 且 min > 0 | **log1p + standard** | T, Q (推力/扭矩,右偏长尾) |
| 3 | \|skew\| > 1.5 或 \|kurt\| > 8 | **quantile (normal)** | H, My (含极端值的力/力矩) |
| 4 | nunique ≤ 20 且 unique/n < 5% | **standard** | RPM, ANGLE (离散网格) |
| 5 | 默认 | **standard** | chord_*, twist_* (几何 sections) |

`decisions.json` 会记录每列实际采用的方法 + 完整 stats,可在 `distribution_grid.png` 上直接核对归一化后是否近似 N(0, 1)。

## Sweep 默认网格

| 超参 | 候选值 |
|---|---|
| `n_experts` | 2, 4, 6, 8 |
| `hidden_dim` | 64, 128, 256 |
| `shared_dim` | 32, 64, 128 |
| `lr` | 1e-3, 5e-4, 1e-4 |
| `weight_decay` | 1e-4, 1e-3 |
| `balance_weight` | 0.01, 0.1, 0.3 |
| `smooth_weight` | 0.0, 0.01 |

全网格 4×3×3×3×2×3×2 = **1296 组**,默认随机抽 32 组。
要做完整网格: `--n-trials 1296`;要自定义: `--grid my_grid.json`。

## 关键设计

- **按几何分割训练/测试**:`split_by_geometry()` 保证测试集的 `geom_idx` 完全没在训练集出现过,避免不同 RPM/Angle 下同一几何带来的数据泄露。
- **专家平衡 + gate 熵正则**:`MoELoss` 同时惩罚专家使用不均衡 (`balance_weight`) 和 gate 过于尖锐 (`smooth_weight`),sweep 同时扫这两个权重。
- **早停+best 模型回滚**:每 trial 记录 `best_test_loss` 对应的 `state_dict`,落到 `best_trial/model.pth`。
- **诊断三图**:对角图 (准确性) + 残差直方图 (偏差) + σ 校准散点 (不确定度量纲是否合理),覆盖 T/H/My/Q 四个输出。
