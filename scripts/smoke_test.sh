#!/usr/bin/env bash
# 端到端最小冒烟：1 几何（baseline）× 全工况 → data.py 处理 → 校验 raw_data.csv。
# 不跑 train.py（默认 5000 epoch，不适合冒烟；如需训练验证请手动 python train.py）。
# 用法：bash scripts/smoke_test.sh [--device CPU|GPU]
set -euo pipefail

DEVICE="CPU"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "未知参数：$1"; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "▸ LFM 根目录：$ROOT"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "${LFM_ENV:-LFM}"; then
        conda activate "${LFM_ENV:-LFM}"
        echo "▸ conda env：${LFM_ENV:-LFM}"
    fi
fi

DATA_DIR="data_for_train/data"
mkdir -p "$DATA_DIR"

# ── 1. 生成 1 几何 × 全工况 ────────────────────────────────────────────
echo
echo "─── 1/3 数据生成（baseline 几何，device=$DEVICE）───"
python scripts/run_data_gen.py --device "$DEVICE" --tag smoke

# 找出最新产物
PKL="$(ls -t "$DATA_DIR"/data_smoke_*.pkl 2>/dev/null | head -n1 || true)"
if [[ -z "$PKL" ]]; then
    echo "✖ 未生成 pkl"; exit 3
fi
SIZE=$(stat -c%s "$PKL" 2>/dev/null || stat -f%z "$PKL")
echo "▸ 产物：$PKL ($SIZE 字节)"

# ── 2. data.py 处理 ────────────────────────────────────────────────────
echo
echo "─── 2/3 LFM data.py 处理 ───"
python data.py

# ── 3. 校验 raw_data.csv ───────────────────────────────────────────────
CSV="data_for_train/geometry_run_results_100/raw_data.csv"
if [[ ! -s "$CSV" ]]; then
    echo "✖ 未生成或为空：$CSV"; exit 4
fi
ROWS=$(($(wc -l < "$CSV") - 1))
echo "▸ 产物：$CSV（$ROWS 行）"

# 校验关键列存在（首行 header）
HEADER="$(head -n1 "$CSV")"
for col in RPM WIND ANGLE Fx Fy Fz Torque; do
    if ! echo "$HEADER" | grep -qE "(^|,)$col(,|$)"; then
        echo "✖ raw_data.csv 缺列：$col"; exit 5
    fi
done
echo "▸ 关键列齐全：RPM/WIND/ANGLE/Fx/Fy/Fz/Torque"

echo
echo "✅ 冒烟通过。如需训练：python train.py（注意默认 5000 epoch）。"
