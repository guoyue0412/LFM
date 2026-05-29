#!/usr/bin/env bash
# LFM 顶层数据生成一键入口：从仓库根目录调用 scripts/run_data_gen.py。
# 透传所有参数；--help 可见完整用法。
# 用法：bash scripts/run_data_gen.sh [--device CPU|GPU] [--geometry-npy ...] [--tag ...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 如果 LFM conda env 已建，自动激活；否则用当前解释器
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "${LFM_ENV:-LFM}"; then
        conda activate "${LFM_ENV:-LFM}"
    fi
fi

exec python scripts/run_data_gen.py "$@"
