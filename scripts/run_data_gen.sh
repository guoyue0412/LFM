#!/usr/bin/env bash
# LFM 顶层数据生成一键入口：从仓库根目录调用 scripts/run_data_gen.py。
# 透传所有参数；--help 可见完整用法。
# 用法：bash scripts/run_data_gen.sh [--device CPU|GPU] [--geometry-npy ...] [--tag ...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 如果 LFM conda env 已建，自动激活；否则用当前解释器
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for _p in /data/xzfang/miniconda3 "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        if [[ -f "$_p/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="$_p"; break
        fi
    done
fi
if [[ -n "$CONDA_BASE" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "${LFM_ENV:-LFM}"; then
        conda activate "${LFM_ENV:-LFM}"
    fi
fi

# QBlade .so 依赖：QBladeCE/Libraries 提供 fortran/openCL/QGLViewer 等；conda env lib 提供 Qt5
QBLADE_LIB_DIR="$ROOT/Propeller_project-main/QBladeCE_2.0.8.6/Libraries"
export LD_LIBRARY_PATH="$QBLADE_LIB_DIR:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

exec python scripts/run_data_gen.py "$@"
