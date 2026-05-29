#!/usr/bin/env bash
# Linux 部署脚本：conda 环境 + Python 依赖 + submodule 校验 + QBlade .so 加载校验。
# 用法：bash scripts/install_linux.sh [--env-name LFM] [--python 3.10] [--skip-conda]
# 设计前提：裸机 Ubuntu，已装 conda；不自动 sudo apt。
set -euo pipefail

ENV_NAME="LFM"
PY_VERSION="3.10"
SKIP_CONDA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --python)   PY_VERSION="$2"; shift 2 ;;
        --skip-conda) SKIP_CONDA=1; shift ;;
        -h|--help)
            sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "未知参数：$1"; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "▸ LFM 根目录：$ROOT"

# ── 1. 校验 submodule 已 init ──────────────────────────────────────────
SUB="Propeller_project-main"
if [[ ! -f "$SUB/code/Simulation_QBlade/config.py" ]]; then
    echo "⚠ 子模块 $SUB 未初始化，执行 git submodule update --init --recursive"
    git submodule update --init --recursive
fi

# 子模块默认是 detached HEAD，新人若改了再 push 会丢；强制 checkout 锁定分支
SUB_BRANCH="$(git config -f .gitmodules submodule.${SUB}.branch || echo main)"
git -C "$SUB" checkout "$SUB_BRANCH" >/dev/null 2>&1 || true
echo "▸ 子模块分支：$SUB_BRANCH ($(git -C "$SUB" rev-parse --short HEAD))"

# ── 2. 校验 .so 在位 ────────────────────────────────────────────────────
# 从子模块 config.py 读 QBlade_dll（变量名虽为 dll，实际指向 .so）
SO_REL="$(python3 - "$SUB/code/Simulation_QBlade" <<'PY'
import os, runpy, sys
cfg_dir = sys.argv[1]
os.chdir(cfg_dir)
sys.path.insert(0, cfg_dir)
cfg = runpy.run_module("config", run_name="__cfg__")
# config 用 os.getcwd() 解析路径，相对当前 cwd 给出绝对路径
print(os.path.relpath(cfg["QBlade_dll"], start=os.path.abspath(os.path.join(cfg_dir, "../.."))))
PY
)"
SO_PATH="$SUB/$SO_REL"
if [[ ! -f "$SO_PATH" ]]; then
    echo "✖ 未找到 QBlade 共享库：$SO_PATH"
    echo "  请确认 submodule 已完整 clone（含 18MB+ 的 .so 文件）"
    exit 3
fi
echo "▸ QBlade 库：$SO_PATH ($(stat -c%s "$SO_PATH" 2>/dev/null || stat -f%z "$SO_PATH") 字节)"

# ── 3. ldd 检查 .so 系统依赖 ────────────────────────────────────────────
if command -v ldd >/dev/null 2>&1; then
    MISSING="$(ldd "$SO_PATH" 2>/dev/null | grep -E "not found" || true)"
    if [[ -n "$MISSING" ]]; then
        echo "⚠ 以下系统库缺失（请用 apt install 补齐，不会自动执行）："
        echo "$MISSING"
        echo "  常见补救：sudo apt install -y libgomp1 libstdc++6 libgfortran5 libblas3 liblapack3"
    else
        echo "▸ ldd 检查通过（无缺失系统依赖）"
    fi
else
    echo "▸ 当前平台无 ldd（非 Linux？），跳过系统依赖检查"
fi

# ── 4. conda 环境 ──────────────────────────────────────────────────────
if [[ $SKIP_CONDA -eq 0 ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "✖ 未检测到 conda；请先安装 Miniconda/Anaconda，或加 --skip-conda 跳过"
        exit 4
    fi
    # 使 conda activate 在脚本中可用
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "▸ 创建 conda 环境：$ENV_NAME (python=$PY_VERSION)"
        conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
    fi
    conda activate "$ENV_NAME"
    echo "▸ 已激活环境：$ENV_NAME ($(python --version 2>&1))"
    pip install -r requirements.txt
else
    echo "▸ --skip-conda：使用当前 Python ($(python3 --version 2>&1))"
fi

# ── 5. ctypes dry-load .so ─────────────────────────────────────────────
python3 - "$SO_PATH" <<'PY'
import ctypes, sys
so = sys.argv[1]
try:
    lib = ctypes.CDLL(so)
    print(f"▸ ctypes.CDLL 加载成功：{so}")
except OSError as e:
    print(f"✖ ctypes.CDLL 加载失败：{e}", file=sys.stderr)
    sys.exit(5)
PY

echo
echo "✅ 环境就绪。下一步可跑："
echo "    bash scripts/smoke_test.sh         # 最小冒烟（1 几何 × 全工况 → data.py）"
echo "    bash scripts/run_data_gen.sh --help"
