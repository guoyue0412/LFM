#!/usr/bin/env bash
set -euo pipefail

# Read-only status check for the two LFM data-generation/training workstations.
# It avoids nested local shell expansion by sending single-quoted scripts to ssh.

ROOT_3660="${ROOT_3660:-/home/gy/gy_2026/graduation/LFM}"
ROOT_7920="${ROOT_7920:-/home/gy/gy_2026/graduation/LFM}"
HOST_3660="${HOST_3660:-gy@100.127.130.104}"
HOST_7920="${HOST_7920:-7920workstation}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"

check_host() {
  local host="$1"
  local root="$2"
  local label="$3"
  local sweep_dir="${4:-}"
  local sweep_log="${5:-}"

  echo "========== ${label} (${host}) =========="
  if ! ssh -o ConnectTimeout="$CONNECT_TIMEOUT" "$host" "cd '$root' && echo CONNECTED && date '+%F %T'"; then
    echo "STATUS=UNREACHABLE"
    echo
    return 0
  fi

  ssh -o ConnectTimeout="$CONNECT_TIMEOUT" "$host" "cd '$root' && bash -s" <<'REMOTE'
set -euo pipefail

echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "nvidia-smi unavailable"

echo "--- data generation processes ---"
pgrep -af run_data_gen_parallel.py || true

echo "--- pkl count ---"
find data_for_train/data_lhs -name '*.pkl' 2>/dev/null | wc -l | awk '{print $1}'

echo "--- latest pkl files ---"
find data_for_train/data_lhs -name '*.pkl' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 5 | cut -d' ' -f2- || true

echo "--- latest processed CSV files ---"
PYTHON_BIN=""
for candidate in python3 python "$HOME/anaconda3/envs/LFM/bin/python" "$HOME/miniconda3/envs/LFM/bin/python"; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import pandas' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done
find data_for_train -path '*processed_lhs*' -name 'dataset*.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 8 | cut -d' ' -f2- | while read -r csv; do
  [ -n "$csv" ] || continue
  rows=$(($(wc -l < "$csv") - 1))
  if [[ -n "$PYTHON_BIN" ]]; then
    geoms=$("$PYTHON_BIN" - "$csv" <<'PY'
import sys
import pandas as pd
p = sys.argv[1]
df = pd.read_csv(p, usecols=["geom_idx"])
print(df["geom_idx"].nunique())
PY
)
  else
    geoms="unknown"
  fi
  printf '%s rows=%s geoms=%s\n' "$csv" "$rows" "$geoms"
done
REMOTE

  if [[ -n "$sweep_dir" ]]; then
    echo "--- sweep status ---"
    ssh -o ConnectTimeout="$CONNECT_TIMEOUT" "$host" "cd '$root' && \
      pgrep -af 'sweep_moe|${sweep_dir}' || true && \
      find '$sweep_dir' -maxdepth 2 -type f 2>/dev/null | sort | sed -n '1,80p' && \
      if [ -f '${sweep_dir}/results.csv' ]; then echo RESULTS_CSV=ready; fi && \
      if [ -f '${sweep_dir}/results_partial.csv' ]; then echo RESULTS_PARTIAL=ready; tail -n 8 '${sweep_dir}/results_partial.csv'; fi"
  fi

  if [[ -n "$sweep_log" ]]; then
    echo "--- sweep log tail ---"
    ssh -o ConnectTimeout="$CONNECT_TIMEOUT" "$host" "cd '$root' && tail -n 40 '$sweep_log' 2>/dev/null || true"
  fi

  echo
}

check_host "$HOST_3660" "$ROOT_3660" "3660" \
  "sweep_results/moe_merged_latest_20260628_1425_formal32x1500" \
  "logs/formal_sweep_merged_latest_20260628_1425.nohup.log"

check_host "$HOST_7920" "$ROOT_7920" "7920"
