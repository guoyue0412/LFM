#!/usr/bin/env bash
set -euo pipefail

# Resume the latest CSV-only data snapshot workflow after remote connectivity
# recovers. This script never transfers raw pkl files.
#
# Default mode is dry-run. Pass --execute to run commands.
#
# Optional environment variables:
#   LOCAL_7920_CSV_GZ=/path/to/dataset.csv.gz
#       Use a locally cached compressed 7920 CSV instead of streaming from 7920.
#   SKIP_7920_EXTRACT=1
#       Skip re-extracting CSV on 7920. Useful when LOCAL_7920_CSV_GZ is set.

MODE="dry-run"
if [[ "${1:-}" == "--execute" ]]; then
  MODE="execute"
elif [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--execute]" >&2
  exit 2
fi

TS="${TS:-$(date +%Y%m%d_%H%M)}"
ROOT_3660="${ROOT_3660:-/home/gy/gy_2026/graduation/LFM}"
ROOT_7920="${ROOT_7920:-/home/gy/gy_2026/graduation/LFM}"
HOST_3660="${HOST_3660:-gy@100.127.130.104}"
HOST_7920="${HOST_7920:-7920workstation}"
LOCAL_7920_CSV_GZ="${LOCAL_7920_CSV_GZ:-}"
SKIP_7920_EXTRACT="${SKIP_7920_EXTRACT:-0}"

TAG="latest_${TS}"
CSV_3660="data_for_train/processed_lhs_current_3660_${TAG}/dataset_current_3660_${TAG}.csv"
CSV_7920="data_for_train/processed_lhs_current_7920_${TAG}/dataset_current_7920_${TAG}.csv"
CSV_7920_ON_3660="data_for_train/processed_lhs_current_7920_${TAG}/dataset_current_7920_${TAG}.csv"
MERGED="data_for_train/processed_lhs_merged_${TAG}/dataset_merged_${TAG}.csv"
NORM_DIR="data_for_train/processed_lhs_merged_${TAG}/norm"

run() {
  echo "+ $*"
  if [[ "$MODE" == "execute" ]]; then
    "$@"
  fi
}

run_shell() {
  local cmd="$1"
  echo "+ $cmd"
  if [[ "$MODE" == "execute" ]]; then
    bash -lc "$cmd"
  fi
}

echo "[resume] mode=${MODE} tag=${TAG}"
echo "[resume] raw pkl transfer is intentionally disabled; CSV only."
if [[ -n "$LOCAL_7920_CSV_GZ" ]]; then
  echo "[resume] using local 7920 CSV gzip: $LOCAL_7920_CSV_GZ"
fi

echo "[check] local -> 3660"
run ssh -o ConnectTimeout=12 "$HOST_3660" "cd '$ROOT_3660' && echo 3660_OK && date '+%F %T'"

if [[ -z "$LOCAL_7920_CSV_GZ" ]]; then
  echo "[check] local -> 7920"
  run ssh -o ConnectTimeout=12 "$HOST_7920" "cd '$ROOT_7920' && echo 7920_OK && date '+%F %T'"
fi

echo "[extract] 3660 pkl -> CSV"
run ssh "$HOST_3660" "cd '$ROOT_3660' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate LFM && python -m optimization_v2.tools.convert_pkl_to_dataset --src data_for_train/data_lhs --dst data_for_train/processed_lhs_current_3660_${TAG} --formats csv --last-n 120 --tag current_3660_${TAG}"

if [[ "$SKIP_7920_EXTRACT" == "1" || -n "$LOCAL_7920_CSV_GZ" ]]; then
  echo "[extract] 7920 pkl -> CSV skipped; using existing CSV artifact."
else
  echo "[extract] 7920 pkl -> CSV"
  run ssh "$HOST_7920" "cd '$ROOT_7920' && source ~/anaconda3/etc/profile.d/conda.sh && conda activate LFM && python -m optimization_v2.tools.convert_pkl_to_dataset --src data_for_train/data_lhs --dst data_for_train/processed_lhs_current_7920_${TAG} --formats csv --last-n 120 --tag current_7920_${TAG}"
fi

if [[ -n "$LOCAL_7920_CSV_GZ" ]]; then
  echo "[transfer] local cached 7920 CSV gzip -> 3660"
  if [[ "$MODE" == "execute" && ! -f "$LOCAL_7920_CSV_GZ" ]]; then
    echo "missing LOCAL_7920_CSV_GZ: $LOCAL_7920_CSV_GZ" >&2
    exit 1
  fi
  run_shell "ssh '$HOST_3660' \"cd '$ROOT_3660' && mkdir -p '$(dirname "$CSV_7920_ON_3660")'\" && gzip -cd '$LOCAL_7920_CSV_GZ' | ssh '$HOST_3660' \"cd '$ROOT_3660' && cat > '$CSV_7920_ON_3660'\""
else
  echo "[transfer] 7920 CSV -> 3660 via local stream"
  run_shell "ssh '$HOST_7920' \"gzip -c '$ROOT_7920/$CSV_7920'\" | ssh '$HOST_3660' \"cd '$ROOT_3660' && mkdir -p '$(dirname "$CSV_7920_ON_3660")' && gunzip -c > '$CSV_7920_ON_3660'\""
fi

echo "[merge] latest 3660 + latest 7920 CSV on 3660"
run ssh "$HOST_3660" "cd '$ROOT_3660' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate LFM && python -m optimization_v2.tools.merge_datasets --inputs '$CSV_3660' '$CSV_7920_ON_3660' --out '$MERGED'"

echo "[normalize] merged CSV"
run ssh "$HOST_3660" "cd '$ROOT_3660' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate LFM && python -m optimization_v2.tools.analyze_and_normalize --csv '$MERGED' --out '$NORM_DIR' --plot"

echo "[status] current formal sweep"
run ssh "$HOST_3660" "cd '$ROOT_3660' && pgrep -af 'moe_merged_latest_20260628_1425_formal32x1500|formal_sweep_merged_latest_20260628_1425' || true && tail -n 80 logs/formal_sweep_merged_latest_20260628_1425.nohup.log 2>/dev/null || true"

cat <<EOF

[next]
1. Compare merged row/geometry counts above with the active snapshot.
2. If the new snapshot is materially larger, decide whether to stop the old sweep and launch a new formal sweep:

ssh $HOST_3660 "cd $ROOT_3660 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate LFM && python -u -m optimization_v2.tools.sweep_moe \\
  --data ./$NORM_DIR/data_normalized.npz \\
  --norm ./$NORM_DIR/normalizer.pkl \\
  --geom-idx-csv ./$MERGED \\
  --out ./sweep_results/moe_merged_${TAG}_formal32x1500 \\
  --epochs 1500 --n-trials 32 --device cuda --seed ${TS//_/} --verbose"

EOF
