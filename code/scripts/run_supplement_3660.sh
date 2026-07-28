#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_supplement_3660.sh MODE --manifest PATH [options]

Modes:
  alpha1-preflight  geom 1000, six condition_ood_alpha1 conditions
  smoke             geoms 1000 and 1010, one exact declared base condition
  formal            all 20 geometries and all 63 conditions; resumes checkpoints
  status            read-only process, output, and log status for this version

Options:
  --manifest PATH          canonical 20 x 63 formal manifest (required)
  --manifest-sha256 SHA    exact digest; default reads PATH.sha256
  --run-version VERSION    stable output version override
  --dry-run                validate and print the launch command (default)
  --execute                launch the selected mode with nohup
  -h, --help               show this help

Environment overrides:
  PYTHON_BIN, RUNNER_PYTHONPATH, EXPECTED_SUBMODULE_COMMIT,
  OUTPUT_ROOT, WORKTREE_ROOT, WORKERS, OMP_THREADS, BATCH_SIZE,
  CONDA_LIB_DIR, LD_LIBRARY_PATH
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
MODE="$1"
shift

case "$MODE" in
  alpha1-preflight|smoke|formal|status) ;;
  -h|--help) usage; exit 0 ;;
  *) die "unknown mode: $MODE" ;;
esac

ACTION="dry-run"
MANIFEST=""
MANIFEST_SHA256=""
RUN_VERSION="${RUN_VERSION:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      [[ $# -ge 2 ]] || die "--manifest requires a path"
      MANIFEST="$2"
      shift 2
      ;;
    --manifest-sha256)
      [[ $# -ge 2 ]] || die "--manifest-sha256 requires a digest"
      MANIFEST_SHA256="$2"
      shift 2
      ;;
    --run-version)
      [[ $# -ge 2 ]] || die "--run-version requires a value"
      RUN_VERSION="$2"
      shift 2
      ;;
    --dry-run)
      ACTION="dry-run"
      shift
      ;;
    --execute)
      ACTION="execute"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$MANIFEST" ]] || die "--manifest is required"
[[ -f "$MANIFEST" ]] || die "manifest does not exist: $MANIFEST"
MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CODE_ROOT="$REPO_ROOT/code"
SUBMODULE_PATH="code/Propeller_project-main"
SUBMODULE_ROOT="$REPO_ROOT/$SUBMODULE_PATH"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/LFM/bin/python}"
RUNNER_PYTHONPATH="${RUNNER_PYTHONPATH:-$CODE_ROOT}"
EXPECTED_SUBMODULE_COMMIT="${EXPECTED_SUBMODULE_COMMIT:-f3e56740b83aa00a36f4d56380b095c5b741437f}"
WORKERS="${WORKERS:-1}"
OMP_THREADS="${OMP_THREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CPU_COUNT="${CPU_COUNT:-$($PYTHON_BIN -c 'import os; print(os.cpu_count() or 1)')}"

[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || die "WORKERS must be a positive integer"
[[ "$OMP_THREADS" =~ ^[1-9][0-9]*$ ]] || die "OMP_THREADS must be a positive integer"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be a positive integer"
[[ "$CPU_COUNT" =~ ^[1-9][0-9]*$ ]] || die "CPU_COUNT must be a positive integer"
(( WORKERS * OMP_THREADS <= CPU_COUNT )) \
  || die "WORKERS * OMP_THREADS exceeds CPU_COUNT=$CPU_COUNT"
(( WORKERS == 1 )) || die "GPU execution requires exactly one worker"
[[ "$EXPECTED_SUBMODULE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || die "EXPECTED_SUBMODULE_COMMIT must be a full lowercase Git SHA"

CHECKOUT_STATUS="$(
  git -C "$REPO_ROOT" status --porcelain --untracked-files=normal
)"
if [[ -n "$CHECKOUT_STATUS" ]]; then
  printf '%s\n' "$CHECKOUT_STATUS" >&2
  die "superproject checkout is not clean"
fi

GENERATOR_COMMIT="$(git -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}')"
SUBMODULE_COMMIT="$(git -C "$SUBMODULE_ROOT" rev-parse --verify 'HEAD^{commit}')"
GITLINK_COMMIT="$(
  git -C "$REPO_ROOT" ls-tree HEAD -- "$SUBMODULE_PATH" | awk '{print $3}'
)"
[[ "$GENERATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || die "superproject HEAD is not a full Git SHA"
[[ "$SUBMODULE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || die "submodule HEAD is not a full Git SHA"
[[ "$GITLINK_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || die "superproject gitlink is not a full Git SHA"
[[ "$SUBMODULE_COMMIT" == "$GITLINK_COMMIT" ]] \
  || die "submodule HEAD does not match superproject gitlink"
[[ "$SUBMODULE_COMMIT" == "$EXPECTED_SUBMODULE_COMMIT" ]] \
  || die "submodule HEAD does not match EXPECTED_SUBMODULE_COMMIT"

if [[ -z "$MANIFEST_SHA256" ]]; then
  SHA_SIDECAR="${MANIFEST}.sha256"
  [[ -s "$SHA_SIDECAR" ]] || die "manifest SHA-256 sidecar is missing: $SHA_SIDECAR"
  read -r MANIFEST_SHA256 _ < "$SHA_SIDECAR"
fi
[[ "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "manifest SHA-256 must be 64 lowercase hexadecimal characters"
ACTUAL_MANIFEST_SHA256="$($PYTHON_BIN - "$MANIFEST" <<'PY'
import hashlib
import sys

print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
[[ "$ACTUAL_MANIFEST_SHA256" == "$MANIFEST_SHA256" ]] \
  || die "manifest SHA-256 mismatch: expected $MANIFEST_SHA256, got $ACTUAL_MANIFEST_SHA256"

PYTHONPATH="$RUNNER_PYTHONPATH" "$PYTHON_BIN" - \
  "$MANIFEST" "$MANIFEST_SHA256" "$MODE" <<'PY'
import sys

from scripts.run_data_gen_parallel import build_manifest_tasks, condition_key, load_manifest

manifest_path, digest, mode = sys.argv[1:]
tasks = build_manifest_tasks(load_manifest(manifest_path, expected_sha256=digest))
if len(tasks) != 20:
    raise SystemExit(f"formal manifest must contain exactly 20 geometries, got {len(tasks)}")
if any(len(task.manifest_conditions) != 63 for task in tasks):
    raise SystemExit("every formal manifest geometry must declare exactly 63 conditions")
task_by_id = {task.geom_id: task for task in tasks}
required_ids = {
    "alpha1-preflight": {1000},
    "smoke": {1000, 1010},
}.get(mode, set())
if not required_ids <= set(task_by_id):
    raise SystemExit(f"manifest lacks required geometry IDs: {sorted(required_ids - set(task_by_id))}")
if mode == "alpha1-preflight":
    alpha_count = sum(
        condition[3] == "condition_ood_alpha1"
        for condition in task_by_id[1000].manifest_conditions
    )
    if alpha_count != 6:
        raise SystemExit(f"geom 1000 must declare six alpha1 conditions, got {alpha_count}")
if mode == "smoke":
    expected_key = "RPM4000.0_Wind10.0_Angle85.0"
    for geom_id in required_ids:
        keys = {
            condition_key(*condition[:3])
            for condition in task_by_id[geom_id].manifest_conditions
        }
        if expected_key not in keys:
            raise SystemExit(f"geom {geom_id} lacks smoke condition {expected_key}")
print(f"validated_manifest geometries={len(tasks)} conditions_per_geometry=63 mode={mode}")
PY

if [[ -z "$RUN_VERSION" ]]; then
  RUN_VERSION="supplement_${MANIFEST_SHA256:0:12}_${GENERATOR_COMMIT:0:7}_${SUBMODULE_COMMIT:0:7}"
fi
[[ "$RUN_VERSION" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "run version contains unsafe characters: $RUN_VERSION"

OUTPUT_ROOT="${OUTPUT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/LFM_supplement_outputs}"
WORKTREE_ROOT="${WORKTREE_ROOT:-/tmp/lfm_supplement_${RUN_VERSION}}"
RUN_ROOT="$OUTPUT_ROOT/$RUN_VERSION"
MODE_OUTPUT="$RUN_ROOT/pkl"
LOG_DIR="$RUN_ROOT/logs"
LOG_PATH="$LOG_DIR/${MODE}.log"
PID_PATH="$RUN_ROOT/${MODE}.pid"
MODE_WORKTREE_ROOT="$WORKTREE_ROOT/$MODE"
TAG="${RUN_VERSION}_${MODE}"

QBLADE_LIB_DIR="$SUBMODULE_ROOT/QBladeCE_2.0.8.6/Libraries"
CONDA_LIB_DIR="${CONDA_LIB_DIR:-${CONDA_PREFIX:-$HOME/miniconda3/envs/LFM}/lib}"
export LD_LIBRARY_PATH="$QBLADE_LIB_DIR:$CONDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$RUNNER_PYTHONPATH"

COMMAND=(
  "$PYTHON_BIN" -u scripts/run_data_gen_parallel.py
  --manifest "$MANIFEST"
  --manifest-sha256 "$MANIFEST_SHA256"
  --workers "$WORKERS"
  --omp-threads "$OMP_THREADS"
  --device GPU
  --batch-size "$BATCH_SIZE"
  --out-dir "$MODE_OUTPUT"
  --tag "$TAG"
  --worktree-root "$MODE_WORKTREE_ROOT"
)
case "$MODE" in
  alpha1-preflight)
    COMMAND+=(--geom-ids 1000 --condition-splits condition_ood_alpha1)
    ;;
  smoke)
    COMMAND+=(
      --geom-ids 1000,1010
      --condition-keys RPM4000.0_Wind10.0_Angle85.0
    )
    ;;
  formal) ;;
esac

echo "mode=$MODE action=$ACTION run_version=$RUN_VERSION"
echo "manifest=$MANIFEST"
echo "manifest_sha256=$MANIFEST_SHA256"
echo "generator_git_commit=$GENERATOR_COMMIT"
echo "generator_submodule_commit=$SUBMODULE_COMMIT"
echo "workers=$WORKERS omp_threads=$OMP_THREADS device=GPU"
echo "output=$MODE_OUTPUT"
echo "log=$LOG_PATH"
echo "tmp=$MODE_WORKTREE_ROOT"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

if [[ "$MODE" == "status" ]]; then
  echo "--- process ---"
  pgrep -af "run_data_gen_parallel.py.*${RUN_VERSION}" || true
  echo "--- checkpoint/output count ---"
  find "$RUN_ROOT/pkl" -name '*.pkl' -type f 2>/dev/null | wc -l | awk '{print $1}'
  echo "--- latest log ---"
  tail -n 80 "$LOG_DIR"/*.log 2>/dev/null || true
  exit 0
fi

if [[ "$ACTION" == "dry-run" ]]; then
  printf '[dry-run] command:'
  printf ' %s' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$MODE_OUTPUT" "$LOG_DIR" "$MODE_WORKTREE_ROOT"
if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  die "mode already running with PID $(cat "$PID_PATH")"
fi
{
  echo "mode=$MODE run_version=$RUN_VERSION"
  echo "manifest_sha256=$MANIFEST_SHA256"
  echo "generator_git_commit=$GENERATOR_COMMIT"
  echo "generator_submodule_commit=$SUBMODULE_COMMIT"
  echo "workers=$WORKERS omp_threads=$OMP_THREADS device=GPU"
} > "$LOG_PATH"
(
  cd "$CODE_ROOT"
  nohup "${COMMAND[@]}" >> "$LOG_PATH" 2>&1 &
  echo "$!" > "$PID_PATH"
)
echo "launched_pid=$(cat "$PID_PATH")"
echo "status_command=$0 status --manifest $MANIFEST --manifest-sha256 $MANIFEST_SHA256"
