"""多进程并行数据生成：用 multiprocessing.Pool(N) 跑 SIMULATION，每进程独立工作目录。

设计要点
========
- QBlade SIL 用 cwd 解析 .sim/.bld/.qpr 相对路径，并发跑同一目录会互相覆写。
- 每个 worker 进程启动时：
    1. 用 PID 在 /tmp/lfm_w<pid>/ 建独立工作树
    2. QBlade_data/ 深拷贝（写操作集中在这里）
    3. code/、QBladeCE_2.0.8.6/ 符号链接共享（只读）
    4. chdir 到 <work>/code/Simulation_QBlade（config.py 用 os.getcwd() 解析）
- 任务粒度：1 个几何 = 1 个 task；几何数 > workers 时排队。
- QBlade GPU 路径仅允许单 worker 串行运行；多 GPU/多进程路径曾出现
  CL_INVALID_DEVICE 或挂起（见 RESEARCH_LOG 与 TECHNICAL_REFERENCE）。

输出格式与单进程版完全一致：
    {f'geometry_{idx}': {'geometry': np.ndarray, 'RPM*_Wind*_Angle*': pd.DataFrame, ...}}

典型用法
========
    # baseline × 16 worker（最小冒烟）
    python scripts/run_data_gen_parallel.py --workers 16 --tag para_smoke

    # 多几何批量生成
    python scripts/run_data_gen_parallel.py --workers 16 --geometry-npy <path> \
        --tag run_$(date +%Y%m%d) --first-n-geoms 32

注意
====
- 每 worker 占用：QBlade SIL 32 thread + 80MB 工作目录（RAM tmpfs OK）
- node6 推荐：--workers 16 → 占 16×32=512 thread > 160 核（OK，超额订阅 QBlade 内部 OpenMP 会自适应）
            实际产能 = 16 个几何 同时跑，每个仍然 ~25 min / 几何
"""

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

LFM_ROOT = Path(__file__).resolve().parent.parent
SUPERPROJECT_ROOT = LFM_ROOT.parent
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from optimization_v2.geometry import GEOMETRY_R, cp_to_sections  # noqa: E402

SUBMODULE_ROOT = LFM_ROOT / "Propeller_project-main"
SUB_CODE_ROOT = SUBMODULE_ROOT / "code" / "Simulation_QBlade"
SUB_DATA_ROOT = SUBMODULE_ROOT / "QBlade_data"
SUB_LIB_ROOT = SUBMODULE_ROOT / "QBladeCE_2.0.8.6"
CONDITION_SPLITS = frozenset(
    {"base42", "condition_id_interp", "condition_ood_alpha1"}
)
CONDITION_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
CONDITION_KEY_PATTERN = re.compile(
    rf"RPM{CONDITION_NUMBER_PATTERN}_Wind{CONDITION_NUMBER_PATTERN}_"
    rf"Angle{CONDITION_NUMBER_PATTERN}"
)
EXPECTED_SPLIT_COUNTS = {
    "base42": 42,
    "condition_id_interp": 15,
    "condition_ood_alpha1": 6,
}
GEOMETRY_CATEGORIES = frozenset({"geometry_id", "geometry_ood"})


class LoadedManifest(dict):
    """Decoded manifest payload carrying the digest of its source bytes."""

    def __init__(self, payload: dict, sha256: str):
        super().__init__(payload)
        self.sha256 = sha256


@dataclass(frozen=True)
class ManifestTask:
    """Serializable, exact work item derived from one manifest geometry."""

    geom_id: int
    category: str
    cp8: np.ndarray
    geometry: np.ndarray
    conditions: tuple[tuple[float, float, float, str], ...]
    declared_conditions: tuple[tuple[float, float, float, str], ...]
    manifest_conditions: tuple[tuple[float, float, float, str], ...]
    seed: int
    manifest_sha256: str

    @property
    def sections(self) -> np.ndarray:
        return np.concatenate((self.geometry[:, 1], self.geometry[:, 2]))

    @property
    def condition_splits(self) -> dict[str, int]:
        return dict(Counter(split for _, _, _, split in self.conditions))


def _checked_digest(actual: str, expected: str, source: str) -> None:
    expected = expected.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"invalid SHA-256 value from {source}: {expected!r}")
    if actual != expected:
        raise ValueError(f"{source} SHA-256 mismatch: expected {expected}, got {actual}")


def _is_full_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def load_manifest(
    path: str | Path, expected_sha256: str | None = None
) -> LoadedManifest:
    """Load JSON and verify any explicit/sidecar digest against exact file bytes."""
    manifest_path = Path(path)
    payload_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if expected_sha256 is not None:
        _checked_digest(digest, expected_sha256, "manifest")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if sidecar.exists():
        sidecar_parts = sidecar.read_text().split()
        if not sidecar_parts:
            raise ValueError(f"empty manifest SHA-256 sidecar: {sidecar}")
        _checked_digest(digest, sidecar_parts[0], "sidecar")
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    return LoadedManifest(payload, digest)


def _manifest_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected numeric {label}") from error
    if array.shape != shape:
        raise ValueError(f"expected {label} shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"expected finite {label}")
    return array


def _manifest_number(record: dict, key: str, label: str) -> float:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"expected numeric {label}.{key}") from error
    if not np.isfinite(value):
        raise ValueError(f"expected finite {label}.{key}")
    return value


def build_manifest_tasks(
    manifest: LoadedManifest, geom_ids: set[int] | None = None
) -> list[ManifestTask]:
    """Build exact geometry/condition tasks from a loaded manifest."""
    if manifest.get("manifest_version") != 1:
        raise ValueError(f"unsupported manifest_version: {manifest.get('manifest_version')!r}")
    records = manifest.get("geometries")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest geometries must be a non-empty list")

    seen_geom_ids: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("manifest geometry records must be objects")
        geom_id = record.get("geom_id")
        if isinstance(geom_id, bool) or not isinstance(geom_id, int):
            raise ValueError(f"geom_id must be an integer: {geom_id!r}")
        if geom_id in seen_geom_ids:
            raise ValueError(f"duplicate geom_id: {geom_id}")
        seen_geom_ids.add(geom_id)

    if geom_ids is not None:
        missing = geom_ids - seen_geom_ids
        if missing:
            raise ValueError(f"requested geom IDs absent from manifest: {sorted(missing)}")

    tasks = []
    for record in records:
        geom_id = record["geom_id"]
        geometry_category = record.get("category")
        if geometry_category not in GEOMETRY_CATEGORIES:
            raise ValueError(
                f"geometry {geom_id}: unknown geometry category {geometry_category!r}"
            )
        cp8 = _manifest_array(record.get("cp8"), (8,), "cp8")
        sections = _manifest_array(record.get("sections"), (44,), "sections")
        rebuilt = cp_to_sections(cp8)
        if not np.array_equal(sections, rebuilt):
            raise ValueError(f"geometry {geom_id}: cp8-section mismatch")
        geometry = np.column_stack((GEOMETRY_R, sections[:22], sections[22:]))
        declared_conditions = record.get("conditions")
        if not isinstance(declared_conditions, list) or not declared_conditions:
            raise ValueError(f"geometry {geom_id}: conditions must be a non-empty list")
        conditions_list = []
        seen_condition_keys: set[tuple[float, float, float]] = set()
        seen_serialized_keys: set[str] = set()
        for index, condition in enumerate(declared_conditions):
            if not isinstance(condition, dict):
                raise ValueError(f"geometry {geom_id}: condition {index} must be an object")
            label = f"geometry {geom_id} condition {index}"
            rpm = _manifest_number(condition, "rpm", label)
            wind = _manifest_number(condition, "wind", label)
            angle = _manifest_number(condition, "angle", label)
            split = condition.get("category")
            if split not in CONDITION_SPLITS:
                raise ValueError(
                    f"geometry {geom_id}: unknown condition split {split!r}"
                )
            if wind != 10.0:
                raise ValueError(f"geometry {geom_id}: wind must equal 10 m/s, got {wind}")
            key = (rpm, wind, angle)
            if key in seen_condition_keys:
                raise ValueError(f"geometry {geom_id}: duplicate condition key {key}")
            serialized_key = condition_key(rpm, wind, angle)
            if serialized_key in seen_serialized_keys:
                raise ValueError(
                    f"geometry {geom_id}: duplicate serialized condition key "
                    f"{serialized_key!r}"
                )
            seen_condition_keys.add(key)
            seen_serialized_keys.add(serialized_key)
            conditions_list.append((rpm, wind, angle, split))
        conditions = tuple(conditions_list)
        split_counts = Counter(split for _, _, _, split in conditions)
        if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
            raise ValueError(
                f"geometry {geom_id}: condition split counts must be "
                f"{EXPECTED_SPLIT_COUNTS}, got {dict(split_counts)}"
            )
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"geometry {geom_id}: seed must be an integer")
        tasks.append(
            ManifestTask(
                geom_id=geom_id,
                category=geometry_category,
                cp8=cp8,
                geometry=geometry,
                conditions=conditions,
                declared_conditions=conditions,
                manifest_conditions=conditions,
                seed=seed,
                manifest_sha256=manifest.sha256,
            )
        )
    if geom_ids is None:
        return tasks
    return [task for task in tasks if task.geom_id in geom_ids]


def parse_geom_ids(value: str | None) -> set[int] | None:
    """Parse a comma-separated exact manifest geometry-ID subset."""
    if value is None:
        return None
    try:
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise ValueError
        return {int(part) for part in parts}
    except ValueError as error:
        raise ValueError("--geom-ids must be comma-separated integers") from error


def parse_condition_splits(value: str | None) -> set[str] | None:
    """Parse a comma-separated set of exact declared condition split names."""
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--condition-splits must be comma-separated split names")
    requested = set(parts)
    unknown = requested - CONDITION_SPLITS
    if unknown:
        raise ValueError(f"unknown condition split: {sorted(unknown)}")
    return requested


def parse_condition_keys(value: str | None) -> set[str] | None:
    """Parse comma-separated exact serialized manifest condition keys."""
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if (
        not parts
        or any(not part for part in parts)
        or any(CONDITION_KEY_PATTERN.fullmatch(part) is None for part in parts)
    ):
        raise ValueError("--condition-keys contains an invalid condition key")
    return set(parts)


def select_manifest_task_conditions(
    tasks: list[ManifestTask],
    *,
    condition_splits: set[str] | None,
    condition_keys: set[str] | None,
) -> list[ManifestTask]:
    """Filter fully validated tasks while retaining their original declarations."""
    if condition_splits is not None and condition_keys is not None:
        raise ValueError("condition split and condition key selectors are mutually exclusive")
    if condition_splits is not None:
        if not condition_splits:
            raise ValueError("condition split selector must not be empty")
        unknown = condition_splits - CONDITION_SPLITS
        if unknown:
            raise ValueError(f"unknown condition split: {sorted(unknown)}")
    if condition_keys is not None:
        if not condition_keys:
            raise ValueError("condition key selector must not be empty")
        invalid = sorted(
            key
            for key in condition_keys
            if not isinstance(key, str) or CONDITION_KEY_PATTERN.fullmatch(key) is None
        )
        if invalid:
            raise ValueError(f"invalid condition key: {invalid}")
    if condition_splits is None and condition_keys is None:
        return tasks

    selected_tasks = []
    for task in tasks:
        manifest_keys = {
            condition_key(*condition[:3]) for condition in task.manifest_conditions
        }
        if condition_keys is not None:
            missing = condition_keys - manifest_keys
            if missing:
                raise ValueError(
                    "condition keys absent from selected manifest geometries: "
                    f"geom_id={task.geom_id}, keys={sorted(missing)}"
                )
            selected_conditions = tuple(
                condition
                for condition in task.manifest_conditions
                if condition_key(*condition[:3]) in condition_keys
            )
        else:
            selected_conditions = tuple(
                condition
                for condition in task.manifest_conditions
                if condition[3] in condition_splits
            )
        if not selected_conditions:
            raise ValueError(
                f"condition selector produced no conditions for geom_id={task.geom_id}"
            )
        selected_tasks.append(
            replace(
                task,
                conditions=selected_conditions,
                declared_conditions=selected_conditions,
            )
        )
    return selected_tasks


def _condition_split_map(task: ManifestTask) -> dict[str, str]:
    return {
        condition_key(rpm, wind, angle): split
        for rpm, wind, angle, split in task.declared_conditions
    }


def _manifest_condition_split_map(task: ManifestTask) -> dict[str, str]:
    return {
        condition_key(rpm, wind, angle): split
        for rpm, wind, angle, split in task.manifest_conditions
    }


def _valid_manifest_condition_keys(
    node: object, task: ManifestTask, expected_provenance: dict[str, str]
) -> set[str]:
    """Return valid declared condition keys from one exact manifest node."""
    if not isinstance(node, dict):
        return set()
    expected_splits = _condition_split_map(task)
    manifest_splits = _manifest_condition_split_map(task)
    condition_keys = {key for key in node if isinstance(key, str) and key.startswith("RPM")}
    try:
        control_points = np.asarray(node["control_points"], dtype=np.float64)
        sections = np.asarray(node["sections"], dtype=np.float64)
        geometry = np.asarray(node["geometry"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return set()
    generator_commit = node.get("generator_git_commit")
    generator_submodule_commit = node.get("generator_submodule_commit")
    split_map = node.get("condition_split")
    metadata_valid = (
        node.get("geom_id") == task.geom_id
        and node.get("manifest_sha256") == task.manifest_sha256
        and node.get("category") == task.category
        and node.get("seed") == task.seed
        and isinstance(split_map, dict)
        and set(split_map) == condition_keys
        and condition_keys.issubset(manifest_splits)
        and all(split_map[key] == manifest_splits[key] for key in condition_keys)
        and _is_full_git_sha(generator_commit)
        and _is_full_git_sha(generator_submodule_commit)
        and generator_commit == expected_provenance["generator_git_commit"]
        and generator_submodule_commit
        == expected_provenance["generator_submodule_commit"]
        and np.array_equal(control_points, task.cp8)
        and np.array_equal(sections, task.sections)
        and np.array_equal(geometry, task.geometry)
    )
    if not metadata_valid:
        return set()
    return {
        key
        for key in condition_keys
        if key in expected_splits
        and _is_nonempty_condition_frame(node.get(key))
    }


def _is_complete_manifest_node(node: object, task: ManifestTask) -> bool:
    expected_keys = set(_condition_split_map(task))
    return (
        _valid_manifest_condition_keys(node, task, _generator_provenance())
        == expected_keys
    )


def find_completed_manifest_conditions(
    output_dir: str | Path, tasks: list[ManifestTask]
) -> dict[int, set[str]]:
    """Union exact valid condition keys across checkpoint and aggregate pkls."""
    expected_provenance = _generator_provenance()
    task_by_id = {task.geom_id: task for task in tasks}
    completed: dict[int, set[str]] = {}
    for path in sorted(Path(output_dir).glob("*.pkl")):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        # Pickle reducers/imports can raise arbitrary ordinary exceptions. A single
        # unreadable file is invalid resume evidence, not a reason to abort scanning.
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for geom_id, task in task_by_id.items():
            valid_keys = _valid_manifest_condition_keys(
                payload.get(f"geometry_{geom_id}"), task, expected_provenance
            )
            if valid_keys:
                completed.setdefault(geom_id, set()).update(valid_keys)
    return completed


def find_completed_manifest_ids(
    output_dir: str | Path, tasks: list[ManifestTask]
) -> set[int]:
    """Return IDs backed by at least one exact, complete manifest result node."""
    condition_keys = find_completed_manifest_conditions(output_dir, tasks)
    return {
        task.geom_id
        for task in tasks
        if condition_keys.get(task.geom_id, set()) == set(_condition_split_map(task))
    }


def prepare_manifest_run(
    manifest_path: str | Path,
    geom_ids: set[int] | None,
    expected_sha256: str | None,
    output_dir: str | Path,
    condition_splits: set[str] | None = None,
    condition_keys: set[str] | None = None,
) -> tuple[list[ManifestTask], set[int], str]:
    """Load, validate, subset, and resume-filter a manifest without QBlade imports."""
    manifest = load_manifest(manifest_path, expected_sha256=expected_sha256)
    selected = build_manifest_tasks(manifest, geom_ids=geom_ids)
    selected = select_manifest_task_conditions(
        selected,
        condition_splits=condition_splits,
        condition_keys=condition_keys,
    )
    completed_conditions = find_completed_manifest_conditions(output_dir, selected)
    completed = {
        task.geom_id
        for task in selected
        if completed_conditions.get(task.geom_id, set())
        == set(_condition_split_map(task))
    }
    pending = []
    for task in selected:
        if task.geom_id in completed:
            continue
        valid_keys = completed_conditions.get(task.geom_id, set())
        missing = tuple(
            condition
            for condition in task.conditions
            if condition_key(*condition[:3]) not in valid_keys
        )
        pending.append(replace(task, conditions=missing))
    return pending, completed, manifest.sha256


def output_filename(
    tag: str,
    stamp: str,
    batch_number: int,
    batched: bool,
    manifest_sha256: str | None = None,
) -> str:
    """Build an output filename, including manifest provenance when present."""
    manifest_tag = (
        f"_manifest-{manifest_sha256[:12]}" if manifest_sha256 is not None else ""
    )
    suffix = f"_b{batch_number:04d}" if batched else ""
    return f"data_{tag}{manifest_tag}_{stamp}{suffix}.pkl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workers", type=int, default=16,
                   help="并行进程数；node6 默认 16；node1 推荐 8")
    p.add_argument("--omp-threads", type=int, default=8,
                   help="每 worker 内部 OMP/MKL/BLAS 线程数；默认 8。"
                        "workers × omp-threads 应 ≤ 物理核心数；"
                        "node6 (160 核) 推荐 16×8=128，留 32 核给系统/QBlade Qt helper")
    p.add_argument("--device", choices=["CPU", "GPU"], default="CPU",
                   help="GPU 使用 OpenCL device_id=1，且仅允许单 worker 串行运行")
    p.add_argument("--geometry-npy", type=str, default=None,
                   help="形状 (N, 22, 3) 的几何 numpy 文件；省略则只跑 baseline 1 个几何")
    p.add_argument("--manifest", type=str, default=None,
                   help="Task 1 生成的 canonical JSON manifest；提供后按精确 geom_id/工况运行")
    p.add_argument("--geom-ids", type=str, default=None,
                   help="逗号分隔的精确 manifest geom_id 子集")
    p.add_argument("--manifest-sha256", type=str, default=None,
                   help="可选的 manifest 文件字节 SHA-256；存在 sidecar 时也会自动校验")
    condition_selector = p.add_mutually_exclusive_group()
    condition_selector.add_argument(
        "--condition-splits",
        type=str,
        default=None,
        help="逗号分隔的精确 manifest 工况 split 子集",
    )
    condition_selector.add_argument(
        "--condition-keys",
        type=str,
        default=None,
        help="逗号分隔的精确序列化 manifest condition_key 子集",
    )
    p.add_argument("--first-n-geoms", type=int, default=None,
                   help="只跑前 N 个几何，用于产能基线测试")
    p.add_argument("--start-idx", type=int, default=0,
                   help="几何起始索引（含），用于断点续传；默认 0")
    p.add_argument("--end-idx", type=int, default=None,
                   help="几何结束索引（不含），用于断点续传或分批；默认全部")
    p.add_argument("--batch-size", type=int, default=0,
                   help="每 N 个几何落一个 pkl（0 = 跑完一次性落盘）。"
                        "推荐大规模时设 16 或 32，配合断点续传；崩了只丢一批")
    p.add_argument("--num-timesteps", type=int, default=None,
                   help="仿真步数，默认取 config.number_of_timesteps（1000）")
    p.add_argument("--out-dir", type=str, default=str(LFM_ROOT / "data_for_train" / "data"),
                   help="输出 pkl 目录；data.py 默认扫这里")
    p.add_argument("--tag", type=str, default="parallel",
                   help="输出 pkl 文件名 tag")
    p.add_argument("--worktree-root", type=str, default="/tmp",
                   help="worker 工作目录的父目录；node6 上 /tmp 是 tmpfs RAM-backed")
    p.add_argument("--keep-worktree", action="store_true",
                   help="跑完不删 /tmp/lfm_w<pid>，便于调试")
    return p.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> None:
    """Reject execution modes known to be unsafe before importing QBlade."""
    if args.device == "GPU" and args.workers != 1:
        raise ValueError("GPU execution requires exactly one worker")


def setup_worker_workdir(worktree_root: str) -> Path:
    """初始化当前 worker 进程的工作目录，返回 cwd 应当切到的位置。

    工作目录结构：
        <worktree_root>/lfm_w<pid>/
            ├── code/                   ← 深拷贝（仅 ~164KB；worker 要 chdir 进去）
            ├── QBladeCE_2.0.8.6/       → 符号链接（read-only，QBlade SIL 入口）
            └── QBlade_data/            ← 深拷贝（每 worker 独立写 .sim/.qpr）

    重要：code/ 必须深拷贝（不能 symlink），否则 chdir(symlink) 后
    os.getcwd() 会跟随到真实路径（submodule 原位），config.py 用 os.getcwd()
    解析的所有 file_path 都会指回 source 目录，worker 之间互相冲突。
    """
    pid = os.getpid()
    work_root = Path(worktree_root) / f"lfm_w{pid}"
    work_root.mkdir(parents=True, exist_ok=True)

    # 共享只读：QBladeCE_2.0.8.6/（worker 不会 chdir 进去，symlink 安全）
    lib_link = work_root / "QBladeCE_2.0.8.6"
    if not lib_link.exists():
        lib_link.symlink_to(SUB_LIB_ROOT)

    # 必须独立：code/（worker 要 chdir 到 code/Simulation_QBlade）
    code_dir = work_root / "code"
    if not code_dir.exists():
        shutil.copytree(SUBMODULE_ROOT / "code", code_dir,
                        symlinks=False, ignore=shutil.ignore_patterns("__pycache__"))

    # 必须独立：QBlade_data/（.sim/.bld/.qpr 写入隔离）
    data_dir = work_root / "QBlade_data"
    if not data_dir.exists():
        shutil.copytree(SUB_DATA_ROOT, data_dir)

    return work_root / "code" / "Simulation_QBlade"


def worker_init(worktree_root: str, omp_threads: int) -> None:
    """Pool worker 启动钩子：建工作目录、chdir、注入 sys.path、import 子模块。

    在 fork 子进程中执行（每个 worker 进程仅一次）。

    限制每 worker 内部并行度（OMP/BLAS/MKL），避免多 worker × 多线程
    在 N 核机器上 over-subscription：实测 4 worker × 32 OMP 线程
    + ~200 Qt/OpenCL helper 线程 = 900+ runnable thread → load avg
    ~200, 单几何 wall clock 从 5min 拖到 20+min。
    """
    # 必须在 import QBlade SIL 之前设置，否则 .so 启动时已读取环境
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["OMP_THREAD_LIMIT"] = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(omp_threads)
    os.environ["QT_THREAD_POOL_MAX_THREAD_COUNT"] = str(omp_threads)

    work_cwd = setup_worker_workdir(worktree_root)
    os.chdir(work_cwd)
    sys.path.insert(0, str(work_cwd))

    # 子模块 import 必须在 chdir 之后，因为 config.py 用 os.getcwd() 解析路径
    global _config, _SIMULATION
    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config as _cfg  # noqa: E402
    from class_sim.simulation import SIMULATION as _SIM  # noqa: E402
    _config = _cfg
    _SIMULATION = _SIM
    print(f"[worker {os.getpid()}] ready, cwd={os.getcwd()}", flush=True)


def _format_condition_number(value: float) -> str:
    """Use Python's shortest lossless binary64 round-trip representation."""
    return repr(float(value))


def condition_key(rpm: float, wind: float, angle: float) -> str:
    """Return the stable legacy-compatible key for one declared condition."""
    return (
        f"RPM{_format_condition_number(rpm)}_"
        f"Wind{_format_condition_number(wind)}_"
        f"Angle{_format_condition_number(angle)}"
    )


def _is_nonempty_condition_frame(value: object) -> bool:
    return isinstance(value, pd.DataFrame) and not value.empty


def _git_commit(repo: Path, label: str) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD^{commit}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot resolve {label} Git commit") from error
    if not _is_full_git_sha(commit):
        raise RuntimeError(
            f"{label} Git commit must be a full 40-character hexadecimal SHA"
        )
    return commit


def _generator_git_commit() -> str:
    return _git_commit(SUPERPROJECT_ROOT, "generator")


def _generator_submodule_commit() -> str:
    return _git_commit(SUBMODULE_ROOT, "generator submodule")


def _generator_gitlink_commit() -> str:
    relative_submodule = SUBMODULE_ROOT.relative_to(SUPERPROJECT_ROOT)
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(SUPERPROJECT_ROOT),
                "ls-tree",
                "HEAD",
                "--",
                str(relative_submodule),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot resolve generator submodule gitlink") from error
    parts = output.split(maxsplit=3)
    if (
        len(parts) != 4
        or parts[0] != "160000"
        or parts[1] != "commit"
        or not _is_full_git_sha(parts[2])
    ):
        raise RuntimeError("generator submodule gitlink is not a full Git commit")
    return parts[2]


def _generator_provenance() -> dict[str, str]:
    generator_commit = _generator_git_commit()
    submodule_commit = _generator_submodule_commit()
    gitlink_commit = _generator_gitlink_commit()
    if submodule_commit != gitlink_commit:
        raise RuntimeError(
            "generator submodule HEAD does not match superproject gitlink: "
            f"{submodule_commit} != {gitlink_commit}"
        )
    return {
        "generator_git_commit": generator_commit,
        "generator_submodule_commit": submodule_commit,
    }


def _condition_checkpoint_path(
    output_dir: str | Path, task: ManifestTask, key: str
) -> Path:
    """Return a deterministic filename for one manifest condition checkpoint."""
    key_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(output_dir) / (
        f"checkpoint_manifest-{task.manifest_sha256}_"
        f"geometry-{task.geom_id}_{key_digest}.pkl"
    )


def _atomic_pickle_dump(payload: object, destination: Path) -> None:
    """Persist one pickle using fsync + atomic rename in its destination directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_condition_checkpoint(
    output_dir: str | Path,
    task: ManifestTask,
    key: str,
    split: str,
    frame: pd.DataFrame,
    generator_provenance: dict[str, str],
) -> Path:
    """Atomically persist one successful declared manifest condition."""
    node = {
        "geometry": task.geometry.copy(),
        "sections": task.sections.copy(),
        key: frame,
        "geom_id": task.geom_id,
        "category": task.category,
        "control_points": task.cp8.tolist(),
        "condition_split": {key: split},
        "seed": task.seed,
        "manifest_sha256": task.manifest_sha256,
        **generator_provenance,
    }
    destination = _condition_checkpoint_path(output_dir, task, key)
    _atomic_pickle_dump({f"geometry_{task.geom_id}": node}, destination)
    return destination


def _worker_run_manifest(
    manifest_task: ManifestTask,
    num_timesteps: int | None,
    device: str,
    checkpoint_dir: str | Path | None = None,
) -> tuple[int, dict | None, str | None]:
    pid = os.getpid()
    t0 = time.time()
    sim = None
    try:
        generator_provenance = _generator_provenance()
        sim = _SIMULATION(
            file_path=_config.file_path,
            geometry_baseline=_config.geometry_baseline,
            device_type=device,
            number_of_timesteps=num_timesteps or _config.number_of_timesteps,
        )
        sim.change_propeller_geometry(section_data=manifest_task.geometry)
        condition_results = {}
        split_map = {}
        for rpm, wind, angle, split in manifest_task.conditions:
            key = condition_key(rpm, wind, angle)
            sim.run_one_simulation(RPM=rpm, WIND_SPEED=wind, ANGLE=angle)
            if key not in sim.all_simulation_data:
                raise RuntimeError(
                    f"absent persisted QBlade result for {(rpm, wind, angle)}"
                )
            frame = sim.all_simulation_data[key]
            if not _is_nonempty_condition_frame(frame):
                raise RuntimeError(f"empty QBlade result for {(rpm, wind, angle)}")
            condition_results[key] = frame
            split_map[key] = split
            if checkpoint_dir is not None:
                _write_condition_checkpoint(
                    checkpoint_dir,
                    manifest_task,
                    key,
                    split,
                    frame,
                    generator_provenance,
                )

        result = {
            "geometry": manifest_task.geometry.copy(),
            "sections": manifest_task.sections.copy(),
            **condition_results,
            "geom_id": manifest_task.geom_id,
            "category": manifest_task.category,
            "control_points": manifest_task.cp8.tolist(),
            "condition_split": split_map,
            "seed": manifest_task.seed,
            "manifest_sha256": manifest_task.manifest_sha256,
            **generator_provenance,
        }
        elapsed = time.time() - t0
        print(
            f"[worker {pid}] geom {manifest_task.geom_id} 完成 "
            f"({elapsed:.1f}s, {len(manifest_task.conditions)} 工况)",
            flush=True,
        )
        return (manifest_task.geom_id, result, None)
    except Exception as error:
        err = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        print(f"[worker {pid}] geom {manifest_task.geom_id} FAIL: {err}", flush=True)
        return (manifest_task.geom_id, None, err)
    finally:
        close = getattr(sim, "close", None)
        if callable(close):
            try:
                close()
            except Exception as close_error:
                print(
                    f"[warn] worker {pid} geom {manifest_task.geom_id} close failed: "
                    f"{type(close_error).__name__}: {close_error}",
                    flush=True,
                )


def worker_run(task: tuple) -> tuple:
    """单 worker 处理一个几何：返回 (geom_idx, label, all_simulation_data 或 None)。

    Pool.map 会自动分发 task 到空闲 worker。
    """
    if len(task) in (3, 4) and isinstance(task[0], ManifestTask):
        return _worker_run_manifest(*task)

    geom_idx, geom_array, num_timesteps, device = task
    pid = os.getpid()
    t0 = time.time()
    try:
        sim = _SIMULATION(
            file_path=_config.file_path,
            geometry_baseline=_config.geometry_baseline,
            device_type=device,
            number_of_timesteps=num_timesteps or _config.number_of_timesteps,
        )
        if not np.array_equal(geom_array, _config.geometry_baseline):
            sim.change_propeller_geometry(section_data=geom_array)
        sim.run_all_simulation()
        elapsed = time.time() - t0
        print(f"[worker {pid}] geom {geom_idx} 完成 ({elapsed:.1f}s, "
              f"{len(sim.all_simulation_data)-1} 工况)", flush=True)
        return (geom_idx, sim.all_simulation_data, None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[worker {pid}] geom {geom_idx} FAIL: {err}", flush=True)
        return (geom_idx, None, err)


def load_geometries(npy_path: str | None, baseline: np.ndarray,
                    first_n: int | None) -> list[np.ndarray]:
    """与单进程版相同的几何加载逻辑。"""
    if npy_path is None:
        return [baseline]
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        sys.exit(f"✖ 几何 npy 形状不对：{arr.shape}，期望 (N, 22, 3)")
    if first_n is not None:
        arr = arr[:first_n]
    return [arr[i] for i in range(arr.shape[0])]


def cleanup_worktrees(worktree_root: str, keep: bool) -> None:
    """跑完后清理所有 /tmp/lfm_w<pid>/，避免 tmpfs 占用堆积。"""
    if keep:
        return
    root = Path(worktree_root)
    for d in root.glob("lfm_w*"):
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f"[warn] 清理 {d} 失败：{e}", flush=True)


def main() -> int:
    args = parse_args()
    try:
        validate_runtime_args(args)
    except ValueError as error:
        sys.exit(f"✖ invalid runtime configuration: {error}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_digest: str | None = None

    if args.manifest is not None:
        legacy_selectors = (
            args.geometry_npy is not None
            or args.first_n_geoms is not None
            or args.start_idx != 0
            or args.end_idx is not None
        )
        if legacy_selectors:
            sys.exit(
                "✖ --manifest cannot be combined with --geometry-npy, --first-n-geoms, "
                "--start-idx, or --end-idx; use --geom-ids for exact manifest IDs"
            )
        try:
            requested_ids = parse_geom_ids(args.geom_ids)
            requested_condition_splits = parse_condition_splits(
                args.condition_splits
            )
            requested_condition_keys = parse_condition_keys(args.condition_keys)
            manifest_tasks, completed_ids, manifest_digest = prepare_manifest_run(
                manifest_path=args.manifest,
                geom_ids=requested_ids,
                expected_sha256=args.manifest_sha256,
                output_dir=out_dir,
                condition_splits=requested_condition_splits,
                condition_keys=requested_condition_keys,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            sys.exit(f"✖ manifest validation failed: {error}")

        selected_count = len(manifest_tasks) + len(completed_ids)
        print(
            f"▸ manifest {manifest_digest[:12]}: selected {selected_count}, "
            f"validated resume {len(completed_ids)}, pending {len(manifest_tasks)}"
        )
        if completed_ids:
            print(f"▸ 跳过已完整 geom_id: {sorted(completed_ids)}")
        if not manifest_tasks:
            print("✅ 所选 manifest 几何均已有完整且元数据匹配的输出")
            return 0
        tasks = [
            (task, args.num_timesteps, args.device, out_dir)
            for task in manifest_tasks
        ]
    else:
        if args.condition_splits is not None or args.condition_keys is not None:
            sys.exit("✖ --condition-splits/--condition-keys require --manifest")
        if args.geom_ids is not None or args.manifest_sha256 is not None:
            sys.exit("✖ --geom-ids/--manifest-sha256 require --manifest")

        # Legacy geometry-npy/baseline mode: preserve positional global indices.
        saved_cwd = os.getcwd()
        try:
            os.chdir(SUB_CODE_ROOT)
            sys.path.insert(0, str(SUB_CODE_ROOT))
            import config as _main_cfg  # noqa: E402
            baseline = _main_cfg.geometry_baseline
        finally:
            os.chdir(saved_cwd)

        geometries = load_geometries(args.geometry_npy, baseline, args.first_n_geoms)
        start = max(0, args.start_idx)
        end = args.end_idx if args.end_idx is not None else len(geometries)
        end = min(end, len(geometries))
        if start >= end:
            sys.exit(f"✖ start-idx={start} >= end-idx={end}，无几何可跑")
        selected = [
            (global_idx, geometry)
            for global_idx, geometry in enumerate(geometries)
            if start <= global_idx < end
        ]
        print(
            f"▸ 几何总数 {len(geometries)}, 本次跑 [{start}, {end}) = {len(selected)} 个"
        )
        tasks = [
            (global_idx, geometry, args.num_timesteps, args.device)
            for global_idx, geometry in selected
        ]

    n_geoms = len(tasks)
    n_workers = min(args.workers, n_geoms) if n_geoms > 0 else 1
    print(f"▸ workers: {n_workers}, omp/worker: {args.omp_threads}, "
          f"device: {args.device}, timesteps: {args.num_timesteps or '<default>'}")
    if args.batch_size > 0:
        print(f"▸ 分批落盘: 每 {args.batch_size} 几何写一个 pkl")
    print(f"▸ 工作目录父级: {args.worktree_root}/lfm_w<pid>/")
    print(f"▸ submodule: {SUBMODULE_ROOT}")

    # 用 fork 上下文（子进程继承 LD_LIBRARY_PATH 与 sys.path）
    # spawn 也可以但启动慢、需要重新 import；fork 在 Linux 上更高效
    ctx = mp.get_context("fork")

    t_start = time.time()
    aggregated: dict[str, dict] = {}   # 当前 batch 缓冲
    all_failed: list[tuple] = []
    n_done_total = 0
    n_batches_written = 0
    stamp = time.strftime("%Y%m%d_%H%M%S")

    def flush_batch():
        """把当前 batch buffer 落盘并清空。"""
        nonlocal aggregated, n_batches_written
        if not aggregated:
            return
        out_pkl = out_dir / output_filename(
            tag=args.tag,
            stamp=stamp,
            batch_number=n_batches_written,
            batched=args.batch_size > 0,
            manifest_sha256=manifest_digest,
        )
        with open(out_pkl, "wb") as f:
            pickle.dump(aggregated, f)
        print(f"  💾 落盘 {out_pkl.name} ({len(aggregated)} 几何)", flush=True)
        aggregated = {}
        n_batches_written += 1

    try:
        with ctx.Pool(processes=n_workers,
                      initializer=worker_init,
                      initargs=(args.worktree_root, args.omp_threads)) as pool:
            for (geom_idx, result, err) in pool.imap_unordered(worker_run, tasks):
                if result is not None:
                    aggregated[f"geometry_{geom_idx}"] = result
                else:
                    all_failed.append((geom_idx, err))
                n_done_total += 1
                print(f"  进度: {n_done_total}/{n_geoms} "
                      f"(成功 {n_done_total - len(all_failed)}, "
                      f"失败 {len(all_failed)})", flush=True)
                # 分批落盘：避免崩了丢全部
                if args.batch_size > 0 and len(aggregated) >= args.batch_size:
                    flush_batch()
        # 收尾：把剩余的也落盘
        flush_batch()
    finally:
        cleanup_worktrees(args.worktree_root, args.keep_worktree)

    total = time.time() - t_start
    print()
    print(f"✅ 总用时 {total:.1f}s（壁钟）={total/60:.1f}min={total/3600:.2f}h")
    print(f"   平均每几何 {total/max(n_geoms,1):.1f}s（壁钟）")
    print(f"   理论加速比 ≈ {n_workers}× vs 单进程")
    print(f"   产出 {n_batches_written} 个 pkl 文件，目录: {out_dir}")
    if all_failed:
        print(f"   ⚠ {len(all_failed)} 个几何失败：{[i for i,_ in all_failed]}")
    print(f"   下一步: cd {LFM_ROOT} && python data.py")
    return 0 if not all_failed else 2


if __name__ == "__main__":
    sys.exit(main())
