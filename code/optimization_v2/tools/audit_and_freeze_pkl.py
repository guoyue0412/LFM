"""Audit QBlade pickle keys and export a provenance-preserving frozen dataset.

The primary identity of a simulation record is ``(geom_id, RPM, WIND, ANGLE)``.
Files and batch numbers are provenance only and never participate in identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from optimization_v2.tools.convert_pkl_to_dataset import aggregate_timeseries
from optimization_v2.tools.supplement_manifest import (
    BASE_ANGLES,
    BASE_RPMS,
    WIND,
    recover_cp8,
)


SEED = 20260727
CASE_PATTERN = re.compile(
    r"RPM(?P<rpm>[+-]?(?:\d+(?:\.\d*)?|\.\d+))_"
    r"Wind(?P<wind>[+-]?(?:\d+(?:\.\d*)?|\.\d+))_"
    r"Angle(?P<angle>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)
GEOMETRY_KEY_PATTERN = re.compile(r"geometry_(\d+)")
BASE_CONDITION_KEYS = tuple(
    (int(rpm), float(WIND), float(angle))
    for rpm in BASE_RPMS
    for angle in BASE_ANGLES
)
KEY_COLUMNS = ["geom_id", "RPM", "WIND", "ANGLE"]
CP_COLUMNS = [f"chord_cp_{index}" for index in range(4)] + [
    f"twist_cp_{index}" for index in range(4)
]
SECTION_COLUMNS = [f"chord_{index}" for index in range(22)] + [
    f"twist_{index}" for index in range(22)
]
OUTPUT_COLUMNS = ["T", "H", "My", "Q"]
PROVENANCE_COLUMNS = [
    "category",
    "condition_split",
    "manifest_sha256",
    "output_sign_source",
    "geom_id_source",
    "generator_git_commit",
    "source_file",
]
DATA_COLUMNS = (
    KEY_COLUMNS + CP_COLUMNS + SECTION_COLUMNS + OUTPUT_COLUMNS + PROVENANCE_COLUMNS
)
EXTERNAL_PARTITION_NAMES = (
    "geometry_id__base42",
    "geometry_id__condition_id_interp",
    "geometry_id__condition_ood_alpha1",
    "geometry_ood__base42",
    "geometry_ood__condition_id_interp",
    "geometry_ood__condition_ood_alpha1",
)
PARTITION_NAMES = (
    "historical_train__base42",
    "historical_train_47d_only__base42",
    *EXTERNAL_PARTITION_NAMES,
)
PARTITION_FILES = {
    "historical_train__base42": "training",
    "historical_train_47d_only__base42": "training_47d_only",
    "geometry_id__base42": "geometry_id_base42",
    "geometry_id__condition_id_interp": "id_interpolation",
    "geometry_id__condition_ood_alpha1": "condition_ood",
    "geometry_ood__base42": "geometry_ood_base42",
    "geometry_ood__condition_id_interp": "geometry_ood_id_interp",
    "geometry_ood__condition_ood_alpha1": "geometry_ood_condition_ood",
}


ConditionKey = tuple[int, float, float]
RecordKey = tuple[int, int, float, float]


@dataclass(frozen=True)
class AuditRecord:
    """One normalized condition candidate retained with its provenance."""

    geom_id: int
    condition_key: ConditionKey
    outputs: dict[str, float]
    output_sign_source: str
    valid_outputs: bool
    geom_id_source: str
    manifest_sha_match: bool
    manifest_sha256: str
    generator_git_commit: str
    mtime: float
    category: str
    condition_split: str
    cp8: tuple[float, ...] | None
    sections: tuple[float, ...] | None
    is_supplement: bool
    exclusion_reasons: tuple[str, ...]
    source_file: str

    @property
    def key(self) -> RecordKey:
        rpm, wind, angle = self.condition_key
        return (self.geom_id, rpm, wind, angle)


@dataclass
class AuditReport:
    """Complete audit evidence plus deterministic selected records."""

    expected_base_ids: tuple[int, ...]
    selected_records: list[AuditRecord]
    duplicate_keys: list[RecordKey]
    missing_keys: list[RecordKey]
    duplicate_decisions: list[dict]
    unreadable_files: list[dict]
    geom_id_disagreements: list[dict]
    metadata_errors: list[dict]
    excluded_cp8_recoveries: list[dict]
    record_exclusions: list[dict]
    per_geometry_coverage: dict[int, dict]
    pkl_counts: dict[str, int]
    input_paths: list[str]
    manifest_sha256: str = ""
    manifest_path: str = ""

    @property
    def unique_geometry_count(self) -> int:
        return len({record.geom_id for record in self.selected_records})

    @property
    def unique_geometry_ids(self) -> list[int]:
        return sorted({record.geom_id for record in self.selected_records})

    def training_records(self) -> list[AuditRecord]:
        """Return only complete, valid, reconstructable historical base42 rows."""
        records: list[AuditRecord] = []
        base_set = set(BASE_CONDITION_KEYS)
        for geom_id in self.expected_base_ids:
            geometry_records = {
                record.condition_key: record
                for record in self.selected_records
                if record.geom_id == geom_id
                and record.category == "historical_train"
                and record.condition_key in base_set
            }
            if set(geometry_records) != base_set:
                continue
            if not all(
                record.valid_outputs
                and record.cp8 is not None
                and record.sections is not None
                for record in geometry_records.values()
            ):
                continue
            records.extend(geometry_records[key] for key in BASE_CONDITION_KEYS)
        return sorted(records, key=lambda record: record.key)

    def historical_47d_only_records(self) -> list[AuditRecord]:
        """Return complete historical base42 geometries lacking recoverable cp8."""
        records: list[AuditRecord] = []
        base_set = set(BASE_CONDITION_KEYS)
        for geom_id in self.expected_base_ids:
            geometry_records = {
                record.condition_key: record
                for record in self.selected_records
                if record.geom_id == geom_id
                and record.category == "historical_train"
                and record.condition_key in base_set
            }
            if set(geometry_records) != base_set:
                continue
            if not all(
                record.valid_outputs and record.sections is not None
                for record in geometry_records.values()
            ):
                continue
            if all(record.cp8 is not None for record in geometry_records.values()):
                continue
            records.extend(geometry_records[key] for key in BASE_CONDITION_KEYS)
        return sorted(records, key=lambda record: record.key)


@dataclass(frozen=True)
class _ManifestInfo:
    digest: str
    path: str
    geometries: dict[int, dict]
    conditions: dict[tuple[int, ConditionKey], str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _condition_tuple(rpm: object, wind: object, angle: object) -> ConditionKey:
    rpm_number = _number(rpm, "RPM")
    if not rpm_number.is_integer():
        raise ValueError(f"RPM must be an integer-valued number, got {rpm_number}")
    return (int(rpm_number), _number(wind, "WIND"), _number(angle, "ANGLE"))


def _parse_condition_key(key: object) -> ConditionKey | None:
    if not isinstance(key, str):
        return None
    match = CASE_PATTERN.fullmatch(key)
    if match is None:
        return None
    return _condition_tuple(match["rpm"], match["wind"], match["angle"])


def _load_manifest(path: str | Path | None) -> _ManifestInfo | None:
    if path is None:
        return None
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    records = payload.get("geometries")
    if not isinstance(records, list):
        raise ValueError("manifest geometries must be a list")
    geometries: dict[int, dict] = {}
    conditions: dict[tuple[int, ConditionKey], str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("manifest geometry entries must be objects")
        geom_id = record.get("geom_id")
        if isinstance(geom_id, bool) or not isinstance(geom_id, int):
            raise ValueError(f"manifest geom_id must be an integer: {geom_id!r}")
        if geom_id in geometries:
            raise ValueError(f"duplicate manifest geom_id: {geom_id}")
        cp8 = np.asarray(record.get("cp8"), dtype=np.float64)
        if cp8.shape != (8,) or not np.all(np.isfinite(cp8)):
            raise ValueError(f"manifest geometry {geom_id} must contain finite cp8")
        geometries[geom_id] = record
        declared = record.get("conditions")
        if not isinstance(declared, list):
            raise ValueError(f"manifest geometry {geom_id} conditions must be a list")
        for condition in declared:
            if not isinstance(condition, dict):
                raise ValueError(f"manifest geometry {geom_id} condition must be an object")
            key = _condition_tuple(
                condition.get("rpm"), condition.get("wind"), condition.get("angle")
            )
            lookup_key = (geom_id, key)
            if lookup_key in conditions:
                raise ValueError(f"duplicate manifest condition key: {lookup_key}")
            split = condition.get("category")
            if split not in {
                "base42",
                "condition_id_interp",
                "condition_ood_alpha1",
            }:
                raise ValueError(f"unknown manifest condition split: {split!r}")
            conditions[lookup_key] = split
    return _ManifestInfo(
        digest=_sha256(manifest_path),
        path=str(manifest_path),
        geometries=geometries,
        conditions=conditions,
    )


def _stored_geom_id(node: Mapping) -> int | None:
    value = node.get("geom_id")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("stored geom_id must be an integer")
    number = _number(value, "stored geom_id")
    if not number.is_integer():
        raise ValueError("stored geom_id must be an integer")
    return int(number)


def _geometry_sections(geometry: object) -> np.ndarray:
    array = np.asarray(geometry, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 22:
        raise ValueError(f"expected geometry with exactly 22 rows, got {array.shape}")
    if array.shape[1] >= 3:
        sections = np.concatenate((array[:, 1], array[:, 2]))
    elif array.shape[1] == 2:
        sections = np.concatenate((array[:, 0], array[:, 1]))
    else:
        raise ValueError(f"expected geometry with at least 2 columns, got {array.shape}")
    if not np.all(np.isfinite(sections)):
        raise ValueError("geometry sections must be finite")
    return sections


def _manifest_geometry(
    node: Mapping, geom_id: int, manifest: _ManifestInfo
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    record = manifest.geometries[geom_id]
    cp8 = np.asarray(record["cp8"], dtype=np.float64)
    stored = node.get("control_points")
    try:
        stored_cp8 = np.asarray(stored, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("stored control_points are not numeric") from error
    if stored_cp8.shape != (8,) or not np.array_equal(stored_cp8, cp8):
        raise ValueError("stored control_points do not match exact manifest cp8")
    expected_sections = np.asarray(record.get("sections"), dtype=np.float64)
    stored_sections = _geometry_sections(node.get("geometry"))
    if expected_sections.shape != (44,) or not np.array_equal(
        stored_sections, expected_sections
    ):
        raise ValueError("stored geometry does not match exact manifest sections")
    return (
        tuple(float(value) for value in cp8),
        tuple(float(value) for value in stored_sections),
    )


def _candidate_rank(record: AuditRecord) -> tuple:
    return (
        record.valid_outputs,
        record.geom_id_source == "stored_geom_id",
        record.manifest_sha_match,
        bool(record.generator_git_commit.strip()),
        record.mtime,
        record.source_file,
    )


def _decision_reason(selected: AuditRecord, discarded: AuditRecord) -> str:
    checks = (
        (selected.valid_outputs, discarded.valid_outputs, "valid_finite_outputs"),
        (
            selected.geom_id_source == "stored_geom_id",
            discarded.geom_id_source == "stored_geom_id",
            "explicit_stored_geom_id",
        ),
        (selected.manifest_sha_match, discarded.manifest_sha_match, "manifest_sha_match"),
        (
            bool(selected.generator_git_commit.strip()),
            bool(discarded.generator_git_commit.strip()),
            "generator_git_metadata",
        ),
    )
    for selected_value, discarded_value, reason in checks:
        if selected_value != discarded_value:
            return reason
    if selected.mtime != discarded.mtime:
        return "newest_mtime"
    return "source_path_tiebreak"


def audit_sources(
    source_dirs: Sequence[str | Path],
    *,
    expected_base_ids: Iterable[int],
    manifest_path: str | Path | None = None,
    last_n: int = 120,
) -> AuditReport:
    """Audit all pickle keys and deterministically select one record per key."""
    manifest = _load_manifest(manifest_path)
    expected_ids = tuple(sorted({int(value) for value in expected_base_ids}))
    input_paths = [str(Path(path).resolve()) for path in source_dirs]
    pkl_counts: dict[str, int] = {}
    paths: dict[str, Path] = {}
    for source_dir in source_dirs:
        resolved = Path(source_dir).resolve()
        found = sorted(path for path in resolved.rglob("*.pkl") if path.is_file())
        pkl_counts[str(resolved)] = len(found)
        paths.update({str(path.resolve()): path.resolve() for path in found})

    candidates: dict[RecordKey, list[AuditRecord]] = defaultdict(list)
    unreadable_files: list[dict] = []
    disagreements: list[dict] = []
    metadata_errors: list[dict] = []
    excluded_cp8: list[dict] = []

    for source_file in sorted(paths.values()):
        try:
            with source_file.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as error:  # pickle compatibility failures are audit evidence
            unreadable_files.append(
                {"source_file": str(source_file), "reason": f"{type(error).__name__}: {error}"}
            )
            continue
        if not isinstance(payload, dict):
            metadata_errors.append(
                {"source_file": str(source_file), "reason": "top-level payload is not a dict"}
            )
            continue

        geometry_items = [
            (key, value)
            for key, value in payload.items()
            if isinstance(key, str) and GEOMETRY_KEY_PATTERN.fullmatch(key)
        ]
        if not geometry_items:
            geometry_items = [(None, payload)]

        for geometry_key, node in geometry_items:
            if not isinstance(node, dict):
                metadata_errors.append(
                    {
                        "source_file": str(source_file),
                        "geometry_key": geometry_key,
                        "reason": "geometry node is not a dict",
                    }
                )
                continue
            key_id = (
                int(GEOMETRY_KEY_PATTERN.fullmatch(geometry_key).group(1))
                if geometry_key is not None
                else None
            )
            try:
                stored_id = _stored_geom_id(node)
            except ValueError as error:
                metadata_errors.append(
                    {"source_file": str(source_file), "reason": str(error)}
                )
                continue
            if key_id is not None and stored_id is not None and key_id != stored_id:
                disagreements.append(
                    {
                        "source_file": str(source_file),
                        "geometry_key_id": key_id,
                        "stored_geom_id": stored_id,
                    }
                )
                continue
            geom_id = stored_id if stored_id is not None else key_id
            if geom_id is None:
                metadata_errors.append(
                    {
                        "source_file": str(source_file),
                        "reason": (
                            "geometry identity requires a geometry_<id> node or stored "
                            "geom_id; filenames and batch IDs are ignored"
                        ),
                    }
                )
                continue
            geom_id_source = "stored_geom_id" if stored_id is not None else "inferred_geom_id"

            manifest_record = manifest.geometries.get(geom_id) if manifest else None
            stored_sha = node.get("manifest_sha256", "")
            stored_sha = stored_sha if isinstance(stored_sha, str) else ""
            manifest_sha_match = bool(
                manifest_record is not None and stored_sha == manifest.digest
            )
            node_exclusion_reasons: list[str] = []
            if manifest_record is not None:
                expected_category = manifest_record.get("category")
                stored_category = node.get("category")
                category = stored_category if isinstance(stored_category, str) else ""
                if category != expected_category:
                    node_exclusion_reasons.append("category_mismatch")
                    metadata_errors.append(
                        {
                            "source_file": str(source_file),
                            "geom_id": geom_id,
                            "reason": "category_mismatch",
                            "stored": stored_category,
                            "expected": expected_category,
                        }
                    )
                stored_split_map = node.get("condition_split")
                if not isinstance(stored_split_map, Mapping):
                    stored_split_map = None
                    node_exclusion_reasons.append("condition_split_not_mapping")
                    metadata_errors.append(
                        {
                            "source_file": str(source_file),
                            "geom_id": geom_id,
                            "reason": "condition_split_not_mapping",
                        }
                    )
                if not manifest_sha_match:
                    node_exclusion_reasons.append("manifest_sha256_mismatch")
                    metadata_errors.append(
                        {
                            "source_file": str(source_file),
                            "geom_id": geom_id,
                            "reason": "manifest_sha256_mismatch",
                            "stored": stored_sha,
                            "expected": manifest.digest,
                        }
                    )
                try:
                    cp8, sections = _manifest_geometry(node, geom_id, manifest)
                except ValueError as error:
                    cp8, sections = None, None
                    node_exclusion_reasons.append("manifest_geometry_mismatch")
                    excluded_cp8.append(
                        {
                            "geom_id": geom_id,
                            "source_file": str(source_file),
                            "reason": str(error),
                        }
                    )
            else:
                category = "historical_train"
                stored_split_map = None
                try:
                    recovered_sections = _geometry_sections(node.get("geometry"))
                    sections = tuple(float(value) for value in recovered_sections)
                except (TypeError, ValueError) as error:
                    cp8, sections = None, None
                    excluded_cp8.append(
                        {
                            "geom_id": geom_id,
                            "source_file": str(source_file),
                            "reason": str(error),
                        }
                    )
                else:
                    try:
                        recovery = recover_cp8(recovered_sections, tolerance=1e-10)
                        cp8 = tuple(float(value) for value in recovery.cp8)
                    except ValueError as error:
                        cp8 = None
                        excluded_cp8.append(
                            {
                                "geom_id": geom_id,
                                "source_file": str(source_file),
                                "reason": str(error),
                            }
                        )
            generator_commit = node.get("generator_git_commit", "")
            generator_commit = generator_commit if isinstance(generator_commit, str) else ""
            mtime = source_file.stat().st_mtime

            for case_key, case_frame in node.items():
                try:
                    condition = _parse_condition_key(case_key)
                except ValueError as error:
                    metadata_errors.append(
                        {
                            "source_file": str(source_file),
                            "geometry_key": geometry_key,
                            "condition_key": str(case_key),
                            "reason": str(error),
                        }
                    )
                    continue
                if condition is None:
                    continue
                if manifest_record is not None:
                    expected_split = manifest.conditions.get((geom_id, condition))
                    stored_split = (
                        stored_split_map.get(case_key)
                        if stored_split_map is not None
                        else None
                    )
                    condition_split = stored_split if isinstance(stored_split, str) else ""
                    exclusion_reasons = list(node_exclusion_reasons)
                    if expected_split is None:
                        exclusion_reasons.append("condition_absent_from_manifest")
                        metadata_errors.append(
                            {
                                "source_file": str(source_file),
                                "geom_id": geom_id,
                                "condition_key": list(condition),
                                "reason": "condition_absent_from_manifest",
                            }
                        )
                    elif stored_split != expected_split:
                        exclusion_reasons.append("condition_split_mismatch")
                        metadata_errors.append(
                            {
                                "source_file": str(source_file),
                                "geom_id": geom_id,
                                "condition_key": list(condition),
                                "reason": "condition_split_mismatch",
                                "stored": stored_split,
                                "expected": expected_split,
                            }
                        )
                else:
                    condition_split = (
                        "base42" if condition in BASE_CONDITION_KEYS else "undeclared"
                    )
                    exclusion_reasons = []
                if isinstance(case_frame, pd.DataFrame):
                    outputs, sign_source = aggregate_timeseries(case_frame, last_n=last_n)
                else:
                    outputs, sign_source = {}, "invalid_non_dataframe"
                valid_outputs = set(outputs) == set(OUTPUT_COLUMNS) and all(
                    np.isfinite(value) for value in outputs.values()
                )
                if manifest_record is not None and not valid_outputs:
                    exclusion_reasons.append("invalid_outputs")
                record = AuditRecord(
                    geom_id=geom_id,
                    condition_key=condition,
                    outputs=outputs,
                    output_sign_source=sign_source,
                    valid_outputs=valid_outputs,
                    geom_id_source=geom_id_source,
                    manifest_sha_match=manifest_sha_match,
                    manifest_sha256=stored_sha,
                    generator_git_commit=generator_commit,
                    mtime=mtime,
                    category=category,
                    condition_split=condition_split,
                    cp8=cp8,
                    sections=sections,
                    is_supplement=manifest_record is not None,
                    exclusion_reasons=tuple(dict.fromkeys(exclusion_reasons)),
                    source_file=str(source_file),
                )
                candidates[record.key].append(record)

    selected_records: list[AuditRecord] = []
    duplicate_decisions: list[dict] = []
    duplicate_keys = sorted(key for key, records in candidates.items() if len(records) > 1)
    for key in sorted(candidates):
        records = candidates[key]
        selected = max(records, key=_candidate_rank)
        selected_records.append(selected)
        for discarded in sorted(
            (record for record in records if record is not selected),
            key=lambda record: record.source_file,
        ):
            duplicate_decisions.append(
                {
                    "geom_id": key[0],
                    "RPM": key[1],
                    "WIND": key[2],
                    "ANGLE": key[3],
                    "selected_source": selected.source_file,
                    "discarded_source": discarded.source_file,
                    "reason": _decision_reason(selected, discarded),
                }
            )

    expected_keys = {
        (geom_id, *condition)
        for geom_id in expected_ids
        for condition in BASE_CONDITION_KEYS
    }
    missing_keys = sorted(expected_keys - set(candidates))
    per_geometry_coverage: dict[int, dict] = {}
    selected_by_key = {record.key: record for record in selected_records}
    observed_ids = {key[0] for key in candidates}
    manifest_ids = set(manifest.geometries) if manifest else set()
    for geom_id in sorted(set(expected_ids) | observed_ids | manifest_ids):
        if geom_id in expected_ids:
            geometry_expected = {
                (geom_id, *condition) for condition in BASE_CONDITION_KEYS
            }
        elif manifest:
            geometry_expected = {
                (geom_id, *condition)
                for candidate_geom_id, condition in manifest.conditions
                if candidate_geom_id == geom_id
            }
        else:
            geometry_expected = set()
        present = {key for key in candidates if key[0] == geom_id}
        missing = sorted(geometry_expected - present)
        per_geometry_coverage[geom_id] = {
            "expected_key_count": len(geometry_expected),
            "present_key_count": len(present & geometry_expected),
            "extra_key_count": len(present - geometry_expected),
            "missing_key_count": len(missing),
            "duplicate_key_count": sum(key[0] == geom_id for key in duplicate_keys),
            "valid_selected_key_count": sum(
                record.valid_outputs
                for key, record in selected_by_key.items()
                if key[0] == geom_id
            ),
            "cp8_available": any(
                record.cp8 is not None
                for key, record in selected_by_key.items()
                if key[0] == geom_id
            ),
            "missing_keys": [list(key) for key in missing],
        }

    excluded_unique = {
        (entry["geom_id"], entry["source_file"], entry["reason"]): entry
        for entry in excluded_cp8
    }
    record_exclusions = [
        {
            "geom_id": record.key[0],
            "RPM": record.key[1],
            "WIND": record.key[2],
            "ANGLE": record.key[3],
            "source_file": record.source_file,
            "reasons": list(record.exclusion_reasons),
        }
        for record in selected_records
        if record.is_supplement and record.exclusion_reasons
    ]
    return AuditReport(
        expected_base_ids=expected_ids,
        selected_records=selected_records,
        duplicate_keys=duplicate_keys,
        missing_keys=missing_keys,
        duplicate_decisions=duplicate_decisions,
        unreadable_files=sorted(unreadable_files, key=lambda entry: entry["source_file"]),
        geom_id_disagreements=sorted(
            disagreements, key=lambda entry: entry["source_file"]
        ),
        metadata_errors=metadata_errors,
        excluded_cp8_recoveries=list(excluded_unique.values()),
        record_exclusions=record_exclusions,
        per_geometry_coverage=per_geometry_coverage,
        pkl_counts=pkl_counts,
        input_paths=input_paths,
        manifest_sha256=manifest.digest if manifest else "",
        manifest_path=manifest.path if manifest else "",
    )


def geometry_grouped_split(
    geometry_ids: Iterable[int], seed: int = SEED
) -> dict[str, list[int]]:
    """Split geometry IDs (never rows) into deterministic 80/10/10 groups."""
    ids = np.asarray(sorted({int(value) for value in geometry_ids}), dtype=np.int64)
    if not len(ids):
        return {"train": [], "validation": [], "test": []}
    shuffled = np.random.default_rng(seed).permutation(ids)
    if len(ids) < 3:
        return {
            "train": sorted(int(value) for value in shuffled),
            "validation": [],
            "test": [],
        }
    validation_count = max(1, int(round(0.1 * len(ids))))
    test_count = max(1, int(round(0.1 * len(ids))))
    if validation_count + test_count >= len(ids):
        validation_count = test_count = 1
    train_count = len(ids) - validation_count - test_count
    return {
        "train": sorted(int(value) for value in shuffled[:train_count]),
        "validation": sorted(
            int(value)
            for value in shuffled[train_count : train_count + validation_count]
        ),
        "test": sorted(int(value) for value in shuffled[train_count + validation_count :]),
    }


def representation_splits(
    common_ids: Iterable[int], only_47d_ids: Iterable[int], seed: int = SEED
) -> dict[str, dict]:
    """Build common/full splits without changing any common-ID assignment."""
    common_eligible = sorted({int(value) for value in common_ids})
    only_47d_eligible = sorted({int(value) for value in only_47d_ids})
    if set(common_eligible) & set(only_47d_eligible):
        raise ValueError("common and 47D-only geometry IDs must be disjoint")
    common = geometry_grouped_split(common_eligible, seed=seed)
    additional = geometry_grouped_split(only_47d_eligible, seed=seed)
    full = {
        name: sorted(common[name] + additional[name])
        for name in ("train", "validation", "test")
    }
    return {
        "common_11d_47d": {
            "eligible_geometry_ids": common_eligible,
            **common,
        },
        "full_47d": {
            "eligible_geometry_ids": sorted(common_eligible + only_47d_eligible),
            **full,
        },
        "relationship": {
            "common_assignments_preserved": True,
            "additional_47d_only_geometry_ids": only_47d_eligible,
            "additional_assignment": "independent_seeded_geometry_split",
        },
    }


def _record_row(record: AuditRecord, *, include_cp8: bool = True) -> dict:
    if record.sections is None or (include_cp8 and record.cp8 is None):
        raise ValueError("record lacks the geometry representation required for export")
    rpm, wind, angle = record.condition_key
    row = {
        "geom_id": record.geom_id,
        "RPM": rpm,
        "WIND": wind,
        "ANGLE": angle,
        **{
            column: record.sections[index]
            for index, column in enumerate(SECTION_COLUMNS)
        },
        **record.outputs,
        "category": record.category,
        "condition_split": record.condition_split,
        "manifest_sha256": record.manifest_sha256,
        "output_sign_source": record.output_sign_source,
        "geom_id_source": record.geom_id_source,
        "generator_git_commit": record.generator_git_commit,
        "source_file": record.source_file,
    }
    if include_cp8:
        row.update(
            {column: record.cp8[index] for index, column in enumerate(CP_COLUMNS)}
        )
    return row


def _frame(
    records: Sequence[AuditRecord], *, include_cp8: bool = True
) -> pd.DataFrame:
    rows = [
        _record_row(record, include_cp8=include_cp8)
        for record in sorted(records, key=lambda item: item.key)
    ]
    columns = DATA_COLUMNS if include_cp8 else [
        column for column in DATA_COLUMNS if column not in CP_COLUMNS
    ]
    return pd.DataFrame(rows, columns=columns)


def _json_ready(value):
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def freeze_datasets(
    report: AuditReport,
    output_dir: str | Path,
    *,
    version: str,
    input_paths: Mapping[str, str | Path] | None = None,
    git_commits: Mapping[str, str] | None = None,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Write disjoint historical/external partitions and their audit evidence."""
    if not version or not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("version must contain only letters, digits, dot, underscore, or dash")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_records = report.training_records()
    historical_47d_only = report.historical_47d_only_records()
    common_ids = sorted({record.geom_id for record in training_records})
    historical_47d_only_ids = sorted(
        {record.geom_id for record in historical_47d_only}
    )
    split_payload = representation_splits(
        common_ids, historical_47d_only_ids, seed=seed
    )
    common_split_ids = split_payload["common_11d_47d"]
    full_47d_split_ids = split_payload["full_47d"]
    partition_records: dict[str, list[AuditRecord]] = {
        name: [] for name in PARTITION_NAMES
    }
    partition_records["historical_train__base42"] = training_records
    partition_records["historical_train_47d_only__base42"] = historical_47d_only
    for record in report.selected_records:
        if not record.is_supplement:
            continue
        if record.exclusion_reasons:
            continue
        partition_name = f"{record.category}__{record.condition_split}"
        if partition_name not in EXTERNAL_PARTITION_NAMES:
            raise ValueError(
                f"record {record.key} has unexportable partition {partition_name!r}"
            )
        partition_records[partition_name].append(record)

    partitions = {
        name: _frame(
            partition_records[name],
            include_cp8=name != "historical_train_47d_only__base42",
        )
        for name in PARTITION_NAMES
    }
    supplement_keys = {
        record.key for record in report.selected_records if record.is_supplement
    }
    exported_supplement_keys = {
        record.key
        for name in EXTERNAL_PARTITION_NAMES
        for record in partition_records[name]
    }
    excluded_supplement_keys = {
        (entry["geom_id"], entry["RPM"], entry["WIND"], entry["ANGLE"])
        for entry in report.record_exclusions
    }
    if exported_supplement_keys & excluded_supplement_keys:
        raise RuntimeError("a supplement key cannot be both exported and excluded")
    if exported_supplement_keys | excluded_supplement_keys != supplement_keys:
        raise RuntimeError("every selected supplement key must be exported or excluded")
    common_train_ids = set(common_split_ids["train"])
    full_47d_train_ids = set(full_47d_split_ids["train"])
    normalizer_11d_records = [
        record for record in training_records if record.geom_id in common_train_ids
    ]
    normalizer_47d_records = [
        record
        for record in [*training_records, *historical_47d_only]
        if record.geom_id in full_47d_train_ids
    ]
    normalizer_11d_columns = ["RPM", "WIND", "ANGLE", *CP_COLUMNS]
    normalizer_47d_columns = ["RPM", "WIND", "ANGLE", *SECTION_COLUMNS]
    normalizer_inputs_11d = _frame(normalizer_11d_records)[normalizer_11d_columns]
    normalizer_inputs_47d = _frame(
        normalizer_47d_records, include_cp8=False
    )[normalizer_47d_columns]

    artifacts: list[Path] = []
    for name, frame in partitions.items():
        path = output_dir / f"{PARTITION_FILES[name]}_{version}.csv"
        frame.to_csv(path, index=False)
        artifacts.append(path)
    normalizer_11d_path = output_dir / f"normalizer_inputs_11d_{version}.csv"
    normalizer_inputs_11d.to_csv(normalizer_11d_path, index=False)
    artifacts.append(normalizer_11d_path)
    normalizer_47d_path = output_dir / f"normalizer_inputs_47d_{version}.csv"
    normalizer_inputs_47d.to_csv(normalizer_47d_path, index=False)
    artifacts.append(normalizer_47d_path)

    split_path = output_dir / f"geometry_split_{version}.json"
    _write_json(split_path, {"seed": seed, **split_payload})
    artifacts.append(split_path)

    coverage_payload = {
        "pkl_counts": report.pkl_counts,
        "unique_geometry_count": report.unique_geometry_count,
        "unique_geometry_ids": report.unique_geometry_ids,
        "per_geometry_coverage": report.per_geometry_coverage,
        "duplicate_keys": report.duplicate_keys,
        "missing_keys": report.missing_keys,
        "unreadable_files": report.unreadable_files,
        "geom_id_disagreements": report.geom_id_disagreements,
        "metadata_errors": report.metadata_errors,
        "excluded_cp8_recoveries": report.excluded_cp8_recoveries,
        "record_exclusions": report.record_exclusions,
    }
    coverage_json_path = output_dir / f"coverage_{version}.json"
    _write_json(coverage_json_path, coverage_payload)
    artifacts.append(coverage_json_path)
    coverage_csv_path = output_dir / f"coverage_{version}.csv"
    coverage_rows = [
        {"geom_id": geom_id, **{key: value for key, value in coverage.items() if key != "missing_keys"}}
        for geom_id, coverage in sorted(report.per_geometry_coverage.items())
    ]
    pd.DataFrame(coverage_rows).to_csv(coverage_csv_path, index=False)
    artifacts.append(coverage_csv_path)

    duplicate_path = output_dir / f"duplicate_decisions_{version}.csv"
    duplicate_columns = [
        "geom_id",
        "RPM",
        "WIND",
        "ANGLE",
        "selected_source",
        "discarded_source",
        "reason",
    ]
    pd.DataFrame(report.duplicate_decisions, columns=duplicate_columns).to_csv(
        duplicate_path, index=False
    )
    artifacts.append(duplicate_path)

    artifact_sha256 = {path.name: _sha256(path) for path in sorted(artifacts)}
    output_row_counts = {
        **{name: len(frame) for name, frame in partitions.items()},
        "normalizer_inputs_11d": len(normalizer_inputs_11d),
        "normalizer_inputs_47d": len(normalizer_inputs_47d),
    }
    split_ids = {
        representation: {
            name: split_payload[representation][name]
            for name in ("train", "validation", "test")
        }
        for representation in ("common_11d_47d", "full_47d")
    }
    eligible_geometry_ids = {
        representation: split_payload[representation]["eligible_geometry_ids"]
        for representation in ("common_11d_47d", "full_47d")
    }
    historical_47d_only_split_ids = {
        name: sorted(set(full_47d_split_ids[name]) & set(historical_47d_only_ids))
        for name in ("train", "validation", "test")
    }
    acceptance = {
        "version": version,
        "seed": seed,
        "input_paths": {
            key: str(Path(value).resolve()) for key, value in (input_paths or {}).items()
        },
        "audited_input_paths": report.input_paths,
        "git_commits": dict(git_commits or {}),
        "manifest_path": report.manifest_path,
        "manifest_sha256": report.manifest_sha256,
        "pkl_counts": report.pkl_counts,
        "unique_geometry_ids": report.unique_geometry_ids,
        "per_geometry_coverage": report.per_geometry_coverage,
        "duplicates": report.duplicate_decisions,
        "missing_keys": report.missing_keys,
        "excluded_cp8_recoveries": report.excluded_cp8_recoveries,
        "record_exclusions": report.record_exclusions,
        "unreadable_files": report.unreadable_files,
        "geom_id_disagreements": report.geom_id_disagreements,
        "metadata_errors": report.metadata_errors,
        "historical_47d_only_geometry_ids": historical_47d_only_ids,
        "historical_47d_only_geometry_count": len(historical_47d_only_ids),
        "historical_47d_only_split_ids": historical_47d_only_split_ids,
        "eligible_geometry_ids": eligible_geometry_ids,
        "representation_eligibility": {
            "common_11d_47d": {
                "eligible_geometry_ids": common_ids,
                "47d_only_geometry_ids": [],
                "47d_only_geometry_count": 0,
            },
            "full_47d": {
                "eligible_geometry_ids": eligible_geometry_ids["full_47d"],
                "47d_only_geometry_ids": historical_47d_only_ids,
                "47d_only_geometry_count": len(historical_47d_only_ids),
            },
        },
        "split_relationship": split_payload["relationship"],
        "supplement_selected_key_count": len(supplement_keys),
        "supplement_exported_key_count": len(exported_supplement_keys),
        "supplement_excluded_key_count": len(excluded_supplement_keys),
        "output_row_counts": output_row_counts,
        "split_ids": split_ids,
        "artifact_sha256": artifact_sha256,
    }
    acceptance_path = output_dir / f"acceptance_{version}.json"
    _write_json(acceptance_path, acceptance)
    acceptance_path.with_suffix(acceptance_path.suffix + ".sha256").write_text(
        f"{_sha256(acceptance_path)}\n"
    )
    return {
        **partitions,
        "normalizer_inputs_11d": normalizer_inputs_11d,
        "normalizer_inputs_47d": normalizer_inputs_47d,
    }


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--expected-base-start", type=int, default=0)
    parser.add_argument("--expected-base-stop", type=int, default=1000)
    parser.add_argument("--last-n", type=int, default=120)
    args = parser.parse_args(argv)
    report = audit_sources(
        [args.historical, args.supplement],
        expected_base_ids=range(args.expected_base_start, args.expected_base_stop),
        manifest_path=args.manifest,
        last_n=args.last_n,
    )
    freeze_datasets(
        report,
        args.out,
        version=args.version,
        input_paths={"historical": args.historical, "supplement": args.supplement},
        git_commits={
            "freezer": _git_commit(Path(__file__).resolve().parents[3]),
            "generator": ",".join(
                sorted(
                    {
                        record.generator_git_commit
                        for record in report.selected_records
                        if record.generator_git_commit
                    }
                )
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
