"""Create deterministic geometry and condition manifests for data supplementation."""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import qmc

from optimization_v2.geometry import (
    CHORD_COLS,
    CHORD_KNOTS,
    N_SECTIONS,
    R_NORM,
    TWIST_COLS,
    TWIST_KNOTS,
    cp_to_sections,
)


BASE_RPMS = (4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0)
BASE_ANGLES = (82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0)
INTERP_RPMS = (4250.0, 5250.0, 6250.0)
INTERP_ANGLES = (82.5, 83.5, 85.0, 86.5, 87.5)
WIND = 10.0
MANIFEST_VERSION = 1


@dataclass(frozen=True, order=True)
class Condition:
    """One QBlade operating condition."""

    rpm: float
    wind: float
    angle: float

    @property
    def key(self) -> tuple[float, float, float]:
        return (self.rpm, self.wind, self.angle)


@dataclass(frozen=True)
class CpRecovery:
    """A recovered CP vector together with its round-trip error."""

    cp8: np.ndarray
    max_abs_error: float


def build_condition_sets() -> dict[str, list[Condition]]:
    """Return the fixed, mutually exclusive grids for each new geometry."""
    base = [Condition(rpm, WIND, angle) for rpm in BASE_RPMS for angle in BASE_ANGLES]
    interpolation = [
        Condition(rpm, WIND, angle)
        for rpm in INTERP_RPMS
        for angle in INTERP_ANGLES
    ]
    alpha1 = [Condition(rpm, WIND, 89.0) for rpm in BASE_RPMS]
    return {
        "base42": base,
        "condition_id_interp": interpolation,
        "condition_ood_alpha1": alpha1,
    }


def _basis_matrix(knots: np.ndarray) -> np.ndarray:
    eye = np.eye(4)
    return np.column_stack([BSpline(knots, eye[index], 3)(R_NORM) for index in range(4)])


def recover_cp8(sections: np.ndarray, tolerance: float = 1e-10) -> CpRecovery:
    """Recover 4 chord and 4 twist control points from 44 historical sections."""
    sections = np.asarray(sections, dtype=np.float64)
    expected_shape = (2 * N_SECTIONS,)
    if sections.shape != expected_shape:
        raise ValueError(f"expected {expected_shape[0]} sections, got {sections.shape}")
    if not np.all(np.isfinite(sections)):
        raise ValueError("non-finite sections")

    chord_basis = _basis_matrix(CHORD_KNOTS)
    twist_basis = _basis_matrix(TWIST_KNOTS)
    chord_sections, twist_sections = np.split(sections, 2)
    cp8 = np.concatenate(
        [
            np.linalg.lstsq(chord_basis, chord_sections, rcond=None)[0],
            np.linalg.lstsq(twist_basis, twist_sections, rcond=None)[0],
        ]
    )
    if not np.all(np.isfinite(cp8)):
        raise ValueError("non-finite recovered cp8")
    rebuilt = cp_to_sections(cp8)
    if not np.all(np.isfinite(rebuilt)):
        raise ValueError("non-finite rebuilt sections")
    max_abs_error = float(np.max(np.abs(rebuilt - sections)))
    if not np.isfinite(max_abs_error):
        raise ValueError("non-finite reconstruction error")
    if max_abs_error > tolerance:
        raise ValueError(
            f"reconstruction error {max_abs_error:.6e} exceeds {tolerance:.6e}"
        )
    return CpRecovery(cp8=cp8, max_abs_error=max_abs_error)


def derive_cp_bounds(recoveries: np.ndarray) -> np.ndarray:
    """Derive unexpanded per-control-point bounds from accepted recoveries."""
    recoveries = np.asarray(recoveries, dtype=np.float64)
    if recoveries.ndim != 2 or recoveries.shape[1] != 8 or not len(recoveries):
        raise ValueError(f"expected recoveries with shape (n, 8), got {recoveries.shape}")
    bounds = np.column_stack([recoveries.min(axis=0), recoveries.max(axis=0)])
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("cannot sample CP bounds with a zero-width dimension")
    return bounds


def sample_id_cp8(bounds: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sample CPs strictly inside the historical data bounds using seeded LHS."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (8, 2):
        raise ValueError(f"expected bounds with shape (8, 2), got {bounds.shape}")
    if n <= 0:
        raise ValueError(f"expected a positive sample count, got {n}")
    unit = qmc.LatinHypercube(d=8, seed=seed).random(n)
    return qmc.scale(0.1 + 0.8 * unit, bounds[:, 0], bounds[:, 1])


def make_ood_cp8(id_cp: np.ndarray, bounds: np.ndarray, sample_index: int) -> np.ndarray:
    """Move exactly one CP dimension 5% beyond its training-data boundary."""
    cp8 = np.asarray(id_cp, dtype=np.float64).copy()
    bounds = np.asarray(bounds, dtype=np.float64)
    if cp8.shape != (8,) or bounds.shape != (8, 2):
        raise ValueError("expected id_cp shape (8,) and bounds shape (8, 2)")
    dimension = sample_index % 8
    side = -1 if (sample_index // 8 + sample_index) % 2 == 0 else 1
    span = bounds[dimension, 1] - bounds[dimension, 0]
    cp8[dimension] = (
        bounds[dimension, 0] - 0.05 * span
        if side < 0
        else bounds[dimension, 1] + 0.05 * span
    )
    return cp8


def _sections_from_row(row: dict[str, str]) -> np.ndarray:
    columns = [*CHORD_COLS, *TWIST_COLS]
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"missing required historical section columns: {', '.join(missing)}")
    try:
        return np.asarray([row[column] for column in columns], dtype=np.float64)
    except ValueError as error:
        raise ValueError("historical section values must be numeric") from error


def _recover_history(history_csv: str | Path) -> tuple[np.ndarray, dict]:
    """Recover one CP vector per geometry, excluding invalid or inconsistent records."""
    geometry_sections: dict[int, np.ndarray] = {}
    rejected: dict[int, str] = {}
    row_count = 0
    with Path(history_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "geom_idx" not in reader.fieldnames:
            raise ValueError("historical CSV must contain geom_idx")
        for row in reader:
            row_count += 1
            try:
                geom_idx = int(row["geom_idx"])
            except (TypeError, ValueError) as error:
                raise ValueError("geom_idx values must be integers") from error
            try:
                sections = _sections_from_row(row)
                recovery = recover_cp8(sections)
            except ValueError as error:
                rejected[geom_idx] = str(error)
                continue
            existing = geometry_sections.get(geom_idx)
            if existing is not None and not np.array_equal(existing, recovery.cp8):
                rejected[geom_idx] = "inconsistent reconstructable sections for geom_idx"
                continue
            geometry_sections[geom_idx] = recovery.cp8

    for geom_idx in rejected:
        geometry_sections.pop(geom_idx, None)
    accepted_ids = sorted(geometry_sections)
    if not accepted_ids:
        raise ValueError("no reconstructable historical geometry records")
    audit = {
        "input_row_count": row_count,
        "accepted_geometry_ids": accepted_ids,
        "rejected_geometry_ids": sorted(rejected),
        "rejected": [
            {"geom_idx": geom_idx, "reason": rejected[geom_idx]}
            for geom_idx in sorted(rejected)
        ],
    }
    return np.vstack([geometry_sections[geom_idx] for geom_idx in accepted_ids]), audit


def _condition_entries() -> list[dict[str, float | str]]:
    entries = []
    for category, conditions in build_condition_sets().items():
        entries.extend(
            {
                "rpm": condition.rpm,
                "wind": condition.wind,
                "angle": condition.angle,
                "category": category,
            }
            for condition in conditions
        )
    if len(entries) != 63 or len({(entry["rpm"], entry["wind"], entry["angle"])
                                   for entry in entries}) != 63:
        raise RuntimeError("configured condition sets must contain 63 unique conditions")
    return entries


def _geometry_entry(geom_id: int, category: str, cp8: np.ndarray, seed: int) -> dict:
    return {
        "geom_id": geom_id,
        "category": category,
        "seed": seed,
        "cp8": np.asarray(cp8, dtype=np.float64).tolist(),
        "sections": cp_to_sections(cp8).tolist(),
        "conditions": _condition_entries(),
    }


def generate_manifest(history_csv: str | Path, seed: int = 20260727) -> dict:
    """Generate the fixed 10 ID and 10 one-dimensional geometry-OOD records."""
    recoveries, recovery_audit = _recover_history(history_csv)
    bounds = derive_cp_bounds(recoveries)
    id_cp8 = sample_id_cp8(bounds, n=10, seed=seed)
    ood_cp8 = [make_ood_cp8(id_cp8[index], bounds, index) for index in range(10)]
    geometries = [
        _geometry_entry(1000 + index, "geometry_id", cp8, seed)
        for index, cp8 in enumerate(id_cp8)
    ]
    geometries.extend(
        _geometry_entry(1010 + index, "geometry_ood", cp8, seed)
        for index, cp8 in enumerate(ood_cp8)
    )
    return {
        "manifest_version": MANIFEST_VERSION,
        "seed": seed,
        "cp_order": ["chord_cp_0", "chord_cp_1", "chord_cp_2", "chord_cp_3",
                     "twist_cp_0", "twist_cp_1", "twist_cp_2", "twist_cp_3"],
        "cp_bounds": bounds.tolist(),
        "recovery_audit": recovery_audit,
        "geometries": geometries,
    }


def canonical_json(payload: dict) -> str:
    """Serialize a manifest in its stable, byte-for-byte canonical form."""
    return json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_manifest(manifest: dict, output_path: str | Path) -> str:
    """Write canonical manifest JSON plus a SHA-256 digest sidecar."""
    output_path = Path(output_path)
    payload = canonical_json(manifest).encode("utf-8")
    output_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(f"{digest}\n")
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args(argv)
    manifest = generate_manifest(args.dataset_csv, seed=args.seed)
    write_manifest(manifest, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
