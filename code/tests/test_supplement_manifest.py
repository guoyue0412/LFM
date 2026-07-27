import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from optimization_v2.geometry import cp_to_sections
from optimization_v2.tools.supplement_manifest import build_condition_sets
from optimization_v2.tools.supplement_manifest import canonical_json, generate_manifest
from optimization_v2.tools.supplement_manifest import recover_cp8


def test_condition_sets_are_exact_and_disjoint():
    sets = build_condition_sets()

    assert len(sets["base42"]) == 42
    assert len(sets["condition_id_interp"]) == 15
    assert len(sets["condition_ood_alpha1"]) == 6

    assert {condition.key for condition in sets["base42"]} == {
        (rpm, 10.0, angle)
        for rpm in (4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0)
        for angle in (82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0)
    }
    assert {condition.key for condition in sets["condition_id_interp"]} == {
        (rpm, 10.0, angle)
        for rpm in (4250.0, 5250.0, 6250.0)
        for angle in (82.5, 83.5, 85.0, 86.5, 87.5)
    }

    keys = [{condition.key for condition in values} for values in sets.values()]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert {condition.rpm for condition in sets["condition_ood_alpha1"]} == {
        4000,
        4500,
        5000,
        5500,
        6000,
        6500,
    }
    assert {condition.angle for condition in sets["condition_ood_alpha1"]} == {89.0}
    assert {
        condition.wind
        for values in sets.values()
        for condition in values
    } == {10.0}


def test_cp8_round_trip_uses_project_bspline():
    cp = np.array([0.018, 0.031, 0.014, 0.006, 52.0, 21.0, 16.0, 11.0])

    recovery = recover_cp8(cp_to_sections(cp))

    np.testing.assert_allclose(recovery.cp8, cp, atol=1e-12, rtol=0)
    assert recovery.max_abs_error <= 1e-10


def test_cp8_recovery_rejects_non_bspline_sections():
    bad = cp_to_sections(np.array([0.018, 0.031, 0.014, 0.006, 52, 21, 16, 11]))
    bad[5] += 1e-4

    with pytest.raises(ValueError, match="reconstruction error"):
        recover_cp8(bad, tolerance=1e-10)


@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
def test_cp8_recovery_rejects_non_finite_sections(non_finite):
    sections = cp_to_sections(np.array([0.018, 0.031, 0.014, 0.006, 52, 21, 16, 11]))
    sections[5] = non_finite

    with pytest.raises(ValueError, match="non-finite sections"):
        recover_cp8(sections)


@pytest.fixture
def history_csv(tmp_path):
    path = tmp_path / "historical_sections.csv"
    fieldnames = ["geom_idx"] + [f"chord_{index}" for index in range(22)]
    fieldnames += [f"twist_{index}" for index in range(22)]

    rows = []
    for geom_idx in range(12):
        cp8 = np.array([
            0.015 + geom_idx * 0.0001,
            0.030 + geom_idx * 0.0001,
            0.012 + geom_idx * 0.0001,
            0.007 + geom_idx * 0.0001,
            45.0 + geom_idx,
            25.0 + geom_idx,
            18.0 + geom_idx,
            10.0 + geom_idx,
        ])
        sections = cp_to_sections(cp8)
        rows.append({
            "geom_idx": geom_idx,
            **{f"chord_{index}": value for index, value in enumerate(sections[:22])},
            **{f"twist_{index}": value for index, value in enumerate(sections[22:])},
        })

    unreconstructable = cp_to_sections(np.array([0.016, 0.031, 0.013, 0.008, 47, 27, 20, 12]))
    unreconstructable[3] += 1e-4
    rows.append({
        "geom_idx": 999,
        **{f"chord_{index}": value for index, value in enumerate(unreconstructable[:22])},
        **{f"twist_{index}": value for index, value in enumerate(unreconstructable[22:])},
    })

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_manifest_is_reproducible_and_classified(history_csv):
    first = generate_manifest(history_csv, seed=20260727)
    second = generate_manifest(history_csv, seed=20260727)

    assert canonical_json(first) == canonical_json(second)
    assert [geometry["geom_id"] for geometry in first["geometries"]] == list(range(1000, 1020))
    assert {geometry["category"] for geometry in first["geometries"][:10]} == {"geometry_id"}
    assert {geometry["category"] for geometry in first["geometries"][10:]} == {"geometry_ood"}
    assert first["recovery_audit"]["rejected_geometry_ids"] == [999]

    for geometry in first["geometries"]:
        assert len(geometry["conditions"]) == 63
        assert len({tuple(condition[axis] for axis in ("rpm", "wind", "angle"))
                    for condition in geometry["conditions"]}) == 63
        assert {condition["category"] for condition in geometry["conditions"]} == {
            "base42",
            "condition_id_interp",
            "condition_ood_alpha1",
        }
        np.testing.assert_allclose(geometry["sections"], cp_to_sections(geometry["cp8"]))


def test_manifest_keeps_id_inside_and_ood_one_dimension_outside(history_csv):
    manifest = generate_manifest(history_csv, seed=20260727)
    bounds = np.asarray(manifest["cp_bounds"])
    spans = bounds[:, 1] - bounds[:, 0]

    for geometry in manifest["geometries"][:10]:
        normalized = (np.asarray(geometry["cp8"]) - bounds[:, 0]) / spans
        assert np.all((normalized >= 0.1) & (normalized <= 0.9))

    for geometry in manifest["geometries"][10:]:
        cp8 = np.asarray(geometry["cp8"])
        outside = (cp8 < bounds[:, 0]) | (cp8 > bounds[:, 1])
        assert outside.sum() == 1
        dimension = int(np.flatnonzero(outside)[0])
        assert np.isclose(abs(cp8[dimension] - np.clip(cp8[dimension], *bounds[dimension])),
                          0.05 * spans[dimension])


def test_cli_writes_canonical_manifest_and_sha256(history_csv, tmp_path):
    output = tmp_path / "manifest.json"
    environment = os.environ | {"PYTHONPATH": "code"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimization_v2.tools.supplement_manifest",
            "--dataset-csv",
            str(history_csv),
            "--out",
            str(output),
            "--seed",
            "20260727",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    payload = output.read_bytes()
    assert json.loads(payload) == generate_manifest(history_csv, seed=20260727)
    assert output.with_suffix(".json.sha256").read_text().strip() == hashlib.sha256(payload).hexdigest()
