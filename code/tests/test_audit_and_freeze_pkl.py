import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optimization_v2.geometry import GEOMETRY_R, cp_to_sections
from optimization_v2.tools.audit_and_freeze_pkl import (
    audit_sources,
    freeze_datasets,
)
from optimization_v2.tools.convert_pkl_to_dataset import aggregate_timeseries


BASE_CONDITIONS = [
    (rpm, 10.0, angle)
    for rpm in (4000, 4500, 5000, 5500, 6000, 6500)
    for angle in (82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0)
]


def _condition_key(condition):
    rpm, wind, angle = condition
    return f"RPM{rpm}_Wind{wind}_Angle{angle}"


def _frame(value=1.0):
    raw = np.array([-value - 1.0, -value])
    return pd.DataFrame(
        {
            "Thrust": raw,
            "Thrust_z": raw - 1.0,
            "My": raw - 2.0,
            "Torque": raw - 3.0,
            "THRUST": [value, value],
            "THRUST_Z": [value + 1.0, value + 1.0],
            "MY": [value + 2.0, value + 2.0],
            "TORQUE": [value + 3.0, value + 3.0],
        }
    )


def _geometry(cp8=None):
    cp8 = np.asarray(
        cp8
        if cp8 is not None
        else [0.018, 0.031, 0.014, 0.006, 52.0, 21.0, 16.0, 11.0]
    )
    sections = cp_to_sections(cp8)
    return np.column_stack((GEOMETRY_R, sections[:22], sections[22:]))


def write_fake_pkl(
    path: Path,
    *,
    geom_id: int,
    conditions,
    stored_geom_id=True,
    frame_factory=_frame,
    category="historical_train",
    manifest_sha256="",
    generator_git_commit="",
    condition_split=None,
    cp8=None,
):
    node = {
        "geometry": _geometry(cp8),
        **{_condition_key(condition): frame_factory() for condition in conditions},
        "category": category,
        "manifest_sha256": manifest_sha256,
        "generator_git_commit": generator_git_commit,
    }
    if stored_geom_id:
        node["geom_id"] = geom_id
    if condition_split is not None:
        node["condition_split"] = condition_split
    if cp8 is not None:
        node["control_points"] = list(cp8)
    with path.open("wb") as handle:
        pickle.dump({f"geometry_{geom_id}": node}, handle)


def test_audit_reports_missing_and_duplicate_keys(tmp_path):
    write_fake_pkl(tmp_path / "a.pkl", geom_id=0, conditions=[(4000, 10, 82)])
    write_fake_pkl(tmp_path / "b.pkl", geom_id=0, conditions=[(4000, 10, 82)])

    report = audit_sources([tmp_path], expected_base_ids=range(1))

    assert report.unique_geometry_count == 1
    assert report.duplicate_keys == [(0, 4000, 10.0, 82.0)]
    assert len(report.missing_keys) == 41
    assert len(report.duplicate_decisions) == 1


def test_audit_rejects_disagreeing_geom_ids_and_records_unreadable_files(tmp_path):
    node = {
        "geometry": _geometry(),
        "geom_id": 8,
        _condition_key((4000, 10, 82)): _frame(),
    }
    with (tmp_path / "mismatch.pkl").open("wb") as handle:
        pickle.dump({"geometry_7": node}, handle)
    (tmp_path / "broken.pkl").write_bytes(b"not a pickle")

    report = audit_sources([tmp_path], expected_base_ids=[7])

    assert report.unique_geometry_count == 0
    assert report.geom_id_disagreements[0]["geometry_key_id"] == 7
    assert report.geom_id_disagreements[0]["stored_geom_id"] == 8
    assert report.unreadable_files[0]["source_file"].endswith("broken.pkl")


def test_aggregate_prefers_uav_positive_qblade_aliases():
    frame = _frame(value=5.0)

    out, provenance = aggregate_timeseries(frame, last_n=120)

    assert out == pytest.approx(
        {
            "T": frame.THRUST.iloc[-1],
            "H": frame.THRUST_Z.iloc[-1],
            "My": frame.MY.iloc[-1],
            "Q": frame.TORQUE.iloc[-1],
        }
    )
    assert provenance == "qblade_uav_positive_aliases"


def test_aggregate_legacy_raw_columns_are_negated_and_never_mixed():
    raw = pd.DataFrame(
        {
            "Thrust": [-2.0, -4.0],
            "Thrust_z": [-1.0, -3.0],
            "My": [-5.0, -7.0],
            "Torque": [-8.0, -10.0],
        }
    )

    out, provenance = aggregate_timeseries(raw, last_n=2)

    assert out == pytest.approx({"T": 3.0, "H": 2.0, "My": 6.0, "Q": 9.0})
    assert provenance == "legacy_raw_converted"
    partial_alias = raw.assign(THRUST=3.0)
    assert aggregate_timeseries(partial_alias, last_n=2) == ({}, "invalid_partial_qblade_aliases")


def test_duplicate_resolution_follows_declared_precedence(tmp_path):
    older_explicit = tmp_path / "older_explicit.pkl"
    newer_inferred = tmp_path / "newer_inferred.pkl"
    write_fake_pkl(
        older_explicit,
        geom_id=0,
        conditions=[(4000, 10, 82)],
        stored_geom_id=True,
    )
    write_fake_pkl(
        newer_inferred,
        geom_id=0,
        conditions=[(4000, 10, 82)],
        stored_geom_id=False,
    )
    os.utime(older_explicit, (100, 100))
    os.utime(newer_inferred, (200, 200))

    report = audit_sources([tmp_path], expected_base_ids=[0])

    selected = report.selected_records[0]
    assert selected.source_file.endswith("older_explicit.pkl")
    assert selected.geom_id_source == "stored_geom_id"
    decision = report.duplicate_decisions[0]
    assert decision["reason"] == "explicit_stored_geom_id"
    assert decision["discarded_source"].endswith("newer_inferred.pkl")


def test_duplicate_resolution_orders_valid_manifest_git_and_mtime(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    cp8 = np.array([0.018, 0.031, 0.014, 0.006, 52.0, 21.0, 16.0, 11.0])
    payload = _manifest_payload(cp8)
    for index, record in enumerate(payload["geometries"]):
        record["geom_id"] = index + 1
        record["category"] = "geometry_id"
    manifest_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def invalid_frame():
        return _frame().drop(columns=["TORQUE"])

    cases = [
        # valid finite outputs outrank newer invalid outputs
        (0, "valid.pkl", {}, "invalid.pkl", {"frame_factory": invalid_frame}, "valid_finite_outputs"),
        # exact manifest digest outranks a newer mismatch
        (1, "manifest.pkl", {"manifest_sha256": digest, "cp8": cp8},
         "mismatch.pkl", {"manifest_sha256": "f" * 64, "cp8": cp8}, "manifest_sha_match"),
        # recorded generator Git metadata outranks missing metadata
        (2, "git.pkl", {"manifest_sha256": digest, "generator_git_commit": "abc", "cp8": cp8},
         "no_git.pkl", {"manifest_sha256": digest, "cp8": cp8}, "generator_git_metadata"),
        # newest mtime is the final declared resolver
        (3, "newest.pkl", {"manifest_sha256": digest, "cp8": cp8},
         "oldest.pkl", {"manifest_sha256": digest, "cp8": cp8}, "newest_mtime"),
    ]
    expected_sources = {}
    for geom_id, selected_name, selected_kwargs, discarded_name, discarded_kwargs, reason in cases:
        selected_path = tmp_path / selected_name
        discarded_path = tmp_path / discarded_name
        write_fake_pkl(
            selected_path,
            geom_id=geom_id,
            conditions=[(4000, 10, 82)],
            **selected_kwargs,
        )
        write_fake_pkl(
            discarded_path,
            geom_id=geom_id,
            conditions=[(4000, 10, 82)],
            **discarded_kwargs,
        )
        os.utime(selected_path, (100, 100 if reason != "newest_mtime" else 300))
        os.utime(discarded_path, (200, 200))
        expected_sources[geom_id] = (selected_name, reason)

    report = audit_sources(
        [tmp_path], expected_base_ids=[0], manifest_path=manifest_path
    )

    selected_by_id = {record.geom_id: record for record in report.selected_records}
    decision_by_id = {decision["geom_id"]: decision for decision in report.duplicate_decisions}
    for geom_id, (source_name, reason) in expected_sources.items():
        assert selected_by_id[geom_id].source_file.endswith(source_name)
        assert decision_by_id[geom_id]["reason"] == reason


def _manifest_payload(cp8):
    conditions = []
    for split, values in {
        "base42": [(4000, 10.0, 82.0)],
        "condition_id_interp": [(4250, 10.0, 82.5)],
        "condition_ood_alpha1": [(4000, 10.0, 89.0)],
    }.items():
        conditions.extend(
            {"rpm": rpm, "wind": wind, "angle": angle, "category": split}
            for rpm, wind, angle in values
        )
    sections = cp_to_sections(cp8).tolist()
    return {
        "manifest_version": 1,
        "seed": 20260727,
        "geometries": [
            {
                "geom_id": geom_id,
                "category": category,
                "seed": 20260727,
                "cp8": list(cp8),
                "sections": sections,
                "conditions": conditions,
            }
            for geom_id, category in ((1000, "geometry_id"), (1010, "geometry_ood"))
        ],
    }


def test_freeze_exports_are_key_disjoint_and_checksum_versioned(tmp_path):
    historical = tmp_path / "historical"
    supplement = tmp_path / "supplement"
    output = tmp_path / "frozen"
    historical.mkdir()
    supplement.mkdir()
    cp8 = np.array([0.018, 0.031, 0.014, 0.006, 52.0, 21.0, 16.0, 11.0])
    for geom_id in range(3):
        write_fake_pkl(
            historical / f"historical_{geom_id}.pkl",
            geom_id=geom_id,
            conditions=BASE_CONDITIONS,
            cp8=None,
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest_payload(cp8), sort_keys=True, separators=(",", ":"))
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    split_map = {
        _condition_key((4000, 10, 82)): "base42",
        _condition_key((4250, 10, 82.5)): "condition_id_interp",
        _condition_key((4000, 10, 89)): "condition_ood_alpha1",
    }
    for geom_id, category in ((1000, "geometry_id"), (1010, "geometry_ood")):
        write_fake_pkl(
            supplement / f"supplement_{geom_id}.pkl",
            geom_id=geom_id,
            conditions=[(4000, 10, 82), (4250, 10, 82.5), (4000, 10, 89)],
            category=category,
            manifest_sha256=digest,
            generator_git_commit="abc123",
            condition_split=split_map,
            cp8=cp8,
        )

    report = audit_sources(
        [historical, supplement],
        expected_base_ids=range(3),
        manifest_path=manifest_path,
    )
    frozen = freeze_datasets(
        report,
        output,
        version="synthetic-v1",
        input_paths={"historical": historical, "supplement": supplement},
        git_commits={"generator": "abc123", "freezer": "def456"},
    )

    key_cols = ["geom_id", "RPM", "WIND", "ANGLE"]
    partition_names = (
        "historical_train__base42",
        "geometry_id__base42",
        "geometry_id__condition_id_interp",
        "geometry_id__condition_ood_alpha1",
        "geometry_ood__base42",
        "geometry_ood__condition_id_interp",
        "geometry_ood__condition_ood_alpha1",
    )
    frames = [frozen[name] for name in partition_names]
    cp_columns = [f"chord_cp_{index}" for index in range(4)] + [
        f"twist_cp_{index}" for index in range(4)
    ]
    section_columns = [f"chord_{index}" for index in range(22)] + [
        f"twist_{index}" for index in range(22)
    ]
    for frame in frames:
        assert {"RPM", "WIND", "ANGLE", *cp_columns} <= set(frame.columns)
        assert {"RPM", "WIND", "ANGLE", *section_columns} <= set(frame.columns)
        for _, row in frame.iterrows():
            np.testing.assert_allclose(
                row[section_columns].to_numpy(dtype=float),
                cp_to_sections(row[cp_columns].to_numpy(dtype=float)),
                atol=1e-12,
                rtol=0,
            )
    for name in partition_names:
        category, condition_split = name.split("__", 1)
        assert set(frozen[name]["category"]) <= {category}
        assert set(frozen[name]["condition_split"]) <= {condition_split}
    sets = [set(map(tuple, frame[key_cols].to_numpy())) for frame in frames]
    assert all(
        not (sets[i] & sets[j])
        for i in range(len(sets))
        for j in range(i + 1, len(sets))
    )
    assert set(frozen["geometry_id__condition_ood_alpha1"]["ANGLE"]) == {89.0}
    assert set(frozen["historical_train__base42"]["ANGLE"]) <= {
        82,
        83,
        84,
        85,
        86,
        87,
        88,
    }
    all_selected_keys = set().union(*sets)
    assert len(all_selected_keys) == sum(map(len, sets)) == len(report.selected_records)
    assert list(frozen["normalizer_inputs"].columns) == [
        "RPM", "WIND", "ANGLE", *cp_columns, *section_columns
    ]
    split_payload = json.loads((output / "geometry_split_synthetic-v1.json").read_text())
    assert set(split_payload["train"]) < {0, 1, 2}

    acceptance_path = output / "acceptance_synthetic-v1.json"
    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance["manifest_sha256"] == digest
    assert acceptance["output_row_counts"] == {
        "geometry_id__base42": 1,
        "geometry_id__condition_id_interp": 1,
        "geometry_id__condition_ood_alpha1": 1,
        "geometry_ood__base42": 1,
        "geometry_ood__condition_id_interp": 1,
        "geometry_ood__condition_ood_alpha1": 1,
        "historical_train__base42": 126,
        "normalizer_inputs": 42,
    }
    assert acceptance["excluded_cp8_recoveries"] == []
    assert set(acceptance["split_ids"]) == {"train", "validation", "test"}
    for name, artifact_digest in acceptance["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == artifact_digest


def test_failed_historical_cp8_recovery_stays_audited_but_not_exported(tmp_path):
    bad_geometry = _geometry()
    bad_geometry[3, 1] += 1e-4
    node = {
        "geometry": bad_geometry,
        "geom_id": 0,
        **{_condition_key(condition): _frame() for condition in BASE_CONDITIONS},
    }
    with (tmp_path / "bad.pkl").open("wb") as handle:
        pickle.dump({"geometry_0": node}, handle)

    report = audit_sources([tmp_path], expected_base_ids=[0])

    assert report.per_geometry_coverage[0]["present_key_count"] == 42
    assert report.excluded_cp8_recoveries[0]["geom_id"] == 0
    assert report.training_records() == []
