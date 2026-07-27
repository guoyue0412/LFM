import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from optimization_v2.geometry import GEOMETRY_R, cp_to_sections
from optimization_v2.tools.supplement_manifest import build_condition_sets
from scripts import run_data_gen_parallel as runner


def _manifest_payload(geom_ids=(1000,)):
    cp8 = np.array([0.018, 0.031, 0.014, 0.006, 52.0, 21.0, 16.0, 11.0])
    conditions = [
        {
            "rpm": condition.rpm,
            "wind": condition.wind,
            "angle": condition.angle,
            "category": category,
        }
        for category, values in build_condition_sets().items()
        for condition in values
    ]
    return {
        "manifest_version": 1,
        "seed": 20260727,
        "geometries": [
            {
                "geom_id": geom_id,
                "category": "geometry_id",
                "seed": 20260727,
                "cp8": cp8.tolist(),
                "sections": cp_to_sections(cp8).tolist(),
                "conditions": conditions,
            }
            for geom_id in geom_ids
        ],
    }


def _write_manifest(tmp_path, payload):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return path


def test_manifest_tasks_preserve_ids_cp8_and_condition_splits(tmp_path):
    manifest_path = _write_manifest(tmp_path, _manifest_payload())

    manifest = runner.load_manifest(manifest_path)
    tasks = runner.build_manifest_tasks(manifest, geom_ids={1000})

    assert len(tasks) == 1
    task = tasks[0]
    assert task.geom_id == 1000
    assert task.category == "geometry_id"
    assert task.cp8.shape == (8,)
    assert task.geometry.shape == (22, 3)
    np.testing.assert_array_equal(task.geometry[:, 0], GEOMETRY_R)
    np.testing.assert_array_equal(task.geometry[:, 1], task.sections[:22])
    np.testing.assert_array_equal(task.geometry[:, 2], task.sections[22:])
    np.testing.assert_array_equal(task.sections, cp_to_sections(task.cp8))
    assert len(task.conditions) == 63
    assert task.condition_splits["condition_ood_alpha1"] == 6


def test_manifest_sha256_checks_exact_bytes_and_sidecar(tmp_path):
    manifest_path = _write_manifest(tmp_path, _manifest_payload())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    assert runner.load_manifest(manifest_path, expected_sha256=digest).sha256 == digest

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.load_manifest(manifest_path, expected_sha256="0" * 64)

    manifest_path.with_suffix(".json.sha256").write_text("f" * 64 + "\n")
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        runner.load_manifest(manifest_path)


def _duplicate_geometry(payload):
    payload["geometries"].append(copy.deepcopy(payload["geometries"][0]))


def _duplicate_condition(payload):
    conditions = payload["geometries"][0]["conditions"]
    conditions.append(copy.deepcopy(conditions[0]))


def _wrong_wind(payload):
    payload["geometries"][0]["conditions"][0]["wind"] = 9.0


def _unknown_condition_split(payload):
    payload["geometries"][0]["conditions"][0]["category"] = "geometry_id"


def _cp8_section_mismatch(payload):
    payload["geometries"][0]["sections"][0] += 1e-9


def _non_finite_cp8(payload):
    payload["geometries"][0]["cp8"][0] = float("nan")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_duplicate_geometry, "duplicate geom_id"),
        (_duplicate_condition, "duplicate condition key"),
        (_wrong_wind, "wind must equal 10"),
        (_unknown_condition_split, "unknown condition split"),
        (_cp8_section_mismatch, "cp8-section mismatch"),
        (_non_finite_cp8, "finite cp8"),
    ],
)
def test_manifest_validation_rejects_invalid_records(tmp_path, mutate, message):
    payload = _manifest_payload()
    mutate(payload)
    manifest = runner.load_manifest(_write_manifest(tmp_path, payload))

    with pytest.raises(ValueError, match=message):
        runner.build_manifest_tasks(manifest)


def test_manifest_subset_rejects_absent_exact_id(tmp_path):
    manifest = runner.load_manifest(_write_manifest(tmp_path, _manifest_payload()))

    with pytest.raises(ValueError, match="1001"):
        runner.build_manifest_tasks(manifest, geom_ids={1001})


def test_manifest_subset_still_validates_unselected_records(tmp_path):
    payload = _manifest_payload((1000, 1001))
    payload["geometries"][1]["sections"][0] += 1e-9
    manifest = runner.load_manifest(_write_manifest(tmp_path, payload))

    with pytest.raises(ValueError, match="geometry 1001.*cp8-section mismatch"):
        runner.build_manifest_tasks(manifest, geom_ids={1000})


def test_manifest_requires_exact_condition_split_counts(tmp_path):
    payload = _manifest_payload()
    payload["geometries"][0]["conditions"].pop()
    manifest = runner.load_manifest(_write_manifest(tmp_path, payload))

    with pytest.raises(ValueError, match="condition split counts"):
        runner.build_manifest_tasks(manifest)


def test_condition_keys_losslessly_distinguish_close_floats(tmp_path):
    payload = _manifest_payload()
    first, second = payload["geometries"][0]["conditions"][:2]
    first["rpm"] = 4000.000001
    second["rpm"] = 4000.000002
    second["angle"] = first["angle"]
    manifest = runner.load_manifest(_write_manifest(tmp_path, payload))

    task = runner.build_manifest_tasks(manifest)[0]
    keys = [runner.condition_key(*condition[:3]) for condition in task.conditions]

    assert len(keys) == len(set(keys)) == 63
    assert keys[0] != keys[1]


class FakeSimulation:
    instances = []
    empty_condition = None
    omit_persisted_condition = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.changed_geometry = None
        self.closed = False
        self.all_simulation_data = {
            "geometry": np.full((22, 3), -1.0),
            "Base_simulation": "must not leak",
        }
        self.__class__.instances.append(self)

    def change_propeller_geometry(self, section_data):
        self.changed_geometry = section_data

    def run_one_simulation(self, RPM, WIND_SPEED, ANGLE):
        condition = (RPM, WIND_SPEED, ANGLE)
        self.calls.append(condition)
        frame = (
            pd.DataFrame()
            if condition == self.empty_condition
            else pd.DataFrame({"value": [RPM + WIND_SPEED + ANGLE]})
        )
        if condition != self.omit_persisted_condition:
            self.all_simulation_data[runner.condition_key(*condition)] = frame
        return frame

    def close(self):
        self.closed = True


class FakeConfig:
    file_path = {"fake": "path"}
    geometry_baseline = np.zeros((22, 3))
    number_of_timesteps = 1000


@pytest.fixture
def manifest_task(tmp_path):
    manifest = runner.load_manifest(_write_manifest(tmp_path, _manifest_payload()))
    return runner.build_manifest_tasks(manifest)[0]


def test_worker_runs_only_declared_conditions_and_stores_metadata(monkeypatch, manifest_task):
    FakeSimulation.instances.clear()
    FakeSimulation.empty_condition = None
    FakeSimulation.omit_persisted_condition = None
    monkeypatch.setattr(runner, "_SIMULATION", FakeSimulation, raising=False)
    monkeypatch.setattr(runner, "_config", FakeConfig, raising=False)

    geom_id, result, error = runner.worker_run((manifest_task, 1000, "CPU"))

    assert error is None and geom_id == manifest_task.geom_id
    simulation = FakeSimulation.instances[-1]
    assert simulation.calls == [condition[:3] for condition in manifest_task.conditions]
    np.testing.assert_array_equal(simulation.changed_geometry, manifest_task.geometry)
    assert simulation.closed
    assert result["geom_id"] == manifest_task.geom_id
    assert result["control_points"] == pytest.approx(manifest_task.cp8)
    assert result["category"] == manifest_task.category
    assert result["seed"] == manifest_task.seed
    assert result["manifest_sha256"] == manifest_task.manifest_sha256
    assert result["generator_git_commit"]
    np.testing.assert_array_equal(result["geometry"], manifest_task.geometry)
    declared_keys = {
        runner.condition_key(rpm, wind, angle)
        for rpm, wind, angle, _ in manifest_task.conditions
    }
    assert {key for key in result if key.startswith("RPM")} == declared_keys
    assert result["condition_split"] == {
        runner.condition_key(rpm, wind, angle): split
        for rpm, wind, angle, split in manifest_task.conditions
    }
    assert "Base_simulation" not in result


def test_worker_rejects_empty_declared_condition(monkeypatch, manifest_task):
    FakeSimulation.instances.clear()
    FakeSimulation.empty_condition = manifest_task.conditions[0][:3]
    FakeSimulation.omit_persisted_condition = None
    monkeypatch.setattr(runner, "_SIMULATION", FakeSimulation, raising=False)
    monkeypatch.setattr(runner, "_config", FakeConfig, raising=False)

    geom_id, result, error = runner.worker_run((manifest_task, 1000, "CPU"))

    assert geom_id == manifest_task.geom_id
    assert result is None
    assert "empty QBlade result" in error
    assert FakeSimulation.instances[-1].closed


class NoneReturningSimulation(FakeSimulation):
    def run_one_simulation(self, RPM, WIND_SPEED, ANGLE):
        super().run_one_simulation(RPM, WIND_SPEED, ANGLE)
        return None


def test_worker_uses_persisted_mapping_when_run_returns_none(monkeypatch, manifest_task):
    NoneReturningSimulation.instances.clear()
    NoneReturningSimulation.empty_condition = None
    NoneReturningSimulation.omit_persisted_condition = None
    monkeypatch.setattr(runner, "_SIMULATION", NoneReturningSimulation, raising=False)
    monkeypatch.setattr(runner, "_config", FakeConfig, raising=False)

    geom_id, result, error = runner.worker_run((manifest_task, 1000, "CPU"))

    assert error is None and geom_id == manifest_task.geom_id
    simulation = NoneReturningSimulation.instances[-1]
    for rpm, wind, angle, _ in manifest_task.conditions:
        key = runner.condition_key(rpm, wind, angle)
        assert result[key] is simulation.all_simulation_data[key]


def test_worker_rejects_returned_frame_absent_from_persisted_mapping(
    monkeypatch, manifest_task
):
    FakeSimulation.instances.clear()
    FakeSimulation.empty_condition = None
    FakeSimulation.omit_persisted_condition = manifest_task.conditions[0][:3]
    monkeypatch.setattr(runner, "_SIMULATION", FakeSimulation, raising=False)
    monkeypatch.setattr(runner, "_config", FakeConfig, raising=False)

    geom_id, result, error = runner.worker_run((manifest_task, 1000, "CPU"))

    assert geom_id == manifest_task.geom_id and result is None
    assert "absent persisted QBlade result" in error


class CloseFailingSimulation(FakeSimulation):
    def close(self):
        self.closed = True
        raise RuntimeError("close failed")


def test_worker_close_error_does_not_override_success(monkeypatch, manifest_task):
    CloseFailingSimulation.instances.clear()
    CloseFailingSimulation.empty_condition = None
    CloseFailingSimulation.omit_persisted_condition = None
    monkeypatch.setattr(runner, "_SIMULATION", CloseFailingSimulation, raising=False)
    monkeypatch.setattr(runner, "_config", FakeConfig, raising=False)

    geom_id, result, error = runner.worker_run((manifest_task, 1000, "CPU"))

    assert error is None and geom_id == manifest_task.geom_id
    assert result is not None
    assert CloseFailingSimulation.instances[-1].closed


def _stored_result(task):
    condition_split = {
        runner.condition_key(rpm, wind, angle): split
        for rpm, wind, angle, split in task.conditions
    }
    return {
        "geometry": task.geometry,
        **{key: pd.DataFrame({"value": [1.0]}) for key in condition_split},
        "geom_id": task.geom_id,
        "category": task.category,
        "control_points": task.cp8.tolist(),
        "condition_split": condition_split,
        "seed": task.seed,
        "manifest_sha256": task.manifest_sha256,
        "generator_git_commit": "abc123",
    }


def test_resume_skips_only_exact_complete_manifest_nodes(tmp_path):
    manifest = runner.load_manifest(
        _write_manifest(tmp_path, _manifest_payload(tuple(range(1000, 1005))))
    )
    tasks = runner.build_manifest_tasks(manifest)
    stored = {f"geometry_{task.geom_id}": _stored_result(task) for task in tasks}
    stored["geometry_1001"]["geom_id"] = 777
    stored["geometry_1002"]["manifest_sha256"] = "f" * 64
    missing_key = runner.condition_key(*tasks[3].conditions[0][:3])
    stored["geometry_1003"].pop(missing_key)
    stored["geometry_1004"]["RPM9999.0_Wind10.0_Angle90.0"] = pd.DataFrame(
        {"value": [1.0]}
    )
    with (tmp_path / "prior.pkl").open("wb") as handle:
        pickle.dump(stored, handle)

    assert runner.find_completed_manifest_ids(tmp_path, tasks) == {1000}


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda node, key: node.__setitem__(key, None),
        lambda node, key: node.__setitem__(key, pd.DataFrame()),
        lambda node, key: node.__setitem__(key, {"not": "a dataframe"}),
        lambda node, key: node.__setitem__("generator_git_commit", ""),
        lambda node, key: node.pop("generator_git_commit"),
    ],
)
def test_resume_rejects_invalid_condition_payload_or_commit(tmp_path, corrupt):
    manifest = runner.load_manifest(_write_manifest(tmp_path, _manifest_payload()))
    task = runner.build_manifest_tasks(manifest)[0]
    node = _stored_result(task)
    first_key = runner.condition_key(*task.conditions[0][:3])
    corrupt(node, first_key)
    with (tmp_path / "corrupt.pkl").open("wb") as handle:
        pickle.dump({f"geometry_{task.geom_id}": node}, handle)

    assert runner.find_completed_manifest_ids(tmp_path, [task]) == set()


def test_manifest_cli_arguments_and_exact_id_parser(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--manifest",
            "manifest.json",
            "--geom-ids",
            "1002,1000,1002",
            "--manifest-sha256",
            "a" * 64,
        ],
    )

    args = runner.parse_args()

    assert args.manifest == "manifest.json"
    assert runner.parse_geom_ids(args.geom_ids) == {1000, 1002}
    assert args.manifest_sha256 == "a" * 64
    with pytest.raises(ValueError, match="comma-separated integers"):
        runner.parse_geom_ids("1000,nope")


def test_manifest_output_name_includes_digest_tag():
    name = runner.output_filename(
        tag="supplement", stamp="20260727_230000", batch_number=3,
        batched=True, manifest_sha256="abcdef1234567890",
    )

    assert name == "data_supplement_manifest-abcdef123456_20260727_230000_b0003.pkl"


def test_prepare_manifest_run_filters_only_valid_completed_ids(tmp_path):
    manifest_path = _write_manifest(tmp_path, _manifest_payload((1000, 1001)))
    loaded = runner.load_manifest(manifest_path)
    first_task = runner.build_manifest_tasks(loaded, geom_ids={1000})[0]
    with (tmp_path / "completed.pkl").open("wb") as handle:
        pickle.dump({"geometry_1000": _stored_result(first_task)}, handle)

    pending, completed, digest = runner.prepare_manifest_run(
        manifest_path=manifest_path,
        geom_ids={1000, 1001},
        expected_sha256=None,
        output_dir=tmp_path,
    )

    assert [task.geom_id for task in pending] == [1001]
    assert completed == {1000}
    assert digest == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


class FakeLegacySimulation:
    instances = []

    def __init__(self, **kwargs):
        self.geometry_baseline = kwargs["geometry_baseline"]
        self.all_simulation_data = {"geometry": self.geometry_baseline, "legacy": 1}
        self.changed_geometry = None
        self.ran_all = False
        self.__class__.instances.append(self)

    def change_propeller_geometry(self, section_data):
        self.changed_geometry = section_data

    def run_all_simulation(self):
        self.ran_all = True


def test_worker_keeps_legacy_geometry_tuple_behavior(monkeypatch):
    FakeLegacySimulation.instances.clear()
    monkeypatch.setattr(runner, "_SIMULATION", FakeLegacySimulation, raising=False)
    monkeypatch.setattr(runner, "_config", FakeConfig, raising=False)
    geometry = np.ones((22, 3))

    geom_id, result, error = runner.worker_run((7, geometry, 500, "CPU"))

    assert error is None and geom_id == 7
    assert result["legacy"] == 1
    assert FakeLegacySimulation.instances[-1].ran_all
    np.testing.assert_array_equal(
        FakeLegacySimulation.instances[-1].changed_geometry, geometry
    )


def test_legacy_help_works_without_pythonpath():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/run_data_gen_parallel.py", "--help"],
        cwd=runner.LFM_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--geometry-npy" in result.stdout
    assert "--manifest" in result.stdout
