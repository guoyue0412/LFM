import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from optimization_v2.geometry import cp_to_sections
from optimization_v2.tools.supplement_manifest import build_condition_sets
from scripts import run_data_gen_parallel as runner


SCRIPT = runner.LFM_ROOT / "scripts" / "run_supplement_3660.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "checkout"
    script_dir = repo / "code" / "scripts"
    submodule = repo / "code" / "Propeller_project-main"
    script_dir.mkdir(parents=True)
    submodule.mkdir(parents=True)
    subprocess.run(["git", "init", str(submodule)], check=True, capture_output=True)
    _git(submodule, "config", "user.email", "test@example.com")
    _git(submodule, "config", "user.name", "Test")
    (submodule / "qblade.txt").write_text("pinned\n")
    _git(submodule, "add", "qblade.txt")
    _git(submodule, "commit", "-m", "pin qblade")
    submodule_commit = _git(submodule, "rev-parse", "HEAD")

    shutil.copy2(SCRIPT, script_dir / SCRIPT.name)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "code")
    _git(repo, "commit", "-m", "test checkout")
    return repo, submodule_commit


def _write_manifest(tmp_path: Path) -> tuple[Path, str]:
    cp8 = np.array([0.018, 0.031, 0.014, 0.006, 52.0, 21.0, 16.0, 11.0])
    conditions = [
        {
            "rpm": condition.rpm,
            "wind": condition.wind,
            "angle": condition.angle,
            "category": split,
        }
        for split, values in build_condition_sets().items()
        for condition in values
    ]
    payload = {
        "manifest_version": 1,
        "seed": 20260727,
        "geometries": [
            {
                "geom_id": 1000 + index,
                "category": "geometry_id" if index < 10 else "geometry_ood",
                "seed": 20260727,
                "cp8": cp8.tolist(),
                "sections": cp_to_sections(cp8).tolist(),
                "conditions": conditions,
            }
            for index in range(20)
        ],
    }
    manifest = tmp_path / "supplement_manifest.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix(".json.sha256").write_text(f"{digest}  {manifest.name}\n")
    return manifest, digest


@pytest.fixture
def launcher_env(tmp_path):
    repo, submodule_commit = _init_repo(tmp_path)
    manifest, digest = _write_manifest(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "RUNNER_PYTHONPATH": str(runner.LFM_ROOT),
            "EXPECTED_SUBMODULE_COMMIT": submodule_commit,
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "WORKTREE_ROOT": str(tmp_path / "worktrees"),
            "WORKERS": "1",
            "OMP_THREADS": "3",
            "CPU_COUNT": "24",
        }
    )
    return repo / "code" / "scripts" / SCRIPT.name, manifest, digest, environment


def test_supplement_launcher_has_valid_shell_syntax_and_no_training_entrypoint():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text()

    assert "train.py" not in source
    assert "sweep_moe" not in source
    assert 'WORKERS="${WORKERS:-1}"' in source
    assert 'OMP_THREADS="${OMP_THREADS:-4}"' in source
    assert (
        'EXPECTED_SUBMODULE_COMMIT="${EXPECTED_SUBMODULE_COMMIT:-'
        'f3e56740b83aa00a36f4d56380b095c5b741437f}"'
    ) in source


@pytest.mark.parametrize(
    ("mode", "required", "forbidden"),
    [
        (
            "alpha1-preflight",
            ("--geom-ids 1000", "--condition-splits condition_ood_alpha1"),
            ("--condition-keys",),
        ),
        (
            "smoke",
            (
                "--geom-ids 1000,1010",
                "--condition-keys RPM4000.0_Wind10.0_Angle85.0",
            ),
            ("--condition-splits",),
        ),
        (
            "formal",
            ("--batch-size", "--device GPU"),
            ("--geom-ids", "--condition-splits", "--condition-keys"),
        ),
    ],
)
def test_supplement_launcher_dry_run_builds_controlled_commands(
    launcher_env, mode, required, forbidden
):
    script, manifest, digest, environment = launcher_env

    result = subprocess.run(
        [str(script), mode, "--manifest", str(manifest), "--dry-run"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"manifest_sha256={digest}" in result.stdout
    assert "workers=1 omp_threads=3 device=GPU" in result.stdout
    assert "LD_LIBRARY_PATH=" in result.stdout
    assert "scripts/run_data_gen_parallel.py" in result.stdout
    assert f"/pkl/{mode}" not in result.stdout
    assert all(value in result.stdout for value in required), result.stdout
    assert all(value not in result.stdout for value in forbidden), result.stdout


def test_supplement_launcher_rejects_manifest_sha_mismatch(launcher_env):
    script, manifest, _, environment = launcher_env
    manifest.with_suffix(".json.sha256").write_text(f"{'f' * 64}\n")

    result = subprocess.run(
        [str(script), "formal", "--manifest", str(manifest), "--dry-run"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "manifest SHA-256 mismatch" in result.stderr


def test_supplement_launcher_rejects_untracked_checkout_file(launcher_env):
    script, manifest, _, environment = launcher_env
    repo = script.parents[2]
    (repo / "untracked.txt").write_text("must fail clean gate\n")

    result = subprocess.run(
        [str(script), "formal", "--manifest", str(manifest), "--dry-run"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "superproject checkout is not clean" in result.stderr
