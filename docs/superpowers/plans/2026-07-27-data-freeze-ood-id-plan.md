# 1000 Geometry Data Freeze and OOD/ID Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, run, and freeze a traceable QBlade supplement batch that repairs the 1000-geometry base grid and adds 10 ID plus 10 geometry-OOD geometries with 42 base conditions and 21 isolated validation conditions each.

**Architecture:** A deterministic manifest generator recovers historical cp8 values, derives training bounds, creates new geometries, and emits per-geometry condition lists. The parallel QBlade runner consumes only that manifest and stores cp8/category metadata beside each geometry. A separate audit/freezing tool treats `(geom_id, RPM, WIND, ANGLE)` as the primary key and exports mutually exclusive training, ID-interpolation, condition-OOD, and geometry-OOD CSVs.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pandas, pytest, QBlade CE 2.0.8.6 SIL, multiprocessing, JSON, pickle, SHA-256, Git.

## Global Constraints

- All QBlade/LLFVW simulations run only on `3660workstation` under `/home/gy/gy_2026/graduation/LFM`.
- Raw pkl files remain on 3660 and are never transferred to the local Mac.
- `WIND` is always exactly `10.0 m/s`.
- Base conditions are exactly `RPM={4000,4500,5000,5500,6000,6500}` crossed with `ANGLE={82,83,84,85,86,87,88}`, totaling 42 conditions per geometry.
- Extra validation conditions are exactly six `ANGLE=89` condition-OOD points at the six base RPM values plus 15 ID interpolation points from `RPM={4250,5250,6250}` crossed with `ANGLE={82.5,83.5,85.0,86.5,87.5}`, totaling 21 conditions per new geometry.
- The 21 extra validation conditions never enter formal training.
- ID geometry IDs are `1000..1009`; geometry-OOD IDs are `1010..1019`.
- Every new pkl geometry node stores `geom_id`, `category`, exact `cp8`, 22-section geometry, condition split, seed, manifest SHA-256, and generator Git commit.
- Historical and generated records are deduplicated by `(geom_id, RPM, WIND, ANGLE)`, not filename.
- Historical cp8 recovery uses the exact knot vectors in `optimization_v2.geometry` and accepts only maximum section reconstruction error `<=1e-10`.
- Official RotorS Iris trim parameters are `mass=1.5 kg`, `l1=l2=0.13 m`, and `d1=d2=0.023 m`.
- Do not launch formal training, sweep, CMA-ES, or change thesis numeric conclusions before the frozen-data acceptance report passes.

---

### Task 1: Deterministic cp8 Recovery and Supplement Manifest

**Files:**
- Create: `code/optimization_v2/tools/supplement_manifest.py`
- Create: `code/tests/test_supplement_manifest.py`
- Modify: `docs/superpowers/specs/2026-07-27-surrogate-data-ood-id-design.md`

**Interfaces:**
- Consumes: a historical flattened CSV containing `geom_idx`, `chord_0..21`, and `twist_0..21`.
- Produces: `recover_cp8(sections: np.ndarray) -> CpRecovery`, `build_condition_sets() -> dict[str, list[Condition]]`, `generate_manifest(...) -> dict`, and a versioned JSON manifest.

- [ ] **Step 1: Write failing tests for condition membership and uniqueness**

```python
def test_condition_sets_are_exact_and_disjoint():
    sets = build_condition_sets()
    assert len(sets["base42"]) == 42
    assert len(sets["condition_id_interp"]) == 15
    assert len(sets["condition_ood_alpha1"]) == 6
    keys = [{c.key for c in values} for values in sets.values()]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert {c.rpm for c in sets["condition_ood_alpha1"]} == {4000, 4500, 5000, 5500, 6000, 6500}
    assert {c.angle for c in sets["condition_ood_alpha1"]} == {89.0}
    assert {c.wind for values in sets.values() for c in values} == {10.0}
```

- [ ] **Step 2: Run the condition test and verify the missing-module failure**

Run: `PYTHONPATH=code /Users/guoyue/anaconda3/bin/python -m pytest code/tests/test_supplement_manifest.py::test_condition_sets_are_exact_and_disjoint -v`

Expected: `FAIL` because `supplement_manifest` or `build_condition_sets` does not exist.

- [ ] **Step 3: Implement immutable condition objects and exact grids**

```python
@dataclass(frozen=True, order=True)
class Condition:
    rpm: float
    wind: float
    angle: float

    @property
    def key(self) -> tuple[float, float, float]:
        return (self.rpm, self.wind, self.angle)

def build_condition_sets() -> dict[str, list[Condition]]:
    base = [Condition(r, 10.0, a) for r in BASE_RPMS for a in BASE_ANGLES]
    interp = [Condition(r, 10.0, a) for r in INTERP_RPMS for a in INTERP_ANGLES]
    alpha1 = [Condition(r, 10.0, 89.0) for r in BASE_RPMS]
    return {"base42": base, "condition_id_interp": interp,
            "condition_ood_alpha1": alpha1}
```

- [ ] **Step 4: Write failing tests for exact cp8 recovery and rejection**

```python
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
```

- [ ] **Step 5: Implement recovery with basis matrices built from project knots**

```python
def _basis_matrix(knots: np.ndarray) -> np.ndarray:
    eye = np.eye(4)
    return np.column_stack([BSpline(knots, eye[i], 3)(R_NORM) for i in range(4)])

def recover_cp8(sections: np.ndarray, tolerance: float = 1e-10) -> CpRecovery:
    sections = np.asarray(sections, dtype=np.float64)
    if sections.shape != (44,):
        raise ValueError(f"expected 44 sections, got {sections.shape}")
    matrices = (_basis_matrix(CHORD_KNOTS), _basis_matrix(TWIST_KNOTS))
    cp = np.concatenate([np.linalg.lstsq(a, y, rcond=None)[0]
                         for a, y in zip(matrices, np.split(sections, 2))])
    rebuilt = cp_to_sections(cp)
    error = float(np.max(np.abs(rebuilt - sections)))
    if error > tolerance:
        raise ValueError(f"reconstruction error {error:.6e} exceeds {tolerance:.6e}")
    return CpRecovery(cp8=cp, max_abs_error=error)
```

- [ ] **Step 6: Write failing deterministic-manifest tests**

```python
def test_manifest_is_reproducible_and_classified(history_csv, tmp_path):
    a = generate_manifest(history_csv, seed=20260727)
    b = generate_manifest(history_csv, seed=20260727)
    assert canonical_json(a) == canonical_json(b)
    assert [g["geom_id"] for g in a["geometries"]] == list(range(1000, 1020))
    assert {g["category"] for g in a["geometries"][:10]} == {"geometry_id"}
    assert {g["category"] for g in a["geometries"][10:]} == {"geometry_ood"}
    assert all(len(g["conditions"]) == 63 for g in a["geometries"])
```

- [ ] **Step 7: Implement training-bound derivation, seeded LHS, and one-dimensional OOD sampling**

```python
def derive_cp_bounds(recoveries: np.ndarray) -> np.ndarray:
    return np.column_stack([recoveries.min(axis=0), recoveries.max(axis=0)])

def sample_id_cp8(bounds: np.ndarray, n: int, seed: int) -> np.ndarray:
    unit = qmc.LatinHypercube(d=8, seed=seed).random(n)
    unit = 0.1 + 0.8 * unit
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])

def make_ood_cp8(id_cp: np.ndarray, bounds: np.ndarray, sample_index: int) -> np.ndarray:
    cp = id_cp.copy()
    dim = sample_index % 8
    side = -1 if (sample_index // 8 + sample_index) % 2 == 0 else 1
    span = bounds[dim, 1] - bounds[dim, 0]
    cp[dim] = bounds[dim, 0] - 0.05 * span if side < 0 else bounds[dim, 1] + 0.05 * span
    return cp
```

- [ ] **Step 8: Add CLI output, canonical JSON, and SHA-256 sidecar**

Run: `PYTHONPATH=code /Users/guoyue/anaconda3/bin/python -m optimization_v2.tools.supplement_manifest --dataset-csv <snapshot.csv> --out <manifest.json> --seed 20260727`

Expected: JSON with 20 geometries, 63 unique conditions per geometry, recovery audit summary, and `<manifest.json>.sha256` whose digest matches `shasum -a 256`.

- [ ] **Step 9: Run Task 1 tests and commit**

Run: `PYTHONPATH=code /Users/guoyue/anaconda3/bin/python -m pytest code/tests/test_supplement_manifest.py -v`

Expected: all tests pass.

Commit: `git add code/optimization_v2/tools/supplement_manifest.py code/tests/test_supplement_manifest.py docs/superpowers/specs/2026-07-27-surrogate-data-ood-id-design.md && git commit -m "feat: add deterministic OOD ID supplement manifest"`

---

### Task 2: Manifest-Driven Parallel QBlade Runner

**Files:**
- Modify: `code/scripts/run_data_gen_parallel.py`
- Create: `code/tests/test_run_data_gen_manifest.py`

**Interfaces:**
- Consumes: Task 1 manifest JSON and optional `--geom-ids` subset.
- Produces: pkl nodes keyed by `geometry_<geom_id>` containing metadata and only the manifest-declared conditions.

- [ ] **Step 1: Write failing loader and task-construction tests**

```python
def test_manifest_tasks_preserve_ids_cp8_and_condition_splits(manifest_path):
    manifest = load_manifest(manifest_path)
    tasks = build_manifest_tasks(manifest, geom_ids={1000})
    assert len(tasks) == 1
    task = tasks[0]
    assert task.geom_id == 1000
    assert task.cp8.shape == (8,)
    assert task.geometry.shape == (22, 3)
    assert len(task.conditions) == 63
    assert task.condition_splits["condition_ood_alpha1"] == 6
```

- [ ] **Step 2: Run the test and verify it fails before implementation**

Run: `PYTHONPATH=code /Users/guoyue/anaconda3/bin/python -m pytest code/tests/test_run_data_gen_manifest.py::test_manifest_tasks_preserve_ids_cp8_and_condition_splits -v`

Expected: `FAIL` because manifest loading is absent.

- [ ] **Step 3: Add manifest arguments and a serializable task dataclass**

```python
p.add_argument("--manifest", type=str, default=None)
p.add_argument("--geom-ids", type=str, default=None,
               help="comma-separated exact geom_id subset")
p.add_argument("--manifest-sha256", type=str, default=None)

@dataclass(frozen=True)
class ManifestTask:
    geom_id: int
    category: str
    cp8: np.ndarray
    geometry: np.ndarray
    conditions: tuple[tuple[float, float, float, str], ...]
    seed: int
    manifest_sha256: str
```

- [ ] **Step 4: Implement strict manifest validation**

Reject wrong SHA-256, duplicate geometry IDs, duplicate condition keys, non-10 m/s wind, geometry-section mismatch, cp8-section mismatch, unknown split names, or a requested geom ID absent from the manifest. Do not silently fall back to the legacy geometry-npy path when `--manifest` is supplied.

- [ ] **Step 5: Write a failing worker test with a fake SIMULATION**

```python
def test_worker_runs_only_declared_conditions_and_stores_metadata(monkeypatch, task):
    monkeypatch.setattr(runner, "_SIMULATION", FakeSimulation)
    monkeypatch.setattr(runner, "_config", FakeConfig)
    geom_id, result, error = runner.worker_run((task, 1000, "CPU"))
    assert error is None and geom_id == task.geom_id
    assert result["geom_id"] == task.geom_id
    assert result["control_points"] == pytest.approx(task.cp8)
    assert result["category"] == task.category
    assert len([k for k in result if k.startswith("RPM")]) == len(task.conditions)
```

- [ ] **Step 6: Replace `run_all_simulation` with declared `run_one_simulation` calls for manifest tasks**

```python
for rpm, wind, angle, split in task.conditions:
    frame = sim.run_one_simulation(RPM=rpm, WIND_SPEED=wind, ANGLE=angle)
    if frame.empty:
        raise RuntimeError(f"empty QBlade result for {(rpm, wind, angle)}")
result = dict(sim.all_simulation_data)
result.update({"geom_id": task.geom_id, "category": task.category,
               "control_points": task.cp8.tolist(), "condition_split": split_map,
               "seed": task.seed, "manifest_sha256": task.manifest_sha256,
               "generator_git_commit": git_commit})
```

- [ ] **Step 7: Make output naming and resume ID-safe**

Each batch filename includes the manifest tag. Before queuing work, scan the chosen output directory and skip only geometry IDs whose pkl node passes metadata and complete-key validation. Never infer global IDs from batch numbers.

- [ ] **Step 8: Run Task 2 tests and legacy argument smoke test**

Run: `PYTHONPATH=code /Users/guoyue/anaconda3/bin/python -m pytest code/tests/test_run_data_gen_manifest.py -v`

Run: `PYTHONPATH=code /Users/guoyue/anaconda3/bin/python code/scripts/run_data_gen_parallel.py --help`

Expected: tests pass and legacy options remain documented.

- [ ] **Step 9: Commit**

Commit: `git add code/scripts/run_data_gen_parallel.py code/tests/test_run_data_gen_manifest.py && git commit -m "feat: run QBlade batches from versioned manifests"`

---

### Task 3: Key-Level pkl Audit and Frozen Dataset Export

**Files:**
- Create: `code/optimization_v2/tools/audit_and_freeze_pkl.py`
- Modify: `code/optimization_v2/tools/convert_pkl_to_dataset.py`
- Create: `code/tests/test_audit_and_freeze_pkl.py`

**Interfaces:**
- Consumes: historical pkl directory, supplement pkl directory, and the exact manifest.
- Produces: coverage JSON/CSV, duplicate decision log, versioned training CSV, ID interpolation CSV, condition-OOD CSV, geometry-OOD CSV, geometry split JSON, and training-only normalizer inputs.

- [ ] **Step 1: Write failing synthetic-pkl tests for geom ID and condition-key audit**

```python
def test_audit_reports_missing_and_duplicate_keys(tmp_path):
    write_fake_pkl(tmp_path / "a.pkl", geom_id=0, conditions=[(4000, 10, 82)])
    write_fake_pkl(tmp_path / "b.pkl", geom_id=0, conditions=[(4000, 10, 82)])
    report = audit_sources([tmp_path], expected_base_ids=range(1))
    assert report.unique_geometry_count == 1
    assert report.duplicate_keys == [(0, 4000.0, 10.0, 82.0)]
    assert len(report.missing_keys) == 41
```

- [ ] **Step 2: Implement tolerant pkl readers but strict normalized keys**

Normalize `geometry_<id>` and stored `geom_id`; reject disagreements. Parse numeric condition keys into exact `(int, float, float)` tuples, record unreadable pkl files, and retain source filename plus mtime for deterministic duplicate decisions.

- [ ] **Step 3: Write and implement explicit QBlade sign-convention tests**

The QBlade wrapper stores both raw time-series columns and post-processed UAV-positive aliases. Require `THRUST`, `THRUST_Z`, `MY`, and `TORQUE` when they are present; these aliases equal `-mean(Thrust)`, `-mean(Thrust_z)`, `-mean(My)`, and `-mean(Torque)` in the pinned wrapper. Never prefer raw `Thrust`, `Thrust_z`, `My`, or `Torque` merely because they appear first in a candidate list. For legacy frames that contain raw columns only, apply the same explicit sign conversion and record `output_sign_source=legacy_raw_converted`.

```python
def test_aggregate_prefers_uav_positive_qblade_aliases():
    frame = fake_qblade_frame_with_raw_and_postprocessed_columns()
    out, provenance = aggregate_timeseries(frame, last_n=120)
    assert out == pytest.approx({"T": frame.THRUST.iloc[-1],
                                 "H": frame.THRUST_Z.iloc[-1],
                                 "My": frame.MY.iloc[-1],
                                 "Q": frame.TORQUE.iloc[-1]})
    assert provenance == "qblade_uav_positive_aliases"
```

- [ ] **Step 4: Define deterministic duplicate resolution and test it**

Resolution order is: valid finite outputs over invalid; explicit stored `geom_id` over filename inference; manifest SHA match over mismatch; newer generator Git metadata over missing metadata; otherwise newest mtime. Every discarded row is written to the duplicate decision log.

- [ ] **Step 5: Extend conversion records with provenance**

```python
row.update({
    "geom_id": geom_id,
    "category": geo_val.get("category", "historical_train"),
    "condition_split": split_for_key(geo_val, condition_key),
    **{f"chord_cp_{i}": cp8[i] for i in range(4)},
    **{f"twist_cp_{i}": cp8[i + 4] for i in range(4)},
    "manifest_sha256": geo_val.get("manifest_sha256", ""),
    "output_sign_source": output_sign_source,
    "source_file": fpath.name,
})
```

Historical rows use recovered cp8 only when reconstruction passes `1e-10`; failures remain in the audit but are excluded from the 11D dataset.

- [ ] **Step 6: Write and implement leakage-free export tests**

```python
def test_freeze_exports_are_key_disjoint(frozen):
    key_cols = ["geom_id", "RPM", "WIND", "ANGLE"]
    sets = [set(map(tuple, df[key_cols].to_numpy())) for df in frozen.values()]
    assert all(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
    assert set(frozen["condition_ood"]["ANGLE"]) == {89.0}
    assert set(frozen["train"]["ANGLE"]) <= {82, 83, 84, 85, 86, 87, 88}
```

- [ ] **Step 7: Implement geometry-grouped split manifests**

Use seed `20260727`; split only historical complete base-grid geometry IDs, never rows. Save exact train/validation/test geometry ID arrays. New geometry ID, geometry OOD, and condition OOD records remain external evaluation sets.

- [ ] **Step 8: Emit checksums and acceptance report**

The acceptance JSON contains input paths, Git commits, manifest digest, pkl counts, unique geometry IDs, per-geometry coverage, duplicates, missing keys, excluded cp8 recoveries, output row counts, split IDs, and SHA-256 for every CSV/JSON artifact.

- [ ] **Step 9: Run Task 3 tests and commit**

Run: `PYTHONPATH=code /Users/guoyue/anaconda3/bin/python -m pytest code/tests/test_audit_and_freeze_pkl.py -v`

Expected: all tests pass.

Commit: `git add code/optimization_v2/tools/audit_and_freeze_pkl.py code/optimization_v2/tools/convert_pkl_to_dataset.py code/tests/test_audit_and_freeze_pkl.py && git commit -m "feat: audit and freeze keyed QBlade datasets"`

---

### Task 4: 3660 Physics Preflight and Batch Generation

**Files:**
- Create locally and copy through Git: `code/scripts/run_supplement_3660.sh`
- Create remotely as generated artifacts, not Git: versioned manifest, logs, pkl batch directory, preflight JSON.

**Interfaces:**
- Consumes: reviewed commits from Tasks 1-3.
- Produces: six alpha=1 preflight simulations, two metadata smoke simulations, repaired geometry 0/999 conditions, and the accepted 1341-condition supplement batch.

- [ ] **Step 1: Verify remote source state without modifying it**

Run: `ssh 3660workstation 'cd /home/gy/gy_2026/graduation/LFM && git status --short && git rev-parse HEAD && pgrep -af run_data_gen_parallel.py || true && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'`

Expected: exact remote commit and dirty state are recorded; no active generator conflicts with the new batch.

- [ ] **Step 2: Push the reviewed branch and fast-forward or clone it into a separate remote run directory**

Do not overwrite the existing dirty remote checkout. Use a versioned run checkout such as `/home/gy/gy_2026/graduation/runs/LFM_datafreeze_20260727_<commit7>` and initialize the pinned submodule.

- [ ] **Step 3: Generate the manifest against the latest audited historical CSV**

Run the Task 1 CLI with seed `20260727`, save the JSON and SHA-256 in the versioned remote output directory, then rerun it and verify byte-for-byte identity with `cmp`.

- [ ] **Step 4: Run the six-condition alpha=1 physics preflight**

Use one baseline geometry and only the six `ANGLE=89` conditions. Save raw QBlade mean values before sign conversion and converted `T,H,My,Q`. Acceptance requires finite nonzero outputs, monotonic/continuous RPM trends without sign discontinuity, and explicit force-balance tables under both fuselage-drag assumptions.

- [ ] **Step 5: Apply official Iris parameters in the preflight branch and compare drag models**

The report must compare `l=0.13`, `d=0.023` with the previous code values and compare current cross-flow equivalent area with the thesis `Cd*A=1.10*0.0517`. This is a sensitivity report, not a claim that either drag model is validated.

- [ ] **Step 6: Run one ID and one geometry-OOD metadata smoke condition**

Run one declared base condition for geom IDs 1000 and 1010. Reload the resulting pkl and verify exact cp8, geometry, category, condition key, manifest digest, and generator commit.

- [ ] **Step 7: Generate exact missing-condition manifests for geom 0 and 999**

Audit the remote raw pkl first. Derive missing keys from stored `geom_id` and parsed condition keys; do not assume the previously observed counts remain 41 and 40. If counts differ, record the live counts and generate only live missing keys.

- [ ] **Step 8: Launch the accepted batch with resumable versioned outputs**

Run the repair manifest and new-geometry manifest into separate directories. Use CPU QBlade, worker/OMP settings derived from the live core count, per-geometry or small-batch pkl writes, and `nohup` logs containing the manifest digest and commit.

- [ ] **Step 9: Monitor until all declared keys finish**

Every status sample records generator PID, CPU/GPU utilization, newest pkl mtime, completed unique geometry IDs, completed unique condition keys, failures, duplicates, and remaining keys. Completion is based only on the Task 3 audit tool.

- [ ] **Step 10: Commit the launcher after the smoke command succeeds**

Commit: `git add code/scripts/run_supplement_3660.sh && git commit -m "ops: add reproducible 3660 supplement launcher"`

---

### Task 5: Freeze Acceptance and Training Handoff

**Files:**
- Generated remotely: `data_for_train/frozen/<version>/...`
- Create locally after transferring only reports/CSV: `docs/experiments/<version>-data-freeze-audit.md`

**Interfaces:**
- Consumes: complete historical and supplement pkl directories on 3660.
- Produces: accepted frozen CSVs and a human-readable evidence report for the later model-training plan.

- [ ] **Step 1: Run final key-level audit on 3660**

Acceptance requires historical geometry IDs `0..999` each have exactly 42 unique base conditions; all 20 new geometries each have exactly 63 unique declared conditions; no unresolved duplicate key; no unreadable accepted pkl; and every new geometry metadata block matches the manifest.

- [ ] **Step 2: Export isolated datasets**

Export formal training base-grid CSV, ID-geometry base-grid evaluation CSV, ID condition-interpolation CSV, alpha=1 condition-OOD CSV, geometry-OOD base-grid CSV, and joint geometry-plus-condition OOD CSV. Never transfer raw pkl.

- [ ] **Step 3: Fit normalization only on the training geometry split**

Run `analyze_and_normalize.py` only after feeding the exact training geometry IDs. Verify validation, test, and all OOD rows do not influence scaler statistics.

- [ ] **Step 4: Transfer only CSVs, manifests, checksums, split lists, and reports to the local Git workspace**

Verify every transferred artifact against its remote SHA-256 before using it.

- [ ] **Step 5: Write the evidence report and commit metadata, not raw simulation data**

The report links each row count and coverage conclusion to an acceptance JSON/CSV artifact. It explicitly separates observed facts from planned training actions and contains no model metrics.

- [ ] **Step 6: Create the next implementation plan only after acceptance**

Write `docs/superpowers/plans/<date>-four-model-retrain-and-sweep-plan.md` with the frozen version, exact geometry counts, grouped split IDs, four 1500-epoch model configurations, result directories, and final-test one-shot rule.
