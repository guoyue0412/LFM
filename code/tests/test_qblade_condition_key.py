import sys
from pathlib import Path


SIMULATION_ROOT = (
    Path(__file__).resolve().parents[1]
    / "Propeller_project-main"
    / "code"
    / "Simulation_QBlade"
)
sys.path.insert(0, str(SIMULATION_ROOT))

from class_sim import simulation as qblade_simulation  # noqa: E402


def test_direct_qblade_condition_uses_canonical_persisted_key():
    assert hasattr(qblade_simulation, "condition_label")
    condition_label = qblade_simulation.condition_label
    assert condition_label(4000.0, 10.0, 89.0) == (
        "RPM4000.0_Wind10.0_Angle89.0"
    )
    assert condition_label(4250, 10, 82.5) == (
        "RPM4250.0_Wind10.0_Angle82.5"
    )
