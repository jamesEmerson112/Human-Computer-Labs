import math

from lamp_sim.backends import KinematicBackend
from lamp_sim.benchmark import run_trial
from lamp_sim.gestures import parse_gesture


def test_benchmark_trace_has_expected_duration_and_no_fake_contact_measurements():
    backend = KinematicBackend()
    gesture = parse_gesture({
        "schema_version": 1, "name": "hold", "label": "Hold", "units": "radians",
        "steps": [{"hold_seconds": 0.5}],
    }, backend.names)
    summary, rows = run_trial(backend, gesture, speed=0.5, record=True)
    assert summary["simulated_s"] >= 1.9  # One-second command plus a one-second settling tail.
    assert summary["simulated_s"] <= 2.1
    assert len(rows) == summary["frames"]
    assert summary["tracking_rmse_deg"] == 0
    assert summary["max_contacts"] is None
    assert summary["max_contact_force_n"] is None
    assert summary["loop_wall_s"] > 0
    assert math.isfinite(summary["realtime_factor"])
