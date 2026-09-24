"""Matched cycle deletions preserve a complete route and checkpoint identity."""
import pytest

from benchmarks.constrained_cycle_intervention import (
    ConstrainedCycleConfig, ConstrainedCycleIntervention)


def test_matched_cycle_reruns_and_resume(tmp_path):
    study = ConstrainedCycleIntervention(ConstrainedCycleConfig(
        units=("square",), seeds=(11,), directions=("x",)))
    path = tmp_path / "cycles.json"
    data = study.run(path)
    case = data["cases"]["square:seed11:x"]
    assert case["status"] == "complete"
    assert case["same_length_candidate_count"] >= 5
    assert case["max_material_class_spread"] == 0
    assert len(case["interventions"]) == 5
    for outcome in case["interventions"].values():
        assert outcome["euler_route_edges"] == outcome["edges"]
        assert outcome["final_raw_reaction"] > 0
    assert study.run(path) == data
    changed = ConstrainedCycleIntervention(ConstrainedCycleConfig(
        units=("square",), seeds=(11,), directions=("x",),
        min_active=0.12))
    with pytest.raises(ValueError, match="incompatible"):
        changed.run(path)
