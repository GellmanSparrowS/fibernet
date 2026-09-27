"""Independent beam-FEM cycle screening is bounded and resumable."""
import pytest

from benchmarks.independent_fem_cycle_intervention import (
    IndependentFemCycleCheck, ROOT)


@pytest.mark.skipif(not (ROOT / "benchmarks" / "results" /
                         "constrained_cycle_intervention.json").exists(),
                    reason="24-case reference study is not installed")
def test_independent_fem_resume_and_matched_grips(tmp_path):
    path = tmp_path / "fem_cycles.json"
    first = IndependentFemCycleCheck().run(path, max_new_cases=1)
    assert len(first["cases"]) == 1
    case = first["cases"]["square:seed11:x"]
    assert case["status"] == "complete"
    assert len(case["interventions"]) == 5
    assert case["baseline"]["fem_balance_relative"] < 1e-5
    second = IndependentFemCycleCheck().run(path, max_new_cases=1)
    assert len(second["cases"]) == 2
    with pytest.raises(ValueError, match="incompatible independent FEM checkpoint"):
        IndependentFemCycleCheck(stretch=1.10).run(path, max_new_cases=1)
