"""AI cycle study resumes candidate reruns and rejects stale checkpoints."""
import pytest

from benchmarks.ai_cycle_selector import AICycleSelector, RESULTS


@pytest.mark.skipif(not (RESULTS / "constrained_cycle_intervention.json").exists(),
                    reason="24-case reference study is not installed")
def test_ai_cycle_candidate_resume_and_signature(tmp_path):
    path = tmp_path / "ai_cycle.json"
    selector = AICycleSelector()
    first = selector.run(path, max_new=1)
    case = first["cases"]["square:seed11:x"]
    assert len(case["options"]) == 1
    item = case["options"]["0"]
    assert item["euler_route_edges"] > 0
    assert len(item["early_features"]) == 8
    second = selector.run(path, max_new=1)
    assert len(second["cases"]["square:seed11:x"]["options"]) == 2
    with pytest.raises(ValueError, match="incompatible AI cycle checkpoint"):
        AICycleSelector(max_options_per_case=99).run(path, max_new=1)
