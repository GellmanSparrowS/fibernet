"""AI cycle study resumes candidate reruns and rejects stale checkpoints."""
from dataclasses import asdict
import json

import pytest

from benchmarks.ai_cycle_selector import (
    AICycleSelector, ENGINE_SRC_HASH, ENGINE_VERSION, RandomForestRegressor)
from benchmarks.constrained_cycle_intervention import (
    ConstrainedCycleConfig, ConstrainedCycleIntervention)


def test_ai_cycle_candidate_resume_and_signature(tmp_path):
    config = ConstrainedCycleConfig(
        units=("square",), seeds=(11,), directions=("x",))
    reference_path = tmp_path / "constrained_reference.json"
    ConstrainedCycleIntervention(config).run(reference_path)
    path = tmp_path / "ai_cycle.json"
    selector = AICycleSelector(reference_path=reference_path, study_config=config)
    first = selector.run(path, max_new=1)
    assert first["study_config"] == json.loads(json.dumps(asdict(config)))
    case = first["cases"]["square:seed11:x"]
    assert len(case["options"]) == 1
    item = case["options"]["0"]
    assert item["euler_route_edges"] > 0
    assert len(item["early_features"]) == 8
    second = selector.run(path, max_new=1)
    assert len(second["cases"]["square:seed11:x"]["options"]) == 2
    with pytest.raises(ValueError, match="incompatible AI cycle checkpoint"):
        AICycleSelector(max_options_per_case=99, reference_path=reference_path,
                        study_config=config).run(path, max_new=1)
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["study_config"]["steps"] += 1
    changed_path = tmp_path / "different_study_checkpoint.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible AI cycle checkpoint"):
        selector.run(changed_path, max_new=1)


@pytest.mark.parametrize("mismatch", [
    "engine_version", "engine_hash", "analysis_hash", "config",
    "missing_case", "extra_case", "wrong_case", "invalid_cases",
])
def test_ai_cycle_rejects_incompatible_reference_before_simulation(
        tmp_path, monkeypatch, mismatch):
    config = ConstrainedCycleConfig(
        units=("square",), seeds=(11,), directions=("x",))
    reference_path = tmp_path / "reference.json"
    selector = AICycleSelector(reference_path=reference_path, study_config=config)
    key = "square:seed11:x"
    reference = {"config": json.loads(json.dumps(asdict(config))),
                 "engine_version": ENGINE_VERSION,
                 "engine_hash": ENGINE_SRC_HASH,
                 "analysis_hash": selector.study.analysis_hash,
                 "cases": {key: {}}}
    if mismatch == "engine_version":
        reference["engine_version"] = "e2v12"
    elif mismatch == "config":
        reference["config"]["steps"] += 1
    elif mismatch == "missing_case":
        reference["cases"].pop(key)
    elif mismatch == "extra_case":
        reference["cases"]["square:seed23:x"] = {}
    elif mismatch == "wrong_case":
        reference["cases"] = {"square:seed23:x": {}}
    elif mismatch == "invalid_cases":
        reference["cases"] = [key]
    else:
        reference[mismatch] = "stale-reference"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    def unexpected_simulation(graph):
        pytest.fail("reference validation must precede every simulation")

    monkeypatch.setattr(selector.study, "_simulate", unexpected_simulation)
    output = tmp_path / "ai_cycle.json"
    with pytest.raises(ValueError, match="incompatible constrained cycle reference") as caught:
        selector.run(output, max_new=1)
    assert "new output path" in str(caught.value)
    assert "do not edit only checkpoint metadata" in str(caught.value)
    assert not output.exists()


def test_ai_cycle_rejects_reference_without_eligible_candidates_before_simulation(
        tmp_path, monkeypatch):
    config = ConstrainedCycleConfig(
        units=("square",), seeds=(11,), directions=("x",))
    reference_path = tmp_path / "reference.json"
    selector = AICycleSelector(reference_path=reference_path, study_config=config)
    reference = {"config": json.loads(json.dumps(asdict(config))),
                 "engine_version": ENGINE_VERSION,
                 "engine_hash": ENGINE_SRC_HASH,
                 "analysis_hash": selector.study.analysis_hash,
                 "cases": {"square:seed11:x": {
                     "status": "no sufficiently populated material class",
                     "same_length_candidate_count": 0}}}
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    def unexpected_simulation(graph):
        pytest.fail("ineligible references must be rejected before simulation")

    monkeypatch.setattr(selector.study, "_simulate", unexpected_simulation)
    output = tmp_path / "ai_cycle.json"
    with pytest.raises(ValueError, match="lack complete eligible candidates") as caught:
        selector.run(output, max_new=1)
    assert "choose a study configuration with eligible cycles" in str(caught.value)
    assert not output.exists()


@pytest.mark.parametrize("units", [("square",), ("square", "square")])
def test_ai_cycle_evaluation_requires_distinct_topologies_before_fit(
        monkeypatch, units):
    selector = AICycleSelector(study_config=ConstrainedCycleConfig(
        units=units, seeds=(11,), directions=("x",)))

    def unexpected_fit(*args, **kwargs):
        pytest.fail("single-topology evaluation must not fit a model")

    monkeypatch.setattr(RandomForestRegressor, "fit", unexpected_fit)
    with pytest.raises(ValueError, match="at least two distinct topologies"):
        selector.evaluate({"cases": {}}, {"cases": {}})
