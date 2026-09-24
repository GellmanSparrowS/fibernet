"""Check case identity, loading direction and safe resumable output."""
import json
import math

from benchmarks.recruitment_sensitivity import (
    RecruitmentSensitivity, SensitivityConfig)


def test_sensitivity_resume_and_direction(tmp_path):
    cfg = SensitivityConfig(units=("kagome",), seeds=(11,),
                            directions=("x", "y"), contact=(False,),
                            alphas=(0.02, 0.05),
                            min_actives=(0.0001, 0.05), grid=2,
                            increments=4, steps=300)
    study = RecruitmentSensitivity(cfg)
    output = tmp_path / "screen.json"
    data = study.run(output)
    assert len(data["cases"]) == 2
    assert data == study.run(output)
    assert data == json.loads(output.read_text(encoding="utf-8"))
    assert not output.with_suffix(".json.tmp").exists()
    for case in data["cases"].values():
        assert case["nodes"] == 121 and case["edges"] == 184
        assert math.isfinite(case["final_raw_reaction"])
        assert case["max_contact_pairs"] == 0
        for item in case["thresholds"].values():
            assert 0 <= item["never_recruited_fraction"] <= 1
            assert 0 <= item["never_recruited_length_fraction"] <= 1
        low = case["absolute_thresholds"]["0.0001"]
        high = case["absolute_thresholds"]["0.0500"]
        assert high["never_recruited_fraction"] >= low["never_recruited_fraction"]
        assert (high["first_spanning_frame"] == -1 or
                high["first_spanning_frame"] >= low["first_spanning_frame"])
