"""Validate the independent FEM checkpoint and recruitment protocol."""
import json
import math

from benchmarks.recruitment_fem_validation import (
    FemRecruitmentValidation, FemValidationConfig)


def test_fem_validation_one_case_resume(tmp_path):
    cfg = FemValidationConfig(seeds=(11,), directions=("x",),
                              stretches=(1.08,), min_actives=(0.05,))
    study = FemRecruitmentValidation(cfg)
    path = tmp_path / "fem.json"
    data = study.run(path)
    assert data == study.run(path)
    assert data == json.loads(path.read_text(encoding="utf-8"))
    assert not path.with_suffix(".json.tmp").exists()
    result = data["cases"]["seed11:x:stretch1.080"]
    assert result["nodes"] == 132 and result["edges"] == 192
    assert math.isfinite(result["max_force"])
    assert 0 <= result["thresholds"]["0.0500"]["recruited_edge_fraction"] <= 1
