"""Example composition with synthetic stubs: no solver or model imports/runs."""
import json
from pathlib import Path
import runpy
import sys
import types

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def workflows(monkeypatch):
    class StubDataset:
        def __init__(self, *, path, seed0, **kwargs):
            self.path = path
            self.next_seed = seed0
            self.config = {"engine_source": "synthetic-test-stub-not-physics"}

        def generate(self, n):
            X = np.arange(n * 14, dtype=float).reshape(n, 14)
            Y = np.arange(n * 3, dtype=float).reshape(n, 3)
            self.next_seed += n
            # Same NPZ keys as DatasetStream; not a real resumable checkpoint.
            np.savez_compressed(self.path, X=X, Y=Y,
                                specs=json.dumps([{} for _ in range(n)]),
                                meta=json.dumps({"fixture": "synthetic test only"}))
            return X, Y

    def forbidden_regressor(*args, **kwargs):
        raise AssertionError("Training is forbidden in this test")

    package = types.ModuleType("fibernet")
    package.__path__ = []
    ml = types.ModuleType("fibernet.ml")
    ml.PlanarPhysicalDataset = StubDataset
    ml.PhysicalRegressor = forbidden_regressor
    package.ml = ml
    monkeypatch.setitem(sys.modules, "fibernet", package)
    monkeypatch.setitem(sys.modules, "fibernet.ml", ml)
    dataset = runpy.run_path(str(EXAMPLES / "physical_dataset_workflow.py"))
    learning = runpy.run_path(str(EXAMPLES / "physical_learning_workflow.py"))
    return dataset["PhysicalDatasetWorkflow"], learning["PhysicalLearningWorkflow"]


def test_constructor_default_meets_actual_learner_minimum(workflows, tmp_path):
    DatasetWorkflow, LearningWorkflow = workflows
    summary = DatasetWorkflow(tmp_path / "labels").run()
    path = tmp_path / "labels" / "physical_labels.npz"
    X, Y, source = LearningWorkflow(tmp_path / "learning", path)._data()
    assert summary["samples"] == len(X) == 10
    assert Y.shape == (10, 3)
    assert source == str(path)
    assert json.loads((path.parent / "summary.json").read_text()) == summary


@pytest.mark.parametrize("samples", [None, 8, 12])
def test_cli_default_and_explicit_count(workflows, monkeypatch, tmp_path, samples):
    _, LearningWorkflow = workflows
    monkeypatch.chdir(tmp_path)
    argv = ["physical_dataset_workflow.py"]
    if samples is not None:
        argv += ["--samples", str(samples), "--output-dir", "custom_labels"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(EXAMPLES / "physical_dataset_workflow.py"), run_name="__main__")
    folder = tmp_path / ("demo_labels" if samples is None else "custom_labels")
    path = folder / "physical_labels.npz"
    count = 10 if samples is None else samples
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {"X", "Y", "specs", "meta"}
        assert saved["X"].shape == (count, 14)
        assert saved["Y"].shape == (count, 3)
    assert json.loads((folder / "summary.json").read_text())["samples"] == count
    learner = LearningWorkflow(tmp_path / "learning", path)
    if count < 10:
        with pytest.raises(ValueError, match="N>=10"):
            learner._data()
    else:
        assert len(learner._data()[0]) == count


@pytest.mark.parametrize("kind", ["nine_rows", "bad_features", "bad_targets", "nan_features", "nan_targets"])
def test_learner_validation_is_not_relaxed(workflows, tmp_path, kind):
    _, LearningWorkflow = workflows
    X, Y = np.ones((10, 14)), np.ones((10, 3))
    if kind == "nine_rows":
        X, Y = X[:9], Y[:9]
    elif kind == "bad_features":
        X = X[:, :13]
    elif kind == "bad_targets":
        Y = Y[:, :2]
    elif kind == "nan_features":
        X[0, 0] = np.nan
    else:
        Y[0, 0] = np.nan
    path = tmp_path / "invalid.npz"
    np.savez_compressed(path, X=X, Y=Y)
    with pytest.raises(ValueError, match="expected finite"):
        LearningWorkflow(tmp_path / "learning", path)._data()


@pytest.mark.parametrize("samples", [1, 8, 12])
def test_constructor_preserves_explicit_counts(workflows, tmp_path, samples):
    DatasetWorkflow, _ = workflows
    summary = DatasetWorkflow(tmp_path, samples=samples).run()
    assert summary["samples"] == samples
