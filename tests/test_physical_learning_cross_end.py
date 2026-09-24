"""Shared APP physical surrogate keeps training and acquisition behavior."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

from fibernet.ml import PhysicalRegressor, light_features
from fibernet.ml.physical_learning import select_candidate
from examples.physical_learning_workflow import PhysicalLearningWorkflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FiberScope"))
from fslab.learning import Regressor as AppRegressor


def test_regressor_public_api_matches_app():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(40, 14))
    targets = np.column_stack((2 * features[:, 0] + features[:, 1],
                               features[:, 2] - features[:, 3],
                               features[:, 4] + 0.5 * features[:, 5]))
    assert PhysicalRegressor is AppRegressor
    expected = {"mlp": "052d896173e1b9bc",
                "ridge": "511c2feab86091bc",
                "random_forest": "99aa146406c63c29"}
    for key, digest in expected.items():
        model = PhysicalRegressor(key=key, seed=23)
        history = model.train(features, targets, epochs=30)
        prediction = np.asarray(model.predict(features[:5]), float)
        assert history["validation_count"] == 8
        assert hashlib.sha256(prediction.tobytes()).hexdigest()[:16] == digest


def test_structural_features_and_acquisition():
    positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], int)
    descriptors = light_features(positions, edges)
    assert descriptors.shape == (14,)
    assert np.isfinite(descriptors).all()
    pool = np.array([[0.0] * 14, [1.0] * 14])
    index = select_candidate(pool, np.empty((0, 14)),
                             np.empty((0, 3)), "random",
                             np.random.default_rng(7))
    assert index in (0, 1)


def test_public_learning_errors_are_clear():
    with pytest.raises(ValueError, match="integer edges"):
        light_features(np.array([[0, 0], [1, 0]]), np.empty((0, 2), int))
    with pytest.raises(ValueError, match="invalid node indices"):
        light_features(np.array([[0, 0], [1, 0]]), np.array([[0, 2]]))
    model = PhysicalRegressor(key="ridge", seed=1)
    with pytest.raises(RuntimeError, match="train"):
        model.predict(np.zeros(14))
    with pytest.raises(ValueError, match="invalid shape"):
        select_candidate(np.empty((0, 14)), np.empty((0, 14)),
                         np.empty((0, 3)), "random", np.random.default_rng(1))
    with pytest.raises(ValueError, match="training controls"):
        model.train(np.ones((5, 14)), np.ones((5, 3)), val_frac=0)


def test_physical_learning_workflow(tmp_path):
    summary = PhysicalLearningWorkflow(tmp_path).run()
    assert summary["samples"] == 80
    assert summary["validation_rows"] == 16
    assert "synthetic" in summary["source"]
    with np.load(tmp_path / "held_out_predictions.npz",
                 allow_pickle=False) as saved:
        assert saved["predicted"].shape == (16, 3)
