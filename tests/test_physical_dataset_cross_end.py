"""Resumable public physical labels agree with the desktop APP."""
from pathlib import Path
import sys

import numpy as np
import pytest

from fibernet.ml import PlanarPhysicalDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FiberScope"))
from fslab.mlmodel import make_sample
from fslab.structure import StructureFactory


def test_library_physical_label_matches_app(tmp_path):
    rng = np.random.default_rng(5)
    factory = StructureFactory(
        unit="square", seed=5,
        line_displacements=rng.uniform(-0.2, 0.2, (5, 2)).tolist(),
        perturbation=0.1 * rng.uniform())
    expected_x, expected_y = make_sample(factory)
    path = tmp_path / "labels.npz"
    stream = PlanarPhysicalDataset(unit="square", grid=3, seed0=5, path=path)
    X, Y = stream.generate(1)
    np.testing.assert_array_equal(X[0], expected_x)
    np.testing.assert_array_equal(Y[0], expected_y)
    resumed = PlanarPhysicalDataset(unit="square", grid=3, seed0=5, path=path)
    assert resumed.next_seed == 6
    np.testing.assert_array_equal(resumed.arrays()[0], X)
    with pytest.raises(ValueError, match="configuration/schema mismatch"):
        PlanarPhysicalDataset(unit="square", grid=3, seed0=7, path=path)


def test_unlabeled_stream_is_explicit_and_bounded(tmp_path):
    stream = PlanarPhysicalDataset(unit="square", grid=1, mode="generate",
                                   path=tmp_path / "unlabeled.npz")
    X, Y = stream.generate(3)
    assert X.shape == (3, 14) and np.isnan(Y).all()
    with pytest.raises(ValueError, match="dataset size"):
        stream.generate(2001)
