"""Resumable public physical-label generation and checkpoint contracts."""

import numpy as np
import pytest

from fibernet.ml import PlanarPhysicalDataset



def test_library_physical_label_matches_app(tmp_path):
    path = tmp_path / "labels.npz"
    stream = PlanarPhysicalDataset(unit="square", grid=3, seed0=5, path=path)
    X, Y = stream.generate(1)
    assert X.shape == (1, 14)
    assert Y.shape == (1, 3)
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
