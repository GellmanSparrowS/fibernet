"""Fresh-fit physical holdouts are independent of model width and initialization."""
import numpy as np
import pytest

from fibernet.ml.physical_learning import PhysicalRegressor
from fibernet.ml.physical_surrogate import MLP


def samples(n):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(n, 14)) + np.arange(n)[:, None] ** 2
    y = rng.normal(size=(n, 3)) + np.arange(n)[:, None] ** 2
    x[:, -1] = 7.0
    y[:, -1] = 11.0
    return x, y


def assert_partition_and_scaling(model, x, y, expected_count):
    ti, vi = model.train_indices, model.val_indices
    assert len(vi) == expected_count
    assert not np.intersect1d(ti, vi).size
    np.testing.assert_array_equal(np.sort(np.concatenate((ti, vi))), np.arange(len(x)))
    for values, prefix in ((x, 'x'), (y, 'y')):
        np.testing.assert_allclose(getattr(model, prefix + '_mean'), values[ti].mean(0))
        std = values[ti].std(0)
        std[std < 1e-12] = 1.0
        np.testing.assert_allclose(getattr(model, prefix + '_std'), std)
    expected = np.random.default_rng(7).permutation(len(x))
    np.testing.assert_array_equal(vi, expected[:expected_count])
    np.testing.assert_array_equal(ti, expected[expected_count:])


@pytest.mark.parametrize('n,val_frac,count', [(20, .2, 4), (5, .2, 2),
                                             (5, .01, 2), (6, .49, 2)])
def test_fresh_adapter_mlp_widths_match_ridge_holdout(n, val_frac, count):
    pytest.importorskip('sklearn')
    x, y = samples(n)
    models = [PhysicalRegressor('mlp', {'hidden': width}, seed=7) for width in (4, 32)]
    models.append(PhysicalRegressor('ridge', seed=7))
    for model in models:
        history = model.train(x, y, epochs=1, val_frac=val_frac)
        assert history['validation_count'] == count
        assert_partition_and_scaling(model, x, y, count)
    for model in models[1:]:
        np.testing.assert_array_equal(model.train_indices, models[0].train_indices)
        np.testing.assert_array_equal(model.val_indices, models[0].val_indices)


@pytest.mark.parametrize('n,val_frac,count', [(20, .2, 4), (5, .2, 2),
                                             (5, .01, 2), (5, .99, 3),
                                             (1, .2, 0), (2, .2, 0),
                                             (3, .2, 0), (4, .2, 0),
                                             (5, 0, 0), (20, 0, 0)])
def test_standalone_mlp_split_edges_and_train_only_normalization(n, val_frac, count):
    x, y = samples(n)
    for width in (4, 32):
        model = MLP(hidden=width, seed=7)
        history = model.train(x, y, epochs=1, val_frac=val_frac)
        assert_partition_and_scaling(model, x, y, count)
        assert np.isfinite(history['train']).all()
        assert np.isfinite(history['val']).all()
        if count == 0:
            assert history['val'] == history['train']


@pytest.mark.parametrize('val_frac', [-.1, 1, 1.1, np.nan, np.inf, -np.inf])
def test_standalone_invalid_fraction_rejected_before_initialization(val_frac):
    x, y = samples(20)
    model = MLP(seed=7)
    with pytest.raises(ValueError, match='val_frac'):
        model.train(x, y, epochs=1, val_frac=val_frac)
    assert model.W1 is None
    assert not hasattr(model, 'train_indices')


@pytest.mark.parametrize('val_frac', [0, -.1, .5, 1, np.nan, np.inf])
def test_adapter_keeps_positive_fraction_contract(val_frac):
    x, y = samples(5)
    for key in ('mlp', 'ridge'):
        with pytest.raises(ValueError, match='training controls'):
            PhysicalRegressor(key, seed=7).train(x, y, epochs=1, val_frac=val_frac)


def test_adapter_still_requires_five_samples():
    x, y = samples(4)
    for key in ('mlp', 'ridge'):
        with pytest.raises(ValueError, match='at least 5'):
            PhysicalRegressor(key, seed=7).train(x, y, epochs=1)
