"""Physical prediction admission with preconfigured state; no model fitting.

Load the exact source module without package initialization. This module itself
imports only importlib and NumPy; no sklearn, native provider, or model training
is needed for the inert estimator fixtures below.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SOURCE = Path(__file__).resolve().parents[1] / 'fibernet/ml/physical_learning.py'
_SPEC = importlib.util.spec_from_file_location('_physical_prediction_contract', _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PhysicalRegressor = _MODULE.PhysicalRegressor


class _EstimatorSpy:
    """Record inputs and return an explicit three-column arithmetic oracle."""
    def __init__(self, single_output):
        self.single_output = single_output
        self.calls = []

    def predict(self, values):
        values = np.asarray(values)
        self.calls.append(values.copy())
        rows = np.atleast_2d(values)
        out = np.column_stack((rows.sum(axis=1), rows[:, 0], rows[:, -1]))
        return out[0] if self.single_output and values.ndim == 1 else out


def _configured(key, width):
    """Represent a completed fit without calling train, fit, or an optimizer."""
    model = PhysicalRegressor(key=key)
    model.x_mean = np.arange(width, dtype=float) * .25
    model.x_std = np.arange(width, dtype=float) + 1.
    model.y_mean = np.array([10., 20., 30.])
    model.y_std = np.array([2., 3., 4.])
    model.estimator = _EstimatorSpy(single_output=key == 'mlp')
    return model


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
@pytest.mark.parametrize('width', [1, 4, 14])
@pytest.mark.parametrize('shape', ['single', 'singleton_batch', 'batch'])
def test_predict_preserves_fitted_width_shape_values_and_inputs(key, width, shape):
    model = _configured(key, width)
    rows = (np.arange(3 * width, dtype=float).reshape(3, width) + .5)
    single = shape == 'single'
    values = (rows[1].copy() if single else
              rows[1:2].copy() if shape == 'singleton_batch' else rows.copy())
    original = values.copy()
    values.setflags(write=False)
    state = {name: getattr(model, name).copy()
             for name in ('x_mean', 'x_std', 'y_mean', 'y_std')}

    result = model.predict(values)

    backend_input = (original if key == 'mlp' else
                     (np.atleast_2d(original) - state['x_mean']) / state['x_std'])
    oracle_rows = np.atleast_2d(backend_input)
    expected = np.column_stack((oracle_rows.sum(axis=1),
                                oracle_rows[:, 0], oracle_rows[:, -1]))
    if key != 'mlp':
        expected = expected * state['y_std'] + state['y_mean']
    if single:
        expected = expected[0]
    assert result.shape == ((3,) if single else (len(values), 3))
    np.testing.assert_allclose(result, expected)
    assert len(model.estimator.calls) == 1
    np.testing.assert_allclose(model.estimator.calls[0], backend_input)
    np.testing.assert_array_equal(values, original)
    assert not values.flags.writeable
    for name, old in state.items():
        np.testing.assert_array_equal(getattr(model, name), old)


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
@pytest.mark.parametrize('values', [
    np.arange(14., dtype=float).reshape(14, 1),
    np.ones((3, 1)),
    np.ones(1),
    np.ones(13),
    np.ones((2, 15)),
])
def test_predict_rejects_wrong_width_before_backend(key, values):
    model = _configured(key, 14)
    original = values.copy()
    with pytest.raises(ValueError, match='feature width.*14.*got'):
        model.predict(values)
    assert model.estimator.calls == []
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
@pytest.mark.parametrize('values', [np.array(1.), np.ones((2, 3, 14))])
def test_predict_rejects_wrong_rank_before_backend(key, values):
    model = _configured(key, 14)
    with pytest.raises(ValueError, match=r'shape \(D,\) or \(N, D\)'):
        model.predict(values)
    assert model.estimator.calls == []


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('single', [True, False])
def test_predict_rejects_nonfinite_before_backend(key, bad, single):
    model = _configured(key, 4)
    values = np.ones(4) if single else np.ones((2, 4))
    values.flat[-1] = bad
    original = values.copy()
    with pytest.raises(ValueError, match='must be finite'):
        model.predict(values)
    assert model.estimator.calls == []
    np.testing.assert_array_equal(values, original)


class _ArithmeticForbidden(np.ndarray):
    def __array_ufunc__(self, *args, **kwargs):
        raise AssertionError('normalization was attempted before admission')


@pytest.mark.parametrize('values', [
    np.ones((14, 1)), np.array(1.), np.full(14, np.nan),
])
def test_invalid_prediction_is_rejected_before_normalization(values):
    model = _configured('ridge', 14)
    model.x_mean = model.x_mean.view(_ArithmeticForbidden)
    model.x_std = model.x_std.view(_ArithmeticForbidden)
    with pytest.raises(ValueError):
        model.predict(values)
    assert model.estimator.calls == []


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
def test_single_feature_batch_is_not_confused_with_multi_feature_sample(key):
    model = _configured(key, 1)
    result = model.predict(np.array([[1.], [2.], [3.]]))
    assert result.shape == (3, 3)
    assert len(model.estimator.calls) == 1
    model.estimator.calls.clear()
    with pytest.raises(ValueError, match='feature width.*1.*got 3'):
        model.predict(np.array([1., 2., 3.]))
    assert model.estimator.calls == []


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
@pytest.mark.parametrize('state', ['fresh', 'missing_mean', 'none_mean'])
def test_missing_fitted_metadata_is_an_unfitted_error(key, state):
    model = PhysicalRegressor(key=key)
    if state != 'fresh':
        model.estimator = _EstimatorSpy(single_output=key == 'mlp')
    if state == 'none_mean':
        model.x_mean = None
    with pytest.raises(RuntimeError, match='train the physical regressor'):
        model.predict(np.ones(14))
    if hasattr(model, 'estimator'):
        assert model.estimator.calls == []
