"""Missing observations must not silently become physical training labels."""
from collections import UserList, deque

import numpy as np
import pytest

from fibernet.ml import PhysicalRegressor


def _data():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(12, 4))
    y = np.column_stack((2*x[:, 0], x[:, 1]-x[:, 2], x[:, 3]))
    return x, y


def _masked(values, layout):
    result = np.ma.array(values, mask=False)
    if layout != 'all_observed':
        result.mask.flat[0] = True
    if layout in ('nested', 'deque', 'userlist'):
        rows = [result[0]] + [row.tolist() for row in values[1:]]
        return deque(rows) if layout == 'deque' else UserList(rows) if layout == 'userlist' else rows
    if layout == 'object':
        rows = np.empty(len(values), dtype=object)
        rows[:] = [result[i] for i in range(len(values))]
        return rows
    return result


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
@pytest.mark.parametrize('field', ['features', 'targets'])
@pytest.mark.parametrize('layout', ['masked', 'all_observed', 'nested', 'object', 'deque', 'userlist'])
def test_training_rejects_mask_before_backend_or_state_publication(key, field, layout):
    x, y = _data()
    model = PhysicalRegressor(key=key)
    if field == 'features':
        x = _masked(x, layout)
    else:
        y = _masked(y, layout)
    with pytest.raises(ValueError, match='must not be masked'):
        model.train(x, y, epochs=2, validation_indices=[10, 11])
    assert not hasattr(model, 'estimator')


@pytest.mark.parametrize('layout', ['masked', 'all_observed', 'nested', 'object', 'deque', 'userlist'])
def test_real_fitted_model_survives_rejected_refit_and_prediction(layout):
    x, y = _data()
    model = PhysicalRegressor(key='mlp', seed=23)
    model.train(x.tolist(), y.tolist(), epochs=3, validation_indices=[10, 11])
    estimator = model.estimator
    means = model.y_mean.copy()
    before = model.predict(x).copy()
    with pytest.raises(ValueError, match='training targets must not be masked'):
        model.train(x, _masked(y, layout), epochs=3, validation_indices=[10, 11])
    with pytest.raises(ValueError, match='prediction features must not be masked'):
        model.predict(_masked(x, layout))
    assert model.estimator is estimator
    np.testing.assert_array_equal(model.y_mean, means)
    np.testing.assert_array_equal(model.predict(x), before)
    assert np.isfinite(before).all()


@pytest.mark.parametrize('container', [deque, UserList])
def test_complete_sequence_buffers_keep_existing_training_behavior(container):
    x, y = _data()
    array_model = PhysicalRegressor(key='mlp', seed=23)
    buffer_model = PhysicalRegressor(key='mlp', seed=23)
    array_model.train(x, y, epochs=3, validation_indices=[10, 11])
    buffer_model.train(container(x.tolist()), container(y.tolist()), epochs=3,
                       validation_indices=[10, 11])
    np.testing.assert_array_equal(buffer_model.predict(container(x.tolist())),
                                  array_model.predict(x))


def test_select_complete_rows_before_explicit_array_conversion():
    x, y = _data()
    x, y = np.ma.array(x, mask=False), np.ma.array(y, mask=False)
    y.mask[0, 0] = True
    complete = ~(np.ma.getmaskarray(x).any(axis=1) |
                 np.ma.getmaskarray(y).any(axis=1))
    observed_x = np.asarray(x[complete], dtype=float)
    observed_y = np.asarray(y[complete], dtype=float)
    model = PhysicalRegressor(key='mlp', seed=23)
    model.train(observed_x, observed_y, epochs=3, validation_indices=[9, 10])
    np.testing.assert_array_equal(model.y_mean, observed_y[:9].mean(axis=0))
    assert np.isfinite(model.predict(observed_x)).all()
