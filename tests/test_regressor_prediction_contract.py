"""Three-target adapter contract using inert plugins, never sklearn training."""
import numpy as np
import pytest
from fibernet.ml import physical_learning as learning


BAD_OUTPUTS = ('column', 'wrong_rows', 'vector', 'nan', 'inf', 'neg_inf',
               'complex', 'masked', 'unmasked_array', 'strings', 'object')


class InertEstimator:
    def __init__(self, mode='array', bad_call=None):
        self.mode, self.bad_call, self.calls = mode, bad_call, 0

    def fit(self, X, Y):
        return self

    def predict(self, X):
        self.calls += 1
        out = np.ones((len(X), 3))
        mode = self.mode if self.bad_call is None or self.calls == self.bad_call else 'array'
        if mode == 'column':
            return out[:, :1]
        if mode == 'wrong_rows':
            return np.zeros((len(X) + 1, 3))
        if mode == 'vector':
            return np.zeros(3)
        if mode in ('nan', 'inf', 'neg_inf'):
            out[0, 0] = {'nan': np.nan, 'inf': np.inf, 'neg_inf': -np.inf}[mode]
        if mode == 'complex':
            return out.astype(complex) + 1j
        if mode in ('masked', 'unmasked_array'):
            return np.ma.array(out, mask=mode == 'masked')
        if mode == 'strings':
            return out.astype(str)
        if mode == 'object':
            return out.astype(object)
        return out.tolist() if mode == 'list' else out


@pytest.fixture
def plugin(monkeypatch):
    key = 'test_prediction_contract'
    # register_model is public; register into fixture-owned copies so teardown
    # restores both original registries, even if the test fails.
    monkeypatch.setattr(learning, 'MODEL_SPECS', dict(learning.MODEL_SPECS))
    monkeypatch.setattr(learning, 'MODEL_PLUGINS', dict(learning.MODEL_PLUGINS))
    config = {'mode': 'array', 'bad_call': None}
    learning.register_model(key, ('test', 'test'), {},
                            lambda params, seed: InertEstimator(**config))
    return key, config


@pytest.fixture
def data():
    return np.arange(20., dtype=float).reshape(10, 2), np.arange(30.).reshape(10, 3)


@pytest.mark.parametrize('mode', BAD_OUTPUTS)
@pytest.mark.parametrize('bad_call', [1, 2], ids=['training_loss', 'validation_loss'])
@pytest.mark.parametrize('refit', [False, True], ids=['initial_fit', 'refit'])
def test_invalid_fit_output_does_not_publish(plugin, data, mode, bad_call, refit, monkeypatch):
    key, config = plugin
    X, Y = data
    model = learning.Regressor(key)
    if refit:
        model.train(X, Y)
    before = dict(model.__dict__)
    config.update(mode=mode, bad_call=bad_call)
    attempted = InertEstimator(**config)
    monkeypatch.setitem(learning.MODEL_PLUGINS, key, lambda params, seed: attempted)
    with pytest.raises(ValueError, match='estimator predictions'):
        model.train(X, Y)
    # Invalid train output must fail before any validation prediction call.
    assert attempted.calls == bad_call
    assert model.__dict__.keys() == before.keys()
    assert all(model.__dict__[name] is value for name, value in before.items())
    if not refit:
        with pytest.raises(RuntimeError, match='before prediction'):
            model.predict(X)
    else:
        assert model.predict(X).shape == (len(X), 3)


@pytest.mark.parametrize('mode', BAD_OUTPUTS)
@pytest.mark.parametrize('single', [False, True])
def test_invalid_prediction_output_rejected(plugin, data, mode, single):
    key, _ = plugin
    X, Y = data
    model = learning.Regressor(key)
    model.train(X, Y)
    model.estimator.mode = mode
    with pytest.raises(ValueError, match='estimator predictions'):
        model.predict(X[0] if single else X)


@pytest.mark.parametrize('mode', ['array', 'list'])
def test_valid_output_preserves_shapes_and_inverse_normalization(plugin, data, mode):
    key, config = plugin
    config['mode'] = mode
    X, Y = data
    model = learning.Regressor(key)
    history = model.train(X, Y)
    assert len(history['train']) == len(history['val']) == 1
    batch = model.predict(X)
    single = model.predict(X[0])
    assert batch.shape == (len(X), 3)
    assert single.shape == (3,)
    np.testing.assert_allclose(batch, np.tile(model.y_mean + model.y_std, (len(X), 1)))
    np.testing.assert_allclose(single, model.y_mean + model.y_std)


def test_reused_prediction_buffer_preserves_immediate_training_loss(plugin, data, monkeypatch):
    key, _ = plugin
    X, Y = data

    class ReusedBufferEstimator(InertEstimator):
        def __init__(self):
            super().__init__()
            self.buffer = np.zeros((len(X), 3))

        def predict(self, rows):
            self.calls += 1
            self.buffer.fill(float(self.calls))
            return self.buffer[:len(rows)]

    monkeypatch.setitem(learning.MODEL_PLUGINS, key,
                        lambda params, seed: ReusedBufferEstimator())
    model = learning.Regressor(key)
    history = model.train(X, Y)
    scaled = (Y - model.y_mean) / model.y_std
    expected_train = np.mean((1. - scaled[model.train_indices]) ** 2)
    expected_val = np.mean((2. - scaled[model.val_indices]) ** 2)
    assert history['train'] == pytest.approx([expected_train])
    assert history['val'] == pytest.approx([expected_val])
