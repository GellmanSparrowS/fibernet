"""Transactional wrapper refits using inert fixtures only; no model training.

The exact wrapper source is loaded directly, without package initialization.
All constructor/fit/partial_fit/train/predict calls below are arithmetic spies;
no real MLP, sklearn, optimizer, native provider, dataset, or solver is imported.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


_SOURCE = Path(__file__).resolve().parents[1] / 'fibernet/ml/physical_learning.py'
_SPEC = importlib.util.spec_from_file_location('_physical_refit_publication', _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Regressor = _MODULE.Regressor
_METADATA = ('val_indices', 'train_indices', 'x_mean', 'x_std', 'y_mean', 'y_std')
_FIT_FIELDS = ('estimator',) + _METADATA
_KEYS = ('mlp', 'ridge', 'knn', 'deep_mlp', 'random_forest', 'extra_trees')


class _InjectedFailure(RuntimeError):
    pass


def _explode(message):
    raise _InjectedFailure(message)


def _data(width):
    X = np.arange(12 * width, dtype=float).reshape(12, width) / 7.
    X[:, -1] = 3.  # Exercise the unchanged zero-standard-deviation rule.
    Y = np.column_stack((np.arange(12.), np.arange(12.) ** 2, np.full(12, 7.)))
    X.setflags(write=False)
    Y.setflags(write=False)
    return X, Y


def _expected_metadata(X, Y, seed, val_frac=.2):
    ids = np.random.default_rng(seed).permutation(len(X))
    nv = max(2, min(len(X) - 2, int(len(X) * val_frac)))
    vi, ti = ids[:nv], ids[nv:]
    xm, xs = X[ti].mean(0), X[ti].std(0)
    ym, ys = Y[ti].mean(0), Y[ti].std(0)
    xs[xs < 1e-12] = 1.
    ys[ys < 1e-12] = 1.
    return dict(val_indices=vi, train_indices=ti,
                x_mean=xm, x_std=xs, y_mean=ym, y_std=ys)


class _PublishedEstimator:
    """A completed preconfigured state: prediction is pure, fitting forbidden."""
    def __init__(self, single_output):
        self.single_output = single_output
        self.marker = object()

    def predict(self, values):
        rows = np.atleast_2d(values)
        out = np.column_stack((rows.sum(1), rows[:, 0], rows[:, -1]))
        return out[0] if self.single_output and np.ndim(values) == 1 else out

    def _forbidden(self, *args, **kwargs):
        raise AssertionError('published estimator was mutated by a refit')

    fit = partial_fit = train = set_params = _forbidden


class _PublicationDict(dict):
    """Record publication without changing the wrapper's update behavior."""
    def __init__(self, values, events):
        super().__init__(values)
        self.commits = []
        self.events = events

    def update(self, mapping):
        self.commits.append((dict(self), dict(mapping)))
        self.events.append('publish')
        return super().update(mapping)


def _wrapper(key, fitted, events=None):
    params = {'trees': 25} if key in ('random_forest', 'extra_trees') else None
    model = Regressor(key=key, params=params, seed=7)
    model.unrelated_marker = object()
    if fitted:
        model.estimator = _PublishedEstimator(single_output=key == 'mlp')
        model.val_indices = np.array([2, 4])
        model.train_indices = np.array([0, 1, 3, 5])
        model.x_mean = np.arange(4, dtype=float) + 100.
        model.x_std = np.arange(4, dtype=float) + 1.
        model.y_mean = np.array([10., 20., 30.])
        model.y_std = np.array([2., 3., 4.])
        for name in _METADATA:
            getattr(model, name).setflags(write=False)
    model.__dict__ = _PublicationDict(model.__dict__, events if events is not None else [])
    return model


def _snapshot(model):
    refs = dict(model.__dict__)
    arrays = {k: (v.copy(), v.flags.writeable) for k, v in refs.items()
              if isinstance(v, np.ndarray)}
    estimator_state = dict(model.estimator.__dict__) if 'estimator' in refs else None
    prediction = model.predict(np.arange(4.)) if 'estimator' in refs else None
    return refs, arrays, estimator_state, prediction


def _assert_previous_state(model, before):
    refs, arrays, estimator_state, prediction = before
    assert set(model.__dict__) == set(refs)
    for name, value in refs.items():
        assert getattr(model, name) is value, name
    for name, (value, writeable) in arrays.items():
        np.testing.assert_array_equal(getattr(model, name), value)
        assert getattr(model, name).flags.writeable is writeable
    assert model.__dict__.commits == []
    if 'estimator' in refs:
        assert model.estimator.__dict__ == estimator_state
        np.testing.assert_array_equal(model.predict(np.arange(4.)), prediction)
    else:
        assert all(name not in model.__dict__ for name in _FIT_FIELDS)
        with pytest.raises(RuntimeError, match='train the physical regressor'):
            model.predict(np.ones(4))


def _assert_old_objects_untouched(before):
    refs, arrays, estimator_state, _ = before
    for name, (value, writeable) in arrays.items():
        np.testing.assert_array_equal(refs[name], value)
        assert refs[name].flags.writeable is writeable
    if 'estimator' in refs:
        assert refs['estimator'].__dict__ == estimator_state


def _assert_publication(model, before, expected, estimator, events):
    refs = before[0]
    assert model.estimator is estimator
    assert len(model.__dict__.commits) == 1
    previous, published = model.__dict__.commits[0]
    assert set(previous) == set(refs)
    assert all(previous[k] is v for k, v in refs.items())
    assert set(published) == set(_FIT_FIELDS)
    assert all(model.__dict__[k] is v for k, v in published.items())
    for name in _METADATA:
        np.testing.assert_array_equal(getattr(model, name), expected[name])
        if name in refs:
            assert getattr(model, name) is not refs[name]
    for name in set(refs) - set(_FIT_FIELDS):
        assert getattr(model, name) is refs[name]
    assert set(model.__dict__) == set(refs) | set(_FIT_FIELDS)
    assert 'history' not in model.__dict__
    assert events[-1] == 'publish'
    _assert_old_objects_untouched(before)


class _BadLen:
    def __len__(self):
        _explode('metadata length')


class _BadHistory(dict):
    def update(self, **kwargs):
        _explode('history update')


class _InertMLP:
    """Publish synthetic progress; no weights, gradients, or optimizer exist."""
    def __init__(self, events, fail=None, fail_step=1, seed=7):
        self.events, self.fail, self.fail_step, self.seed = events, fail, fail_step, seed
        self.calls = []
        self.progress = []
        self.history = None

    def __getattribute__(self, name):
        if name in _METADATA:
            fail = object.__getattribute__(self, 'fail')
            if fail == 'metadata:' + name:
                _explode('metadata ' + name)
        return object.__getattribute__(self, name)

    def train(self, X, Y, **kwargs):
        self.calls.append((X, Y, kwargs))
        self.events.append('inert_mlp_train')
        self.__dict__.update(_expected_metadata(X, Y, self.seed, kwargs['val_frac']))
        tr, va = [], []
        for ep in range(1, min(int(kwargs['epochs']), 3) + 1):
            if kwargs['stop_cb'] is not None and kwargs['stop_cb']():
                break
            if self.fail == 'fit' and ep == self.fail_step:
                _explode('inert MLP train')
            tr.append(ep / 10.)
            va.append(ep / 5.)
            self.progress.append((ep, tr[-1], va[-1]))
            if kwargs['epoch_cb'] is not None:
                kwargs['epoch_cb'](ep, tr[-1], va[-1])
        history = dict(train=tr, val=va, epochs_run=len(tr),
                       best_epoch=1 if tr else 0, early_stopped=False)
        if self.fail == 'history_train':
            del history['train']
        if self.fail == 'history_update':
            history = _BadHistory(history)
        if self.fail == 'metadata_len':
            self.val_indices = _BadLen()
        self.history = history
        return history

    def predict(self, values):
        rows = np.atleast_2d(values)
        out = np.column_stack((rows.sum(1), rows[:, 0], rows[:, -1]))
        return out[0] if np.ndim(values) == 1 else out


class _InertEstimator:
    """Records normalized arrays and synthetic losses; has no fitting algorithm."""
    def __init__(self, events, fail=None, fail_step=1, biases=None):
        self.events, self.fail, self.fail_step = events, fail, fail_step
        self.biases = biases if biases is not None else (0., 0., 0.)
        self.fit_count = 0
        self.fit_calls, self.predict_calls, self.checkpoints, self.param_calls = [], [], [], []
        self.n_estimators = 0
        self.checkpoint_of = None

    def set_params(self, **kwargs):
        self.param_calls.append(kwargs)
        self.events.append('set_params')
        if self.fail == 'set_params' and self.fit_count + 1 == self.fail_step:
            _explode('set params')
        self.n_estimators = kwargs['n_estimators']
        return self

    def _fit(self, kind, X, Y):
        self.fit_count += 1
        self.fit_calls.append((kind, X.copy(), Y.copy()))
        self.events.append(kind)
        if self.fail == 'fit' and self.fit_count == self.fail_step:
            _explode(kind)
        return self

    def fit(self, X, Y):
        return self._fit('fit', X, Y)

    def partial_fit(self, X, Y):
        return self._fit('partial_fit', X, Y)

    def predict(self, X):
        which = 'predict_train' if len(self.predict_calls) % 2 == 0 else 'predict_val'
        self.predict_calls.append(X.copy())
        self.events.append(which)
        if self.fail == which and self.fit_count == self.fail_step:
            _explode(which)
        return np.full((len(X), 3), self.biases[min(self.fit_count - 1, len(self.biases) - 1)])

    def __deepcopy__(self, memo):
        self.events.append('deepcopy')
        if self.fail == 'deepcopy' and self.fit_count == self.fail_step:
            _explode('deepcopy')
        saved = _InertEstimator(self.events, self.fail, self.fail_step, self.biases)
        saved.fit_count, saved.n_estimators = self.fit_count, self.n_estimators
        saved.checkpoint_of = self
        self.checkpoints.append(saved)
        return saved

    @property
    def estimators_(self):
        self.events.append('forest_history')
        if self.fail == 'forest_history':
            _explode('forest history property')
        if self.fail == 'forest_history_len':
            return _BadLen()
        return [None] * self.n_estimators


@pytest.fixture(autouse=True)
def _no_backend_imports(monkeypatch):
    def forbidden(name):
        raise AssertionError('real backend import forbidden: ' + name)
    monkeypatch.setattr(_MODULE, 'importlib', SimpleNamespace(import_module=forbidden))


def _install_factory(monkeypatch, key, estimator, factory_calls, fail=False):
    if key == 'mlp':
        def factory(**kwargs):
            factory_calls.append(kwargs)
            if fail:
                _explode('factory')
            return estimator
        root, ml, surrogate = (ModuleType(name) for name in
                               ('fibernet', 'fibernet.ml', 'fibernet.ml.physical_surrogate'))
        root.__path__, ml.__path__ = [], []
        surrogate.MLP = factory
        root.ml, ml.physical_surrogate = ml, surrogate
        for module in (root, ml, surrogate):
            monkeypatch.setitem(sys.modules, module.__name__, module)
    else:
        def factory(params, seed):
            factory_calls.append((params, seed))
            if fail:
                _explode('factory')
            return estimator
        monkeypatch.setitem(_MODULE.MODEL_PLUGINS, key, factory)


def _new_estimator(key, events, fail=None, fail_step=1, biases=None):
    if key == 'mlp':
        return _InertMLP(events, fail, fail_step)
    return _InertEstimator(events, fail, fail_step, biases)


def _assert_factory(model, calls):
    assert len(calls) == 1
    if model.key == 'mlp':
        assert calls[0] == dict(hidden=model.params['hidden'], seed=model.seed)
    else:
        assert calls[0][0] is model.params
        assert calls[0][1] == model.seed


@pytest.mark.parametrize('key', _KEYS)
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('width', [4, 7])
@pytest.mark.parametrize('failure', ['factory', 'fit', 'epoch_callback', 'stop_callback', 'zero_stop'])
def test_failures_leave_fresh_or_completed_state_untouched(monkeypatch, key, fitted, width, failure):
    events, calls = [], []
    model = _wrapper(key, fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator(key, events, 'fit' if failure == 'fit' else None)
    _install_factory(monkeypatch, key, candidate, calls, fail=failure == 'factory')
    X, Y = _data(width)
    originals = X.copy(), Y.copy()
    observed = []

    def epoch_cb(*args):
        _assert_previous_state(model, before)
        observed.append(args)
        if failure == 'epoch_callback':
            _explode('epoch callback')

    def stop_cb():
        _assert_previous_state(model, before)
        if failure == 'stop_callback':
            _explode('stop callback')
        return failure == 'zero_stop'

    error = InterruptedError if failure == 'zero_stop' else _InjectedFailure
    with pytest.raises(error):
        model.train(X, Y, epochs=3, epoch_cb=epoch_cb, stop_cb=stop_cb)
    _assert_previous_state(model, before)
    _assert_factory(model, calls)
    np.testing.assert_array_equal(X, originals[0])
    np.testing.assert_array_equal(Y, originals[1])
    assert not X.flags.writeable and not Y.flags.writeable
    assert (len(observed) == 1) if failure == 'epoch_callback' else not observed
    assert 'publish' not in events


@pytest.mark.parametrize('key', ['ridge', 'deep_mlp', 'random_forest', 'extra_trees'])
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('width', [4, 7])
@pytest.mark.parametrize('failure', ['predict_train', 'predict_val', 'deepcopy'])
def test_estimator_prediction_and_copy_failures_never_publish(monkeypatch, key, fitted, width, failure):
    events, calls = [], []
    model = _wrapper(key, fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator(key, events, failure)
    _install_factory(monkeypatch, key, candidate, calls)
    with pytest.raises(_InjectedFailure):
        model.train(*_data(width), epochs=3)
    _assert_previous_state(model, before)
    assert candidate.fit_count == 1
    assert 'publish' not in events


@pytest.mark.parametrize('key', ['mlp', 'deep_mlp', 'random_forest', 'extra_trees'])
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('width', [4, 7])
@pytest.mark.parametrize('failure', ['fit', 'epoch_callback', 'stop_callback'])
def test_ordinary_error_after_progress_discards_staged_fit(monkeypatch, key, fitted, width, failure):
    events, calls = [], []
    model = _wrapper(key, fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator(key, events, 'fit' if failure == 'fit' else None, fail_step=2)
    _install_factory(monkeypatch, key, candidate, calls)
    progress, stop_calls = [], []

    def epoch_cb(*args):
        _assert_previous_state(model, before)
        progress.append(args)
        if failure == 'epoch_callback' and len(progress) == 2:
            _explode('late epoch callback')

    def stop_cb():
        _assert_previous_state(model, before)
        stop_calls.append(None)
        if failure == 'stop_callback' and len(stop_calls) == 2:
            _explode('late stop callback')
        return False

    with pytest.raises(_InjectedFailure):
        model.train(*_data(width), epochs=3, epoch_cb=epoch_cb, stop_cb=stop_cb)
    _assert_previous_state(model, before)
    assert len(progress) == (2 if failure == 'epoch_callback' else 1)
    assert 'publish' not in events


@pytest.mark.parametrize('key', ['deep_mlp', 'random_forest', 'extra_trees'])
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('width', [4, 7])
@pytest.mark.parametrize('failure', ['predict_train', 'predict_val', 'deepcopy'])
def test_late_predict_or_copy_failure_discards_earlier_checkpoint(monkeypatch, key, fitted, width, failure):
    events, calls = [], []
    model = _wrapper(key, fitted, events)
    before = _snapshot(model)
    # Improving synthetic losses force a second neural checkpoint/deepcopy.
    candidate = _new_estimator(key, events, failure, fail_step=2, biases=(100., 0., 0.))
    _install_factory(monkeypatch, key, candidate, calls)
    with pytest.raises(_InjectedFailure):
        model.train(*_data(width), epochs=3)
    _assert_previous_state(model, before)
    assert candidate.fit_count == 2
    assert len(candidate.checkpoints) == 1
    assert 'publish' not in events


@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('width', [4, 7])
@pytest.mark.parametrize('failure', ['metadata:' + name for name in _METADATA] +
                         ['metadata_len', 'history_train', 'history_update'])
def test_mlp_metadata_and_history_fail_before_publication(monkeypatch, fitted, width, failure):
    events, calls = [], []
    model = _wrapper('mlp', fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator('mlp', events, failure)
    _install_factory(monkeypatch, 'mlp', candidate, calls)
    error = KeyError if failure == 'history_train' else _InjectedFailure
    with pytest.raises(error):
        model.train(*_data(width), epochs=2)
    _assert_previous_state(model, before)
    assert candidate.progress == [(1, .1, .2), (2, .2, .4)]
    assert 'publish' not in events


@pytest.mark.parametrize('key', ['random_forest', 'extra_trees'])
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('width', [4, 7])
@pytest.mark.parametrize('failure', ['set_params', 'forest_history', 'forest_history_len'])
def test_forest_parameter_and_history_failures_are_unpublished(monkeypatch, key, fitted, width, failure):
    events, calls = [], []
    model = _wrapper(key, fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator(key, events, failure)
    _install_factory(monkeypatch, key, candidate, calls)
    with pytest.raises(_InjectedFailure):
        model.train(*_data(width), epochs=3)
    _assert_previous_state(model, before)
    assert candidate.fit_count == (0 if failure == 'set_params' else 3)
    assert len(candidate.checkpoints) == (0 if failure == 'set_params' else 3)
    assert 'publish' not in events


@pytest.mark.parametrize('key', _KEYS)
@pytest.mark.parametrize('width', [4, 7])
def test_factory_cannot_return_exact_published_estimator(monkeypatch, key, width):
    events, calls = [], []
    model = _wrapper(key, True, events)
    before = _snapshot(model)
    _install_factory(monkeypatch, key, model.estimator, calls)
    callbacks = []
    with pytest.raises(ValueError, match='factory must return a new estimator'):
        model.train(*_data(width), epoch_cb=lambda *args: callbacks.append(args),
                    stop_cb=lambda: callbacks.append('stop'))
    _assert_previous_state(model, before)
    _assert_factory(model, calls)
    assert callbacks == []
    assert events == []


@pytest.mark.parametrize('key', _KEYS)
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('width', [4, 7])
def test_success_publishes_once_with_complete_metadata_and_history(monkeypatch, key, fitted, width):
    events, calls = [], []
    model = _wrapper(key, fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator(key, events)
    _install_factory(monkeypatch, key, candidate, calls)
    X, Y = _data(width)
    original = X.copy(), Y.copy()
    expected = _expected_metadata(X, Y, model.seed)
    progress = []

    def epoch_cb(*args):
        _assert_previous_state(model, before)
        progress.append(args)
        events.append('epoch_callback')

    def stop_cb():
        _assert_previous_state(model, before)
        events.append('stop_callback')
        return False

    history = model.train(X, Y, epochs=3, lr=.02, batch=3, patience=4,
                          min_delta=.0002, epoch_cb=epoch_cb, stop_cb=stop_cb)
    _assert_factory(model, calls)
    estimator = candidate if key == 'mlp' else candidate.checkpoints[-1]
    _assert_publication(model, before, expected, estimator, events)
    assert history['validation_count'] == len(expected['val_indices'])
    assert history['schema'] is None
    assert history['feature_count'] == width
    assert history['epochs_run'] == len(history['train']) == len(history['val'])
    assert [(ep, tr, va) for (ep, _, _), tr, va in zip(progress, history['train'], history['val'])] == progress
    forest = key in ('random_forest', 'extra_trees')
    expected_steps = [10, 20, 25] if forest else [1, 2, 3] if key in ('mlp', 'deep_mlp') else [1]
    assert [entry[0] for entry in progress] == expected_steps
    assert history['progress_kind'] == ('trees' if forest else 'epoch' if key in ('mlp', 'deep_mlp') else 'fit')
    assert history['best_epoch'] == (25 if forest else 1)
    assert not history['early_stopped']
    if key == 'mlp':
        assert history is candidate.history
        assert 'trees_fitted' not in history
        for name in _METADATA:
            assert getattr(model, name) is getattr(candidate, name)
        call_X, call_Y, controls = candidate.calls[0]
        assert call_X is X and call_Y is Y
        assert controls == dict(lr=.02, epochs=3, batch=3, val_frac=.2,
                                patience=4, min_delta=.0002, epoch_cb=epoch_cb, stop_cb=stop_cb)
        assert progress == candidate.progress
    else:
        assert history['trees_fitted'] == (25 if forest else None)
        assert estimator is not candidate
        assert estimator.checkpoint_of is candidate
        ti, vi = expected['train_indices'], expected['val_indices']
        x = (X - expected['x_mean']) / expected['x_std']
        y = (Y - expected['y_mean']) / expected['y_std']
        for kind, fit_X, fit_Y in candidate.fit_calls:
            assert kind == ('partial_fit' if key == 'deep_mlp' else 'fit')
            np.testing.assert_allclose(fit_X, x[ti])
            np.testing.assert_allclose(fit_Y, y[ti])
        np.testing.assert_allclose(candidate.predict_calls[0], x[ti])
        np.testing.assert_allclose(candidate.predict_calls[1], x[vi])
        np.testing.assert_allclose(history['train'], [float(np.mean(y[ti] ** 2))] * len(expected_steps))
        np.testing.assert_allclose(history['val'], [float(np.mean(y[vi] ** 2))] * len(expected_steps))
        assert candidate.param_calls == ([{'n_estimators': n} for n in expected_steps] if forest else [])
        if forest:
            assert events[-2] == 'forest_history'
    # Read the event-order assertion above before this ordinary post-publication prediction.
    assert model.predict(np.arange(float(width))).shape == (3,)
    if width != 4:
        with pytest.raises(ValueError, match='feature width'):
            model.predict(np.ones(4))
    np.testing.assert_array_equal(X, original[0])
    np.testing.assert_array_equal(Y, original[1])
    assert not X.flags.writeable and not Y.flags.writeable


@pytest.mark.parametrize('key', ['mlp', 'deep_mlp', 'random_forest', 'extra_trees'])
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('width', [4, 7])
def test_stop_after_accepted_progress_publishes_partial_checkpoint(monkeypatch, key, fitted, width):
    events, calls = [], []
    model = _wrapper(key, fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator(key, events)
    _install_factory(monkeypatch, key, candidate, calls)
    X, Y = _data(width)
    progress, stop_calls = [], []

    def stop_cb():
        _assert_previous_state(model, before)
        stop_calls.append(None)
        return len(stop_calls) == 2

    def epoch_cb(*args):
        _assert_previous_state(model, before)
        progress.append(args)

    history = model.train(X, Y, epochs=3, epoch_cb=epoch_cb, stop_cb=stop_cb)
    estimator = candidate if key == 'mlp' else candidate.checkpoints[0]
    _assert_publication(model, before, _expected_metadata(X, Y, model.seed), estimator, events)
    assert len(progress) == history['epochs_run'] == 1
    assert len(stop_calls) == 2
    forest = key in ('random_forest', 'extra_trees')
    assert history['best_epoch'] == (10 if forest else 1)
    assert progress[0] == (history['best_epoch'], history['train'][0], history['val'][0])
    assert not history['early_stopped']
    if key != 'mlp':
        assert candidate.fit_count == 1
        assert len(candidate.checkpoints) == 1
        assert history['trees_fitted'] == (10 if forest else None)


@pytest.mark.parametrize('mode', ['full', 'early_stop', 'stop_callback'])
@pytest.mark.parametrize('fitted', [False, True])
def test_neural_checkpoint_choice_and_early_stop_are_unchanged(monkeypatch, mode, fitted):
    events, calls = [], []
    model = _wrapper('deep_mlp', fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator('deep_mlp', events, biases=(0., 100., 200.))
    _install_factory(monkeypatch, 'deep_mlp', candidate, calls)
    X, Y = _data(7)
    stop_calls, progress = [], []

    def stop_cb():
        _assert_previous_state(model, before)
        stop_calls.append(None)
        return mode == 'stop_callback' and len(stop_calls) == 3

    def epoch_cb(*args):
        _assert_previous_state(model, before)
        progress.append(args)

    history = model.train(X, Y, epochs=3, patience=1 if mode == 'early_stop' else 15,
                          epoch_cb=epoch_cb, stop_cb=stop_cb)
    _assert_publication(model, before, _expected_metadata(X, Y, model.seed), candidate.checkpoints[0], events)
    assert len(candidate.checkpoints) == 1
    assert model.estimator.fit_count == history['best_epoch'] == 1
    assert history['epochs_run'] == candidate.fit_count == (3 if mode == 'full' else 2)
    assert history['early_stopped'] is (mode == 'early_stop')
    assert len(progress) == history['epochs_run']
    assert history['val'][1] > history['val'][0]


@pytest.mark.parametrize('key', ['ridge', 'knn', 'deep_mlp', 'random_forest', 'extra_trees'])
def test_builtin_constructor_arguments_are_unchanged_without_importing_backends(monkeypatch, key):
    events, imports, constructor_calls = [], [], []
    model = _wrapper(key, False, events)
    candidate = _new_estimator(key, events)
    monkeypatch.delitem(_MODULE.MODEL_PLUGINS, key, raising=False)

    def constructor(**kwargs):
        constructor_calls.append(kwargs)
        return candidate

    namespace = SimpleNamespace(MLPRegressor=constructor, Ridge=constructor,
                                KNeighborsRegressor=constructor,
                                RandomForestRegressor=constructor, ExtraTreesRegressor=constructor)

    def fake_import(name):
        imports.append(name)
        return namespace

    monkeypatch.setattr(_MODULE, 'importlib', SimpleNamespace(import_module=fake_import))
    model.train(*_data(4), lr=.02, batch=99, epochs=1)
    expected = {
        'ridge': ('sklearn.linear_model', dict(alpha=1.)),
        'knn': ('sklearn.neighbors', dict(n_neighbors=5, weights='distance', n_jobs=1)),
        'deep_mlp': ('sklearn.neural_network', dict(hidden_layer_sizes=(64, 64, 64),
                    learning_rate_init=.02, batch_size=10, max_iter=1, random_state=7)),
        'random_forest': ('sklearn.ensemble', dict(n_estimators=25, max_depth=8,
                         random_state=7, n_jobs=1, warm_start=True)),
        'extra_trees': ('sklearn.ensemble', dict(n_estimators=25, max_depth=8,
                       random_state=7, n_jobs=1, warm_start=True)),
    }[key]
    assert imports == [expected[0]]
    assert constructor_calls == [expected[1]]


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('bad_input', ['nan', 'few_rows', 'wrong_target', 'epochs', 'batch', 'val_frac', 'lr'])
def test_input_validation_failure_precedes_factory_and_publication(monkeypatch, key, fitted, bad_input):
    events, calls = [], []
    model = _wrapper(key, fitted, events)
    before = _snapshot(model)
    candidate = _new_estimator(key, events)
    _install_factory(monkeypatch, key, candidate, calls)
    X, Y = _data(7)
    kwargs = {}
    if bad_input == 'nan':
        X = X.copy()
        X[0, 0] = np.nan
    elif bad_input == 'few_rows':
        X, Y = X[:4], Y[:4]
    elif bad_input == 'wrong_target':
        Y = Y[:, :2]
    else:
        kwargs[bad_input] = 0
    with pytest.raises(ValueError):
        model.train(X, Y, **kwargs)
    _assert_previous_state(model, before)
    assert calls == []
    assert events == []
