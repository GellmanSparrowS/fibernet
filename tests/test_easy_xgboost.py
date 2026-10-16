"""Optional CPU scalar interface checks; fixtures are explicitly synthetic."""
import builtins
import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from fibernet import easy
from fibernet.ml.dataset import DatasetSample, FiberNetDataset


def dataset():
    samples, groups = [], []
    for g in range(8):
        for j in range(3):
            x, z = (g*3+j)/10, (g % 3)-j/4
            samples.append(DatasetSample(f'g{g}-r{j}', 'synthetic', (1, 1), {},
                           {'x': x, 'z': z}, {'synthetic_y': x*x+.5*z},
                           {'origin': 'synthetic_fixture'}))
            groups.append(f'g{g}')
    return FiberNetDataset(samples, feature_names=['x', 'z'],
                           target_names=['synthetic_y']), groups


def train(ds, groups, path=None):
    return easy.train_model(ds, target='synthetic_y', features=['z', 'x'],
                            model_type='xgb', split='group', groups=groups,
                            test_size=.25, save_path=path, metadata_keys=['origin'])


def test_missing_dependency_no_fallback_or_replacement(monkeypatch, tmp_path):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'xgboost':
            raise ImportError('blocked optional backend')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    p = tmp_path/'model.pkl'
    p.write_bytes(b'original')
    ds, groups = dataset()
    with pytest.raises(ImportError, match='optional XGBoost'):
        train(ds, groups, p)
    assert p.read_bytes() == b'original'


@pytest.mark.parametrize('raw', [np.zeros((6, 1)), np.full(6, np.nan),
                                 np.full(6, np.inf), np.ones(6, dtype=complex),
                                 np.ma.array(np.ones(6), mask=False),
                                 np.ma.array(np.ones(6), mask=True)])
def test_bad_predictions_do_not_replace_artifact(monkeypatch, tmp_path, raw):
    class BadModel:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return raw
    monkeypatch.setattr(easy, '_new_xgb_regressor', BadModel)
    ds, groups = dataset()
    p = tmp_path/'model.pkl'
    p.write_bytes(b'original')
    with pytest.raises(ValueError):
        train(ds, groups, p)
    assert p.read_bytes() == b'original'


@pytest.mark.parametrize('kind', ['lr', 'rf', 'gb'])
def test_legacy_does_not_import_xgb(monkeypatch, kind):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'xgboost':
            raise AssertionError('legacy path imported optional XGBoost')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    ds, groups = dataset()
    result = easy.train_model(ds, target='synthetic_y', model_type=kind,
                              split='group', groups=groups, test_size=.25)
    assert np.isfinite(result['rmse'])


def _real_smoke(directory):
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    directory = Path(directory)
    ds, groups = dataset()
    before = copy.deepcopy(ds.samples)
    calls = []
    real_fit = XGBRegressor.fit
    def spy(self, X, y, **kwargs):
        calls.append((X.copy(), y.copy(), dict(kwargs)))
        return real_fit(self, X, y, **kwargs)
    XGBRegressor.fit = spy
    p = directory/'model.pkl'
    metrics = train(ds, groups, p)
    with p.open('rb') as f:
        artifact = pickle.load(f)  # Trusted artifact produced immediately above.
    assert set(artifact) == {'model', 'features', 'metrics', 'artifact_version', 'target'}
    assert artifact['features'] == ['z', 'x']
    split = metrics['training_data']['split']
    tr, te = split['train_rows'], split['test_rows']
    assert not ({groups[i] for i in tr} & {groups[i] for i in te})
    X = np.array([[s.features['z'], s.features['x']] for s in ds.samples])
    y = np.array([s.labels['synthetic_y'] for s in ds.samples])
    assert len(calls) == 1 and calls[0][2] == {}
    np.testing.assert_array_equal(calls[0][0], X[tr])
    np.testing.assert_array_equal(calls[0][1], y[tr])
    assert ds.samples == before
    direct = artifact['model'].predict(X[te])
    named = easy.predict_named(artifact, [dict(x=X[i, 1], z=X[i, 0]) for i in te])
    np.testing.assert_array_equal(named['predictions'], direct)
    np.testing.assert_allclose(metrics['r2'], r2_score(y[te], direct), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(metrics['rmse'], np.sqrt(mean_squared_error(y[te], direct)),
                               rtol=1e-12, atol=1e-12)
    assert all(r['metadata']['origin'] == 'synthetic_fixture'
               for r in metrics['training_data']['rows'])
    for data in [{'z': 1.}, {'x': 1., 'z': 1., 'extra': 1.}]:
        try:
            easy.predict_named(artifact, data)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid named input accepted')
    for i in te:
        ds.samples[i].labels['synthetic_y'] += 1000
    p2 = directory/'perturbed.pkl'
    metrics2 = train(ds, groups, p2)
    with p2.open('rb') as f:
        perturbed = pickle.load(f)
    np.testing.assert_array_equal(artifact['model'].predict(X), perturbed['model'].predict(X))
    assert metrics2['rmse'] != metrics['rmse']
    independent = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                               device='cpu', n_estimators=32, max_depth=3,
                               learning_rate=.1, n_jobs=1, random_state=42, verbosity=0)
    independent.fit(X[tr], y[tr])
    np.testing.assert_array_equal(independent.predict(X), artifact['model'].predict(X))
    # CSV path uses the existing row split; exercise the same optional backend.
    import pandas as pd
    csv = directory/'input.csv'
    pd.DataFrame({'x': X[:, 1], 'z': X[:, 0], 'synthetic_y': y}).to_csv(csv,index=False)
    easy.train_model(str(csv), target='synthetic_y', features=['z','x'],
                      model_type='xgb', test_size=.25, save_path=directory/'csv.pkl')
    return {'status': 'passed', 'train_rows': tr, 'test_rows': te,
            'fit_calls': len(calls), 'fixture': 'synthetic_only',
            'max_rss_kib': __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss}


def test_real_cpu_smoke_bounded(tmp_path):
    if importlib.util.find_spec('xgboost') is None:
        pytest.skip('optional XGBoost unavailable; real fit not validated')
    if sys.platform != 'linux':
        pytest.skip('bounded pilot resource gate currently Linux only')
    script = '''
import resource, runpy, sys, json
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
scope = runpy.run_path(sys.argv[1])
report = scope['_real_smoke'](sys.argv[2])
print(json.dumps(report))
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(tmp_path/'mpl'))
    result = subprocess.run([sys.executable, '-c', script, str(Path(__file__).resolve()), str(tmp_path)],
                             env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
