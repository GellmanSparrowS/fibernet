"""Inert metadata/caller tests: no real model, UI, provider or solver imports.

Reuse the versioned refit suite's arithmetic spies; isolate caller definitions
with AST extraction rather than importing their application modules.
"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import test_physical_refit_publication_v2 as refit


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = refit._MODULE.FEATURE_SCHEMA


@pytest.fixture(autouse=True)
def no_backend_imports(monkeypatch):
    def forbidden(name):
        raise AssertionError('real backend import forbidden: ' + name)
    monkeypatch.setattr(refit._MODULE, 'importlib', SimpleNamespace(import_module=forbidden))


@pytest.mark.parametrize('key', refit._KEYS)
@pytest.mark.parametrize('width', [1, 4, 14])
@pytest.mark.parametrize('fitted', [False, True])
def test_default_width_never_supplies_identity(monkeypatch, key, width, fitted):
    events, calls = [], []
    model = refit._wrapper(key, fitted, events)
    before = refit._snapshot(model)
    candidate = refit._new_estimator(key, events)
    refit._install_factory(monkeypatch, key, candidate, calls)
    X, Y = refit._data(width)
    history = model.train(X, Y, epochs=3)
    assert history['schema'] is None
    assert history['feature_count'] == width
    refit._assert_factory(model, calls)
    estimator = candidate if key == 'mlp' else candidate.checkpoints[-1]
    refit._assert_publication(model, before, refit._expected_metadata(X, Y, 7),
                              estimator, events)
    assert not {'feature_schema', 'feature_count', 'schema', 'history'} & model.__dict__.keys()
    if key == 'mlp':
        assert 'feature_schema' not in candidate.calls[0][2]


@pytest.mark.parametrize('key', refit._KEYS)
@pytest.mark.parametrize('declaration,width', [
    (SCHEMA, 14), ('my-features:v1', 1), ('my-features:v1', 4),
    ('my-features:v1', 14), ('declared features / v2', 4), ('x' * 128, 4),
    ('结构特征:v1', 4),
])
def test_explicit_identity_roundtrips_without_numerical_changes(monkeypatch, key, declaration, width):
    X, Y = refit._data(width)
    outcomes = []
    for schema in (None, declaration):
        events, calls, progress = [], [], []
        model = refit._wrapper(key, True, events)
        before = refit._snapshot(model)
        candidate = refit._new_estimator(key, events)
        refit._install_factory(monkeypatch, key, candidate, calls)
        history = model.train(X, Y, epochs=3, feature_schema=schema,
                              epoch_cb=lambda *values: progress.append(values))
        assert history['schema'] == schema
        assert history['feature_count'] == width
        estimator = candidate if key == 'mlp' else candidate.checkpoints[-1]
        refit._assert_publication(model, before, refit._expected_metadata(X, Y, 7),
                                  estimator, events)
        outcomes.append((history, model.predict(X), progress, events, model))
    h0, p0, progress0, events0, m0 = outcomes[0]
    h1, p1, progress1, events1, m1 = outcomes[1]
    assert {k: v for k, v in h0.items() if k != 'schema'} == {
        k: v for k, v in h1.items() if k != 'schema'}
    np.testing.assert_array_equal(p0, p1)
    assert progress0 == progress1 and events0 == events1
    for name in refit._METADATA:
        np.testing.assert_array_equal(getattr(m0, name), getattr(m1, name))


@pytest.mark.parametrize('key', refit._KEYS)
@pytest.mark.parametrize('width', [1, 4, 7])
@pytest.mark.parametrize('fitted', [False, True])
def test_canonical_wrong_width_rejected_before_factory_or_callbacks(monkeypatch, key, width, fitted):
    events, calls, callbacks = [], [], []
    model = refit._wrapper(key, fitted, events)
    before = refit._snapshot(model)
    refit._install_factory(monkeypatch, key, refit._new_estimator(key, events), calls)
    with pytest.raises(ValueError, match='14 feature columns'):
        model.train(*refit._data(width), feature_schema=SCHEMA,
                    epoch_cb=lambda *args: callbacks.append(args),
                    stop_cb=lambda: callbacks.append('stop'))
    assert calls == callbacks == events == []
    refit._assert_previous_state(model, before)


@pytest.mark.parametrize('key', ['mlp', 'ridge'])
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('invalid', [
    False, True, 0, 1.5, np.nan, b'schema', [], {}, np.array(['schema']),
    '', ' ', ' leading', 'trailing ', '\ttab', 'line\nbreak', 'nul\0byte',
    'zero\u200bwidth', 'x' * 129,
])
def test_invalid_declaration_rejected_without_coercion_or_old_fit_changes(monkeypatch, key, fitted, invalid):
    events, calls, callbacks = [], [], []
    model = refit._wrapper(key, fitted, events)
    before = refit._snapshot(model)
    refit._install_factory(monkeypatch, key, refit._new_estimator(key, events), calls)
    with pytest.raises(ValueError, match='feature_schema'):
        model.train(*refit._data(14), feature_schema=invalid,
                    epoch_cb=lambda *args: callbacks.append(args),
                    stop_cb=lambda: callbacks.append('stop'))
    assert calls == callbacks == events == []
    refit._assert_previous_state(model, before)


def test_signature_preserves_old_positional_controls_and_makes_declaration_keyword_only():
    params = list(inspect.signature(refit.Regressor.train).parameters.values())
    assert [p.name for p in params] == [
        'self', 'X', 'Y', 'lr', 'epochs', 'batch', 'val_frac', 'patience',
        'min_delta', 'epoch_cb', 'stop_cb', 'feature_schema']
    assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params[:-1])
    assert params[-1].kind == inspect.Parameter.KEYWORD_ONLY
    assert params[-1].default is None


def _definition(relative, name, namespace, method=None):
    """Load only one named source definition, with explicitly supplied inert names."""
    source = (ROOT / relative).read_text(encoding='utf-8')
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    if method is not None:
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    if (relative, name) == ('examples/physical_learning_workflow.py', '_target_report'):
        # Execute the reviewed literal-only source helper, never a report stub.
        assert hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() == (
            'e78d335f24689cf6f7573adfeb6f06feba553daf63aa436984385f03ea4b0437'), 'review changed workflow helper before extraction'
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, relative, 'exec'), namespace)
    return namespace[node.name]


class _Signal:
    def __init__(self, *args):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


class _Thread:
    def __init__(self, parent=None):
        self.parent = parent


@pytest.mark.parametrize('declaration', [None, SCHEMA, 'plugin:v3'])
def test_generic_train_worker_forwards_only_explicit_declaration(declaration):
    calls = []

    class Model:
        def __init__(self, *args):
            pass

        def train(self, X, Y, **kwargs):
            calls.append((X, Y, kwargs))
            return {'schema': kwargs['feature_schema'], 'feature_count': X.shape[1]}

    Worker = _definition('FiberScope/studio/ml_tab.py', 'TrainWorker',
                         dict(QThread=_Thread, Signal=_Signal, Regressor=Model))
    X, Y = refit._data(14)
    kwargs = {} if declaration is None else {'feature_schema': declaration}
    worker = Worker(X, Y, 32, .003, 3, 2, .0001, **kwargs)
    worker.run()
    assert worker.failed.values == []
    assert calls[0][0] is X and calls[0][1] is Y
    assert calls[0][2]['feature_schema'] == declaration
    assert worker.done.values[0][0]['history']['schema'] == declaration
    assert inspect.signature(Worker).parameters['feature_schema'].kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize('cached', ['present', 'missing', 'malformed'])
def test_ui_fresh_generation_declares_canonical_but_unchecked_cache_resets(tmp_path, cached):
    path = tmp_path / 'cached.npz'
    X, Y = refit._data(14)
    if cached == 'present':
        np.savez(path, X=X, Y=Y)
    elif cached == 'malformed':
        np.savez(path, unrecognized=X)
    noop = lambda *args: None
    tab = SimpleNamespace(
        gen_worker=SimpleNamespace(label_mode='physics', deleteLater=noop),
        _read_specs=noop, _set_busy=noop, _refresh_r2=noop,
        progress=SimpleNamespace(setValue=noop), status=SimpleNamespace(setText=noop),
        _train_after_generation=False, mode_combo=SimpleNamespace(currentData=lambda: 'physics'),
        unit_combo=SimpleNamespace(currentData=lambda: 'hexagon'), _dataset_path=lambda: str(path))
    namespace = dict(os=os, np=np, FEATURE_SCHEMA=SCHEMA)
    generated = _definition('FiberScope/studio/ml_tab.py', 'MLTab', namespace, '_gen_done')
    generated(tab, X, Y, str(path))
    assert tab.feature_schema == SCHEMA
    loaded = _definition('FiberScope/studio/ml_tab.py', 'MLTab', namespace, '_load_existing')
    loaded(tab)
    assert tab.feature_schema is None
    if cached == 'present':
        np.testing.assert_array_equal(tab.X, X)
    else:
        assert tab.X is tab.Y is None


def test_ui_initializes_unknown_and_forwards_its_current_declaration():
    tree = ast.parse((ROOT / 'FiberScope/studio/ml_tab.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MLTab')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    assert any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is None
               and any(isinstance(t, ast.Attribute) and t.attr == 'feature_schema' for t in n.targets)
               for n in ast.walk(init))
    start = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'start_train')
    call = next(n for n in ast.walk(start) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'TrainWorker')
    declaration = next(k.value for k in call.keywords if k.arg == 'feature_schema')
    assert ast.unparse(declaration) == 'self.feature_schema'


@pytest.mark.parametrize('source', ['synthetic', 'npz'])
@pytest.mark.parametrize('returned_schema', [None, 'caller-history:v1'])
def test_workflow_summary_uses_history_not_width(tmp_path, source, returned_schema):
    calls = []

    class Model:
        def __init__(self, **kwargs):
            self.val_indices = np.array([0, 1])

        def train(self, X, Y, **kwargs):
            calls.append(kwargs)
            return dict(schema=returned_schema, feature_count=X.shape[1], best_epoch=1)

        def predict(self, X):
            return np.zeros((len(X), 3))

    Workflow = _definition('examples/physical_learning_workflow.py', 'PhysicalLearningWorkflow',
                           dict(Path=Path, np=np, os=os, json=json, PhysicalRegressor=Model,
                                FEATURE_SCHEMA=refit._MODULE.FEATURE_SCHEMA,
                                _target_report=_definition(
                                    'examples/physical_learning_workflow.py', '_target_report', {})))
    data = None
    if source == 'npz':
        data = tmp_path / 'unknown.npz'
        np.savez(data, X=refit._data(14)[0], Y=refit._data(14)[1])
    result = Workflow(tmp_path / 'out', data).run()
    assert result['schema'] == returned_schema
    assert result['feature_count'] == 14
    assert calls == [{}]  # Neither random data nor an unchecked NPZ declares identity.
    assert json.loads((tmp_path / 'out/summary.json').read_text()) == result


def test_workflow_retains_existing_fourteen_column_gate(tmp_path):
    Workflow = _definition('examples/physical_learning_workflow.py', 'PhysicalLearningWorkflow',
                           dict(Path=Path, np=np))
    data = tmp_path / 'wrong-width.npz'
    np.savez(data, X=refit._data(4)[0], Y=refit._data(4)[1])
    with pytest.raises(ValueError, match='expected finite'):
        Workflow(tmp_path, data)._data()


def test_web_known_generator_passes_canonical_schema(tmp_path):
    calls = []
    X, Y = refit._data(14)

    class Model:
        def __init__(self, *args, **kwargs):
            self.val_indices = np.array([0, 1])

        def train(self, x, y, **kwargs):
            calls.append((x, y, kwargs))

        def predict(self, x):
            return np.zeros((len(x), 3))

    figure = SimpleNamespace(add_trace=lambda *a, **k: None, update_layout=lambda **k: None)
    namespace = dict(Path=Path, json=json, np=np, FEATURE_SCHEMA=SCHEMA,
                     ACQUISITIONS={'random': None}, MODEL_SPECS={'ridge': None},
                     Regressor=Model, gen_dataset=lambda *a, **k: (X, Y),
                     TARGET_NAMES=('peak', 'stiffness', 'toughness'), COLORS={},
                     gr=SimpleNamespace(Progress=lambda: lambda *a, **k: None, Error=ValueError),
                     go=SimpleNamespace(Figure=lambda: figure, Scatter=lambda **k: k))
    train = _definition('web_demo/app.py', 'Demo', namespace, 'train')
    demo = SimpleNamespace(require=lambda state: SimpleNamespace(unit='hexagon', seed=0))
    train(demo, {'directory': str(tmp_path)}, 12, 'ridge')
    assert calls[0][0] is X and calls[0][1] is Y
    assert calls[0][2] == {'epochs': 150, 'feature_schema': SCHEMA}


def test_schema_constant_and_feature_names_remain_the_canonical_contract():
    assert SCHEMA == 'structure14-v4-geometry27'
    assert refit._MODULE.FEATURE_NAMES == (
        'length_mean', 'length_std', 'length_cv', 'orientation_entropy',
        'degree2', 'degree3', 'degree4', 'degree_other', 'gyration_ratio',
        'nodes_scaled', 'edges_scaled', 'edges_per_node', 'aspect', 'length_ratio')


def test_unchanged_cache_identity_and_npz_checkpoint_roundtrip(tmp_path):
    # Read the unchanged repository sources; hashes pin the cache/NPZ contract.
    # No separate fixture directory or application-module import is required.
    references = ROOT
    adapter = references / 'FiberScope/fslab/dataset_stream.py'
    dataset = references / 'fibernet/ml/physical_dataset.py'
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == (
        '9de802cb59bad35836864c88bff02e4d47c673ab31604be6c1c5b72ffa427863')
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == (
        '01c71485fa5b816a6f3ac934e3ec68972958cdbb4cbbe01a9689e8da2884a5c0')
    config = dict(schema=SCHEMA, topology='topnet26', unit='hexagon', amp=.2,
                  pert=.1, seed0=0, mode='physics', acquisition='random',
                  simulation={}, custom_cell=None, engine_source='inert-fixture')
    adapter_tree = ast.parse(adapter.read_text())
    config_id = next(n for n in adapter_tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == 'config_id')
    namespace = dict(hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=[config_id], type_ignores=[]), str(adapter), 'exec'), namespace)
    assert namespace['config_id'](config) == '0d648a9b54da'
    # Resume/save only inert arrays; generate() is never called and its provider
    # imports are therefore never evaluated.
    dataset_tree = ast.parse(dataset.read_text())
    stream_class = next(n for n in dataset_tree.body
                        if isinstance(n, ast.ClassDef) and n.name == 'DatasetStream')
    namespace = dict(np=np, json=json, os=os, tempfile=tempfile, warnings=warnings)
    exec(compile(ast.Module(body=[stream_class], type_ignores=[]), str(dataset), 'exec'), namespace)
    X, Y = refit._data(14)
    path = tmp_path / 'legacy.npz'
    specs = [{} for _ in X]
    # This legacy random checkpoint intentionally omits acquisition_pool_size.
    np.savez(path, X=X, Y=Y, specs=json.dumps(specs),
             meta=json.dumps(dict(config=config, next_seed=12, failures=[])))
    noop = lambda *args: None
    stream = namespace['DatasetStream'](config, str(path),
        factory_builder=noop, feature_fn=noop, sample_fn=noop, spec_fn=noop)
    np.testing.assert_array_equal(stream.arrays()[0], X)
    np.testing.assert_array_equal(stream.arrays()[1], Y)
    stream._save()
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {'X', 'Y', 'specs', 'meta'}
        assert json.loads(str(saved['meta']))['config'] == config
        assert json.loads(str(saved['specs'])) == specs
        np.testing.assert_array_equal(saved['X'], X)
        np.testing.assert_array_equal(saved['Y'], Y)
