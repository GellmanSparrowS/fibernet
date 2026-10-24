"""Declared-input compatibility only: fixed untrained models, no fit or physics."""
import importlib.abc
import json
import sys
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.ml.graph_data import graph_batch
from fibernet.ml.gnn import FiberGNN, graphs_from_batch, predict_gnn


def networks():
    """Distinct 3D branched/cyclic networks, parallel curves and an isolated node."""
    result = []
    for scale in (1., 1.7, .6):
        graph = StructureGraph(3)
        points = {40: (2, 1, 1), 4: (0, 0, 0), 19: (1, 0, 1),
                  73: (2, 0, 3), 102: (0, 2, 2), 200: (4, 4, 2)}
        for nid, point in points.items():
            graph.nodes[nid] = SNode(np.asarray(point, dtype=float)*scale, node_id=nid)
        for eid, left, right in ((90, 4, 19), (3, 19, 4), (21, 19, 40),
                                 (8, 40, 73), (51, 73, 19), (60, 40, 102)):
            curve = np.array([[.2, .5, .6]])*scale if eid == 90 else None
            graph.edges[eid] = SEdge(left, right, edge_id=eid,
                                    radius=None if eid == 90 else .1*scale,
                                    internal_points=curve)
        graph.metadata['origin'] = 'fixed_untrained_interface_fixture'
        result.append(graph)
    return result


def batch(order=(0, 1, 2), **kwargs):
    graphs = networks()
    return graph_batch([graphs[i] for i in order], graph_ids=['sample-'+str(i) for i in order],
                       length_unit=kwargs.get('length_unit', 'mm'),
                       coordinate_frame=kwargs.get('coordinate_frame', 'specimen'),
                       edge_features=('radius', 'chord_length', 'polyline_length'))


def model(mode='gcn', pooling='mean'):
    torch.manual_seed(42)
    return FiberGNN(4, hidden=8, n_layers=2, n_heads=2, n_outputs=1,
                    layer_type=mode, pooling=pooling, dropout=.2)


def declaration(mode='gcn', pooling='mean'):
    # A fresh caller declaration, deliberately not a purported training result.
    layers = [{'class': {'gcn': 'GraphConvLayer', 'gat': 'GraphAttentionLayer',
                         'sage': 'GraphSAGELayer'}[mode]} for _ in range(2)]
    if mode == 'gat':
        for i, layer in enumerate(layers):
            layer.update(n_heads=2, attention_dropout=.1, concat=i == 0,
                         out_dim=4 if i == 0 else 8)
    elif mode == 'sage':
        for layer in layers:
            layer['aggregator'] = 'mean'
    return {'input_schema': 'fibernet.fibergnn.xyz_dimension.v1',
            'feature_names': ['x', 'y', 'z', 'dimension'],
            'length_unit': 'mm', 'coordinate_frame': 'specimen',
            'model': {'class': 'FiberGNN', 'node_dim': 4, 'hidden': 8,
                      'n_outputs': 1, 'n_layers': 2, 'layer_type': mode,
                      'pooling': pooling, 'dropout': .2, 'head_dropout': .2,
                      'layers': layers}}


def predict(m=None, b=None, c=None, groups=None, **kwargs):
    return predict_gnn(model() if m is None else m, batch() if b is None else b,
                       input_contract=declaration() if c is None else c,
                       source_groups=['origin-a', 'origin-a', 'origin-b'] if groups is None else groups,
                       target_name=kwargs.get('target_name', 'caller_property'))


@pytest.fixture(autouse=True)
def no_training_or_solver(monkeypatch):
    forbidden = {'taichi', 'gmsh', 'lammps', 'torch_geometric'}
    loaded = forbidden & set(sys.modules)
    def prohibited(*args, **kwargs):
        raise AssertionError('training, optimizer or backward must not execute')
    for name in ('Adam', 'AdamW', 'SGD'):
        monkeypatch.setattr(torch.optim, name, prohibited)
    monkeypatch.setattr(torch.Tensor, 'backward', prohibited)
    monkeypatch.setattr('fibernet.ml.gnn.train_gnn', prohibited)
    yield
    assert forbidden & set(sys.modules) == loaded


@pytest.mark.parametrize('mode', ['gcn', 'gat', 'sage'])
@pytest.mark.parametrize('pooling', ['mean', 'max', 'attention'])
def test_untrained_parity_order_identity_and_state(mode, pooling):
    m = model(mode, pooling)
    b = batch()
    groups = ['origin-a', 'origin-a', 'origin-b']
    inputs = graphs_from_batch(b, source_groups=groups)
    assert inputs[0]['node_ids'].tolist() == [4, 19, 40, 73, 102, 200]
    assert inputs[0]['physical_edge_ids'].tolist() == [90, 3, 21, 8, 51, 60]
    assert inputs[0]['message_parent_ids'].tolist()[:4] == [90, 90, 3, 3]
    assert not inputs[0]['edge_feature_available'][0, 0]
    m.eval()
    with torch.no_grad():
        direct = m(inputs).reshape(-1).tolist()
    m.train()
    m.norms[0].training = False
    flags = [module.training for module in m.modules()]
    before = {key: value.clone() for key, value in m.state_dict().items()}
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    grads = [p.grad.clone() for p in m.parameters()]
    rng = torch.get_rng_state().clone()
    seen = []
    hook = m.register_forward_pre_hook(lambda mdl, args: seen.append(
        (mdl.training, torch.is_grad_enabled(), [g['graph_id'] for g in args[0]])))
    answer = predict(m, b, declaration(mode, pooling))
    hook.remove()
    assert answer['predictions'] == direct
    assert seen == [(False, False, list(b.graph_ids))]
    assert [module.training for module in m.modules()] == flags
    assert torch.is_grad_enabled()
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert all(torch.equal(p.grad, grad) for p, grad in zip(m.parameters(), grads))
    changed = predict(m, batch((2, 1, 0)), declaration(mode, pooling), list(reversed(groups)))
    assert changed['predictions'] == list(reversed(direct))
    assert changed['graph_ids'] == list(reversed(answer['graph_ids']))
    assert changed['source_groups'] == list(reversed(groups))
    assert answer['prediction_space'] == 'model_output'
    assert answer['target_name_source'] == 'caller_declared'
    assert answer['target_unit'] is answer['target_transform'] is None
    assert answer['target_unit_status'] == answer['target_transform_status'] == 'unknown'
    assert answer['validation'] == 'declared_input_contract_only'
    json.dumps(answer, allow_nan=False)


def test_one_graph_group_and_irrelevant_bookkeeping_not_validated():
    c = declaration()
    # Optional old bookkeeping can be opaque; prediction neither validates nor copies it.
    c.update(schema='fibernet.gnn_group_training.v1', graph_ids=['sample-0'],
             split={'groups': ['origin-a']}, training=object(), model_selection=object())
    result = predict(b=batch((0,)), c=c, groups=['origin-a'])
    assert result['graph_ids'] == ['sample-0'] and len(result['predictions']) == 1
    assert result['input_contract'] == declaration()
    result['input_contract']['feature_names'].reverse()
    assert c['feature_names'] == declaration()['feature_names']


@pytest.mark.parametrize('change', [
    {'input_schema': 'unknown'}, {'input_schema': 'caller-legacy'},
    {'feature_names': ['y', 'x', 'z', 'dimension']},
    {'feature_names': ('x', 'y', 'z', 'dimension')},
    {'length_unit': 'm'}, {'coordinate_frame': 'world'},
    {'schema': 'fibernet.gnn_group_training.v2'}, {'node_scale': [1, 1, 1, 1]},
    {'preprocessing': None}, {'length_unit': ''}, {'coordinate_frame': None},
])
def test_bad_declaration_rejects_before_forward(change):
    c = declaration(); c.update(change)
    m = model(); calls = []
    hook = m.register_forward_pre_hook(lambda *args: calls.append(True))
    with pytest.raises(ValueError):
        predict(m, c=c)
    hook.remove()
    assert not calls


@pytest.mark.parametrize('field', ['input_schema', 'feature_names', 'length_unit', 'coordinate_frame', 'model'])
def test_missing_contract_fields(field):
    c = declaration(); del c[field]
    with pytest.raises(ValueError):
        predict(c=c)


@pytest.mark.parametrize('field,value', [('pooling', 'max'), ('hidden', 9), ('n_outputs', True),
                                         ('dropout', False), ('dropout', 10**400), ('head_dropout', float('nan')),
                                         ('layers', []), ('class', 'GINERegressor')])
def test_recorded_model_mismatch_before_forward(field, value):
    c = declaration(); c['model'][field] = value
    m = model(); calls = []
    hook = m.register_forward_pre_hook(lambda *args: calls.append(True))
    with pytest.raises(ValueError):
        predict(m, c=c)
    hook.remove(); assert not calls


@pytest.mark.parametrize('bad', ['', ' ', None, 1])
def test_target_name_must_be_explicit(bad):
    with pytest.raises(ValueError, match='target_name'):
        predict(target_name=bad)


@pytest.mark.parametrize('groups', [[], ['only'], ['a', '', 'b'], None])
def test_group_declaration_must_align(groups):
    with pytest.raises(ValueError):
        predict_gnn(model(), batch(), source_groups=groups,
                    input_contract=declaration(), target_name='property')


@pytest.mark.parametrize('kind', ['shape', 'nan', 'double', 'list', 'raise'])
def test_forward_failures_restore_individual_modes(kind):
    m = model(); m.train(); m.norms[0].eval()
    flags = [module.training for module in m.modules()]
    def alter(_model, _args, out):
        if kind == 'raise':
            raise RuntimeError('caller hook failed')
        return {'shape': out[:, 0], 'nan': out*float('nan'),
                'double': out.double(), 'list': out.tolist()}[kind]
    hook = m.register_forward_hook(alter)
    with pytest.raises((ValueError, RuntimeError)):
        predict(m)
    hook.remove()
    assert [module.training for module in m.modules()] == flags
    assert torch.is_grad_enabled()


def test_ambient_cpu_autocast_is_disabled_and_restored():
    m = model('gat'); c = declaration('gat')
    expected = predict(m, c=c)
    observed = []
    hook = m.register_forward_pre_hook(lambda *_: observed.append(torch.is_autocast_enabled('cpu')))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        assert torch.is_autocast_enabled('cpu')
        actual = predict(m, c=c)
        assert torch.is_autocast_enabled('cpu')
    hook.remove()
    assert observed == [False] and expected == actual


def test_low_level_forward_remains_generic():
    m = model().eval()
    inputs = graphs_from_batch(batch(), source_groups=['a', 'a', 'b'])
    changed = [dict(g, length_unit='anything', input_schema='caller-custom') for g in inputs]
    with torch.no_grad():
        assert torch.equal(m(inputs), m(changed))


def test_no_optional_dependency_import_or_training():
    # In-process guard catches new imports without a resource-unbounded subprocess.
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('sklearn', 'taichi', 'gmsh', 'lammps', 'torch_geometric'):
                raise AssertionError('prohibited optional import: '+fullname)
    guard = Block(); sys.meta_path.insert(0, guard)
    try:
        predict()
    finally:
        sys.meta_path.remove(guard)


def test_existing_group_record_generation_without_training(monkeypatch):
    # Exercise the refactored preflight only; a deterministic splitter double
    # avoids importing sklearn or fitting anything. The record is not a trained result.
    class Split:
        def __init__(self, **kwargs):
            assert kwargs == {'n_splits': 1, 'test_size': .25, 'random_state': 42}
        def split(self, rows, groups):
            yield np.array([0, 1]), np.array([2])
    fake = ModuleType('sklearn.model_selection'); fake.GroupShuffleSplit = Split
    monkeypatch.setitem(sys.modules, 'sklearn.model_selection', fake)
    from fibernet.ml.gnn import _gnn_group_preflight
    m = model(); b = batch(); groups = ['a', 'a', 'b']
    inputs = graphs_from_batch(b, source_groups=groups)
    _, train, val, record = _gnn_group_preflight(m, inputs, np.array([1., 2., 3.]),
                                               groups, 2, .001, 32, .25, 3)
    assert train.tolist() == [0, 1] and val.tolist() == [2]
    assert record['model'] == declaration()['model']
    assert record['feature_names'] == declaration()['feature_names']
    assert predict(m, b, record, groups)['input_contract'] == declaration()


def test_public_lazy_export():
    from fibernet.ml import predict_gnn as exported
    assert exported is predict_gnn


def test_declared_strings_cannot_be_equality_spoofs():
    class Anything:
        def __eq__(self, other):
            return True
        def __ne__(self, other):
            return False
    for field in ('input_schema', 'schema'):
        c = declaration(); c[field] = Anything()
        with pytest.raises(ValueError):
            predict(c=c)
    c = declaration(); c['feature_names'][0] = Anything()
    with pytest.raises(ValueError):
        predict(c=c)


@pytest.mark.parametrize('kind', ['adapted_dicts', 'v2', 'empty'])
def test_unsupported_graph_entry_before_forward(kind):
    from dataclasses import replace
    b = batch()
    if kind == 'adapted_dicts':
        b = graphs_from_batch(b, source_groups=['a', 'a', 'b'])
    elif kind == 'v2':
        b = replace(b, schema_version='fibernet.graph_batch.v2')
    else:
        b = graph_batch([], graph_ids=[], length_unit='mm', coordinate_frame='specimen')
    m = model(); seen = []
    hook = m.register_forward_pre_hook(lambda *args: seen.append(True))
    with pytest.raises(ValueError):
        predict(m, b)
    hook.remove(); assert not seen


def test_missing_torch_guard(monkeypatch):
    import fibernet.ml.gnn as gnn
    monkeypatch.setattr(gnn, 'HAS_TORCH', False)
    with pytest.raises(ImportError, match='PyTorch required'):
        predict_gnn(None, None, source_groups=None, input_contract=None, target_name=None)


def test_documented_fresh_untrained_example():
    from pathlib import Path
    doc = Path(__file__).resolve().parents[1]/'docs/architecture/gnn-checked-prediction-usage.md'
    example = doc.read_text().split('## Fresh untrained interface example\n', 1)[1]
    code = example.split('```python\n')[1].split('```')[0]
    namespace = {}
    exec(compile(code, str(doc), 'exec'), namespace)
    assert 'schema' not in namespace['contract']
    assert namespace['result']['graph_ids'] == ['example-a']
    assert namespace['result']['target_transform_status'] == 'unknown'
