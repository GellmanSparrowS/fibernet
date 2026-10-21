"""Identity and synthetic integration checks, not physical-accuracy benchmarks."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.ml.graph_data import graph_batch


def fixture(z=2., dimension=3):
    g = StructureGraph(dimension)
    g.nodes[20] = SNode(np.array([1., 0., z]), node_id=20)
    g.nodes[4] = SNode(np.array([1., 0., 0.]), node_id=4)
    g.edges[9] = SEdge(20, 4, edge_id=9, radius=None,
                      internal_points=np.array([[2., 0., z / 2]]))
    g.edges[2] = SEdge(4, 20, edge_id=2, radius=.3)
    g.metadata['origin'] = 'synthetic_fixture'
    return g


def snapshot(*graphs):
    return graph_batch(graphs, graph_ids=[str(i) for i in range(len(graphs))],
                       length_unit='mm', coordinate_frame='specimen',
                       edge_features=('radius', 'chord_length', 'polyline_length'))


def adapt(b, groups=None):
    pytest.importorskip('torch')
    from fibernet.ml.gnn import graphs_from_batch
    return graphs_from_batch(b, source_groups=groups or ['source'] * len(b.graph_ids))


def test_identity_masks_copy_and_parallel_adjacency():
    b = snapshot(fixture(), fixture(dimension=2))
    gs = adapt(b, ['a', 'b'])
    import torch
    for i, gd in enumerate(gs):
        assert gd['snapshot'] is b and gd['snapshot_graph_row'] == i
        assert gd['node_ids'].tolist() == [4, 20]
        assert gd['physical_edge_ids'].tolist() == [9, 2]
        assert gd['message_parent_rows'].tolist() == [0, 0, 1, 1]
        assert gd['message_parent_ids'].tolist() == [9, 9, 2, 2]
        assert gd['message_orientation'].tolist() == [1, -1, 1, -1]
        assert gd['edge_index'].tolist() == [[1, 0, 0, 1], [0, 1, 1, 0]]
        assert gd['edge_feature_available'][:, 0].tolist() == [False, False, True, True]
        assert gd['edge_features'].shape == (4, 3)
        assert gd['n_edges'] == 2
        # Two parallel members plus a computational loop: D = 3.
        torch.testing.assert_close(gd['adjacency'].to_dense(),
                                   torch.tensor([[1/3, 2/3], [2/3, 1/3]]))
        assert gd['adjacency']._nnz() == 6
    assert gs[0]['source_group'] == 'a' and gs[1]['source_group'] == 'b'
    assert gs[0]['node_features'][:, 2].tolist() == [0., 2.]
    before = b.positions.copy()
    gs[0]['node_features'].fill_(42)
    gs[0]['edge_features'].fill_(99)
    np.testing.assert_array_equal(b.positions, before)
    assert b.edge_features[0, 0] == 0


@pytest.mark.parametrize('field', ['node_ptr', 'edge_ptr', 'internal_ptr', 'node_ids',
    'edge_ids', 'edge_index', 'message_index', 'message_parent_rows', 'message_parent_ids',
    'message_orientation', 'dimensions'])
@pytest.mark.parametrize('dtype', [bool, float])
def test_strict_integer_arrays(field, dtype):
    b = snapshot(fixture())
    with pytest.raises(ValueError):
        adapt(replace(b, **{field: getattr(b, field).astype(dtype)}))


@pytest.mark.parametrize('field', ['positions', 'internal_points', 'edge_features',
                                  'edge_feature_available', 'node_ptr'])
@pytest.mark.parametrize('mask', [False, True])
def test_masked_arrays_rejected(field, mask):
    b = snapshot(fixture())
    with pytest.raises(ValueError):
        adapt(replace(b, **{field: np.ma.array(getattr(b, field), mask=mask)}))


@pytest.mark.parametrize('value', [1e-200, 1e200, np.nan])
@pytest.mark.parametrize('field', ['positions', 'edge_features'])
def test_float32_range_rejected(field, value):
    b = snapshot(fixture())
    a = getattr(b, field).copy()
    a[-1, -1] = value
    with pytest.raises(ValueError):
        adapt(replace(b, **{field: a}))


def test_forged_and_empty_inputs():
    b = snapshot(fixture(), fixture())
    changes = [dict(schema_version='unknown'), dict(node_ptr=np.array([0, 3, 4])),
               dict(message_parent_rows=np.zeros(8, dtype=np.int64)),
               dict(message_orientation=np.ones(8, dtype=np.int8)),
               dict(edge_index=np.full((2, 4), -1, dtype=np.int64)),
               dict(edge_feature_available=np.ones((4, 3), dtype=int)),
               dict(node_ids=np.array([4, 4, 4, 20]))]
    for change in changes:
        with pytest.raises(ValueError):
            adapt(replace(b, **change))
    for empty in (snapshot(), snapshot(StructureGraph(3))):
        with pytest.raises(ValueError):
            adapt(empty)
    from fibernet.ml.gnn import graphs_from_batch
    for groups in (None, ['only'], ['a', ''], [True, 'b']):
        with pytest.raises(ValueError):
            graphs_from_batch(b, source_groups=groups)
    edgeless = fixture()
    edgeless.edges.clear()
    gd = adapt(snapshot(edgeless))[0]
    assert gd['edge_features'].shape == (0, 3)
    assert gd['adjacency']._nnz() == 0


def test_unsigned_out_of_range_and_cross_graph_messages():
    b = snapshot(fixture(), fixture())
    bad = b.node_ids.astype(np.uint64)
    bad[0] = 2**63
    with pytest.raises(ValueError, match='int64'):
        adapt(replace(b, node_ids=bad))
    physical = b.edge_index.copy()
    physical[0, 0] = 2
    messages = b.message_index.copy()
    messages[:, ::2], messages[:, 1::2] = physical, physical[::-1]
    with pytest.raises(ValueError, match='within their graph'):
        adapt(replace(b, edge_index=physical, message_index=messages))


def test_permutations_and_reversed_parallel_members():
    torch = pytest.importorskip('torch')
    from fibernet.ml.gnn import FiberGNN
    g = fixture()
    changed = copy.deepcopy(g)
    changed._nodes = dict(reversed(list(changed.nodes.items())))
    changed._edges = dict(reversed(list(changed.edges.items())))
    for edge in changed.edges.values():
        edge.node_i, edge.node_j = edge.node_j, edge.node_i
        if edge.internal_points is not None:
            edge.internal_points = edge.internal_points[::-1].copy()
    torch.manual_seed(42)
    model = FiberGNN(4, hidden=8, n_layers=1, dropout=0).eval()
    gs = adapt(snapshot(g, changed, fixture(3)))
    with torch.no_grad():
        p = model(gs)
        torch.testing.assert_close(p[0], p[1], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(model(gs[::-1]), p.flip(0), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.cat([model([gd]) for gd in gs]), p,
                                   rtol=1e-5, atol=1e-6)
    assert gs[1]['physical_edge_ids'].tolist() == [2, 9]
    assert gs[1]['physical_edge_index'].tolist() == [[1, 0], [0, 1]]


def test_lazy_and_missing_torch():
    code = '''
import sys
from fibernet.ml.graph_data import graph_batch
import fibernet
assert 'torch' not in sys.modules and 'torch_geometric' not in sys.modules
sys.modules['torch'] = None
from fibernet.ml.gnn import graphs_from_batch
try:
    graphs_from_batch(None, source_groups=[])
except ImportError as exc:
    assert 'PyTorch required' in str(exc)
else:
    raise AssertionError('missing dependency silently accepted')
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=15)


def cpu_smoke(tmp):
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    from fibernet.ml.gnn import FiberGNN, graphs_from_batch
    from fibernet.ml.training import save_model_bundle, load_model_bundle
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    b = snapshot(*[fixture(1 + i / 20) for i in range(16)])
    groups = ['source-' + str(i//2) for i in range(16)]
    gs = graphs_from_batch(b, source_groups=groups)
    train, held = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
                      .split(np.arange(16), groups=groups))
    assert not {groups[i] for i in train} & {groups[i] for i in held}
    config = dict(node_dim=4, hidden=8, n_layers=1, n_outputs=1, layer_type='gcn', dropout=0)
    labels = torch.arange(16, dtype=torch.float32).reshape(-1, 1) / 10

    def fit(y):
        torch.manual_seed(42)
        model = FiberGNN(**config)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        seen = []
        hook = model.register_forward_pre_hook(lambda _m, args: seen.append(
            [gd['graph_id'] for gd in args[0]]))
        opt = torch.optim.SGD(model.parameters(), lr=.001)
        for _ in range(2):
            opt.zero_grad()
            inputs = [gs[i] for i in train]
            used_y = y[train]
            torch.testing.assert_close(used_y, labels[train], rtol=0, atol=0)
            pred = model(inputs)
            loss = torch.nn.functional.mse_loss(pred, used_y)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
        hook.remove()
        assert seen == [[str(i) for i in train]] * 2
        assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
        model.eval()
        return model, float(loss.detach())

    model, loss = fit(labels)
    changed_y = labels.clone()
    changed_y[held] += 1000
    second, _ = fit(changed_y)
    for key, val in model.state_dict().items():
        assert torch.equal(val, second.state_dict()[key])
    with torch.no_grad():
        pred = model([gs[i] for i in held])
        assert pred.shape == (4, 1) and torch.isfinite(pred).all()
        assert torch.equal(pred, second([gs[i] for i in held]))
    path = str(Path(tmp) / 'own-model.pt')
    metadata = dict(input_schema=gs[0]['input_schema'], model_config=config,
                    graph_ids=list(b.graph_ids), groups=groups, train=train.tolist(),
                    held=held.tolist(), origin='synthetic_fixture', length_unit=b.length_unit,
                    coordinate_frame=b.coordinate_frame)
    save_model_bundle(model, path, feature_names=list(gs[0]['node_feature_names']),
                      target_names=['synthetic_target'], metadata=metadata)
    restored, info = load_model_bundle(path, model_class=FiberGNN, **config)
    assert info['metadata'] == metadata
    assert info['feature_names'] == ['x', 'y', 'z', 'dimension']
    with torch.no_grad():
        assert torch.equal(pred, restored([gs[i] for i in held]))
    import resource
    print(json.dumps(dict(status='passed', train_graphs=len(train), held_graphs=len(held),
          steps=2, train_loss=loss, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))


def test_bounded_actual_cpu_smoke(tmp_path):
    pytest.importorskip('torch')
    if sys.platform != 'linux':
        pytest.skip('Linux address-space budget fixture')
    code = '''
import resource, runpy, sys
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
ns = runpy.run_path(sys.argv[1])
ns['cpu_smoke'](sys.argv[2])
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', code, __file__, str(tmp_path)],
                            env=env, text=True, capture_output=True, timeout=30)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
