"""Synthetic CPU integration of existing GAT/SAGE, not accuracy benchmarks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

# Share geometry-only fixtures with the adapter contract tests.
from test_gnn_graph_batch import fixture, snapshot


def config(mode):
    return dict(node_dim=4, hidden=8, n_heads=4, n_layers=2,
                n_outputs=1, dropout=0, layer_type=mode)


@pytest.mark.parametrize('mode', ['gat', 'sage'])
def test_forward_gradient_permutations_and_nonconsumption(mode):
    torch = pytest.importorskip('torch')
    from fibernet.ml.gnn import FiberGNN, graphs_from_batch
    g = fixture()
    reordered = copy.deepcopy(g)
    reordered._nodes = dict(reversed(list(reordered.nodes.items())))
    reordered._edges = dict(reversed(list(reordered.edges.items())))
    for edge in reordered.edges.values():
        edge.node_i, edge.node_j = edge.node_j, edge.node_i
        if edge.internal_points is not None:
            edge.internal_points = edge.internal_points[::-1].copy()
    relabeled = copy.deepcopy(g)
    mapping = {4: 100, 20: 3}
    relabeled._nodes = {mapping[k]: v for k, v in relabeled.nodes.items()}
    for key, node in relabeled.nodes.items():
        node.node_id = key
    for edge in relabeled.edges.values():
        edge.node_i, edge.node_j = mapping[edge.node_i], mapping[edge.node_j]
    edgeless = fixture()
    edgeless.edges.clear()
    b = snapshot(g, reordered, relabeled, edgeless)
    gs = graphs_from_batch(b, source_groups=['same', 'same', 'same', 'other'])
    torch.manual_seed(42)
    model = FiberGNN(**config(mode)).eval()
    pred = model(gs)
    assert pred.shape == (4, 1) and torch.isfinite(pred).all()
    torch.testing.assert_close(pred[:3], pred[0].expand(3, 1), rtol=1e-5, atol=1e-6)
    order = [3, 1, 0, 2]
    torch.testing.assert_close(model([gs[i] for i in order]), pred[order], rtol=1e-5, atol=1e-6)
    assert [(gs[i]['graph_id'], gs[i]['source_group']) for i in order] == [
        ('3', 'other'), ('1', 'same'), ('0', 'same'), ('2', 'same')]
    assert gs[2]['node_ids'].tolist() == [3, 100]
    assert gs[1]['physical_edge_ids'].tolist() == [2, 9]
    assert gs[1]['message_parent_ids'].tolist() == [2, 2, 9, 9]
    assert gs[1]['message_orientation'].tolist() == [1, -1, 1, -1]
    assert gs[1]['edge_feature_available'][:, 0].tolist() == [True, True, False, False]
    altered = [dict(gd) for gd in gs]
    for gd in altered:
        gd['edge_features'] = torch.full_like(gd['edge_features'], 123)
        gd['edge_feature_available'] = ~gd['edge_feature_available']
        gd['adjacency'] = None  # These modes use messages, not GCN adjacency.
    torch.testing.assert_close(model(altered), pred, rtol=0, atol=0)
    pred.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    assert any(torch.any(x != 0) for x in grads)
    assert gs[0]['snapshot'] is b and b.edge_features[0, 0] == 0


@pytest.mark.parametrize('mode', ['gat', 'sage'])
def test_analytic_message_multiplicity(mode):
    torch = pytest.importorskip('torch')
    from fibernet.ml.gnn import GraphAttentionLayer, GraphSAGELayer
    x = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    original = torch.tensor([[0, 1], [2, 2]])
    duplicated = torch.tensor([[0, 0, 1], [2, 2, 2]])
    if mode == 'gat':
        layer = GraphAttentionLayer(2, 2, n_heads=1, dropout=0)
        with torch.no_grad():
            layer.W[0].copy_(torch.eye(2))
            layer.a_src.zero_()
            layer.a_dst.zero_()
        want = torch.tensor([2/3, 1/3])
    else:
        layer = GraphSAGELayer(2, 2)
        with torch.no_grad():
            layer.self_weight.weight.zero_()
            layer.neigh_weight.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        want = torch.tensor([2., 1.]) / np.sqrt(5)
    a = layer(x, original)[2]
    b = layer(x, duplicated)[2]
    torch.testing.assert_close(b, want)
    assert not torch.allclose(a, b)
    torch.testing.assert_close(a, layer(x, original.repeat(1, 2))[2])


def test_existing_gat_configuration_boundaries():
    torch = pytest.importorskip('torch')
    from fibernet.ml.gnn import FiberGNN, graphs_from_batch
    model = FiberGNN(**config('gat'))
    assert model.dropout.p == 0
    assert all(layer.dropout.p == .1 for layer in model.layers)
    assert all(layer.dropout.training for layer in model.layers)
    bad = FiberGNN(node_dim=4, hidden=7, n_heads=4, n_layers=2, layer_type='gat')
    gs = graphs_from_batch(snapshot(fixture()), source_groups=['s'])
    with pytest.raises(RuntimeError, match='normalized_shape'):
        bad(gs)


def cpu_smoke(mode, directory):
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    from fibernet.ml.gnn import FiberGNN, graphs_from_batch
    from fibernet.ml.training import save_model_bundle, load_model_bundle
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    b = snapshot(*[fixture(1 + i/20) for i in range(16)])
    groups = ['source-' + str(i//2) for i in range(16)]
    gs = graphs_from_batch(b, source_groups=groups)
    train, held = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
                      .split(np.arange(16), groups=groups))
    assert not {groups[i] for i in train} & {groups[i] for i in held}
    labels = torch.arange(16, dtype=torch.float32).reshape(-1, 1)/10

    def fit(y):
        torch.manual_seed(42)
        model = FiberGNN(**config(mode)).train()
        if mode == 'gat':
            assert all(layer.dropout.p == .1 and layer.dropout.training for layer in model.layers)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        seen = []
        hook = model.register_forward_pre_hook(lambda m, args: seen.append(
            [gd['graph_id'] for gd in args[0]]))
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            target = y[train]
            assert torch.equal(target, labels[train])
            pred = model([gs[i] for i in train])
            loss = torch.nn.functional.mse_loss(pred, target)
            assert torch.isfinite(loss)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(torch.any(g != 0) for g in grads)
            optimizer.step()
        hook.remove()
        assert seen == [[str(i) for i in train]] * 2
        assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
        return model.eval(), float(loss.detach())

    model, loss = fit(labels)
    changed = labels.clone()
    changed[held] += 1000
    repeated, _ = fit(changed)
    assert all(torch.equal(v, repeated.state_dict()[k]) for k, v in model.state_dict().items())
    with torch.no_grad():
        prediction = model([gs[i] for i in held])
        assert prediction.shape == (4, 1) and torch.isfinite(prediction).all()
        assert torch.equal(prediction, repeated([gs[i] for i in held]))
    metadata = dict(input_schema=gs[0]['input_schema'], model_config=config(mode),
                    attention_dropout=.1 if mode == 'gat' else None,
                    train=train.tolist(), held=held.tolist(), groups=groups,
                    graph_ids=list(b.graph_ids), origin='synthetic_fixture',
                    length_unit=b.length_unit, coordinate_frame=b.coordinate_frame)
    path = str(Path(directory) / (mode + '-own.pt'))
    save_model_bundle(model, path, feature_names=list(gs[0]['node_feature_names']),
                      target_names=['synthetic_target'], metadata=metadata)
    restored, info = load_model_bundle(path, model_class=FiberGNN, **config(mode))
    assert info['metadata'] == metadata and info['feature_names'] == ['x', 'y', 'z', 'dimension']
    with torch.no_grad():
        assert torch.equal(prediction, restored([gs[i] for i in held]))
    import resource
    print(json.dumps(dict(mode=mode, status='passed', steps=2, train=12, held=4,
          train_loss=loss, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))


@pytest.mark.parametrize('mode', ['gat', 'sage'])
def test_bounded_training_smoke(mode, tmp_path):
    pytest.importorskip('torch')
    if sys.platform != 'linux':
        pytest.skip('Linux address-space budget fixture')
    code = '''
import resource, runpy, sys
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
sys.path.insert(0, sys.argv[1])
ns = runpy.run_path(sys.argv[2])
ns['cpu_smoke'](sys.argv[3], sys.argv[4])
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    r = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent),
                        __file__, mode, str(tmp_path)], env=env, capture_output=True,
                        text=True, timeout=30)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr
