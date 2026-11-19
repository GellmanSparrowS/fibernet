"""Read native v3.1 fracture geometry without inferring its physical inputs."""
from dataclasses import asdict
import json
import math
import sys

import numpy as np
import pytest

from fibernet import Material, StructureGraph, pattern_2d
from fibernet.sim.fracture import FractureConfig, FractureSolver
from fibernet.sim.fracture.__main__ import _load_structure_graph, main


def _config():
    return FractureConfig(1e9, .3, 1e6, 1e6, axis=0, maximum_displacement_m=3e-4)


def _run_cli(monkeypatch, tmp_path, graph_path, fixed, loaded, *opts):
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(asdict(_config())), encoding='utf-8')
    result_path = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', [
        'fibernet.sim.fracture', '--graph', str(graph_path), '--config', str(config_path),
        '--fixed', *map(str, fixed), '--loaded', *map(str, loaded),
        '--length-unit', 'mm', '--output', str(result_path), *opts,
    ])
    main()
    return result_path


def _chain():
    graph = StructureGraph(dimension=3)
    left = graph.add_node([0, 0, 0])
    temporary = graph.add_node([0, 0, 2])
    graph.remove_node(temporary)
    right = graph.add_node([50, 0, 0])
    graph.add_edge(left, right, radius=1., material=Material(youngs_modulus=1e9, poissons_ratio=.3))
    return graph


def test_same_generated_graph_and_initial_notch_through_native_cli(tmp_path, monkeypatch, capsys):
    graph = pattern_2d(
        unit='honeycomb', box=(10, 10), grid=(2, 2), radius=.5, seed=23,
        material=Material(youngs_modulus=1e9, poissons_ratio=.3, extra={'batch': [23]}),
    )
    data = graph.to_dict()
    data['next_node_id'] = 100
    data['next_edge_id'] = 300
    for node in data['nodes']:
        node['id'] = 10 + 3 * node['id']
        node['metadata']['joint_type'] = 'welded'
    for edge in data['edges']:
        edge['id'] = 100 + 7 * edge['id']
        edge['node_i'] = 10 + 3 * edge['node_i']
        edge['node_j'] = 10 + 3 * edge['node_j']
    data['metadata']['study'] = {'seed': 23}
    graph = StructureGraph.from_dict(data)
    first = graph.edges[100]
    first.metadata['junction_type'] = 'welded'
    first.internal_points = np.array([
        (graph.nodes[first.node_i].position + graph.nodes[first.node_j].position) / 2,
    ])
    fixed = [i for i, node in graph.nodes.items() if node.position[0] == 0.]
    loaded = [i for i, node in graph.nodes.items() if node.position[0] == 20.]
    notch = 107
    snapshot = graph.to_dict()
    native_path = tmp_path / 'graph.json'
    graph.save_json(native_path)
    original_bytes = native_path.read_bytes()
    assert _load_structure_graph(native_path).to_dict() == snapshot

    cli_path = _run_cli(monkeypatch, tmp_path, native_path, fixed, loaded,
                        '--graph-format', 'structure-graph-3.1', '--initially-broken', str(notch))
    direct = FractureSolver(graph, _config(), fixed_nodes=fixed, loaded_nodes=loaded,
                            length_unit='mm', initially_broken_edges=[notch]).run()
    direct_path = tmp_path / 'direct.json'
    direct.save(direct_path)
    assert cli_path.read_bytes() == direct_path.read_bytes()
    assert graph.to_dict() == snapshot
    assert native_path.read_bytes() == original_bytes
    assert (len(graph.nodes), len(graph.edges)) == (18, 22)
    assert direct.data['complete'] and direct.events
    assert direct.observations[0]['mask'][list(graph.edges).index(notch)] is False
    assert set(map(int, direct.data['state']['positions_m'])) == set(graph.nodes)
    assert {event['segment_id'] for event in direct.events} <= set(graph.edges) - {notch}
    assert direct.data['binding']['geometry_sha256']
    assert set(direct.data['binding']['sources']) == {'__init__.py', '_beam.py', '_sparse.py', '_engine.py'}
    assert json.loads(capsys.readouterr().out)['events'] == len(direct.events)


def test_dedicated_schema_remains_the_default_cli_format(tmp_path, monkeypatch):
    graph = _chain()
    native_path = tmp_path / 'graph.json'
    graph.save_json(native_path)
    with pytest.raises(ValueError, match='Explicit fibernet-fracture-geometry-1'):
        _run_cli(monkeypatch, tmp_path, native_path, [0], [2])
    assert not (tmp_path / 'result.json').exists()

    data = graph.to_dict()
    dedicated = dict(schema='fibernet-fracture-geometry-1',
                     nodes=[{k: n[k] for k in ('id', 'position')} for n in data['nodes']],
                     edges=[{**{k: e[k] for k in ('id', 'node_i', 'node_j', 'radius')},
                             'material': {k: e['material'][k] for k in
                                          ('youngs_modulus', 'poissons_ratio', 'shear_modulus')}}
                            for e in data['edges']])
    old_path = tmp_path / 'dedicated.json'
    old_path.write_text(json.dumps(dedicated), encoding='utf-8')
    result_path = _run_cli(monkeypatch, tmp_path, old_path, [0], [2])
    result = json.loads(result_path.read_text(encoding='utf-8'))
    assert result['termination_reason'] == 'material_path_loss'
    assert [event['segment_id'] for event in result['events']] == [0]
    assert result['events'][0]['displacement_m'] == pytest.approx(5e-5)
    peak = next(row for row in result['observations'] if row['phase'] == 'event_pre')
    assert peak['loaded_reaction_force_N'] == pytest.approx(math.pi)


@pytest.mark.parametrize('change, reason', [
    ('old-version', 'version 3.1'), ('both-schemas', 'version 3.1'),
    ('mixed-representations', 'version 3.1'), ('missing-node-id', 'Explicit node IDs'),
    ('repeated-node-id', 'duplicate node ID'), ('fractional-id', 'literal integer ID'),
    ('nonfinite-coordinate', 'coordinate for node 0'), ('xy-only', 'finite XYZ'),
    ('repeated-edge-id', 'duplicate edge ID'), ('bool-endpoint', 'literal integer ID'),
    ('unknown-endpoint', 'existing endpoint IDs'), ('missing-radius', 'radius and material'),
    ('text-radius', 'actual finite numeric radius'), ('missing-material', 'radius and material'),
    ('missing-elastic', 'explicit isotropic E, nu and G'), ('null-G', 'explicit isotropic E, nu and G'),
    ('duplicate-key', 'duplicate JSON object key'),
])
def test_v31_input_cannot_infer_missing_or_ambiguous_physical_data(tmp_path, change, reason):
    body = _chain().to_dict()
    node = body['nodes'][0]
    edge = body['edges'][0]
    if change == 'old-version':
        body['version'] = '3.0'
    elif change == 'both-schemas':
        body['schema'] = 'fibernet-fracture-geometry-1'
    elif change == 'mixed-representations':
        body['crosslinks'] = []
    elif change == 'missing-node-id':
        node.pop('id')
    elif change == 'repeated-node-id':
        body['nodes'].append(node)
    elif change == 'fractional-id':
        node['id'] = .25
    elif change == 'nonfinite-coordinate':
        node['position'][0] = float('inf')
    elif change == 'xy-only':
        node['position'].pop()
    elif change == 'repeated-edge-id':
        body['edges'].append(edge)
    elif change == 'bool-endpoint':
        edge['node_i'] = False
    elif change == 'unknown-endpoint':
        edge['node_j'] = 99
    elif change == 'missing-radius':
        edge.pop('radius')
    elif change == 'text-radius':
        edge['radius'] = '1.0'
    elif change == 'missing-material':
        edge.pop('material')
    elif change == 'missing-elastic':
        edge['material'].pop('youngs_modulus')
    elif change == 'null-G':
        edge['material']['shear_modulus'] = None
    raw = json.dumps(body)
    if change == 'duplicate-key':
        raw = raw.replace('"radius":', '"radius": 1, "radius":', 1)
    path = tmp_path / 'invalid.json'
    path.write_text(raw, encoding='utf-8')
    with pytest.raises(ValueError, match='(?i)' + reason):
        _load_structure_graph(path)


@pytest.mark.parametrize('change, reason', [
    ('node-hinge', 'welded'), ('edge-hinge', 'welded'),
    ('parallel', 'Parallel'), ('curve', 'Curved'), ('anisotropy', 'homogeneous'),
])
def test_native_mode_delegates_fracture_physics_admission(tmp_path, monkeypatch, change, reason):
    graph = _chain()
    edge = graph.edges[0]
    if change == 'node-hinge':
        graph.nodes[0].metadata['joint_type'] = 'hinged'
    elif change == 'edge-hinge':
        edge.metadata['junction_type'] = 'hinged'
    elif change == 'parallel':
        graph.add_edge(edge.node_i, edge.node_j, radius=edge.radius,
                       material=edge.material, allow_parallel=True)
    elif change == 'curve':
        edge.internal_points = np.array([[25., 1., 0.]])
    else:
        edge.material.stiffness_tensor = np.identity(6)
    path = tmp_path / 'graph.json'
    graph.save_json(path)
    with pytest.raises(ValueError, match=reason):
        _run_cli(monkeypatch, tmp_path, path, [0], [2], '--graph-format', 'structure-graph-3.1')
    assert not (tmp_path / 'result.json').exists()
