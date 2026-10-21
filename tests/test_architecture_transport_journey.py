"""Explicit electrical module journey, kept separate from mechanical simulate."""
import json

import numpy as np
import pytest

import fibernet as fn
from fibernet.sim.linear_transport import LinearTransportResult, solve_linear_transport


@pytest.mark.parametrize('sparse_reversed', [False, True])
def test_generated_graph_native_reload_transport_json(tmp_path, sparse_reversed):
    graph = fn.pattern_2d(unit='square', box=(1, 1), grid=(1, 1),
                          n_pts_per_side=0)
    if sparse_reversed:
        data = graph.to_dict()
        mapping = {nid: 7 + 11 * nid for nid in graph.nodes}
        for node in data['nodes']:
            node['id'] = mapping[node['id']]
        for edge in data['edges']:
            edge['node_i'] = mapping[edge['node_i']]
            edge['node_j'] = mapping[edge['node_j']]
            edge['id'] += 3
        data['nodes'].reverse()
        data.pop('next_node_id'); data.pop('next_edge_id')
        graph = fn.StructureGraph.from_dict(data)
    graph.save_json(tmp_path / 'graph.json')
    graph = fn.StructureGraph.load_json(tmp_path / 'graph.json')
    left = [nid for nid, node in graph.nodes.items() if node.position[0] == 0]
    right = [nid for nid, node in graph.nodes.items() if node.position[0] == 1]
    assert fn.solve_linear_transport is solve_linear_transport
    result = fn.solve_linear_transport(
        graph, {eid: 1. for eid in graph.edges},
        [(nid, 1.) for nid in left] + [(nid, 0.) for nid in right],
        length_unit='mm')
    assert isinstance(result, LinearTransportResult)
    assert not isinstance(result, fn.SimResult)
    np.testing.assert_array_equal(result.node_ids, sorted(graph.nodes))
    np.testing.assert_allclose(result.current_A, [1, 0, -1, 0])
    assert result.dissipated_power_W == 2.
    assert result.boundary_power_W == 2.
    assert result.global_current_balance_A == 0.
    assert result.free_node_residual_A == 0.
    path = tmp_path / 'electrical-result.json'
    path.write_text(json.dumps(result.to_dict(), allow_nan=False))
    saved = json.loads(path.read_text())
    assert saved['schema'] == 'fibernet-linear-conductance-1'
    assert saved['provenance']['units']['current'] == 'A'
    assert saved['provenance']['geometry_conversion'] == 'none'
    assert saved['edge_ids'] == list(graph.edges)


@pytest.mark.parametrize('reverse_fields', [False, True])
def test_transport_id_keyed_fields_vtu_reload(tmp_path, reverse_fields):
    meshio = pytest.importorskip('meshio')
    from fibernet.io.meshio_adapter import write_vtu_bundle

    graph = fn.pattern_2d(unit='square', box=(1, 1), grid=(1, 1),
                          n_pts_per_side=0)
    # Native reload reverses record insertion and makes IDs sparse.
    data = graph.to_dict()
    node_map = {nid: 9 + 13 * nid for nid in graph.nodes}
    for node in data['nodes']:
        node['id'] = node_map[node['id']]
    for edge in data['edges']:
        edge['node_i'] = node_map[edge['node_i']]
        edge['node_j'] = node_map[edge['node_j']]
        edge['id'] += 17
    data['nodes'].reverse()
    data.pop('next_node_id'); data.pop('next_edge_id')
    graph = fn.StructureGraph.from_dict(data)
    result = fn.solve_linear_transport(
        graph, {eid: 1. for eid in graph.edges},
        [(nid, 1. - node.position[0]) for nid, node in graph.nodes.items()],
        length_unit='mm')
    potentials = dict(zip(map(int, result.node_ids), result.voltage_V))
    currents = dict(zip(map(int, result.edge_ids), result.current_A))
    if reverse_fields:
        potentials = dict(reversed(list(potentials.items())))
        currents = dict(reversed(list(currents.items())))
    # The existing adapter consumes row arrays. Explicit ID lookup bridges it;
    # dictionary iteration order must never be treated as field identity.
    node_ids, edge_ids = sorted(graph.nodes), list(graph.edges)
    bundle = write_vtu_bundle(
        tmp_path / 'electrical', graph, length_unit='mm',
        point_data={'potential_V': [potentials[nid] for nid in node_ids]},
        cell_data={'current_A': [currents[eid] for eid in edge_ids]},
        field_metadata={
            'point': {'potential_V': {'units': 'V', 'meaning': 'electric potential'}},
            'cell': {'current_A': {'units': 'A', 'meaning': 'signed current node_i to node_j'}}},
        provenance={'electrical_result': result.to_dict()})
    mesh = meshio.read(bundle / 'network.vtu')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    np.testing.assert_array_equal(mesh.point_data['node_id'], node_ids)
    np.testing.assert_array_equal(mesh.cell_data['edge_id'][0], edge_ids)
    for nid, value in zip(mesh.point_data['node_id'], mesh.point_data['potential_V']):
        assert value == potentials[int(nid)]
    for eid, value in zip(mesh.cell_data['edge_id'][0], mesh.cell_data['current_A'][0]):
        assert value == currents[int(eid)]
    np.testing.assert_allclose(mesh.points, [graph.nodes[nid].position for nid in node_ids])
    rows = {nid: i for i, nid in enumerate(node_ids)}
    np.testing.assert_array_equal(mesh.cells_dict['line'],
        [[rows[e.node_i], rows[e.node_j]] for e in graph.edges.values()])
    assert manifest['length_unit'] == 'mm'
    assert manifest['field_metadata']['cell']['current_A']['units'] == 'A'
    assert manifest['provenance']['electrical_result']['schema'] == 'fibernet-linear-conductance-1'
