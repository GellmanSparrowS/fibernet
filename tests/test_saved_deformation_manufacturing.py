"""Saved geometry-to-export regression, with fabricated (not solved) deformation.

The result is not a graph/restart file. Keep the native graph separately, map
saved rows by node ID, and edit a detached graph through the existing API.
No physical simulation, training, or fabrication qualification is performed.
"""
import copy
import csv
import hashlib
import json
import math

import numpy as np
import pytest

from fibernet import StructureGraph, manufacturing_trails_from_graph, export_mapped_trails
from fibernet.analysis import analyze_geometry_paths, analyze_topology
from fibernet.sim.accelerated import SimResult


def _straight_spatial_fixture():
    """64 nodes, 144 lattice connections, and two reverse parallel fibers."""
    nodes, edges = [], []
    node_id = lambda x, y, z: 101 + 7 * (16 * x + 4 * y + z)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                nid = node_id(x, y, z)
                nodes.append({'id': nid, 'position': [.01*x, .02*y, .03*z]})
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    if max(x+dx, y+dy, z+dz) < 4:
                        edges.append({'id': 1001 + 11*len(edges), 'node_i': nid,
                                      'node_j': node_id(x+dx, y+dy, z+dz),
                                      'radius': .0001})
    first = edges[0]
    for _ in range(2):
        edges.append({'id': 1001 + 11*len(edges), 'node_i': first['node_j'],
                      'node_j': first['node_i'], 'radius': .0002})
    graph = StructureGraph.from_dict({'version': '3.0', 'dimension': 3,
                                     'nodes': nodes, 'edges': edges})
    graph.metadata.update(length_unit='m', coordinate_frame='synthetic_xyz')
    for index, edge in enumerate(graph.edges.values()):
        edge.material.name = 'synthetic-parallel' if index >= 144 else 'synthetic-grid'
        edge.metadata['fixture_edge_id'] = edge.edge_id
    return graph


@pytest.mark.parametrize('node_shift,edge_shift', [(1, 17), (23, 1)])
def test_saved_deformation_to_mapped_trails(tmp_path, node_shift, edge_shift):
    graph = _straight_spatial_fixture()
    original = copy.deepcopy(graph.to_dict())
    node_ids, edge_ids = sorted(graph.nodes), sorted(graph.edges)
    # Independent permutations, neither matching the saved graph's sorted rows.
    node_ids = node_ids[node_shift:] + node_ids[:node_shift]
    edge_ids = edge_ids[edge_shift:] + edge_ids[:edge_shift]
    positions = np.array([graph.nodes[nid].position for nid in node_ids])
    # Deterministic affine geometry only; NOT a predicted mechanical response.
    deformed = positions @ np.array([[1.1, .1, .05], [.2, .9, .1], [.1, .15, 1.2]])
    deformed += [.001, -.002, .003]
    result = SimResult(mode='fabricated_geometry_fixture', n_nodes=64, n_edges=146,
                       displacements=deformed-positions, deformed_positions=deformed,
                       metadata={'node_ids': node_ids, 'edge_ids': edge_ids,
                                 'fixture_scope': 'fabricated deformation; no solver'})
    graph_path, result_path = tmp_path/'graph.json', tmp_path/'result.json'
    graph.save_json(str(graph_path))
    result.save(str(result_path), detailed=True)
    saved_bytes = (graph_path.read_bytes(), result_path.read_bytes())
    restored = StructureGraph.load_json(str(graph_path))
    saved_result = SimResult.load(str(result_path))
    before_result = copy.deepcopy(saved_result.to_dict(detailed=True))
    assert saved_result.metadata['node_ids'] == node_ids
    assert saved_result.metadata['edge_ids'] == edge_ids
    assert list(restored.nodes) != node_ids
    assert list(restored.edges) != edge_ids
    assert set(node_ids) == set(restored.nodes) and len(node_ids) == 64
    assert set(edge_ids) == set(restored.edges) and len(edge_ids) == 146
    np.testing.assert_array_equal(saved_result.deformed_positions, deformed)

    # The cross-journey handoff: never zip graph iteration order with result rows.
    by_id = {nid: saved_result.deformed_positions[row].copy()
             for row, nid in enumerate(saved_result.metadata['node_ids'])}
    assert any(not np.array_equal(by_id[nid], saved_result.deformed_positions[row])
               for row, nid in enumerate(restored.nodes))  # positional zip is wrong
    deformed_graph = restored.copy()
    deformed_graph.set_node_positions(by_id)
    for nid in restored.nodes:
        np.testing.assert_array_equal(deformed_graph.nodes[nid].position, by_id[nid])
    assert deformed_graph.to_dict()['edges'] == original['edges']
    assert all(edge.internal_points is None for edge in deformed_graph.edges.values())

    topology = analyze_topology(deformed_graph)
    assert (topology.n_nodes, topology.n_physical_edges,
            topology.n_unique_connections, topology.n_components) == (64, 146, 144, 1)
    assert sum(degree >= 3 for degree in topology.degree_physical_by_node) == 64
    geometry = analyze_geometry_paths(deformed_graph, length_unit='m',
                                      coordinate_frame='synthetic_xyz')
    expected_lengths = []
    for eid in geometry.edge_ids:
        edge = restored.edges[eid]
        expected_lengths.append(math.dist(by_id[edge.node_i], by_id[edge.node_j]))
    assert geometry.chord_length_by_edge == pytest.approx(expected_lengths)
    assert geometry.polyline_length_by_edge == pytest.approx(expected_lengths)
    assert not geometry.degenerate_edge_ids
    assert (geometry.length_unit, geometry.coordinate_frame) == ('m', 'synthetic_xyz')

    trails, snapshot = manufacturing_trails_from_graph(
        deformed_graph, graph_id='saved-fabricated-deformation',
        length_unit='m', coordinate_frame='synthetic_xyz')
    csv_path, sidecar_path = tmp_path/'trails.csv', tmp_path/'trails.json'
    mapping = export_mapped_trails(csv_path, trails, snapshot, sidecar_path=sidecar_path)
    assert json.loads(sidecar_path.read_text()) == mapping
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == mapping['route_csv_sha256']
    assert (mapping['source_length_unit'], mapping['route_length_unit']) == ('m', 'mm')
    assert mapping['conversion']['factor'] == 1000.
    payload = mapping['source_snapshot']['graph_batch']
    for row, nid in enumerate(payload['node_ids']):
        assert payload['positions'][row] == by_id[nid].tolist()
    for row, eid in enumerate(payload['edge_ids']):
        edge = restored.edges[eid]
        assert payload['edge_features'][row] == [edge.radius]
        assert payload['edge_metadata'][row] == edge.metadata
        assert payload['provenance']['edge_material_records'][row] == edge.material.to_dict()
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == mapping['route_row_count'] == 146 + trails.stroke_count
    visited, previous = [], {}
    for row in rows:
        stroke, step, nid = (int(row[key]) for key in ('stroke', 'step', 'node_id'))
        assert nid == int(snapshot.node_ids[int(row['node_row'])])
        # Exact binary64 round trip, anchored to saved deformation, not snapshot alone.
        assert [float(row[key]) for key in ('x_mm', 'y_mm', 'z_mm')] == [float(v)*1000. for v in by_id[nid]]
        if step == 0:
            assert [int(row[key]) for key in ('incoming_edge_row', 'incoming_edge_id',
                                            'source_edge_direction')] == [-1, -1, 0]
        else:
            eid, direction = int(row['incoming_edge_id']), int(row['source_edge_direction'])
            assert eid == int(snapshot.edge_ids[int(row['incoming_edge_row'])])
            edge = restored.edges[eid]
            assert direction in (-1, 1)
            endpoints = [edge.node_i, edge.node_j]
            assert [previous[stroke], nid] == (endpoints if direction == 1 else endpoints[::-1])
            visited.append(eid)
        previous[stroke] = nid
    assert sorted(visited) == sorted(restored.edges)  # every physical edge once
    assert graph.to_dict() == restored.to_dict() == original
    assert saved_result.to_dict(detailed=True) == before_result
    assert (graph_path.read_bytes(), result_path.read_bytes()) == saved_bytes


def test_saved_curve_is_not_silently_flattened_for_manufacturing(tmp_path):
    graph = _straight_spatial_fixture()
    edge = graph.edges[min(graph.edges)]
    midpoint = (graph.nodes[edge.node_i].position + graph.nodes[edge.node_j].position)/2
    edge.internal_points = np.array([midpoint + [0., .001, .002]])
    path = tmp_path/'curved.json'
    graph.save_json(str(path))
    restored = StructureGraph.load_json(str(path))
    before = copy.deepcopy(restored.to_dict())
    with pytest.raises(ValueError, match='internal-point geometry'):
        manufacturing_trails_from_graph(restored, graph_id='curve-refusal',
                                        length_unit='m', coordinate_frame='synthetic_xyz')
    assert restored.to_dict() == before
    np.testing.assert_array_equal(restored.edges[edge.edge_id].internal_points,
                                  edge.internal_points)
