"""Generate, analyze and export a synthetic mixed-material spatial lattice.

    python -m examples.multistroke_manufacturing NEW_OUTPUT_DIRECTORY

The 3 x 2 x 2 cubic-cell lattice contains 36 nodes and 75 connections. Two
additional reverse parallel fibers give 77 physical edges. Nine edge-once
strokes preserve every physical edge and its source identity. Coordinates and
radii are stored in metres; route CSV coordinates are millimetres. Material
records are provenance, not calibrated solver or fabrication parameters.
"""
import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from fibernet import (Material, StructureGraph, pattern_3d,
                      manufacturing_trails_from_graph, export_mapped_trails)
from fibernet.analysis import analyze_topology, analyze_geometry_paths


def build_graph():
    """Build the synthetic scene using existing generation and record APIs."""
    first = Material(name='synthetic-A', youngs_modulus=1e9, density=1000.)
    second = Material(name='synthetic-B', youngs_modulus=2e9, density=1200.)
    graph = pattern_3d(unit='cubic', grid=(3, 2, 2), box=(.01, .02, .03),
                       n_internal=0, radius=.0001, material=first)
    graph.metadata.update(length_unit='m', coordinate_frame='lab_xyz',
                          description='Synthetic mixed-material cubic-cell lattice')
    # Select an X-directed physical edge without relying on generator row order.
    edge = next(e for e in graph.edges.values()
                if graph.nodes[e.node_i].position[1:].tolist()
                == graph.nodes[e.node_j].position[1:].tolist())
    for index in range(2):
        graph.add_edge(edge.node_j, edge.node_i, radius=.0002, material=second,
                       n_internal=0, allow_parallel=True, synthetic_parallel=index)
    record = graph.to_dict()
    node_ids = {node['id']: 101 + 7 * i for i, node in enumerate(record['nodes'])}
    for node in record['nodes']:
        node['id'] = node_ids[node['id']]
    for i, item in enumerate(record['edges']):
        item.update(id=1001 + 11 * i, node_i=node_ids[item['node_i']],
                    node_j=node_ids[item['node_j']])
    record['next_node_id'] = max(node_ids.values()) + 1
    record['next_edge_id'] = max(e['id'] for e in record['edges']) + 1
    return StructureGraph.from_dict(record)


def main(output_directory):
    output = Path(output_directory)
    output.mkdir()  # Fresh directory: never overwrite an earlier export.
    graph = build_graph()
    before = copy.deepcopy(graph.to_dict())
    material_objects = {eid: edge.material for eid, edge in graph.edges.items()}
    topology = analyze_topology(graph)
    geometry = analyze_geometry_paths(graph, length_unit='m', coordinate_frame='lab_xyz')
    assert (topology.n_nodes, topology.n_physical_edges,
            topology.n_unique_connections, topology.n_components) == (36, 77, 75, 1)
    assert sum(topology.degree_physical_by_node) == 154
    assert sum(d % 2 for d in topology.degree_physical_by_node) == 18
    assert topology.cycle_rank_physical == 42
    assert geometry.edge_ids == topology.edge_ids
    assert geometry.length_unit == 'm' and geometry.coordinate_frame == 'lab_xyz'
    assert not geometry.degenerate_edge_ids
    assert math.isclose(math.fsum(geometry.polyline_length_by_edge), 1.49,
                        rel_tol=1e-12, abs_tol=1e-15)
    source_path = output / 'source_graph.json'
    graph.save_json(str(source_path))
    assert StructureGraph.load_json(str(source_path)).to_dict() == before
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    trails, snapshot = manufacturing_trails_from_graph(
        graph, graph_id='synthetic-cubic-multistroke', length_unit='m',
        coordinate_frame='lab_xyz')
    assert trails.stroke_count == 9
    assert sorted(trails.route_edges.tolist()) == list(range(77))
    assert set(snapshot.node_ids.tolist()) == set(graph.nodes)
    assert set(snapshot.edge_ids.tolist()) == set(graph.edges)
    mapping = export_mapped_trails(output / 'trails.csv', trails, snapshot,
                                   sidecar_path=output / 'trails.json',
                                   source_input_sha256=source_sha256)
    assert hashlib.sha256((output / 'trails.csv').read_bytes()).hexdigest() == mapping['route_csv_sha256']
    assert json.loads((output / 'trails.json').read_text()) == mapping
    assert mapping['source_input_sha256'] == source_sha256
    assert mapping['source_input_sha256_origin'] == 'caller_asserted_not_file_verified'
    assert mapping['conversion']['factor'] == 1000.
    assert (mapping['source_length_unit'], mapping['route_length_unit']) == ('m', 'mm')
    payload = mapping['source_snapshot']['graph_batch']
    assert payload['node_ids'] == snapshot.node_ids.tolist()
    assert payload['edge_ids'] == snapshot.edge_ids.tolist()
    assert payload['positions'] == snapshot.positions.tolist()
    for row, eid in enumerate(snapshot.edge_ids.tolist()):
        edge = graph.edges[eid]
        assert payload['provenance']['edge_material_records'][row] == edge.material.to_dict()
        assert payload['edge_features'][row] == [edge.radius]
        assert payload['edge_metadata'][row] == edge.metadata
    with (output / 'trails.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == mapping['route_row_count'] == 86
    physical_rows, previous_by_stroke = [], {}
    for row in rows:
        stroke, step, node = (int(row[k]) for k in ('stroke', 'step', 'node_row'))
        assert int(row['node_id']) == int(snapshot.node_ids[node])
        assert [float(row[k]) for k in ('x_mm', 'y_mm', 'z_mm')] == [float(v) * 1000. for v in snapshot.positions[node]]
        if step == 0:
            assert [int(row[k]) for k in ('incoming_edge_row', 'incoming_edge_id', 'source_edge_direction')] == [-1, -1, 0]
        else:
            edge_row = int(row['incoming_edge_row'])
            physical_rows.append(edge_row)
            assert int(row['incoming_edge_id']) == int(snapshot.edge_ids[edge_row])
            endpoints = snapshot.edge_index[:, edge_row].tolist()
            traversal = [previous_by_stroke[stroke], node]
            direction = int(row['source_edge_direction'])
            assert direction in (-1, 1)
            assert traversal == (endpoints if direction == 1 else endpoints[::-1])
        previous_by_stroke[stroke] = node
    assert sorted(physical_rows) == list(range(77))
    assert graph.to_dict() == before
    assert all(graph.edges[eid].material is material for eid, material in material_objects.items())
    report = {'topology': asdict(topology), 'geometry': asdict(geometry),
              'source_file_sha256': source_sha256,
              'physical_edges': 77, 'strokes': 9, 'csv_rows': 86,
              'scope': 'synthetic generation, topology, geometry and edge-once export'}
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print('36 nodes, 77 physical fibers, 9 strokes; source m to route mm')
    print(f'Graph, analysis, route CSV and mapping: {output}')
    print('Process-specific travel and fabrication planning are separate steps')
    return mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', help='A new directory whose parent already exists')
    main(parser.parse_args().output_directory)
