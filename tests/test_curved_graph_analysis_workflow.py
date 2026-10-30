"""One synthetic full-XYZ native graph workflow; no physical model or training."""
import importlib.util
import json
import math
from pathlib import Path

import pytest


def test_curved_graph_analysis_native_roundtrip(tmp_path):
    path = Path(__file__).resolve().parents[1]/'examples/curved_graph_analysis.py'
    spec = importlib.util.spec_from_file_location('curved_graph_analysis_example', path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    output = tmp_path/'curved-scene'
    report = example.run(output)
    assert (output/'source_graph.json').read_bytes() == (output/'restored_graph.json').read_bytes()
    graph = json.loads((output/'source_graph.json').read_text())
    assert graph['version'] == '3.1' and graph['dimension'] == 3
    assert [n['id'] for n in graph['nodes']] == [10, 30, 70, 110, 190]
    assert [e['id'] for e in graph['edges']] == [101, 205, 309, 407, 503, 601]
    assert (graph['next_node_id'], graph['next_edge_id']) == (200, 700)
    edges = {e['id']: e for e in graph['edges']}
    assert [(edges[e]['node_i'], edges[e]['node_j']) for e in (101, 205)] == [(10, 30), (30, 10)]
    assert edges[101]['internal_points'] == [[1, 0, 1]]
    assert edges[205]['internal_points'] == [[1, 0, -2]]
    assert edges[407]['internal_points'] == [[1, 2, 2]]
    for eid, kind in [(101,0),(205,1),(309,0),(407,1),(503,0),(601,1)]:
        edge = edges[eid]
        assert edge['radius'] == .01+.002*kind
        assert edge['material']['name'] == ('synthetic-A', 'synthetic-B')[kind]
        assert edge['material']['youngs_modulus'] == (2e6, 5e6)[kind]
        assert edge['material']['density'] == (900, 1100)[kind]
        assert edge['segments'] == 4 and edge['metadata']['label'] == 'edge-'+str(eid)
    topology = report['topology']
    assert (topology['n_nodes'], topology['n_physical_edges'], topology['n_unique_connections'],
            topology['n_components'], topology['n_isolates'], topology['cycle_rank_physical']) == (5,6,5,2,1,3)
    assert sum(topology['degree_physical_by_node']) == 12
    poly, chord = report['polyline'], report['chord']
    expected_chords = [2,2,math.sqrt(8),math.sqrt(8),2,math.sqrt(12)]
    expected_polylines = [2*math.sqrt(2),2*math.sqrt(5),math.sqrt(8),1+math.sqrt(5),2,math.sqrt(12)]
    assert list(poly['edge_ids']) == [101,205,309,407,503,601]
    assert list(poly['node_ids']) == [10,30,70,110,190]
    assert poly['chord_length_by_edge'] == pytest.approx(expected_chords)
    assert poly['polyline_length_by_edge'] == pytest.approx(expected_polylines)
    assert poly['curve_straightness_by_edge'] == pytest.approx([a/b for a,b in zip(expected_chords,expected_polylines)])
    assert poly['pair_distance'][:3] == pytest.approx([2*math.sqrt(2),math.sqrt(12),2+2*math.sqrt(2)])
    assert chord['pair_distance'][:3] == pytest.approx([2,math.sqrt(12),4])
    assert poly['pair_distance'][3:] == (None, 0)
    assert poly['pair_status'] == ('ok','ok','ok','unreachable','zero_chord')
    assert poly['pair_tortuosity'][3:] == (None, None)
    assert poly['length_unit'] == 'm' and poly['coordinate_frame'] == 'synthetic XYZ'
    assert 'material' not in poly  # Materials are checked in graph schema, not metrics.
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        example.run(output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
