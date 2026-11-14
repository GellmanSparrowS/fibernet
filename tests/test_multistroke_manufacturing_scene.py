"""Generation-to-analysis-to-export workflow using the existing example."""
import importlib.util
import json
from pathlib import Path

import pytest


def test_spatial_lattice_multistroke_workflow(tmp_path):
    # The example belongs to the source checkout; fibernet imports must resolve
    # normally (the installed gate separately verifies their installed origins).
    path = Path(__file__).resolve().parents[1] / 'examples' / 'multistroke_manufacturing.py'
    spec = importlib.util.spec_from_file_location('multistroke_manufacturing_example', path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    output = tmp_path / 'scene'
    mapping = example.main(output)
    report = json.loads((output / 'analysis.json').read_text())
    source = json.loads((output / 'source_graph.json').read_text())
    assert len(source['nodes']) == 4 * 3 * 3
    assert len(source['edges']) == 3 * 3 * 3 + 4 * 2 * 3 + 4 * 3 * 2 + 2
    assert report['topology']['n_unique_connections'] == 75
    assert report['topology']['cycle_rank_physical'] == 42
    assert mapping['stroke_count'] == 9 and mapping['route_row_count'] == 86
    assert set(node['id'] for node in source['nodes']) == {101 + 7 * i for i in range(36)}
    assert set(edge['id'] for edge in source['edges']) == {1001 + 11 * i for i in range(77)}
    materials = [edge['material']['name'] for edge in source['edges']]
    assert materials.count('synthetic-A') == 75 and materials.count('synthetic-B') == 2
    parallel = [e for e in source['edges'] if 'synthetic_parallel' in e['metadata']]
    assert len(parallel) == 2
    assert parallel[0]['node_i'] == parallel[1]['node_i']
    assert parallel[0]['node_j'] == parallel[1]['node_j']
    assert any(e['node_i'] == parallel[0]['node_j'] and e['node_j'] == parallel[0]['node_i']
               for e in source['edges'] if e not in parallel)
    assert {e['radius'] for e in source['edges']} == {.0001, .0002}
    # Fresh output admission must not damage an already completed export.
    saved = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        example.main(output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == saved
