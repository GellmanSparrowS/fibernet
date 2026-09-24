"""Verify the documented public API journey and its saved artifacts."""
import csv
import json

import pytest

import fibernet as fn
from examples.open_source_workflow import OpenSourceWorkflow


def test_documented_workflow(tmp_path):
    summary = OpenSourceWorkflow(tmp_path).run()
    assert summary['nodes'] == 18
    assert summary['edges'] == 22
    assert 0 <= summary['recruited_edge_fraction'] <= 1
    assert (tmp_path / 'network.png').stat().st_size > 1000
    assert fn.SimResult.load(tmp_path / 'fem_result.json').max_force > 0
    assert json.loads((tmp_path / 'summary.json').read_text()) == summary


def test_batch_fem_columns_and_invalid_backend(tmp_path):
    graph = fn.pattern_2d(unit='honeycomb', grid=(2, 2))
    with pytest.raises(ValueError, match='backend'):
        fn.simulate(graph, backend='missing')
    output = tmp_path / 'results.csv'
    fn.batch_simulate([{'unit': 'honeycomb', 'grid': (2, 2)}],
                      str(output), strain=1.01)
    with output.open(newline='') as file:
        row = next(csv.DictReader(file))
    assert float(row['max_force']) > 0
    assert float(row['max_displacement']) > 0
    assert 'E_star' not in row
