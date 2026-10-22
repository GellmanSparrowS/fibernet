"""Public one-way Joule API integration on a tiny analytic resistor graph."""
import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import fibernet as fn
from fibernet.sim import joule_heating
from fibernet.sim.transient_thermal import ThermalSolveError


def fixture():
    graph = fn.StructureGraph.from_dict({
        'dimension': 2,
        'nodes': [{'id': 101, 'position': [0, 0, 0]},
                  {'id': 907, 'position': [1, 0, 0]}],
        'edges': [{'id': 11, 'node_i': 101, 'node_j': 907},
                  {'id': 31, 'node_i': 907, 'node_j': 101},
                  {'id': 67, 'node_i': 101, 'node_j': 907}],
    })
    electrical = fn.solve_linear_transport(
        graph, {11: 1., 31: 3., 67: 0.}, {101: 2., 907: 0.}, length_unit='m')
    weights = {eid: {101: .25, 907: .75} for eid in graph.edges}
    return graph, electrical, weights


def thermal_inputs(weights):
    return dict(capacity_J_per_K={101: 2., 907: 3.},
                conductance_W_per_K={11: .2, 31: .3, 67: 0.},
                initial_temperature_K={101: 300., 907: 300.},
                times_s=[0., .1], endpoint_weights=weights, length_unit='m')


def test_root_exports_match_production_module():
    for name in ('JouleHeatingLoad', 'JouleHeatingRun',
                 'assemble_joule_heating', 'run_joule_heating'):
        assert getattr(fn, name) is getattr(joule_heating, name)
        assert name in fn.__all__


def test_actual_one_interval_matches_analytic_backward_euler():
    graph, electrical, weights = fixture()
    graph_before = copy.deepcopy(graph.to_dict())
    electrical_before = electrical.to_dict()
    kwargs = thermal_inputs(weights)
    inputs_before = copy.deepcopy(kwargs)
    result = fn.run_joule_heating(graph, electrical, **kwargs)
    assert isinstance(result, fn.JouleHeatingRun)
    assert result.heat_load.joule_source_W == {101: 4., 907: 12.}
    # For C=(2,3), g=.5, dt=.1: mean T=300.32 and T101-T907=-.192.
    np.testing.assert_allclose(result.thermal.temperature_K[-1],
                               [300.2048, 300.3968], rtol=0., atol=2e-12)
    assert float(result.thermal.node_stored_heat_change_J[-1].sum()) == pytest.approx(1.6, abs=2e-12)
    assert graph.to_dict() == graph_before
    assert electrical.to_dict() == electrical_before
    assert kwargs == inputs_before


def test_public_delegation_preserves_reservoir_sources_and_errors():
    graph, electrical, weights = fixture()
    kwargs = thermal_inputs(weights)
    kwargs['fixed_temperature_K'] = {907: 300.}
    sentinel = object()
    with patch('fibernet.sim.transient_thermal.solve_transient_thermal', return_value=sentinel) as solve:
        result = fn.run_joule_heating(graph, electrical, **kwargs)
        assert result.thermal is sentinel
        solve.assert_called_once()
        assert solve.call_args.kwargs['source_W'] == {101: 4., 907: 12.}
        assert solve.call_args.kwargs['fixed_temperature_K'] is kwargs['fixed_temperature_K']
    failure = ThermalSolveError('test-only thermal failure', None, 1, .1)
    with patch('fibernet.sim.transient_thermal.solve_transient_thermal', side_effect=failure):
        with pytest.raises(ThermalSolveError) as caught:
            fn.run_joule_heating(graph, electrical, **kwargs)
        assert caught.value is failure


def test_example_import_performs_no_solve():
    path = Path(__file__).resolve().parents[1] / 'examples/one_way_joule_heating.py'
    spec = importlib.util.spec_from_file_location('_joule_example_import', path)
    example = importlib.util.module_from_spec(spec)
    with patch.object(fn, 'solve_linear_transport', side_effect=AssertionError('unexpected solve')):
        spec.loader.exec_module(example)
    assert callable(example.run_example)


def test_strict_json_reopen_joule_workflow(tmp_path):
    import json
    graph, electrical, weights = fixture()
    kwargs = thermal_inputs(weights)
    kwargs['fixed_temperature_K'] = {907: 300.}
    heating = fn.run_joule_heating(graph, electrical, **kwargs)
    path = tmp_path / 'joule_heating.json'
    path.write_text(json.dumps({
        'electrical': electrical.to_dict(),
        'heat_load': heating.heat_load.to_dict(),
        'thermal': heating.thermal.to_dict(),
    }, allow_nan=False), encoding='utf-8')
    # Reopening and consuming saved fields is not another physical solve.
    with patch.object(fn, 'solve_linear_transport', side_effect=AssertionError('unexpected electrical solve')), \
            patch('fibernet.sim.transient_thermal.solve_transient_thermal',
                  side_effect=AssertionError('unexpected thermal solve')):
        reopened = json.loads(path.read_text(encoding='utf-8'))
        thermal = reopened['thermal']
        assert thermal['completed'] is True
        assert thermal['input_sha256'] == heating.thermal.input_sha256
        assert thermal['node_ids'] == reopened['electrical']['node_ids']
        assert thermal['edge_ids'] == [11, 31, 67]
        assert thermal['endpoints_by_id'] == [[101, 907], [907, 101], [101, 907]]
        assert thermal['source_W'] == reopened['heat_load']['total_node_power_W']
        assert thermal['interval_diagnostics_available'] == [False, True]
        assert thermal['free_node_residual_J'][0] == [None, None]
        assert thermal['free_node_residual_J'][1][1] is None
        assert thermal['provenance']['units']['stored'] == 'J'
        stored = sum(thermal['node_stored_heat_change_J'][-1])
        supplied = .1 * (sum(thermal['source_W']) + sum(thermal['reservoir_injection_W'][-1]))
        assert abs(stored - supplied) <= sum(thermal['component_acceptance_bound_J'][-1])
        np.testing.assert_array_equal(thermal['temperature_K'], heating.thermal.temperature_K)
