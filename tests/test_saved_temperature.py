"""Saved-temperature binding contracts; every mechanics call is a test double."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.saved_temperature import (prepare_saved_temperature,
                                            read_saved_thermal_json)


def fixture():
    graph = StructureGraph.from_dict({'dimension': 2, 'nodes': [
        {'id': 91, 'position': [.1, 0., 0.]}, {'id': 7, 'position': [0., 0., 0.]}],
        'edges': [{'id': 83, 'node_i': 91, 'node_j': 7, 'radius': .001},
                  {'id': 11, 'node_i': 7, 'node_j': 91, 'radius': .002}]})
    data = {'node_ids': [91, 7], 'edge_ids': [11, 83],
        'endpoints_by_id': [[7, 91], [91, 7]],
        'positions': [[.1, 0., 0.], [0., 0., 0.]],
        'times_s': [0., 1.], 'temperature_K': [[300., 300.], [301., 302.]],
        'completed': True, 'input_sha256': 'a'*64,
        'provenance': {'model': 'constant_lumped_thermal', 'length_unit': 'm',
                       'units': {'T': 'K'}},
        'free_node_residual_J': [[None, None], [0., 0.]]}
    return graph, data


def prepared(graph, data, **kw):
    return prepare_saved_temperature(graph, data, frame_index=1, length_unit='m', **kw)


def test_detached_order_and_explicit_delegation():
    graph, data = fixture(); original = copy.deepcopy(data)
    snap = prepared(graph, data)
    assert snap.mechanical_kwargs(graph) == {'length_unit': 'm', 'temperature_K': {91: 301., 7: 302.}}
    assert snap.to_dict()['edge_ids'] == [11, 83]
    data['temperature_K'][1][0] = 999.
    detached = snap.to_dict(); detached['temperature_K'][0] = 777.
    with patch('fibernet.sim._thermoelastic.solve_thermoelastic_network_snapshot',
               return_value=SimpleNamespace(to_dict=lambda: {'profile': 'stub'})) as owner:
        out = snap.solve(graph, profile='network', E_Pa=1e6,
            expansion_alpha_per_K={11: 1e-5, 83: 1e-5},
            stress_free_temperature_K=300., constraints={7: (0., 0., 0.)})
    assert owner.call_count == 1
    assert owner.call_args.kwargs['temperature_K'] == {91: 301., 7: 302.}
    assert out['feedback_to_electrical_or_thermal'] is False
    assert out['execution'] == 'new_mechanical_solve_from_saved_temperature'
    assert out['temperature_source']['source_completed'] is True
    assert original['temperature_K'][1] == [301., 302.]


@pytest.mark.parametrize('kind', ['position', 'radius', 'endpoint', 'internal', 'dimension'])
def test_graph_mutation_rejected_before_owner(kind):
    graph, data = fixture(); snap = prepared(graph, data)
    if kind == 'position': graph.nodes[7].position[0] += 1e-8
    if kind == 'radius': graph.edges[11].radius *= 2
    if kind == 'endpoint': graph.edges[11].node_i = 91
    if kind == 'internal': graph.edges[11].internal_points = np.array([[.05, .001, 0.]])
    if kind == 'dimension':
        replacement = graph.to_dict(); replacement['dimension'] = 3
        graph = StructureGraph.from_dict(replacement)
    with patch('fibernet.sim._thermoelastic.solve_thermoelastic_network_snapshot') as owner:
        with pytest.raises(ValueError, match='graph changed'):
            snap.solve(graph, profile='network', E_Pa=1e6,
                expansion_alpha_per_K={}, stress_free_temperature_K=300., constraints={})
        owner.assert_not_called()


@pytest.mark.parametrize('kind', ['duplicate_node', 'bool_node', 'duplicate_edge',
    'missing_edge', 'orientation', 'position', 'temperature_shape', 'null_temperature',
    'bool_temperature', 'zero_temperature', 'nonfinite', 'time_order', 'bool_time',
    'wrong_unit', 'wrong_model', 'bad_hash', 'missing_field', 'not_completed_bool'])
def test_invalid_transfer_rejected(kind):
    graph, data = fixture()
    if kind == 'duplicate_node': data['node_ids'] = [91, 91]
    if kind == 'bool_node': data['node_ids'][0] = True
    if kind == 'duplicate_edge': data['edge_ids'] = [11, 11]
    if kind == 'missing_edge': data['edge_ids'] = [11]
    if kind == 'orientation': data['endpoints_by_id'][0].reverse()
    if kind == 'position': data['positions'][0][0] += 1e-12
    if kind == 'temperature_shape': data['temperature_K'][0].pop()
    if kind == 'null_temperature': data['temperature_K'][0][0] = None
    if kind == 'bool_temperature': data['temperature_K'][1][0] = True
    if kind == 'zero_temperature': data['temperature_K'][1][0] = 0.
    if kind == 'nonfinite': data['temperature_K'][1][0] = float('inf')
    if kind == 'time_order': data['times_s'] = [1., 0.]
    if kind == 'bool_time': data['times_s'] = [False, 1.]
    if kind == 'wrong_unit': data['provenance']['units']['T'] = 'C'
    if kind == 'wrong_model': data['provenance']['model'] = 'unknown'
    if kind == 'bad_hash': data['input_sha256'] = 'bad'
    if kind == 'missing_field': del data['positions']
    if kind == 'not_completed_bool': data['completed'] = 1
    with pytest.raises(ValueError): prepared(graph, data)


@pytest.mark.parametrize('index', [-1, 2, True, 1., None])
def test_frame_requires_explicit_existing_row(index):
    graph, data = fixture()
    with pytest.raises(ValueError):
        prepare_saved_temperature(graph, data, frame_index=index, length_unit='m')


def test_partial_and_units_remain_explicit():
    graph, data = fixture(); data['completed'] = False
    with pytest.raises(ValueError, match='allow_partial'): prepared(graph, data)
    snap = prepared(graph, data, allow_partial=True)
    assert snap.to_dict()['source_completed'] is False
    with pytest.raises(ValueError): prepared(graph, data, allow_partial=1)
    for units in ('mm', 'cm', None):
        with pytest.raises(ValueError):
            prepare_saved_temperature(graph, data, frame_index=1,
                                      length_unit=units, allow_partial=True)


@pytest.mark.parametrize('raw', ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}',
                                '{"x": 1, "x": 2}', '{"a": {"x":1,"x":2}}', '[]'])
def test_strict_json_reader(tmp_path, raw):
    path = tmp_path/'saved.json'; path.write_text(raw)
    with pytest.raises(ValueError): read_saved_thermal_json(path)


def test_bundle_reopen_and_graph_order_independence(tmp_path):
    graph, data = fixture(); path = tmp_path/'saved.json'
    path.write_text(json.dumps({'thermal': data}, allow_nan=False))
    snap = prepared(graph, read_saved_thermal_json(path))
    nodes = list(reversed(list(graph.nodes.items())))
    edges = list(reversed(list(graph.edges.items())))
    graph.nodes.clear(); graph.nodes.update(nodes)
    graph.edges.clear(); graph.edges.update(edges)
    assert snap.mechanical_kwargs(graph)['temperature_K'] == {91: 301., 7: 302.}


def test_owner_error_propagates_without_partial_success():
    graph, data = fixture(); snap = prepared(graph, data)
    failure = ValueError('existing profile rejected graph')
    with patch('fibernet.sim._thermoelastic.solve_thermoelastic_snapshot', side_effect=failure):
        with pytest.raises(ValueError) as caught:
            snap.solve(graph, profile='elementary', E_Pa=1e6,
                expansion_alpha_per_K={}, stress_free_temperature_K=300., constraints={})
    assert caught.value is failure


def test_import_is_stdlib_only():
    import builtins
    path = Path(__file__).resolve().parents[1]/'fibernet/sim/saved_temperature.py'
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        assert not name.startswith(('numpy', 'scipy', 'torch', 'taichi', 'fibernet'))
        return original(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location('_temperature_import_only', path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, '__import__', side_effect=guard): spec.loader.exec_module(module)


def test_example_import_does_not_load_solver():
    import builtins
    path = Path(__file__).resolve().parents[1]/'examples/saved_thermal_mechanics.py'
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        assert not name.startswith(('fibernet', 'numpy', 'scipy'))
        return original(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location('_saved_temperature_example', path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, '__import__', side_effect=guard): spec.loader.exec_module(module)
    assert callable(module.main)


def test_example_inspection_and_explicit_solve(tmp_path, capsys):
    graph, data = fixture()
    spec = importlib.util.spec_from_file_location('_saved_temperature_example_run',
        Path(__file__).resolve().parents[1]/'examples/saved_thermal_mechanics.py')
    example = importlib.util.module_from_spec(spec); spec.loader.exec_module(example)
    graph.save_json(tmp_path/'graph.json')
    (tmp_path/'thermal.json').write_text(json.dumps(data, allow_nan=False))
    (tmp_path/'mechanics.json').write_text(json.dumps({'profile':'network','length_unit':'m',
        'E_Pa':1e6,'expansion_alpha_per_K':{'11':1e-5,'83':1e-5},
        'stress_free_temperature_K':300.,'constraints':{'7':[0.,0.,0.]}}))
    args=['--graph',str(tmp_path/'graph.json'),'--thermal',str(tmp_path/'thermal.json'),
          '--mechanics',str(tmp_path/'mechanics.json'),'--frame','1']
    with patch('fibernet.sim.saved_temperature.SavedTemperatureSnapshot.solve') as solve:
        assert example.main(args)==0;solve.assert_not_called()
    assert json.loads(capsys.readouterr().out)['temperature_K']==[301.,302.]
    target=tmp_path/'out.json'
    with patch('fibernet.sim.saved_temperature.SavedTemperatureSnapshot.solve',return_value={'stub':True}) as solve:
        assert example.main(args+['--solve','--output',str(target)])==0
        assert solve.call_count==1
        assert solve.call_args.kwargs['constraints']=={7:[0.,0.,0.]}
        assert solve.call_args.kwargs['expansion_alpha_per_K']=={11:1e-5,83:1e-5}
    with patch('fibernet.sim.saved_temperature.SavedTemperatureSnapshot.solve') as solve:
        with pytest.raises(SystemExit):example.main(args+['--solve','--output',str(target)])
        solve.assert_not_called()
    assert json.loads(target.read_text())=={'stub':True}
