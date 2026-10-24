"""Pure export-contract checks; no native runtime or physics steps."""
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from fibernet.sim import accelerated as m


def load(tmp_path, payload):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(payload))
    return m.SimResult.load(path)


def mechanical():
    return dict(mode='dynamics', energy=1., n_nodes=2, n_edges=1,
                displacements=[[0., 0., 0.], [1., 0., 0.]],
                deformed_positions=[[0., 0., 0.], [2., 0., 0.]],
                edge_forces=[1.], edge_stretches=[2.])


@pytest.mark.parametrize('payload', [{}, [], None, 3, {'hello': 'world'},
    {'mode': 'fem_beam_frame', 'reactions': [[0, 0, 0]]},
    {'schema': 'fibernet.viscoelastic_result/1', 'displacements': [[[0, 0, 0]]]},
    {'schema': 'fibernet-linear-conductance-1'}, {'schema': 'unknown/1'}])
def test_reject_wrong_or_unrecognizable_result(tmp_path, payload):
    with pytest.raises(ValueError):
        load(tmp_path, payload)


@pytest.mark.parametrize('field,value', [
    ('displacements', [[[0, 0, 0], [0, 0, 0]]]),
    ('displacements', [[0, 0], [0, 0]]),
    ('displacements', [[0, 0, 0]]), ('displacements', [[True, 0, 0], [0, 0, 0]]),
    ('deformed_positions', [[0, 0, 0, 0], [0, 0, 0, 0]]),
    ('edge_forces', [[1, 2, 3]]), ('edge_stretches', [1, 2]),
    ('positions_trajectory', [[[0, 0, 0]]]), ('positions_trajectory', {}),
    ('n_nodes', 0), ('n_nodes', True), ('n_edges', -1), ('n_edges', 1.),
    ('energy', float('nan')), ('energy', '1'), ('mode', None),
    ('node_ids', [7, 7]), ('edge_ids', []), ('history', [1]),
])
def test_reject_tampered_mechanical_arrays(tmp_path, field, value):
    payload = mechanical(); payload[field] = value
    with pytest.raises(ValueError):
        load(tmp_path, payload)


def test_schema_less_summary_and_detailed_roundtrip(tmp_path):
    result = load(tmp_path, mechanical())
    for detailed in (False, True):
        payload = result.to_dict(detailed)
        assert 'schema' not in payload
        restored = load(tmp_path, payload)
        assert restored.to_dict(detailed) == payload
        assert (restored.displacements is not None) == detailed


def test_legacy_fem_resultants_remain_supported(tmp_path):
    payload = mechanical(); payload.update(mode='fem_beam_frame', edge_forces=[[1, 2, 3]])
    result = load(tmp_path, payload)
    assert result.edge_forces.tolist() == [1]
    assert result.metadata['beam_resultants'] == [[1, 2, 3]]


def test_retained_public_material_summary_without_rewrite(tmp_path):
    path = Path(__file__).parent / 'fixtures/dynamic_viscoelastic_generalized_summary.json'
    original = path.read_bytes()
    result = m.SimResult.load(path)
    for detailed in (False, True):
        payload = result.to_dict(detailed)
        assert load(tmp_path, payload).to_dict(detailed) == payload
    assert path.read_bytes() == original


def test_actual_electrical_export_is_not_mechanical(tmp_path):
    path = Path(__file__).parent / 'data/joule_torus/run01/torus_electrical.json'
    with pytest.raises(ValueError, match='Electrical'):
        m.SimResult.load(path)


@pytest.mark.parametrize('explicit', [False, True])
def test_dynamics_legacy_damping_warns_only_when_explicit(explicit):
    # Straight geometry preflight precedes the legacy warning and observer check.
    graph = SimpleNamespace(edges={17: SimpleNamespace(internal_points=None)})
    kwargs = {'damping': .3} if explicit else {}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        with pytest.raises(TypeError, match='state_observer'):
            m.TaichiEngine.dynamics(object(), graph, state_observer=1, **kwargs)
    assert len(seen) == int(explicit)
    if explicit:
        assert 'metadata-only' in str(seen[0].message)


@pytest.mark.parametrize('explicit', [False, True])
def test_stretch_preserves_omission_and_active_defaults(monkeypatch, explicit):
    graph = SimpleNamespace(edges={17: SimpleNamespace(internal_points=None)})
    captured = {}
    class Engine:
        def dynamics(self, graph, **kwargs):
            captured.update(kwargs)
            return m._legacy_damping(kwargs['damping'])
    monkeypatch.setattr(m, '_graph_to_arrays', lambda g: (np.array([[0., 0., 0.], [1., 0., 0.]]), np.array([[0, 1]]), None, None))
    monkeypatch.setattr(m, '_get_boundary_indices', lambda p: {'left': [0], 'right': [1]})
    kwargs = {'damping': .6} if explicit else {}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        value = m.TaichiEngine.stretch_test(Engine(), graph, num_steps=1, **kwargs)
    assert value == (.6 if explicit else .3)
    assert len(seen) == int(explicit)
    assert 'dashpot' not in captured and 'drag' not in captured
    assert captured['geometry_policy'] == 'reject_internal_points'


def test_legacy_fem_unavailable_stretches_stay_empty(tmp_path):
    payload = mechanical(); payload.update(mode='fem_beam_frame', edge_stretches=[])
    result = load(tmp_path, payload)
    assert result.edge_stretches.tolist() == []
    assert result.to_dict(True)['edge_stretches'] == []


def test_raw_fem_producer_empty_stretches_roundtrip(tmp_path):
    from fibernet.ml.beam_frame_fem import BeamFrameFEM
    raw = dict(u=np.zeros((2, 3)), n_nodes=2, n_edges=1,
               edge_list=np.array([0]), edge_forces=np.array([[1., 2., 3.]]))
    result = BeamFrameFEM().to_sim_result(raw)
    assert result.edge_stretches.tolist() == []
    path = tmp_path / 'raw.json'
    result.save(path, detailed=True)
    restored = m.SimResult.load(path)
    assert restored.edge_stretches.tolist() == []
    assert restored.n_edges == 1
    payload = json.loads(path.read_text()); payload['edge_stretches'] = [1., 1.]
    with pytest.raises(ValueError, match='n_edges'):
        load(tmp_path, payload)
