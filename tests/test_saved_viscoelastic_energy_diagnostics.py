"""Saved-record algebra only; reuse provider and material-advancement tripwires."""
import copy
from fractions import Fraction
import json
import sys

import pytest

from test_dynamic_viscoelastic_persistence import api, _saved, SAVED


def expected(record):
    """Independent exact stored-number oracle, without production helpers."""
    rows = []
    initial = record['samples'][0]
    f = Fraction
    for state in record['samples']:
        totals = state['cumulative']
        c = [f(u)-f(u0)+f(p)+f(b)-f(w) for u, u0, p, b, w in zip(
            state['stored_energy_J'], initial['stored_energy_J'],
            totals['physical_dissipation_J'], totals['be_numerical_dissipation_J'],
            totals['endpoint_work_J'])]
        a = state['dynamic_cumulative']
        balance = (f(a['external_force_work_J']) + f(a['numerical_drag_energy_change_J'])
            + f(a['projection_kinetic_energy_change_J']) - f(a['projection_force_work_J'])
            + f(a['kick_drift_residual_J']) + f(a['old_stress_endpoint_residual_J']))
        d = f(state['kinetic_energy_J'])-f(initial['kinetic_energy_J']) + sum(map(f, totals['endpoint_work_J']))-balance
        rows.append(([float(value) for value in c], float(sum(c)), float(d), float(sum(c)+d)))
    return rows


def assert_curves(report, record):
    rows = expected(record)
    assert report['constitutive_cumulative_residual_J_by_edge'] == [r[0] for r in rows]
    assert report['constitutive_cumulative_residual_J'] == [r[1] for r in rows]
    assert report['dynamic_bookkeeping_cumulative_residual_J'] == [r[2] for r in rows]
    assert report['combined_cumulative_residual_J'] == [r[3] for r in rows]


def test_saved_curves_are_detached_sparse_observations(api):
    SimResult, _ = api
    raw, payload = _saved()
    result = SimResult.load(str(SAVED))
    before = result.to_dict(detailed=True)
    report = result.viscoelastic_energy_diagnostics()
    record = payload['native_axial_viscoelastic']
    assert_curves(report, record)
    assert report['steps'] == list(range(0, 257, 32))
    assert report['times_s'] == [s['time'] for s in record['samples']]
    assert report['edge_ids'] == [91]
    assert report['initial_kinetic_energy_J'] == 5e-7
    assert report['units'] == {'energy': 'J', 'time': 's'}
    assert report['coverage'] == {'record_validation': 'passed',
        'cumulative_consistency': 'not_certified', 'omitted_intervals': 'not_reconstructed'}
    assert report['constitutive_cumulative_residual_J_by_edge'][0] == [0.]
    assert report['dynamic_bookkeeping_cumulative_residual_J'][0] == 0.
    json.dumps(report, allow_nan=False)
    report['edge_ids'][0] = -1
    report['constitutive_cumulative_residual_J_by_edge'][0][0] = 99.
    report['coverage']['cumulative_consistency'] = 'changed'
    assert result.to_dict(detailed=True) == before
    assert_curves(result.viscoelastic_energy_diagnostics(), record)
    assert SAVED.read_bytes() == raw


@pytest.mark.parametrize('case', ['physical', 'be', 'work', 'physical_and_work', 'projection'])
def test_cumulative_mutations_are_reported_not_certified(api, tmp_path, case):
    SimResult, _ = api
    _, payload = _saved()
    record = payload['native_axial_viscoelastic']
    for state in record['samples'][1:]:
        cumulative = state['cumulative']
        if case in {'physical', 'physical_and_work'}:
            cumulative['physical_dissipation_J'][0] += 1.
        if case == 'be':
            cumulative['be_numerical_dissipation_J'][0] += 1.
        if case in {'work', 'physical_and_work'}:
            cumulative['endpoint_work_J'][0] += 1.
        if case == 'projection':
            state['dynamic_cumulative']['projection_force_work_J'] += 1.
    record['final'] = copy.deepcopy(record['samples'][-1])
    path = tmp_path / (case + '.json')
    path.write_text(json.dumps(payload, allow_nan=False))
    result = SimResult.load(str(path))
    report = result.viscoelastic_energy_diagnostics()
    assert_curves(report, record)
    assert report['coverage']['cumulative_consistency'] == 'not_certified'
    if case in {'physical', 'be'}:
        assert report['constitutive_cumulative_residual_J'][-1] > .5
    if case == 'work':
        assert report['constitutive_cumulative_residual_J'][-1] < -.5
    if case in {'work', 'physical_and_work', 'projection'}:
        assert report['dynamic_bookkeeping_cumulative_residual_J'][-1] > .5


@pytest.mark.parametrize('detailed', [False, True])
def test_diagnostics_survive_existing_roundtrip_without_schema_changes(api, tmp_path, detailed):
    SimResult, _ = api
    result = SimResult.load(str(SAVED))
    before = result.to_dict(detailed=detailed)
    report = result.viscoelastic_energy_diagnostics()
    assert result.to_dict(detailed=detailed) == before
    path = tmp_path / 'roundtrip.json'
    result.save(str(path), detailed=detailed)
    assert SimResult.load(str(path)).viscoelastic_energy_diagnostics() == report


def test_missing_and_mutated_records_fail_clearly(api):
    SimResult, _ = api
    result = SimResult.load(str(SAVED))
    result.metadata.pop('native_axial_viscoelastic')
    with pytest.raises(ValueError, match='require native axial viscoelastic'):
        result.viscoelastic_energy_diagnostics()
    result = SimResult.load(str(SAVED))
    result.metadata['native_axial_viscoelastic']['schema'] = 'native_axial_viscoelastic/99'
    with pytest.raises(ValueError):
        result.viscoelastic_energy_diagnostics()


def test_finite_channels_cannot_silently_overflow_residual(api):
    SimResult, _ = api
    result = SimResult.load(str(SAVED))
    record = result.metadata['native_axial_viscoelastic']
    for state in record['samples'][1:]:
        for key in state['dynamic_cumulative']:
            state['dynamic_cumulative'][key] = sys.float_info.max if key == 'projection_force_work_J' else -sys.float_info.max
    record['final'] = copy.deepcopy(record['samples'][-1])
    before = copy.deepcopy(record)
    with pytest.raises(ValueError, match='diagnostic residual is outside finite binary64 range'):
        result.viscoelastic_energy_diagnostics()
    assert record == before
