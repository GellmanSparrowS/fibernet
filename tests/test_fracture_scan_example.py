"""Checks for the public notched-network fracture scan."""
import csv
import hashlib
import json

import numpy as np
import pytest

from examples import fracture_scan
from fibernet import StructureGraph
from fibernet.sim.fracture import FractureConfig, FractureResult, FractureSolver


def _read_case(root, row, kind):
    payload = (root / row[f'{kind}_file']).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == row[f'{kind}_sha256']
    return json.loads(payload)


def test_scan_tracks_a_common_observed_failure_prefix(tmp_path):
    pytest.importorskip('psutil')
    pytest.importorskip('threadpoolctl')
    out = tmp_path / 'scan'
    # This bound includes actual events without reaching the seed's path loss.
    short, full = fracture_scan.run_scan(out, seeds=(11,), short_um=50)
    settings = [_read_case(out, row, 'input') for row in (short, full)]
    before, after = [_read_case(out, row, 'result') for row in (short, full)]
    assert settings[0]['graph'] == settings[1]['graph']
    assert settings[0]['config']['reference_displacement_m'] == settings[1]['config']['reference_displacement_m']
    assert settings[0]['fixed_nodes'] == settings[1]['fixed_nodes'] == list(range(9))
    assert settings[0]['loaded_nodes'] == settings[1]['loaded_nodes'] == list(range(54, 63))
    assert settings[0]['initially_broken_edges'] == settings[1]['initially_broken_edges']
    assert len(settings[0]['initially_broken_edges']) == 6
    assert len(before['state']['positions_m']) == 63

    assert short['status'] == 'right_censored_loading_bound'
    assert short['path_loss_observed'] is False and short['path_loss_displacement_m'] is None
    assert short['complete'] and full['complete']
    assert full['status'] == 'path_loss_observed' and full['path_loss_observed'] is True
    assert 50e-6 < full['path_loss_displacement_m'] < 400e-6
    assert before['events']
    assert before['events'] == after['events'][:len(before['events'])]

    for row in (short, full):
        assert row['first_failure_displacement_m'] == before['events'][0]['displacement_m']

    saved = settings[0]
    restored = StructureGraph.from_dict(saved['graph'])
    assert restored.to_dict() == saved['graph']
    replay = FractureSolver(
        restored, FractureConfig(**saved['config']), fixed_nodes=saved['fixed_nodes'],
        loaded_nodes=saved['loaded_nodes'], length_unit=saved['length_unit'],
        initially_broken_edges=saved['initially_broken_edges'],
    ).run()
    assert json.loads(json.dumps(replay.data, allow_nan=False)) == before

    for settings_row, result, row in zip(settings, (before, after), (short, full)):
        assert result['binding'] == row['result_binding']
        notch = set(settings_row['initially_broken_edges'])
        failed = {event['segment_id'] for event in result['events']}
        assert notch.isdisjoint(failed)
        assert len(result['state']['mask']) == 158
        assert {i for i, active in enumerate(result['state']['mask']) if not active} == notch | failed
        trace = result['observations']
        assert row['peak_reaction_N'] == max(o['loaded_reaction_force_N'] for o in trace)
        # Each intact-mask segment is a line through the origin in force–displacement
        # space. Integrate ½k (b²−a²) between distinct physical displacements.
        work = sum(
            b['loaded_reaction_force_N'] / (2 * b['displacement_m'])
            * (b['displacement_m'] ** 2 - a['displacement_m'] ** 2)
            for a, b in zip(trace, trace[1:]) if b['displacement_m'] > a['displacement_m']
        )
        release = work - result['state']['strain_energy_J']
        assert row['external_work_J'] == pytest.approx(work, rel=1e-12)
        assert row['released_elastic_energy_J'] == pytest.approx(release, rel=1e-12)
        assert row['event_elastic_energy_drop_J'] == pytest.approx(release, rel=1e-12)
        assert abs(row['elastic_energy_ledger_residual_J']) < 1e-15

    assert json.loads((out / 'summary.json').read_bytes())['cases'] == [short, full]
    with (out / 'scan.csv').open(newline='') as stream:
        table = list(csv.DictReader(stream))
    assert [case['status'] for case in table] == [short['status'], full['status']]
    assert table[0]['path_loss_displacement_m'] == ''
    with pytest.raises(FileExistsError):
        fracture_scan.run_scan(out, seeds=(11,))


def test_geometry_repeatable_with_bounded_seed_jitter():
    graph, fixed, loaded, notch = fracture_scan.make_network(23)
    again, *identifiers = fracture_scan.make_network(23)
    other, *_ = fracture_scan.make_network(37)
    assert graph.to_dict() == again.to_dict()
    assert identifiers == [fixed, loaded, notch]
    assert len(graph.nodes) == 63 and len(graph.edges) == 158
    assert [(e.node_i, e.node_j) for e in graph.edges.values()] == [
        (e.node_i, e.node_j) for e in other.edges.values()
    ]
    np.testing.assert_array_equal([graph.nodes[i].position for i in fixed + loaded],
                                  [other.nodes[i].position for i in fixed + loaded])
    assert not np.array_equal(graph.nodes[10].position, other.nodes[10].position)


def _partial_result(*, complete=True, energy=.5):
    state = dict(displacement_m=1., strain_energy_J=energy, topology={'path_present': True})
    observations = [
        dict(state, displacement_m=0., loaded_reaction_force_N=0., strain_energy_J=0.,
             mask=[True], path_present=True),
        dict(state, loaded_reaction_force_N=1., mask=[True], path_present=True),
    ]
    return FractureResult(dict(
        state=state, observations=observations, events=[], complete=complete,
        config={'maximum_displacement_m': 2.}, termination_reason='zero_stress_or_no_load_path',
    ))


def test_early_stop_is_distinct_from_censoring_and_ledger_mismatch_errors():
    for complete, status in ((True, 'stopped_without_path_loss'), (False, 'incomplete')):
        row = fracture_scan.summarize_result(_partial_result(complete=complete))
        assert row['status'] == status and row['path_loss_displacement_m'] is None
    with pytest.raises(RuntimeError, match='energy-drop ledger'):
        fracture_scan.summarize_result(_partial_result(energy=1.))


def test_failed_solves_retain_inputs_and_missing_metrics(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('simulated failed equilibrium')

    monkeypatch.setattr(fracture_scan, 'FractureSolver', fail)
    out = tmp_path / 'scan'
    rows = fracture_scan.run_scan(out, seeds=(11,))
    for row in rows:
        assert row['status'] == 'error' and row['complete'] is False
        assert row['error'] == 'RuntimeError: simulated failed equilibrium'
        _read_case(out, row, 'input')
        assert 'result_file' not in row and 'peak_reaction_N' not in row
