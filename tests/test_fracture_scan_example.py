"""Checks for the public notched-network fracture scan."""
import csv
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples import fracture_scan
from fibernet import Material, StructureGraph
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

    report = json.loads((out / 'summary.json').read_bytes())
    assert report['cases'] == [short, full] and 'comparison' not in report
    assert [row['case'] for row in report['cases']] == ['seed-11-horizon-short', 'seed-11-horizon-full']
    with (out / 'scan.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        table = list(reader)
        assert {'pair_id', 'notch_condition', 'source_graph_sha256'}.isdisjoint(reader.fieldnames)
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
    bounded = _partial_result()
    bounded.data['config']['maximum_displacement_m'] = 1.
    bounded.data['termination_reason'] = 'loading_bound'
    row = fracture_scan.summarize_result(bounded)
    assert row['status'] == 'right_censored_loading_bound'
    assert row['post_peak_observation_status'] == 'no_later_observation'
    assert row['post_peak_external_work_J'] is None
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


def test_opt_in_paired_cli_retains_shared_inputs_and_distinct_masks(tmp_path, monkeypatch):
    pytest.importorskip('psutil')
    pytest.importorskip('threadpoolctl')
    out = tmp_path / 'paired'
    monkeypatch.setattr(sys, 'argv', [
        'fracture_scan', '--output-dir', str(out), '--compare-notch', '--seeds', '23',
    ])
    assert fracture_scan.main() == 0
    summary = json.loads((out / 'summary.json').read_bytes())
    assert summary['comparison']['conditions'] == ['unnotched', 'notched']
    cases = summary['cases']
    assert len(cases) == 4
    with (out / 'scan.csv').open(newline='') as stream:
        table = list(csv.DictReader(stream))

    all_shared_sources = set()
    for index, horizon in enumerate(('short', 'full')):
        a, b = cases[2*index:2*index + 2]
        assert a['pair_id'] == b['pair_id'] == f'seed-23-horizon-{horizon}'
        assert [a['notch_condition'], b['notch_condition']] == ['unnotched', 'notched']
        inputs = [_read_case(out, row, 'input') for row in (a, b)]
        assert inputs[0]['initially_broken_edges'] == []
        assert len(inputs[1]['initially_broken_edges']) == 6
        different = {key for key in inputs[0] if inputs[0][key] != inputs[1][key]}
        assert different == {'initially_broken_edges', 'notch_condition'}
        canonical = json.dumps(inputs[0]['graph'], sort_keys=True, separators=(',', ':'), allow_nan=False)
        source_sha = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        assert a['source_graph_sha256'] == b['source_graph_sha256'] == source_sha
        all_shared_sources.add(source_sha)
        for pos, row in enumerate((a, b), 2*index):
            assert table[pos]['pair_id'] == row['pair_id']
            assert table[pos]['source_graph_sha256'] == source_sha
            assert table[pos]['notch_condition'] == row['notch_condition']

        results = [_read_case(out, row, 'result') for row in (a, b)]
        assert {key for key in results[0]['binding']
                if results[0]['binding'][key] != results[1]['binding'][key]} == {'geometry_sha256'}
        for item, result in zip(inputs, results):
            excluded = {i for i, active in enumerate(result['observations'][0]['mask']) if not active}
            failures = {event['segment_id'] for event in result['events']}
            assert excluded == set(item['initially_broken_edges']) and excluded.isdisjoint(failures)

    assert len(all_shared_sources) == 1
    assert all(row['status'] == 'right_censored_loading_bound' and row['path_loss_displacement_m'] is None
               for row in cases[:2])
    assert all(row['status'] == 'path_loss_observed' and row['path_loss_displacement_m'] is not None
               for row in cases[2:])


@pytest.mark.parametrize('notch,load_paths', [((), 2), ((0,), 1)])
def test_paired_analytic_axial_force_and_elastic_energy(notch, load_paths):
    pytest.importorskip('psutil')
    pytest.importorskip('threadpoolctl')
    graph = StructureGraph(dimension=3)
    for position in ((0, 0, 0), (.1, 0, 0), (0, .01, 0), (.1, .01, 0)):
        graph.add_node(position)
    material = Material(youngs_modulus=1e9, poissons_ratio=.3)
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
        graph.add_edge(a, b, radius=.001, material=material)
    sigma = 1e6
    critical = .1 * sigma / 1e9
    full_peak = load_paths * sigma * math.pi * .001**2
    for bound in (critical/2, critical*2):
        config = FractureConfig(1e9, .3, sigma, sigma, axis=0, maximum_displacement_m=bound)
        result = FractureSolver(graph, config, fixed_nodes=[0, 2], loaded_nodes=[1, 3],
                                length_unit='m', initially_broken_edges=notch).run()
        row = fracture_scan.summarize_result(result)
        end = min(critical, bound)
        peak = full_peak * end / critical
        work = .5 * peak * end
        spans = bound < critical
        assert row['final_displacement_m'] == pytest.approx(end)
        assert row['peak_reaction_N'] == pytest.approx(peak)
        assert row['external_work_J'] == pytest.approx(work, rel=1e-10)
        assert row['released_elastic_energy_J'] == pytest.approx(0. if spans else work, abs=1e-15)
        assert row['final_strain_energy_J'] == pytest.approx(work if spans else 0., abs=1e-15)
        assert row['failed_edge_count'] == (0 if spans else load_paths)
        if spans:
            assert row['status'] == 'right_censored_loading_bound'
            assert row['path_loss_displacement_m'] is None
        else:
            assert row['status'] == 'path_loss_observed'
            assert row['path_loss_displacement_m'] == pytest.approx(critical)


def test_actual_paused_notched_trace_is_incomplete_not_loading_bound():
    pytest.importorskip('psutil')
    pytest.importorskip('threadpoolctl')
    graph, fixed, loaded, notch = fracture_scan.make_network(11)
    config = FractureConfig(1e9, .3, 1e6, 1e6, maximum_displacement_m=400e-6)
    result = FractureSolver(graph, config, fixed_nodes=fixed, loaded_nodes=loaded,
                            length_unit='m', initially_broken_edges=notch).run(stop_after_epochs=1)
    row = fracture_scan.summarize_result(result)
    assert row['status'] == 'incomplete' and row['termination_reason'] == 'paused'
    assert row['complete'] is False and row['path_loss_displacement_m'] is None
    assert 0 < row['final_displacement_m'] < config.maximum_displacement_m
    assert row['peak_reaction_N'] > 0 and row['failed_edge_count'] > 0


def _ordered_peak_fixture():
    # Exact equilibria with k=4, 2, 0.5 and then no surviving load path.
    # The two peaks are tied at distinct displacements and two masks.
    trace = [(0., 0., 0., [True]*3, True), (1., 4., 2., [True]*3, True),
             (1., 2., 1., [False, True, True], True),
             (2., 4., 4., [False, True, True], True),
             (2., 1., 1., [False, False, True], True),
             (3., 1.5, 2.25, [False, False, True], True),
             (3., 0., 0., [False]*3, False)]
    observations = [dict(displacement_m=d, loaded_reaction_force_N=f, strain_energy_J=u,
                         mask=mask, path_present=path) for d, f, u, mask, path in trace]
    last = observations[-1]
    return FractureResult(dict(
        observations=observations, events=[], complete=True, termination_reason='material_path_loss',
        state=dict(displacement_m=last['displacement_m'], strain_energy_J=last['strain_energy_J'],
                   topology={'path_present': last['path_present']}),
        config={'maximum_displacement_m': 4., 'output_quota_bytes': 10240},
        binding={'test': 'untouched'},
    ))


def test_peak_work_splits_earliest_exact_tie_and_zero_displacement_drops():
    result = _ordered_peak_fixture()
    row = fracture_scan.summarize_result(result)
    assert row['status'] == 'path_loss_observed'
    assert row['peak_observation_index'] == 1 and row['peak_displacement_m'] == 1.
    assert row['peak_reaction_N'] == 4.
    assert row['post_peak_observation_status'] == 'positive_displacement_span'
    assert row['post_peak_displacement_span_m'] == 2.
    assert row['pre_peak_external_work_J'] == 2.
    assert row['post_peak_external_work_J'] == 3. + 1.25
    assert row['external_work_J'] == row['pre_peak_external_work_J'] + row['post_peak_external_work_J']
    assert row['released_elastic_energy_J'] == 1. + 3. + 2.25
    result.data['observations'][2]['displacement_m'] = 1.01
    with pytest.raises(RuntimeError, match='cannot be integrated'):
        fracture_scan.summarize_result(result)


def test_peak_work_terminal_null_and_same_displacement_zero_reach_csv(tmp_path, monkeypatch):
    terminal = _ordered_peak_fixture()
    terminal.data.update(observations=terminal.observations[:2], complete=False,
                         termination_reason='paused')
    terminal.data['state'] = dict(displacement_m=1., strain_energy_J=2.,
                                  topology={'path_present': True})
    plateau = _ordered_peak_fixture()
    plateau.data.update(observations=plateau.observations[:3], complete=False,
                        termination_reason='paused')
    plateau.data['state'] = dict(displacement_m=1., strain_energy_J=1.,
                                 topology={'path_present': True})
    results = iter((terminal, plateau))
    monkeypatch.setattr(fracture_scan, 'FractureSolver',
                        lambda *a, **k: SimpleNamespace(run=lambda: next(results)))
    out = tmp_path / 'scan'
    a, b = fracture_scan.run_scan(out, seeds=(11,))
    for row in (a, b):
        assert row['status'] == 'incomplete'
        assert row['peak_displacement_m'] == 1. and row['pre_peak_external_work_J'] == 2.
        assert row['post_peak_displacement_span_m'] == 0.
        assert row['result_binding'] == {'test': 'untouched'}
    assert a['post_peak_observation_status'] == 'no_later_observation'
    assert a['post_peak_external_work_J'] is None
    assert b['post_peak_observation_status'] == 'constant_displacement_only'
    assert b['post_peak_external_work_J'] == 0. and b['released_elastic_energy_J'] == 1.
    with (out / 'scan.csv').open(newline='') as handle:
        before, after = list(csv.DictReader(handle))
    assert before['post_peak_external_work_J'] == '' and after['post_peak_external_work_J'] == '0.0'
    assert json.loads((out / 'summary.json').read_bytes())['cases'] == [a, b]
