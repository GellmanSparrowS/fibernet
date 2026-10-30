"""Runner artifact/resume contract; no generator, solver, or trainer is executed."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from fibernet import StructureGraph
import fibernet


def snapshot(directory):
    return {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def forbidden(*args, **kwargs):
    raise AssertionError('Unexpected generator, solver, or trainer execution')


@pytest.fixture
def runner_module(monkeypatch):
    # Install before loading the tutorial: never import the real accelerator.
    accelerated = types.ModuleType('fibernet.sim.accelerated')
    accelerated.TaichiEngine = forbidden
    accelerated._graph_to_arrays = forbidden
    accelerated._get_boundary_indices = forbidden
    monkeypatch.setitem(sys.modules, accelerated.__name__, accelerated)
    monkeypatch.setitem(fibernet.__dict__, 'TaichiEngine', forbidden)
    path = Path(__file__).resolve().parents[1] / 'tutorials' / 'run_pipeline.py'
    spec = importlib.util.spec_from_file_location('_tutorial_resume_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'pattern_2d', forbidden)
    monkeypatch.setattr(module.PipelineRunner, 'stage_ml', forbidden)
    return module


def graph():
    g = StructureGraph(dimension=2, box_size=[1, 1])
    for p in ([99, 99], [0, 0], [88, 88], [1, 0]):
        g.add_node(p, merge=False)
    g.remove_node(0)
    g.remove_node(2)
    g.add_edge(1, 3, radius=.12, internal_points=np.array([[.4, .2, 0.]]))
    g.add_edge(1, 3, radius=.21, allow_parallel=True)
    g.metadata['saved_authority'] = {'not_a_seed_reconstruction': True}
    return g


def new_runner(module, tmp_path):
    return module.PipelineRunner(tmp_path / 'data', tmp_path / 'viz', num_structures=2)


def generated(module, tmp_path, monkeypatch):
    originals = [graph(), graph()]
    originals[1].metadata['second'] = True
    it = iter(originals)
    monkeypatch.setattr(module, 'pattern_2d', lambda **kw: next(it))
    runner = new_runner(module, tmp_path)
    runner.stage_generation()
    monkeypatch.setattr(module, 'pattern_2d', forbidden)
    return runner, originals


def save_results(runner, count=2, complete=True):
    records = [dict(index=i, name=runner.all_metadata[i]['name'], max_force=10+i,
                    max_stretch=1.5, energy=3+i, graph_sha256=runner._graph_hashes[i],
                    pipeline_config=runner._simulation_config()) for i in range(count)]
    directory = runner.DATA_OUT / 'simulations'
    directory.mkdir(exist_ok=True)
    runner._write_json(directory / 'checkpoint.json', records)
    if complete:
        runner._save_checkpoint('simulation', {'count': count})
    return records


def test_exact_native_graph_roundtrip_and_repeat_load(runner_module, tmp_path, monkeypatch):
    runner, originals = generated(runner_module, tmp_path, monkeypatch)
    runner.stage_generation()
    runner.stage_generation()
    assert len(runner.all_structures) == 2
    assert [g.to_dict() for g in runner.all_structures] == [g.to_dict() for g in originals]
    assert list(runner.all_structures[0].nodes) == [1, 3]
    assert len(runner.all_structures[0].edges) == 2


@pytest.mark.parametrize('damage', ['missing', 'changed', 'count', 'config', 'legacy'])
def test_bad_graph_artifacts_fail_without_regeneration(runner_module, tmp_path, monkeypatch, damage):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    path = runner.DATA_OUT / 'structures/honeycomb_001.json'
    if damage == 'missing':
        path.unlink()
    elif damage == 'changed':
        path.write_text('{}')
    else:
        cp = runner._checkpoint()
        if damage == 'count':
            cp['generation']['artifacts'].pop()
        elif damage == 'config':
            cp['generation']['config']['perturbation'] = .9
        else:
            cp['generation'] = {'count': 2, 'time': 1}
        runner._write_json(runner.DATA_OUT / 'pipeline_checkpoint.json', cp)
    before = {p: p.read_bytes() for p in runner.DATA_OUT.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        runner.stage_generation()
    assert snapshot(runner.DATA_OUT) == before


def test_completed_simulation_recovers_all_metrics_without_solver(runner_module, tmp_path, monkeypatch):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    save_results(runner)
    resumed = new_runner(runner_module, tmp_path)
    resumed.stage_simulation()
    assert [(m['max_force'], m['max_stretch'], m['energy'])
            for m in resumed.all_metadata] == [(10, 1.5, 3), (11, 1.5, 4)]


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'wrong_identity', 'wrong_config', 'nonfinite'])
def test_bad_simulation_checkpoint_fails_closed(runner_module, tmp_path, monkeypatch, damage):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    records = save_results(runner)
    path = runner.DATA_OUT / 'simulations/checkpoint.json'
    if damage == 'missing':
        path.unlink()
    else:
        if damage == 'truncated':
            records.pop()
        elif damage == 'wrong_identity':
            records[1]['graph_sha256'] = 'wrong'
        elif damage == 'wrong_config':
            records[1]['pipeline_config']['num_steps'] = 9
        else:
            records[1]['energy'] = float('nan')
        path.write_text(json.dumps(records))
    before = snapshot(runner.DATA_OUT)
    with pytest.raises(ValueError):
        new_runner(runner_module, tmp_path).stage_simulation()
    assert snapshot(runner.DATA_OUT) == before


def test_partial_simulation_restores_prefix_and_runs_only_suffix_stub(runner_module, tmp_path, monkeypatch):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    prefix = save_results(runner, count=1, complete=False)
    calls = []
    class Result:
        max_force, max_stretch, energy = 20, 1.5, 8
        def to_dict(self):
            return dict(max_force=self.max_force, max_stretch=self.max_stretch, energy=self.energy)
        def save(self, path, detailed=False):
            Path(path).write_text(json.dumps(self.to_dict()))
    class Engine:
        def dynamics(self, graph, **kwargs):
            # Native API uses array indices (0, 1), NOT sparse graph IDs (1, 3).
            assert list(graph.nodes) == [1, 3]
            assert kwargs['fixed_nodes'] == [0]
            schedule = kwargs.pop('displacement_schedule')
            assert list(schedule) == [1]
            assert [step for step, _ in schedule[1]] == [0, runner.NUM_STEPS]
            np.testing.assert_array_equal(schedule[1][0][1], [0, 0, 0])
            np.testing.assert_array_equal(schedule[1][1][1], [.5, 0, 0])
            assert kwargs == dict(fixed_nodes=[0], stiffness=1e5, damping=.3,
                                  dt=1e-4, num_steps=8000, save_interval=1000,
                                  dashpot=10., drag=1.)
            calls.append(graph.to_dict())
            return Result()
    monkeypatch.setattr(runner_module, 'TaichiEngine', Engine)
    monkeypatch.setattr(runner_module, '_graph_to_arrays', lambda g: (np.array([[0., 0., 0.], [1., 0., 0.]]), [], [], None))
    monkeypatch.setattr(runner_module, '_get_boundary_indices', lambda pos, pct: {'left': [0], 'right': [1]})
    resumed = new_runner(runner_module, tmp_path)
    resumed.stage_simulation()
    assert calls == [runner.all_structures[1].to_dict()]
    assert [(m['max_force'], m['max_stretch'], m['energy'])
            for m in resumed.all_metadata] == [(10, 1.5, 3), (20, 1.5, 8)]
    persisted = json.loads((runner.DATA_OUT / 'simulations/checkpoint.json').read_text())
    assert persisted == prefix + [dict(index=1, name='honeycomb_001',
        max_force=20, max_stretch=1.5, energy=8, graph_sha256=runner._graph_hashes[1],
        pipeline_config=runner._simulation_config())]
    resumed.stage_simulation()  # All complete now; must not call solver again.
    assert len(calls) == 1


@pytest.mark.parametrize('stage', ['simulation', 'features'])
def test_from_stage_restores_prerequisites_before_dispatch(runner_module, tmp_path, monkeypatch, stage):
    runner, originals = generated(runner_module, tmp_path, monkeypatch)
    save_results(runner)
    resumed = new_runner(runner_module, tmp_path)
    calls = []
    def inspect():
        assert [g.to_dict() for g in resumed.all_structures] == [g.to_dict() for g in originals]
        if stage == 'features':
            assert [m['max_force'] for m in resumed.all_metadata] == [10, 11]
        calls.append(stage)
    monkeypatch.setattr(resumed, 'stage_simulation', inspect)
    monkeypatch.setattr(resumed, 'stage_features', inspect)
    monkeypatch.setattr(resumed, 'stage_ml', lambda: calls.append('trainer_stub'))
    resumed.run(from_stage=stage)
    assert calls[-1] == 'trainer_stub'


def test_from_features_missing_results_never_dispatches(runner_module, tmp_path, monkeypatch):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'stage_features', forbidden)
    with pytest.raises(ValueError):
        runner.run(from_stage='features')


def test_completed_features_missing_csv_is_not_silently_skipped(runner_module, tmp_path, monkeypatch):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    save_results(runner)
    runner._save_checkpoint('features', {'count': 2, 'sha256': 'missing'})
    with pytest.raises(ValueError):
        runner.stage_features()


@pytest.mark.parametrize('has_checkpoint', [False, True])
def test_uncommitted_simulation_outputs_are_not_recomputed(runner_module, tmp_path, monkeypatch, has_checkpoint):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    save_results(runner, count=1, complete=False)
    orphan = runner.DATA_OUT / 'simulations/honeycomb_001_sim.json'
    orphan.write_text('{"uncommitted": true}')
    if not has_checkpoint:
        (runner.DATA_OUT / 'simulations/checkpoint.json').unlink()
    before = snapshot(runner.DATA_OUT)
    with pytest.raises(ValueError):
        runner.stage_simulation()
    assert snapshot(runner.DATA_OUT) == before
    assert orphan.read_text() == '{"uncommitted": true}'


def test_uncommitted_features_are_not_overwritten(runner_module, tmp_path, monkeypatch):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    save_results(runner)
    path = runner.DATA_OUT / 'structures_features.csv'
    path.write_text('preserve this evidence\n')
    monkeypatch.setattr(runner_module, 'GraphFeatureExtractor', forbidden)
    before = snapshot(runner.DATA_OUT)
    with pytest.raises(ValueError, match='Uncommitted'):
        runner.stage_features()
    assert snapshot(runner.DATA_OUT) == before
    assert path.read_text() == 'preserve this evidence\n'


@pytest.mark.parametrize('artifact', ['simulations/old_sim.json', 'structures_features.csv'])
def test_fresh_generation_rejects_downstream_evidence(runner_module, tmp_path, artifact):
    runner = new_runner(runner_module, tmp_path)
    path = runner.DATA_OUT / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('preserve downstream evidence')
    before = snapshot(runner.DATA_OUT)
    with pytest.raises(ValueError):
        runner.stage_generation()
    assert snapshot(runner.DATA_OUT) == before
    assert path.read_text() == 'preserve downstream evidence'


def save_features(runner):
    path = runner.DATA_OUT / 'structures_features.csv'
    path.write_text('name,max_force,max_stretch,energy,feature\nhoneycomb_000,10,1.5,3,1\nhoneycomb_001,11,1.5,4,2\n')
    runner._save_checkpoint('features', dict(count=2, sha256=runner._sha256(path),
        simulation_sha256=runner._sha256(runner.DATA_OUT / 'simulations/checkpoint.json')))


@pytest.mark.parametrize('stage', ['features', 'ml'])
def test_features_reject_changed_finite_simulation_inputs(runner_module, tmp_path, monkeypatch, stage):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    records = save_results(runner)
    save_features(runner)
    records[0]['energy'] = 123.0
    runner._write_json(runner.DATA_OUT / 'simulations/checkpoint.json', records)
    before = snapshot(runner.DATA_OUT)
    with pytest.raises(ValueError, match='features artifact'):
        runner.run(from_stage=stage)
    assert snapshot(runner.DATA_OUT) == before


@pytest.mark.parametrize('stage', ['features', 'ml'])
@pytest.mark.parametrize('marker', [None, {}, {'count': 1}])
def test_malformed_simulation_completion_blocks_downstream(runner_module, tmp_path, monkeypatch, stage, marker):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    save_results(runner)
    save_features(runner)
    runner._save_checkpoint('simulation', marker)
    before = snapshot(runner.DATA_OUT)
    with pytest.raises(ValueError, match='completion count'):
        runner.run(from_stage=stage)
    assert snapshot(runner.DATA_OUT) == before


def test_from_ml_restores_complete_saved_inputs_before_trainer_stub(runner_module, tmp_path, monkeypatch):
    runner, originals = generated(runner_module, tmp_path, monkeypatch)
    save_results(runner)
    save_features(runner)
    resumed = new_runner(runner_module, tmp_path)
    calls = []
    def trainer_stub():
        assert [g.to_dict() for g in resumed.all_structures] == [g.to_dict() for g in originals]
        assert [(m['max_force'], m['max_stretch'], m['energy'])
                for m in resumed.all_metadata] == [(10, 1.5, 3), (11, 1.5, 4)]
        calls.append('called')
    monkeypatch.setattr(resumed, 'stage_ml', trainer_stub)
    before = snapshot(runner.DATA_OUT)
    resumed.run(from_stage='ml')
    assert calls == ['called']
    assert snapshot(runner.DATA_OUT) == before


@pytest.mark.parametrize('missing', ['generation', 'simulation', 'features'])
def test_from_ml_missing_prerequisite_never_calls_trainer(runner_module, tmp_path, monkeypatch, missing):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    save_results(runner)
    save_features(runner)
    paths = {'generation': 'structures/honeycomb_001.json',
             'simulation': 'simulations/checkpoint.json',
             'features': 'structures_features.csv'}
    (runner.DATA_OUT / paths[missing]).unlink()
    before = snapshot(runner.DATA_OUT)
    with pytest.raises(ValueError):
        runner.run(from_stage='ml')
    assert snapshot(runner.DATA_OUT) == before


def test_all_simulation_records_without_marker_recovers_without_engine(runner_module, tmp_path, monkeypatch):
    runner, _ = generated(runner_module, tmp_path, monkeypatch)
    records = save_results(runner, complete=False)
    before = snapshot(runner.DATA_OUT)
    resumed = new_runner(runner_module, tmp_path)
    resumed.stage_simulation()  # The default fixture engine is fail-on-call.
    assert resumed._checkpoint()['simulation']['count'] == 2
    assert [(m['max_force'], m['max_stretch'], m['energy'])
            for m in resumed.all_metadata] == [(10, 1.5, 3), (11, 1.5, 4)]
    assert json.loads((runner.DATA_OUT / 'simulations/checkpoint.json').read_text()) == records
    after = snapshot(runner.DATA_OUT)
    completion_path = runner.DATA_OUT / 'pipeline_checkpoint.json'
    assert set(after) == set(before)
    assert {p: b for p, b in after.items() if p != completion_path} == {
        p: b for p, b in before.items() if p != completion_path}
