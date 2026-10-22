"""Provider-free full-history admission and serialization, not physics evidence."""
import copy
from dataclasses import replace
from itertools import combinations
import json
from unittest.mock import patch

import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import lammps_backend as backend
from fibernet.integrations import _harmonic_profile as profiles
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult
from test_lammps_backend_contract import prepared, fixture
from test_lammps_fixed_state import typed_raw, RUNTIME

NAME = 'harmonic_full_history_v1'
MIB = 1024**2
LIMITS = backend.ResourceLimits(memory_bytes=2*1024**3, output_bytes=64*MIB,
                                file_size_bytes=64*MIB)


@pytest.fixture(autouse=True)
def forbid_native_launch():
    with patch('subprocess.Popen', side_effect=AssertionError('provider launch forbidden')):
        yield


def case(n=64, e=512, steps=2048, **changes):
    graph = StructureGraph(dimension=3)
    for i in range(n):
        graph.add_node([i*.001, 0., 0.])
    for i, j in list(combinations(range(n), 2))[:e]:
        graph.add_edge(i, j, n_internal=0)
    options = dict(length_unit='m', maximum_bond_length_m=.1,
                   masses_kg={i: 1e-6 for i in graph.nodes},
                   bonds={i: backend.HarmonicBond(1., .001) for i in graph.edges},
                   run=backend.RunSpec(1e-6, steps, 1, ((-1., 1.),)*3),
                   profile=NAME)
    options.update(changes)
    return graph, options


def small(steps=2, **changes):
    return prepared(profile=NAME, run=backend.RunSpec(1e-6, steps, 1, ((-.01,.01),)*3), **changes)


@pytest.mark.parametrize('steps', [1, 512, 1024, 2048])
@pytest.mark.parametrize('stride', [None, 1])
def test_boundary_compiler_and_task_preserve_every_frame(tmp_path, steps, stride):
    graph, options = case(steps=steps, output_stride=stride)
    p = backend.prepare_harmonic_beads(graph, **options)
    assert backend.validate_prepared(p) is p
    c = json.loads(p.contract_json)
    assert (c['output_stride'], c['native_frames'], c['output_frames']) == (1, steps+1, steps+1)
    assert c['profile']['name'] == NAME
    assert (len(p.node_ids), len(p.edge_ids)) == (64, 512)
    assert 'dump trajectory all custom 1 ' in p.input_text
    with patch.object(backend.RuntimeSpec, 'checked', return_value=RUNTIME):
        task = backend.LAMMPSTask(p, RUNTIME, tmp_path/'never-created', LIMITS)
    assert task.state == backend.TaskState.CREATED
    assert task.output_forecast['output_frames'] == steps+1
    assert task.output_forecast['detailed_export_forecast_bytes'] <= 128*MIB
    assert not task.workdir.exists()


@pytest.mark.parametrize('steps', [0, 2049, True, 2.0])
def test_step_bound_rejected(steps):
    with pytest.raises(ValueError):
        small(steps)


@pytest.mark.parametrize('stride', [0, 2, -1, True, 1.0, '1'])
def test_output_stride_rejected(stride):
    with pytest.raises(ValueError):
        small(output_stride=stride)


@pytest.mark.parametrize('n,e', [(65,512), (64,513)])
def test_topology_bound_rejected(n,e):
    graph, options = case(n=n,e=e)
    with pytest.raises(ValueError, match='node/edge counts'):
        backend.prepare_harmonic_beads(graph, **options)


def test_sampling_mass_and_load_bounds():
    with pytest.raises(ValueError):
        prepared(profile=NAME, run=backend.RunSpec(1e-6,2,2,((-.01,.01),)*3))
    graph, options = case(steps=2)
    backend.prepare_harmonic_beads(graph, **{**options, 'masses_kg': {i: 1e-6*(1+i%2) for i in graph.nodes},
                                            'forces_N': {i: (1e-9,0.,0.) for i in range(20)}})
    for changes in ({'masses_kg': {i: 1e-6*(1+i%3) for i in graph.nodes}},
                    {'forces_N': {i: (1e-9,0.,0.) for i in range(21)}}):
        with pytest.raises(ValueError):
            backend.prepare_harmonic_beads(graph, **{**options, **changes})


@pytest.mark.parametrize('name,cap', [(None,16),('harmonic_large_v1',16),('harmonic_sampled_v1',64),(NAME,64)])
def test_explicit_native_limits_and_legacy_caps(name,cap):
    assert backend.ResourceLimits().checked(profile=name) == backend.ResourceLimits()
    backend.ResourceLimits(output_bytes=cap*MIB, file_size_bytes=cap*MIB).checked(profile=name)
    for key in ('output_bytes','file_size_bytes'):
        with pytest.raises(ValueError):
            backend.ResourceLimits(**{key:cap*MIB+1}).checked(profile=name)


def test_legacy_schedules_still_reject_full_history():
    for name, steps, stride in [('harmonic_large_v1',512,None),
                                ('harmonic_sampled_v1',2048,16),
                                ('harmonic_sampled_v1',1024,1)]:
        with pytest.raises(ValueError):
            prepared(profile=name, output_stride=stride,
                     run=backend.RunSpec(1e-6,steps,1,((-.01,.01),)*3))
    with pytest.raises(ValueError):
        backend.ResourceLimits(output_bytes=64*MIB).checked()


def test_forecast_rejects_before_runtime_directory_or_launch(tmp_path):
    graph, options = case()
    p = backend.prepare_harmonic_beads(graph, **options)
    for limits in (replace(LIMITS,memory_bytes=1024**3),
                   replace(LIMITS,output_bytes=16*MIB),
                   replace(LIMITS,file_size_bytes=16*MIB)):
        with patch.object(backend.RuntimeSpec,'checked') as checked:
            with pytest.raises(ValueError,match='planned'):
                backend.LAMMPSTask(p,RUNTIME,tmp_path/'absent',limits)
        checked.assert_not_called()
        assert not (tmp_path/'absent').exists()


def test_profile_schedule_and_record_tampering_rejected():
    p=small()
    for key,value in [('output_stride',2),('native_frames',2),('output_frames',True)]:
        c=json.loads(p.contract_json);c[key]=value
        with pytest.raises(ValueError):
            backend.validate_prepared(replace(p,contract_json=json.dumps(c)))
    c=json.loads(p.contract_json);c['profile']['max_steps']=4096
    with pytest.raises(ValueError):
        backend.validate_prepared(replace(p,contract_json=json.dumps(c)))
    with pytest.raises(ValueError):
        backend.validate_prepared(replace(p,profile='harmonic_sampled_v1'))


def test_parser_converter_and_persistence_match(tmp_path):
    p=small(2048)
    dump,log=fixture(p)
    raw=backend.parse_outputs(p,dump,log,max_output_bytes=64*MIB)
    r=adapter.to_sim_result(p,raw,RUNTIME,0.,True)
    assert r.dynamics_state['steps'] == list(range(2049))
    assert len(r.positions_trajectory)==2049
    for detailed in (False,True):
        path=tmp_path/'history.json';r.save(path,detailed=detailed)
        restored=SimResult.load(path,profile=NAME)
        assert restored.to_dict(detailed)==r.to_dict(detailed)
        with pytest.raises(ValueError):
            SimResult.load(path,profile='harmonic_sampled_v1')
        with pytest.raises(ValueError):
            SimResult.load(path)
    for key,value in [('output_stride',2),('native_frames',1),('output_frames',True)]:
        bad=copy.deepcopy(r);bad.dynamics_state[key]=value
        with pytest.raises(ValueError):bad.to_dict(True)
    for raw_bad in (replace(raw,trajectory=raw.trajectory[::2]),
                    replace(raw,thermo=raw.thermo[:-1])):
        with pytest.raises(ValueError):adapter.to_sim_result(p,raw_bad,RUNTIME,0.,True)
    with pytest.raises(ValueError):backend.parse_outputs(p,dump,log,max_output_bytes=64*MIB+1)


def test_profile_specific_export_cap_is_used_for_state_and_result(monkeypatch):
    p=small();r=adapter.to_sim_result(p,typed_raw(p),RUNTIME,0.,True)
    observed=[];original=profiles.json_bytes
    def observed_check(value,limit,**kwargs):
        observed.append(limit)
        return original(value,limit,**kwargs)
    monkeypatch.setattr(profiles,'json_bytes',observed_check)
    adapter.validate_state(r.dynamics_state)
    assert 128*MIB in observed and 40*MIB not in observed
    observed.clear();r.to_dict(True)
    assert 128*MIB in observed and 40*MIB not in observed


def test_public_dispatch_uses_existing_task_and_profile(monkeypatch,tmp_path):
    import fibernet
    from fibernet.easy import simulate
    assert fibernet.simulate is simulate
    calls=[]
    class InertTask(backend.LAMMPSTask):
        def start(self):
            calls.append(self.prepared)
            self.result=typed_raw(self.prepared)
            self.state=backend.TaskState.SUCCEEDED
            return self
        def wait(self):return self.state
    monkeypatch.setattr(backend,'LAMMPSTask',InertTask)
    monkeypatch.setattr(backend.RuntimeSpec,'checked',lambda self:self)
    graph,options=case(n=2,e=1,steps=2)
    result=fibernet.simulate(graph,mode='dynamics',backend='lammps',record_state=True,
                            runtime=RUNTIME,workdir=tmp_path/'unused',limits=LIMITS,**options)
    assert len(calls)==1 and calls[0].profile==NAME
    assert result.metadata['profile']['name']==NAME
    assert result.dynamics_state['steps']==[0,1,2]
    assert not (tmp_path/'unused').exists()


def test_capabilities_describe_limits_without_claiming_physics():
    caps=adapter.source_capabilities()
    entry=caps['harmonic_profiles'][NAME]
    assert entry=={'max_nodes':64,'max_edges':512,'max_steps':2048,
                   'sample_every':1,'output_stride':1,'max_output_frames':2049,
                   'max_mass_types':2,'max_loaded_nodes':20}
    entry['max_nodes']=1
    assert adapter.source_capabilities()['harmonic_profiles'][NAME]['max_nodes']==64
    assert caps['runtime_readiness']=='unknown'
    assert caps['physics_qualification']=='not_established_by_dispatch'


def test_full_size_detailed_export_exceeds_legacy_limit_and_roundtrips(tmp_path):
    # Deliberately wide finite tokens. This is an inert serialization fixture,
    # not an integrated trajectory or evidence for any network scenario.
    graph, options=case(n=64,e=1)
    for i,node in graph.nodes.items():
        node.position[:]=[(i+1)*.0003141592653589793,.0012345678901234567,.0023456789012345678]
    options['velocities_m_s']={i:(1.2345678901234567e-100,)*3 for i in graph.nodes}
    p=backend.prepare_harmonic_beads(graph,**options)
    raw=typed_raw(p)
    frames=tuple(replace(frame,forces_N=((1.2345678901234567e-100,)*3,)*64)
                 for frame in raw.trajectory)
    result=adapter.to_sim_result(p,replace(raw,trajectory=frames),RUNTIME,0.,True)
    path=tmp_path/'full.json'
    result.save(path,detailed=True)
    assert 40*MIB < path.stat().st_size <= 128*MIB
    loaded=SimResult.load(path,profile=NAME)
    assert loaded.metadata['profile']['name']==NAME
    assert loaded.dynamics_state['steps']==list(range(2049))
    assert loaded.dynamics_state['positions']==result.dynamics_state['positions']
    assert loaded.dynamics_state['nodal_forces']==result.dynamics_state['nodal_forces']
    assert len(loaded.positions_trajectory)==2049


def test_load_predecode_rejects_profile_byte_overflow(tmp_path):
    path=tmp_path/'oversize.json'
    with path.open('wb') as stream:
        stream.truncate(128*MIB+1)
    with patch.object(json,'loads',side_effect=AssertionError('must reject before decode')):
        with pytest.raises(ValueError,match='predecode byte bound'):
            SimResult.load(path,profile=NAME)


@pytest.mark.parametrize('key,value', [('output_stride',None),('output_stride',1.0),
                                       ('output_stride',True),('native_frames',None),
                                       ('output_frames',None)])
def test_persisted_schedule_requires_explicit_integer_fields(key,value):
    p=small();r=adapter.to_sim_result(p,typed_raw(p),RUNTIME,0.,True)
    bad=copy.deepcopy(r.dynamics_state);bad[key]=value
    with pytest.raises(ValueError):adapter.validate_state(bad)
    contract=json.loads(p.contract_json);contract[key]=value
    with pytest.raises(ValueError):profiles.check_contract(contract,2,1)


def test_documented_prepare_example():
    from pathlib import Path
    doc=(Path(__file__).resolve().parents[1]/'docs/LAMMPS_FULL_HISTORY.md').read_text()
    code=doc.split('```python\n',1)[1].split('```',1)[0]
    namespace={};exec(compile(code,'documented-full-history','exec'),namespace)
    p=namespace['prepared']
    assert p.profile==NAME and p.steps==2048
    assert backend.validate_prepared(p) is p


@pytest.mark.parametrize('step', [1, 2047])
def test_full_history_checks_nonfinal_fixed_and_mobile_states(step):
    from test_lammps_fixed_state import edit_raw, edit_dump
    p=small(2048,fixed_node_ids=(10,))
    dump,log=fixture(p);raw=typed_raw(p)
    for nid,value in ((10,1e-6),(40,.02)):
        bad=edit_raw(raw,step,nid,'positions_m',value)
        with pytest.raises(ValueError):adapter.to_sim_result(p,bad,RUNTIME,0.,True)
        with pytest.raises(ValueError):backend.parse_outputs(p,edit_dump(p,dump,step,nid,2,value),log)
