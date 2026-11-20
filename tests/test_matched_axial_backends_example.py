"""Shared-model contracts and independent reference for the optional native example."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[1]/'examples/matched_axial_backends.py'
spec = importlib.util.spec_from_file_location('matched_axial_example', PATH)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize('side,nodes,edges,fixed,loaded', [(1,2,1,1,1),(5,25,56,5,5)])
def test_shared_model(side,nodes,edges,fixed,loaded):
    case = example.build_case(side)
    assert len(case['node_ids']) == nodes
    assert len(case['edge_ids']) == edges
    assert len(case['fixed_rows']) == fixed
    assert np.count_nonzero(np.linalg.norm(case['force'],axis=1)) == loaded
    np.testing.assert_array_equal(case['mass'], np.full(nodes,1e-9))
    assert np.all(case['rest'] > 1e-12)
    prepared = example.prepare_case(case,256)
    contract = json.loads(prepared.contract_json)
    assert prepared.steps == 256
    assert len(prepared.node_ids) == nodes
    assert contract['fixed_node_ids'] == [case['node_ids'][i] for i in case['fixed_rows']]
    # Native harmonic's coefficient is k/2, in SI J/m² = N/m.
    coefficients = [line.split() for line in prepared.input_text.splitlines() if line.startswith('bond_coeff ')]
    assert len(coefficients) == edges
    assert all(float(row[2]) == .125 for row in coefficients)


def test_force_and_energy_mapping_under_finite_extension():
    case=example.build_case(5)
    length=case['rest']*np.linspace(.8,1.3,len(case['rest']))
    strain=length/case['rest']-1
    taichi_k=case['stiffness_N_per_m']*case['rest']
    np.testing.assert_allclose(taichi_k*strain,case['stiffness_N_per_m']*(length-case['rest']),rtol=2e-14)
    np.testing.assert_allclose(.5*taichi_k*strain**2*case['rest'],.5*case['stiffness_N_per_m']*(length-case['rest'])**2,rtol=2e-14)


def test_forced_spring_exact_reference_and_work():
    case=example.build_case(1)
    w=np.sqrt(.25/1e-9)
    times=np.array([0.,np.pi/w,2*np.pi/w])
    x=example.reference(case,times)
    np.testing.assert_allclose(x[:,1,0]-case['x0'][1,0],[0.,8e-7,0.],atol=3e-19)
    np.testing.assert_array_equal(x[:,0],np.zeros((3,3)))
    # At integer half-periods v=0, K+U equals constant-force displacement work.
    metrics=example.metrics(case,x,np.zeros_like(x),x)
    assert metrics['position_error_peak_normalized']==0.
    assert metrics['energy_work_residual_peak_normalized']<1e-11


def test_network_reference_is_dynamic_with_fixed_supports():
    case=example.build_case(5)
    x=example.reference(case,np.linspace(0.,case['duration_s'],17))
    np.testing.assert_array_equal(x[0],case['x0'])
    np.testing.assert_array_equal(x[:,case['fixed_rows']],np.broadcast_to(case['x0'][case['fixed_rows']],(17,5,3)))
    assert np.max(np.linalg.norm(x[-1]-case['x0'],axis=1))>1e-6
    assert np.all(x[:,:,2]==0.)


@pytest.mark.parametrize('side',[0,2,6])
def test_unknown_geometry_refused(side):
    with pytest.raises(ValueError):example.build_case(side)


def test_start_failure_still_cancels_owned_running_task(monkeypatch,tmp_path):
    from fibernet.integrations import TaskState
    class StartFailureTask:
        instance=None
        def __init__(self,*args):
            type(self).instance=self
            self.state=None
            self.cancelled=False
        def start(self):
            self.state=TaskState.RUNNING
            raise RuntimeError('injected startup cleanup failure')
        def cancel(self):
            self.cancelled=True
            self.state=TaskState.CANCELLED
        def wait(self,timeout):
            pass
    monkeypatch.setattr(example,'LAMMPSTask',StartFailureTask)
    with pytest.raises(RuntimeError,match='injected startup'):
        example.run_pair(example.build_case(1),256,None,tmp_path/'fresh')
    assert StartFailureTask.instance.cancelled


def test_braced_3d_geometry_is_deterministic_connected_and_same_physics():
    a, b = example.build_braced_3d_case(), example.build_braced_3d_case()
    assert a['graph'].to_dict() == b['graph'].to_dict()
    assert len(a['node_ids']) == 120 and len(a['edge_ids']) == 573
    assert len(a['fixed_rows']) == 20
    assert np.count_nonzero(np.any(a['force'],axis=1)) == 20
    np.testing.assert_allclose(np.ptp(a['x0'],axis=0), [.005,.004,.003])
    np.testing.assert_array_equal(a['mass'],np.full(120,1e-9))
    np.testing.assert_array_equal(a['stiffness_N_per_m'],np.full(573,.25))
    neighbors={i:set() for i in range(120)}
    for i,j in a['edges']:neighbors[i].add(j);neighbors[j].add(i)
    seen={0};pending=[0]
    while pending:
        for j in neighbors[pending.pop()]-seen:seen.add(j);pending.append(j)
    assert len(seen)==120 and a['duration_s']==.001


@pytest.mark.parametrize('steps,stride,identity',[
    (256,2,'491fe2c4d790089c1d0b2496fe81ad5f209c1b6c0ad96c4ebc5d2ab318cbdc9d'),
    (512,4,'d2c7c0755535c330e495769c403547412b7922ac2a77008322d05d348cc7a10d'),
    (1024,8,'b42fb8ef097f2e205633d9730ba62f0463b3a96404ef8858d4583ad2bccde828')])
def test_braced_3d_uses_existing_sampled_profile(steps,stride,identity):
    case=example.build_braced_3d_case();prepared=example.prepare_case(case,steps)
    contract=json.loads(prepared.contract_json)
    assert prepared.identity==identity
    assert prepared.profile=='harmonic_sampled_v1'
    assert contract['native_frames']==steps+1
    assert contract['output_frames']==129 and contract['output_stride']==stride
    assert contract['length_unit']=='m'
    assert len(contract['forces_N'])==20
    assert len(contract['fixed_node_ids'])==20


@pytest.mark.parametrize('steps',[True,128,255,2048])
def test_braced_3d_resolution_is_bounded(steps):
    with pytest.raises(ValueError,match='requires 256, 512 or 1024'):
        example.output_stride(example.build_braced_3d_case(),steps)


def aligned_fixture():
    from types import SimpleNamespace as S
    case=example.build_case(1);steps=256;stride=2
    times=np.arange(steps+1)*case['duration_s']/steps
    frames=[S(step=i,time_s=t,node_ids=case['node_ids']) for i,t in enumerate(times)]
    native=S(trajectory=frames,thermo=[S(step=i) for i in range(steps+1)])
    state=dict(node_ids=case['node_ids'],times=times[::stride],positions=np.broadcast_to(case['x0'],(129,2,3)).copy(),velocities=np.zeros((129,2,3)),potential_energy=np.zeros(129),kinetic_energy=np.zeros(129))
    return case,native,state,steps,stride


def test_full_native_history_selects_exact_common_times():
    case,native,state,steps,stride=aligned_fixture()
    frames,thermo,times=example.aligned_frames(case,native,state,steps,stride)
    assert len(frames)==len(thermo)==129
    assert frames[1] is native.trajectory[2]
    assert [f.step for f in thermo]==list(range(0,257,2))
    np.testing.assert_array_equal(times,state['times'])


@pytest.mark.parametrize('defect,match',[
    ('missing_native','frame counts'),('wrong_step','integration steps'),
    ('unsampled_id','stable node IDs'),('taichi_id','stable node IDs'),
    ('unsampled_time','native physical times'),('sample_times','sample times'),
    ('positions','positions shape'),('velocities','velocities shape'),
    ('energy','potential_energy shape')])
def test_alignment_refuses_corrupted_or_mismatched_output(defect,match):
    case,native,state,steps,stride=aligned_fixture()
    if defect=='missing_native':native.trajectory.pop()
    elif defect=='wrong_step':native.thermo[1].step=7
    elif defect=='unsampled_id':native.trajectory[1].node_ids=[99,100]
    elif defect=='taichi_id':state['node_ids']=list(reversed(state['node_ids']))
    elif defect=='unsampled_time':native.trajectory[1].time_s+=.1
    elif defect=='sample_times':state['times']=state['times'][:-1]
    elif defect in ('positions','velocities'):state[defect]=state[defect][:-1]
    elif defect=='energy':state['potential_energy']=state['potential_energy'][:-1]
    with pytest.raises(RuntimeError,match=match):example.aligned_frames(case,native,state,steps,stride)


def test_dry_plan_never_reads_runtime_or_starts_provider(monkeypatch,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('provider activity in dry plan')
    monkeypatch.setattr(example,'load_runtime',forbidden)
    monkeypatch.setattr(example,'LAMMPSTask',forbidden)
    example.main(['--case','braced3d','--steps','256'])
    result=json.loads(capsys.readouterr().out)['braced3d']
    assert result['nodes']==120 and result['bonds']==573
    assert result['plans'][0]['output_stride']==2


@pytest.mark.parametrize('args',[
    ['--case','braced3d','--runtime','missing.json'],
    ['--case','braced3d','--execute'],
    ['--case','braced3d','--steps','256','256'],
    ['--case','braced3d','--steps','128']])
def test_cli_refuses_incomplete_or_ambiguous_execution(args):
    with pytest.raises(SystemExit) as err:example.main(args)
    assert err.value.code==2


def test_runtime_uses_exact_caller_inventory_and_checks_identity(monkeypatch,tmp_path):
    payload=dict(mode='caller-mode',python_executable='caller-python',file_sha256=[['caller-file','caller-hash']],environment=[['UCX_TLS','tcp,self']])
    path=tmp_path/'reviewed-runtime.json';path.write_text(json.dumps(payload),encoding='utf-8')
    seen={}
    class Checked:
        def __init__(self,**fields):seen.update(fields)
        def checked(self):seen['checked']=True;return self
    monkeypatch.setattr(example,'RuntimeSpec',Checked)
    example.load_runtime(path)
    assert seen['python_executable']=='caller-python'
    assert seen['file_sha256']==(('caller-file','caller-hash'),)
    assert seen['environment']==(('UCX_TLS','tcp,self'),)
    assert seen['checked']


def test_missing_runtime_leaves_output_absent(tmp_path):
    output=tmp_path/'new-result'
    with pytest.raises(FileNotFoundError):
        example.main(['--case','braced3d','--execute','--steps','256','--runtime',str(tmp_path/'absent.json'),'--output',str(output)])
    assert not output.exists()
