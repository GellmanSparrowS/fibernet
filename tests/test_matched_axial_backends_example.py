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
