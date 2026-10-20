"""Regression tests for 2-D linear viscoelastic frame history."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def beam():
    graph=fn.StructureGraph(dimension=2)
    left=graph.add_node([0.,0.]);unused=graph.add_node([99.,99.]);graph.remove_node(unused)
    right=graph.add_node([10.,0.]);graph.add_edge(left,right,radius=.1)
    return graph,left,right


def record(name,result,reference=None):
    directory=os.environ.get('FIBERNET_VISCO_OUTPUT')
    if directory:
        directory=Path(directory);result.save(directory/(name+'.json'))
        if reference is not None:
            (directory/(name+'.reference.json')).write_text(json.dumps(reference,allow_nan=False)+'\n')


class NetworkAcceptance(unittest.TestCase):
    def test_axial_ramp_hold_analytic(self):
        g,left,right=beam();A=math.pi*(1e-4)**2;L=.01
        law=fn.LinearViscoelasticMaterial.standard_linear_solid(1e6,2e6,.2)
        errors=[]
        for n in (50,100,200):
            t=np.linspace(0.,1.,n+1);eps=.001*np.minimum(t/.2,1.)
            constraints={left:[[0.,0.,0.] for _ in t],right:[[float(L*x),None,None] for x in eps]}
            result=fn.simulate(g,backend='fem',mode='history',material=law,times=t,
                               constraints=constraints,length_unit='mm')
            q=np.where(t<=.2,2e6*.001*(-np.expm1(-t/.2)),
                       2e6*.001*(-math.expm1(-1))*np.exp(-(t-.2)/.2))
            expected=1e6*eps+q
            error=float(np.max(np.abs(result.axial_stress[:,0]-expected)))/3000.
            errors.append(error)
            np.testing.assert_allclose(result.axial_force[:,0],A*result.axial_stress[:,0],rtol=1e-14)
            record('axial_relax_'+str(n),result,{'stress_Pa':expected.tolist(),'normalized_error':error})
        self.assertLess(errors[-1],.01);self.assertLess(errors[1],.65*errors[0]);self.assertLess(errors[2],.65*errors[1])

    def test_force_ramp_creep_three_materials(self):
        g,left,right=beam();A=math.pi*1e-8;L=.01;t=np.linspace(0.,1.,201)
        tr=.2;stress0=1000.;stress=stress0*np.minimum(t/tr,1.);duration=np.minimum(t,tr)
        laws={'sls':fn.LinearViscoelasticMaterial.standard_linear_solid(1e6,2e6,.2),
              'kv':fn.LinearViscoelasticMaterial.kelvin_voigt(1e6,2e5),
              'maxwell':fn.LinearViscoelasticMaterial.maxwell(3e6,6e5)}
        for name,law in laws.items():
            constraints={left:[[0.,0.,0.] for _ in t]}
            loads={right:np.column_stack((stress*A,np.zeros_like(t),np.zeros_like(t))).tolist()}
            result=BeamFrameFEM().history(g,material=law,times=t,constraints=constraints,loads=loads,length_unit='mm')
            if name in ('sls','kv'):
                tau=.6 if name=='sls' else .2;b=2/3 if name=='sls' else 1.
                expected=stress0/tr/1e6*(duration-b*tau*(np.exp(-(t-duration)/tau)-np.exp(-t/tau)))
            else:
                integral=np.where(t<=tr,stress0*t*t/(2*tr),stress0*(t-tr/2))
                expected=stress/3e6+integral/6e5
            actual=result.displacements[:,result.node_ids.index(right),0]/L
            error=float(np.max(np.abs(actual-expected)))/max(float(np.max(np.abs(expected))),1e-30)
            record('creep_'+name,result,{'strain':expected.tolist(),'normalized_error':error})
            self.assertLess(error,.015,name)

    def test_bending_gauss_and_existing_elastic_solver(self):
        g,left,right=beam();law=fn.LinearViscoelasticMaterial(3e6)
        force=1e-7;t=[0.,1.]
        result=BeamFrameFEM().history(g,material=law,times=t,constraints={left:[[0.,0.,0.]]*2},
                     loads={right:[[0.,0.,0.],[0.,force,0.]]},length_unit='mm')
        row=result.node_ids.index(right);I=math.pi*1e-16/4;L=.01
        self.assertAlmostEqual(result.displacements[-1,row,1]/(force*L**3/(3*3e6*I)),1.,places=10)
        self.assertAlmostEqual(result.displacements[-1,row,2]/(force*L**2/(2*3e6*I)),1.,places=10)
        self.assertNotAlmostEqual(result.section_strain[-1,0,1],result.section_strain[-1,0,2],places=8)
        existing=BeamFrameFEM(E=3e6).solve_2d(np.array([[0],[1]]),np.array([[0.,0.],[L,0.]]),
                     np.array([1e-4]),forces=np.array([[0.,0.],[0.,force]]),fixed_nodes=[0],damping=0.)
        np.testing.assert_allclose(result.displacements[-1],existing['u'],rtol=1e-11,atol=1e-20)
        record('bending_elastic',result,{'tip_deflection_m':force*L**3/(3*3e6*I)})

    def test_transient_bending_sls_analytic(self):
        # Cantilever correspondence principle: force history convolves with the
        # same uniaxial SLS creep compliance, multiplied by L^3/(3I), L^2/(2I).
        g,left,right=beam();L=.01;I=math.pi*1e-16/4;force0=1e-9;tr=.2
        law=fn.LinearViscoelasticMaterial.standard_linear_solid(1e6,2e6,.2)
        tau_c=.6;b=2/3;errors=[]
        for n in (50,100,200):
            t=np.linspace(0.,1.,n+1);duration=np.minimum(t,tr)
            force=force0*np.minimum(t/tr,1.)
            compliance_integral=(duration-b*tau_c*(np.exp(-(t-duration)/tau_c)-np.exp(-t/tau_c)))/1e6
            expected_y=force0/tr*L**3/(3*I)*compliance_integral
            expected_rz=force0/tr*L**2/(2*I)*compliance_integral
            result=BeamFrameFEM().history(g,material=law,times=t,
                         constraints={left:[[0.,0.,0.] for _ in t]},
                         loads={right:np.column_stack((np.zeros_like(t),force,np.zeros_like(t))).tolist()},
                         length_unit='mm')
            row=result.node_ids.index(right)
            error=max(float(np.max(np.abs(result.displacements[:,row,1]-expected_y)))/max(abs(expected_y)),
                      float(np.max(np.abs(result.displacements[:,row,2]-expected_rz)))/max(abs(expected_rz)))
            errors.append(error)
            record('bending_sls_'+str(n),result,{'tip_deflection_m':expected_y.tolist(),
                   'tip_rotation_rad':expected_rz.tolist(),'normalized_error':error})
            self.assertGreater(float(np.max(np.abs(result.section_strain[:,:,1]-result.section_strain[:,:,2]))),0.)
        self.assertLess(errors[-1],.015)
        self.assertLess(errors[1],.65*errors[0]);self.assertLess(errors[2],.65*errors[1])

    def test_late_failure_rollback_and_restart_mismatches(self):
        import fibernet.sim._viscoelastic_fem as history_module
        g,left,right=beam();law=fn.LinearViscoelasticMaterial.standard_linear_solid(1e6,2e6,.2)
        first=BeamFrameFEM().history(g,material=law,times=[0.,.1],
                 constraints={left:[[0.,0.,0.]]*2},loads={right:[[0.,0.,0.],[0.,1e-9,0.]]},length_unit='mm')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.json';first.save(path)
            state=fn.ViscoelasticResult.load(path).final_state
        controls={left:[[0.,0.,0.]]*3};loads={right:[[0.,1e-9,0.],[0.,1.5e-9,0.],[0.,2e-9,0.]]}
        before=copy.deepcopy((state,g.to_dict(),controls,loads))
        real_solve=history_module.spsolve;calls=[]
        def fail_second_solve(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('injected late sparse solve failure')
            return real_solve(*args,**kwargs)
        args=dict(material=law,times=[.1,.2,.3],constraints=controls,loads=loads,
                  length_unit='mm',initial_state=state)
        with patch.object(history_module,'spsolve',side_effect=fail_second_solve):
            with self.assertRaisesRegex(RuntimeError,'injected late sparse solve failure'):
                BeamFrameFEM().history(g,**args)
        self.assertEqual(len(calls),2)  # A successful local advance precedes the failure.
        self.assertEqual((state,g.to_dict(),controls,loads),before)
        changed_graph=fn.StructureGraph.from_dict(copy.deepcopy(g.to_dict()))
        changed_graph.edges[next(iter(changed_graph.edges))].radius*=1.1
        changed_law=fn.LinearViscoelasticMaterial.standard_linear_solid(1.1e6,2e6,.2)
        corrupt=copy.deepcopy(state);corrupt['stored_energy']+=1.
        cases=((changed_graph,args,'material/geometry/identity/support'),
               (g,dict(args,material=changed_law),'material/geometry/identity/support'),
               (g,dict(args,initial_state=corrupt),'checksum/schema'))
        for target,parameters,message in cases:
            saved=copy.deepcopy((target.to_dict(),parameters['initial_state']))
            with patch.object(history_module,'spsolve',side_effect=AssertionError('invalid restart advanced')) as solve:
                with self.assertRaisesRegex(ValueError,message):BeamFrameFEM().history(target,**parameters)
                solve.assert_not_called()
            self.assertEqual((target.to_dict(),parameters['initial_state']),saved)
        self.assertEqual((state,g.to_dict(),controls,loads),before)
        record('rollback_checkpoint',first,{'late_failure_after_accepted_local_steps':1,
               'solver_calls_before_injected_failure':2,'restart_mismatch_refusals':['graph','material','checksum'],
               'caller_graph_controls_loads_and_state_unchanged':True})

    def test_serialized_restart_all_families(self):
        g,left,right=beam();t=np.linspace(0.,.2,21)
        laws=(fn.LinearViscoelasticMaterial(1e6),fn.LinearViscoelasticMaterial.kelvin_voigt(1e6,1e4),
              fn.LinearViscoelasticMaterial(1e6,((2e6,.1),(1e6,.3))))
        for index,law in enumerate(laws):
            controls={left:[[0.,0.,0.] for _ in t],right:[[float(1e-6*np.sin(10*x)),None,None] for x in t]}
            args=dict(material=law,constraints=controls,length_unit='mm')
            full=BeamFrameFEM().history(g,times=t,**args)
            first=BeamFrameFEM().history(g,times=t[:11],**dict(args,constraints={k:v[:11] for k,v in controls.items()}))
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'state.json';first.save(path);state=fn.ViscoelasticResult.load(path).final_state
            before=copy.deepcopy(state)
            second=BeamFrameFEM().history(g,times=t[10:],initial_state=state,
                        **dict(args,constraints={k:v[10:] for k,v in controls.items()}))
            self.assertEqual(state,before)
            np.testing.assert_array_equal(full.displacements[10:],second.displacements)
            np.testing.assert_array_equal(full.axial_stress[10:],second.axial_stress)
            self.assertEqual(full.final_state,second.final_state)
            record('restart_full_'+str(index),full);record('restart_first_'+str(index),first);record('restart_second_'+str(index),second)

    def test_refusals_and_no_mutation(self):
        g,left,right=beam();before=copy.deepcopy(g.to_dict());law=fn.LinearViscoelasticMaterial(1e6)
        args=dict(material=law,times=[0.,.1],length_unit='mm')
        for controls in ({},{left:[[0.,None,None]]*2},{left:[[0.,0.,0.],[None,0.,0.]]}):
            with self.assertRaises(ValueError): BeamFrameFEM().history(g,constraints=controls,**args)
        with self.assertRaises(ValueError):
            BeamFrameFEM().history(g,constraints={left:[[0.,0.,0.]]*2},loads={right:[[1.,0.,0.]]*2},**args)
        self.assertEqual(g.to_dict(),before)

    def test_representative_network_three_protocols(self):
        path=Path(__file__).resolve().parents[1]/'examples/viscoelastic_network.py'
        spec=importlib.util.spec_from_file_location('viscoelastic_example',path)
        example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)
        g=example.make_network();original=copy.deepcopy(g.to_dict())
        self.assertTrue(any(b-a>1 for a,b in zip(sorted(g.nodes),sorted(g.nodes)[1:])))
        for protocol in ('relaxation','creep','cyclic'):
            results=[]
            for n in (40,80,160):
                inputs=example.inputs(g,protocol,np.linspace(0.,2.,n+1))
                result=fn.simulate(g,backend='fem',mode='history',**inputs);results.append(result)
                record('network_'+protocol+'_'+str(n),result)
                self.assertLess(float(result.free_residual.max()),1e-8)
                self.assertGreater(float(np.max(np.abs(result.section_strain[:,:,1:]))),0.)
                self.assertGreaterEqual(min(result.energies['physical_dissipation']),0.)
                scale=max(max(map(abs,result.energies['work'])),max(result.energies['stored']),1e-30)
                self.assertLess(max(map(abs,result.energies['balance_residual']))/scale,1e-8)
            # Uniform material under displacement control can have dt-independent
            # displacement: include reactions, which retain physical memory.
            k_reference=4e6*math.pi*(4e-5)**2/.004
            arrays=[np.concatenate((x.displacements*np.array([1.,1.,.004]),
                     x.reactions/np.array([1.,1.,.004])/k_reference),axis=2) for x in results]
            coarse=float(np.linalg.norm(arrays[1][::2]-arrays[0]));fine=float(np.linalg.norm(arrays[2][::2]-arrays[1]))
            self.assertLess(fine,.8*coarse+1e-20,protocol)
            if protocol=='cyclic':
                times=np.linspace(0.,2.,161)
                first=fn.simulate(g,backend='fem',mode='history',**example.inputs(g,protocol,times[:81]))
                with tempfile.TemporaryDirectory() as d:
                    path=Path(d)/'network_restart.json';first.save(path)
                    checkpoint=fn.ViscoelasticResult.load(path).final_state
                second=fn.simulate(g,backend='fem',mode='history',initial_state=checkpoint,
                                   **example.inputs(g,protocol,times[80:]))
                self.assertEqual(second.final_state,results[-1].final_state)
                np.testing.assert_array_equal(second.bending_moment,results[-1].bending_moment[80:])
                record('network_cyclic_restart_first',first);record('network_cyclic_restart_second',second)
        self.assertEqual(g.to_dict(),original)


if __name__=='__main__':unittest.main()
