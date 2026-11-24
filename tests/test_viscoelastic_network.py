"""Regression tests for 2-D linear viscoelastic frame history."""
import copy
import hashlib
import cmath
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
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

    def test_mapped_history_serial_continuity_and_in_memory_resume(self):
        graph = fn.StructureGraph(dimension=2)
        left, joint, right = [graph.add_node([x, 0.0]) for x in (0, 8, 20)]
        left_edge = graph.add_edge(left, joint, radius=0.1)
        right_edge = graph.add_edge(right, joint, radius=0.12)
        # Heterogeneous Maxwell counts, nonuniform dt, and reverse insertion.
        sls = fn.LinearViscoelasticMaterial.standard_linear_solid(1e6, 4e6, 0.08)
        kv = fn.LinearViscoelasticMaterial.kelvin_voigt(2e6, 1e4)
        assignment = {right_edge: kv, left_edge: sls}
        times = np.array([0., 0.1, 0.25, 0.8])
        stroke = np.array([0., 2e-6, 2e-6, 2e-6])
        before = graph.to_dict()

        def run(time, dx, material=assignment, **kw):
            constraints = {left: [[0., 0., 0.]]*len(time),
                           right: [[x, None, None] for x in dx]}
            return fn.simulate(graph, backend='fem', mode='history', material=material,
                               times=time, constraints=constraints, length_unit='mm', **kw)

        result = run(times, stroke)
        row = result.node_ids.index(joint)
        # Independent serial compliance using the first-step implicit tangents:
        # memory is zero at t0, displacements add, common axial force.
        stiffness = math.pi * np.array([1e-4, 1.2e-4])**2 * np.array([
            1e6 + 4e6 * 0.08/(0.08+0.1), 2e6 + 1e4/0.1
        ]) / np.array([0.008, 0.012])
        force = stroke[1] / np.sum(1/stiffness)
        np.testing.assert_allclose(result.axial_force[1], [force, force], rtol=1e-12, atol=0)
        self.assertAlmostEqual(result.displacements[1, row, 0] / (force/stiffness[0]), 1.)
        self.assertGreater(abs(result.displacements[-1,row,0]-result.displacements[1,row,0]),
                           5e-8)
        ledger = result.energies
        np.testing.assert_allclose(ledger['work'][1:], ledger['work'][1], rtol=1e-12, atol=0)
        head = run(times[:3], stroke[:3])
        self.assertIn([], head.final_state['viscous_strain'])  # Zero Maxwell branches.
        tail = run(times[2:], stroke[2:], dict(reversed(list(assignment.items()))),
                   initial_state=head.final_state)
        np.testing.assert_array_equal(tail.displacements, result.displacements[2:])
        np.testing.assert_array_equal(tail.axial_force, result.axial_force[2:])
        self.assertEqual(tail.final_state, result.final_state)
        with self.assertRaisesRegex(ValueError, 'material/geometry'):
            run(times[2:], stroke[2:], {left_edge: kv, right_edge: sls},
                initial_state=head.final_state)
        self.assertEqual(graph.to_dict(), before)

        def rechecksum(state):
            record = {key: value for key, value in state.items() if key != 'checksum'}
            state['checksum'] = hashlib.sha256(json.dumps(
                record, sort_keys=True, separators=(',', ':'), allow_nan=False
            ).encode('utf-8')).hexdigest()

        for key, message in [('metadata', 'mapped result metadata'),
                             ('energies', 'mapped result energy fields')]:
            malformed = result.to_dict()
            malformed[key] = None
            with patch('fibernet.sim._viscoelastic_fem._json',
                       side_effect=AssertionError('must reject before JSON clone')):
                with self.assertRaisesRegex(ValueError, message):
                    fn.ViscoelasticResult(malformed)

        for timestamp in (10**330, float('inf'), float('nan'), True):
            with self.subTest(timestamp=timestamp):
                malformed = result.to_dict()
                malformed['times'][1] = timestamp
                with self.assertRaisesRegex(ValueError, 'mapped result times requires finite'):
                    fn.ViscoelasticResult(malformed)
                for initial, previous, displacements in (
                    (None, 0., [0., stroke[1]]),
                    (head.final_state, times[2], stroke[2:]),
                ):
                    with patch('fibernet.sim._viscoelastic_fem._geometry',
                               side_effect=AssertionError('must reject malformed time before geometry')):
                        with self.assertRaisesRegex(ValueError, 'mapped history times require finite'):
                            run([previous, timestamp], displacements, initial_state=initial)
        typed_times = [np.int64(0), np.float64(times[1]), np.float64(times[2]), np.float64(times[3])]
        self.assertEqual(run(typed_times, stroke).to_dict(), result.to_dict())

        malformed = result.to_dict()
        state = malformed['final_state']
        malformed['metadata']['material'][0][1]['branches'] = [[5e5, 0.1]]*9
        state['material_by_edge'] = copy.deepcopy(malformed['metadata']['material'])
        state['viscous_strain'][0] = [[[0., 0., 0.]] for _ in range(9)]
        rechecksum(state)
        with patch.object(fn.LinearViscoelasticMaterial, 'from_dict',
                          side_effect=AssertionError('must bound branches before material construction')):
            with self.assertRaisesRegex(ValueError, 'at most eight branches'):
                fn.ViscoelasticResult(malformed)

        bad_state = head.final_state
        bad_state['stored_energy'] = str(bad_state['stored_energy'])
        rechecksum(bad_state)
        with self.assertRaisesRegex(ValueError, 'mapped restart stored_energy requires finite'):
            run(times[2:], stroke[2:], initial_state=bad_state)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / 'new-parent' / 'mapped.json'
            with self.assertRaisesRegex(ValueError, 'persistence is unsupported'):
                result.save(path)
            with patch.object(BeamFrameFEM, 'history', side_effect=AssertionError('must not start')):
                with self.assertRaisesRegex(ValueError, 'persistence is unsupported'):
                    run(times, stroke, save_path=path)
            self.assertFalse(path.parent.exists())
        with patch('fibernet.sim._viscoelastic_fem._geometry',
                   side_effect=AssertionError('must not allocate geometry')):
            with self.assertRaisesRegex(ValueError, 'sample budget'):
                run(np.linspace(0., 1., 514), np.zeros(514))

    def test_frequency_sweep_cli_physical_loop_and_discrete_ledger(self):
        script = Path(__file__).resolve().parents[1] / 'examples/viscoelastic_frequency_sweep.py'
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(script), '--quick', '--amplitude-um', '0.001',
                       '--relaxation-time-s', '1e6']
            env = dict(os.environ)
            # Use the same package as this test, whether installed or in-tree.
            roots = [str(Path(fn.__file__).resolve().parents[1])]
            if env.get('PYTHONPATH'):
                roots.append(env['PYTHONPATH'])
            env['PYTHONPATH'] = os.pathsep.join(roots)
            run = subprocess.run(command, cwd=directory, env=env, capture_output=True,
                                 text=True, check=True, timeout=60)
            self.assertEqual(list(Path(directory).iterdir()), [])
            for flag, value in [('--amplitude-um', 'nan'), ('--amplitude-um', '1e-200'),
                                ('--relaxation-time-s', '1e308')]:
                with self.subTest(flag=flag, value=value):
                    invalid = subprocess.run(command + [flag, value], cwd=directory,
                                             env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(invalid.returncode, 2)
                    self.assertEqual(invalid.stdout, '')
                    self.assertIn(f'{flag} must be', invalid.stderr)
                    self.assertNotIn('RuntimeWarning', invalid.stderr)
        result = json.loads(run.stdout)
        self.assertEqual((result['nodes'], result['edges'], result['samples_per_cycle']),
                         (16, 33, 32))
        self.assertEqual(result['amplitude_m'], 1e-9)
        self.assertEqual(result['relaxation_time_s'], 1e6)
        cases = result['cases']
        self.assertEqual([case['omega_tau'] for case in cases], [0.2, 1.0, 5.0])
        # The unknown frame geometry factor cancels in these ratios. With
        # harmonic strain, a single Maxwell branch has BE transfer H.
        predictions = []
        phase_step = 2 * math.pi / 32
        lag = cmath.exp(-1j * phase_step)
        for case in cases:
            x = case['omega_tau']
            self.assertAlmostEqual(case['frequency_Hz'], x / (2 * math.pi * 1e6), places=14)
            physical, numerical = case['physical_loss_J'], case['numerical_loss_J']
            self.assertGreater(min(physical, numerical), 0.)
            self.assertAlmostEqual(case['previous_physical_loss_J'] / physical, 1., places=4)
            self.assertAlmostEqual((case['stored_energy_change_J'] + physical + numerical)
                                   / case['endpoint_work_J'], 1., places=10)
            a = 1 / (1 + phase_step / x)
            h = a * (1 - lag) / (1 - a * lag)
            expected_physical_scale = abs(h)**2 / x
            predictions.append(expected_physical_scale)
            self.assertAlmostEqual(case['trapezoid_loop_work_J'] / physical,
                                   h.imag * math.sin(phase_step) /
                                   (phase_step * expected_physical_scale), places=4)
        for end in (0, 2):
            actual = cases[1]['physical_loss_J'] / cases[end]['physical_loss_J']
            self.assertGreater(actual, 2.2)
            self.assertAlmostEqual(actual, predictions[1] / predictions[end], places=4)

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
