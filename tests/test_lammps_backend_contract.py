"""Pure protocol tests. Synthetic parser fixtures are NOT native evidence."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations.lammps_backend import (
    HarmonicBond, RunSpec, RuntimeSpec, ResourceLimits, PreparedRun,
    prepare_harmonic_beads, validate_prepared, parse_outputs, LAMMPSTask,
    TaskState, LAMMPSContractError,
)


def graph():
    g = StructureGraph(dimension=3)
    a=g.add_node([0,0,0], boundary=(True,False,False,False,False,False))
    b=g.add_node([1,0,0])
    g.add_edge(a,b,n_internal=0)
    d=g.to_dict()
    d['nodes'][0]['id']=10; d['nodes'][1]['id']=40
    d['edges'][0].update(id=77,node_i=10,node_j=40)
    d['next_node_id']=41;d['next_edge_id']=78
    return StructureGraph.from_dict(d)


def prepared(**changes):
    args=dict(length_unit='mm',maximum_bond_length_m=.01,masses_kg={10:1e-9,40:2e-9},
              bonds={77:HarmonicBond(2.0,0.001)},
              run=RunSpec(1e-6,2,1,((-0.01,0.01),)*3))
    args.update(changes)
    return prepare_harmonic_beads(graph(),**args)


def fixture(p):
    contract=json.loads(p.contract_json)
    g=json.loads(p.graph_json)
    initial_velocity=dict(contract['velocities_m_s'])
    expected_types=dict(contract['atom_to_type_id'])
    dump=['ITEM: UNITS','si']
    thermo=['Step Atoms PotEng KinEng TotEng Pxx Pyy Pzz Pxy Pxz Pyz']
    for step in range(0,p.steps+1,p.sample_every):
        dump += ['ITEM: TIME',str(step*p.timestep_s),'ITEM: TIMESTEP',str(step),
                 'ITEM: NUMBER OF ATOMS',str(len(p.node_ids)),'ITEM: BOX BOUNDS ff ff ff']
        dump += [' '.join(map(str,row)) for row in contract['box_m']]
        dump += ['ITEM: ATOMS id type x y z vx vy vz fx fy fz']
        # Deliberately reverse row order to test stable-ID reassembly.
        for i in reversed(range(len(g['nodes']))):
            node=g['nodes'][i]
            pos=[x*contract['length_to_m'] for x in node['position']]
            vel=initial_velocity.get(node['id'],[0,0,0])
            dump += [' '.join(map(str,[i+1,expected_types[i+1],*pos,*vel,0,0,0]))]
        thermo += [f'{step} {len(p.node_ids)} 1 2 3 4 5 6 7 8 9']
    thermo += ['Loop time of 0.01 on 1 procs',p.completion_marker]
    return '\n'.join(dump)+'\n','\n'.join(thermo)+'\n'


class ContractTests(unittest.TestCase):
    def test_complete_bond_rows_and_coefficient_factor(self):
        p=prepared()
        atoms=p.data_text.split('Atoms # bond\n\n')[1].split('\n\n')[0].splitlines()
        self.assertTrue(all(len(a.split())==6 for a in atoms))
        self.assertIn('bond_coeff 1 1 0.001',p.input_text)
        self.assertEqual(p.manifest()['atom_to_node_id'],{'1':10,'2':40})
        self.assertEqual(p.manifest()['bond_to_edge_id'],{'1':77})
        self.assertIn('thermo_modify norm no',p.input_text)
        self.assertIn('units yes time yes',p.input_text)

    def test_shared_junction_has_one_atom(self):
        g=graph();third=g.add_node([1,1,0]);eid=g.add_edge(40,third,n_internal=0)
        p=prepare_harmonic_beads(g,length_unit='mm',maximum_bond_length_m=.01,masses_kg={10:1,40:2,third:3},
          bonds={77:HarmonicBond(1,.001),eid:HarmonicBond(2,.001)},run=RunSpec(1e-6,2,1,((-1,1),)*3))
        self.assertIn('3 atoms',p.data_text)
        self.assertIn('1 1 1 2\n2 2 2 3',p.data_text)
        self.assertEqual(p.node_ids,(10,40,third))

    def test_library_exports_are_callable(self):
        import fibernet.integrations as api
        self.assertIs(api.prepare_harmonic_beads,prepare_harmonic_beads)
        self.assertIs(api.LAMMPSTask,LAMMPSTask)

    def test_source_graph_unmodified(self):
        g=graph();before=json.dumps(g.to_dict(),sort_keys=True)
        prepare_harmonic_beads(g,length_unit='mm',maximum_bond_length_m=.01,masses_kg={10:1,40:2},bonds={77:HarmonicBond(1,.001)},run=RunSpec(.001,0,1,((-1,1),)*3))
        self.assertEqual(json.dumps(g.to_dict(),sort_keys=True),before)

    def test_material_is_provenance_only(self):
        p=prepared()
        self.assertIn('no inferred force-field',json.loads(p.contract_json)['material_mapping'])
        self.assertEqual(json.loads(p.graph_json)['edges'][0]['material'],graph().to_dict()['edges'][0]['material'])

    def test_constraint_force_velocity_contract(self):
        p=prepared(fixed_node_ids=(10,),forces_N={40:(1,2,3)},velocities_m_s={40:(.1,.2,.3)})
        self.assertIn('group held id 1',p.input_text)
        self.assertIn('fix load0 loaded0 addforce 1 2 3',p.input_text)
        self.assertIn('fix integrate mobile nve',p.input_text)
        self.assertNotIn('setforce',p.input_text)
        self.assertEqual(validate_prepared(p),p)

    def test_run_zero_compiles_and_parses(self):
        p=prepared(run=RunSpec(1e-6,0,1,((-0.01,0.01),)*3))
        result=parse_outputs(p,*fixture(p))
        self.assertEqual([f.step for f in result.trajectory],[0])

    def test_output_remaps_unsorted_native_rows(self):
        p=prepared();r=parse_outputs(p,*fixture(p))
        self.assertEqual(r.trajectory[0].node_ids,(10,40))
        self.assertEqual(r.trajectory[0].positions_m,((0,0,0),(.001,0,0)))
        self.assertEqual(r.thermo[0].pressure_tensor_Pa,(4,5,6,7,8,9))

    def test_reject_invalid_masses(self):
        for mass in (0,-1,float('nan'),float('inf'),True,'1'):
            with self.subTest(mass=mass),self.assertRaises(ValueError):prepared(masses_kg={10:mass,40:2})

    def test_reject_invalid_coefficients(self):
        for value in (0,-1,float('inf'),True):
            with self.subTest(value=value),self.assertRaises(ValueError):prepared(bonds={77:HarmonicBond(value,.001)})
            with self.subTest(value=value),self.assertRaises(ValueError):prepared(bonds={77:HarmonicBond(1,value)})

    def test_reject_alias_keys(self):
        for changes in [dict(masses_kg={10.0:1,40:2}),dict(bonds={77.0:HarmonicBond(1,.001)}),dict(forces_N={40.0:(1,0,0)})]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):prepared(**changes)

    def test_reject_missing_unknown_and_fixed_load(self):
        cases=[dict(masses_kg={10:1}),dict(bonds={}),dict(forces_N={9:(1,0,0)}),
               dict(fixed_node_ids=(10,10)),dict(fixed_node_ids=(10,40)),
               dict(fixed_node_ids=(10,),forces_N={10:(1,0,0)}),
               dict(fixed_node_ids=(10,),velocities_m_s={10:(1,0,0)}),dict(length_unit='real')]
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):prepared(**case)

    def test_reject_curves_parallel_edges(self):
        for curved in (True,False):
            g=graph()
            if curved:
                import numpy as np
                g.edges[77].internal_points=np.array([[.5,.1,0]])
            else:g.add_edge(10,40,allow_parallel=True,n_internal=0)
            with self.assertRaises(ValueError):prepare_harmonic_beads(g,length_unit='mm',maximum_bond_length_m=.01,masses_kg={10:1,40:2},bonds={e:HarmonicBond(1,.001) for e in g.edges},run=RunSpec(.001,2,1,((-1,1),)*3))

    def test_reject_script_or_contract_tampering(self):
        p=prepared()
        for forged in [replace(p,input_text=p.input_text+'shell touch forbidden\n'),replace(p,data_text=p.data_text+'\n'),replace(p,node_ids=(40,10)),replace(p,steps=3)]:
            with self.subTest(forged=forged.identity),self.assertRaises(ValueError):validate_prepared(forged)

    def test_reject_bad_dump_variants(self):
        p=prepared();dump,log=fixture(p)
        variants=[dump.replace('ITEM: UNITS\nsi','ITEM: UNITS\nreal'),dump.replace('ITEM: TIME\n0.0','ITEM: TIME\n1.0'),dump.replace('ff ff ff','pp pp pp'),dump.replace('id type x y z','id type xs ys zs'),dump.replace('0.001','nan'),dump[:-12],dump+dump,dump.replace('2 2 0.001','1 1 0.001'),dump.replace('2 2 0.001','2 2 0.002')]
        for bad in variants:
            with self.subTest(bad=bad[:40]),self.assertRaises(ValueError):parse_outputs(p,bad,log)

    def test_reject_bad_thermo_variants(self):
        p=prepared();dump,log=fixture(p)
        for bad in [log.replace(p.completion_marker,'wrong'),log.replace('1 2 3 4','1 2 4 4'),log.replace('1 2 3 4','4 -1 3 4'),log.replace('1 2 3 4','nan 2 3 4'),log.replace('0 2 1 2','0.5 2 1 2'),log+log]:
            with self.subTest(bad=bad[:40]),self.assertRaises(ValueError):parse_outputs(p,dump,bad)

    def test_budget_validation(self):
        p=prepared()
        with self.assertRaises(ValueError):parse_outputs(p,*fixture(p),max_output_bytes=1)
        for limits in [ResourceLimits(cpu_seconds=0),ResourceLimits(cpu_seconds=True),ResourceLimits(wall_seconds=float('inf')),ResourceLimits(output_bytes=100)]:
            with self.assertRaises(ValueError):limits.checked()

    def test_reject_nonrepresentable_derived_values(self):
        cases=[dict(bonds={77:HarmonicBond(5e-324,.001)}),
               dict(run=RunSpec(1e308,100000,1,((-1,1),)*3)),
               dict(run=RunSpec(1e-6,2,1,((-1e308,1e308),)*3))]
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):prepared(**case)

    def test_pair_none_has_no_pair_coeff_and_explicit_reach(self):
        p=prepared(maximum_bond_length_m=.0015)
        self.assertIn('pair_style none\n',p.input_text)
        self.assertNotIn('pair_coeff',p.input_text)
        self.assertNotIn('pair_style zero',p.input_text)
        self.assertIn('comm_modify cutoff 0.0015\n',p.input_text)
        self.assertNotIn('pair',json.loads(p.contract_json)['required_styles'])

    def test_reject_unbounded_or_undersized_reach(self):
        for reach in (None,True,0,-1,float('inf'),float('nan'),.0009,1.):
            with self.subTest(reach=reach),self.assertRaises(ValueError):
                prepared(maximum_bond_length_m=reach)
        with self.assertRaises(ValueError):
            prepared(maximum_bond_length_m=.001,bonds={77:HarmonicBond(1,.002)})

    def test_reject_decimated_pair_none_trajectory(self):
        with self.assertRaises(ValueError):
            prepared(run=RunSpec(1e-6,4,2,((-.01,.01),)*3))

    def test_reject_later_out_of_reach_or_box_site(self):
        p=prepared(maximum_bond_length_m=.0015);dump,log=fixture(p)
        for value in ('0.002','0.02','0.0'):
            bad=dump.rsplit('2 2 0.001',1)
            bad=('2 2 '+value).join(bad)
            with self.subTest(value=value),self.assertRaises(ValueError):parse_outputs(p,bad,log)

    def test_native_file_limit_is_separate_from_artifacts(self):
        limits=ResourceLimits(output_bytes=128*1024,file_size_bytes=8*1024*1024).checked()
        self.assertEqual(limits.output_bytes,128*1024)
        self.assertEqual(limits.file_size_bytes,8*1024*1024)
        with self.assertRaises(ValueError):ResourceLimits(file_size_bytes=0).checked()

    def test_invalid_runtime_fails_before_files(self):
        with self.assertRaises(ValueError):RuntimeSpec('cli','/missing','/missing').checked()
        with self.assertRaises(ValueError):RuntimeSpec('python',sys.executable).checked()


class FakeProcess:
    pid=123456789
    returncode=0
    def poll(self): return self.returncode
    def wait(self,timeout=None):return self.returncode


@unittest.skipUnless(os.name == 'posix', 'process-group lifecycle tests require POSIX')
class LifecycleTests(unittest.TestCase):
    def setUp(self):
        # Never send a real signal from a pure lifecycle fixture.
        group_patch=patch('os.killpg')
        group_patch.start()
        self.addCleanup(group_patch.stop)

    def make_task(self,root):
        with patch.object(RuntimeSpec,'checked',lambda self:self):
            return LAMMPSTask(prepared(),RuntimeSpec('cli',sys.executable,'/fake/native',expected_banner='LAMMPS (test fixture)'),Path(root)/'run')

    def test_cancel_before_start_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root);task.cancel()
            self.assertEqual(task.wait(),TaskState.CANCELLED)
            self.assertFalse(task.workdir.exists())
            with self.assertRaises(ValueError):task.start()

    def test_mock_success_is_strictly_parsed(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root)
            def fake_spawn(*args,**kw):
                self.assertEqual(args[0][1:3],['-I','-S'])
                dump,log=fixture(task.prepared)
                (task.workdir/'trajectory.dump').write_text(dump)
                (task.workdir/'log.lammps').write_text('LAMMPS (test fixture)\n'+log)
                return FakeProcess()
            with patch.object(RuntimeSpec,'checked',lambda self:self),patch('subprocess.Popen',side_effect=fake_spawn):
                task.start();self.assertEqual(task.wait(2),TaskState.SUCCEEDED)
            self.assertEqual(task.result.identity,task.prepared.identity)

    def test_fast_exit_artifact_limit(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root);task.limits=ResourceLimits(output_bytes=8192)
            def fake_spawn(*args,**kw):
                (task.workdir/'oversized').write_bytes(b'x'*8193)
                return FakeProcess()
            with patch.object(RuntimeSpec,'checked',lambda self:self),patch('subprocess.Popen',side_effect=fake_spawn):
                task.start();self.assertEqual(task.wait(2),TaskState.RESOURCE_LIMIT)

    def test_missing_output_is_failed_not_empty_success(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root)
            with patch.object(RuntimeSpec,'checked',lambda self:self),patch('subprocess.Popen',return_value=FakeProcess()),patch('os.killpg'):
                task.start();self.assertEqual(task.wait(2),TaskState.FAILED)
            self.assertIsNone(task.result)

    def test_group_kill_after_leader_exit(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root);task._process=FakeProcess();task.state=TaskState.RUNNING
            with patch('os.killpg') as kill:
                task.cancel()
                self.assertEqual(kill.call_count,2)
            self.assertEqual(task.state,TaskState.CANCELLED)

    def test_watchdog_timeout_without_user_poll(self):
        import io,time
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root);task.workdir.mkdir();task._stream=io.BytesIO()
            task._process=FakeProcess();task._process.poll=lambda:None
            task._started=time.monotonic()-20;task.state=TaskState.RUNNING
            with patch('os.killpg') as kill:task._watch()
            self.assertEqual(task.wait(),TaskState.TIMED_OUT)
            self.assertGreaterEqual(kill.call_count,2)

    def test_native_nonzero_exit_fails_and_cleans(self):
        import io
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root);task.workdir.mkdir();task._stream=io.BytesIO()
            task._process=FakeProcess();task._process.returncode=17;task.state=TaskState.RUNNING
            with patch('os.killpg') as kill:task._watch()
            self.assertEqual(task.wait(),TaskState.FAILED)
            self.assertIn('17',task.error);self.assertEqual(kill.call_count,1)

    def test_recursive_artifact_accounting_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root);(task.workdir/'scratch').mkdir(parents=True)
            (task.workdir/'scratch'/'native.tmp').write_bytes(b'x'*42)
            self.assertEqual(task._artifact_bytes(),42)
            (task.workdir/'link').symlink_to(task.workdir/'scratch'/'native.tmp')
            with self.assertRaises(ValueError):task._artifact_bytes()

    def test_existing_workdir_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root);task.workdir.mkdir();sentinel=task.workdir/'sentinel';sentinel.write_text('keep')
            with patch.object(RuntimeSpec,'checked',lambda self:self),self.assertRaises(FileExistsError):task.start()
            self.assertEqual(sentinel.read_text(),'keep')

    def test_postspawn_failure_kills_group(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make_task(root)
            with patch.object(RuntimeSpec,'checked',lambda self:self),patch('subprocess.Popen',return_value=FakeProcess()),patch('threading.Thread.start',side_effect=RuntimeError('injected')),patch('os.killpg') as kill:
                with self.assertRaises(RuntimeError):task.start()
                self.assertEqual(kill.call_count,2)
                self.assertEqual(task.wait(),TaskState.FAILED)


if __name__=='__main__':unittest.main()
