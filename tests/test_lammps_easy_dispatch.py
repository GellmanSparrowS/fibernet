"""Source/mocked adapter checks. No native engine execution or qualification."""
import ast
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import lammps_backend as backend
from fibernet.sim.accelerated import SimResult, _validate_dynamics_state
from fibernet.sim import _lammps_result as adapter


def graph(unit='mm', reverse=True):
    g = StructureGraph(dimension=3)
    g.add_node([0,0,0]); g.add_node([1.1 if unit == 'mm' else .0011,0,0]);g.add_edge(0,1,n_internal=0)
    d=g.to_dict();d['nodes'][0]['id']=10;d['nodes'][1]['id']=40
    d['edges'][0].update(id=77,node_i=10,node_j=40);d['next_node_id']=41;d['next_edge_id']=78
    if reverse: d['nodes'].reverse()
    return StructureGraph.from_dict(d)


def options(unit='mm'):
    return dict(length_unit=unit,maximum_bond_length_m=.01,masses_kg={10:1.,40:2.},
                bonds={77:backend.HarmonicBond(2.,.001)},run=backend.RunSpec(.01,2,1,((-.01,.01),)*3))


def raw_result(p):
    frames=[];thermo=[]
    for step in range(3):
        x=.0011+step*.0001
        frames.append(backend.TrajectoryFrame(step,step*.01,(10,40),((0,0,0),(x,0,0)),((0,0,0),(.01,0,0)),((7,8,9),(-4,-5,-6))))
        pe=(x-.001)**2
        thermo.append(backend.ThermoFrame(step,pe,.0001,pe+.0001,(1,2,3,4,5,6)))
    return backend.LAMMPSRunResult(p.identity,tuple(frames),tuple(thermo),'private fixture path not exported')


RUNTIME=backend.RuntimeSpec('cli_serial_static','/fixture/python','/fixture/lammps',expected_version=20250722,
    expected_banner='LAMMPS (22 Jul 2025)',file_sha256=(('/fixture/python','a'*64),('/fixture/lammps','b'*64)))


def mapped(unit='mm', record=True):
    p=backend.prepare_harmonic_beads(graph(unit),**options(unit))
    return adapter.to_sim_result(p,raw_result(p),RUNTIME,.125,record)


class AdapterTests(unittest.TestCase):
    def test_mapping_units_identity_and_graph_unchanged(self):
        g=graph();before=g.to_dict();p=backend.prepare_harmonic_beads(g,**options())
        r=adapter.to_sim_result(p,raw_result(p),RUNTIME,.125,True)
        self.assertEqual(before,g.to_dict());self.assertEqual(r.metadata['node_ids'],[10,40]);self.assertEqual(r.metadata['edge_ids'],[77])
        np.testing.assert_allclose(r.deformed_positions,[[0,0,0],[1.3,0,0]])
        np.testing.assert_allclose(r.displacements,[[0,0,0],[.2,0,0]])
        np.testing.assert_allclose(r.edge_forces,[.0006]);np.testing.assert_allclose(r.edge_stretches,[1.3])
        self.assertAlmostEqual(r.energy,9e-8);self.assertEqual(r.time_seconds,.125)
        self.assertEqual(r.dynamics_state['final']['time'],.02)
        self.assertEqual(r.dynamics_state['final']['nodal_forces'],[[7,8,9],[-4,-5,-6]])
        self.assertEqual(r.dynamics_state['final']['pressure_tensor'],[1,2,3,4,5,6])
        for key in ('stress','reactions','work'):self.assertIsNone(r.metadata[key])
        self.assertIsNone(r.algorithmic_energy_ledger)
        self.assertNotIn('/fixture',json.dumps(r.to_dict(True)))
        self.assertNotIn('private fixture',json.dumps(r.to_dict(True)))

    def test_m_mm_equivalence_and_row_order(self):
        mm,m=mapped(),mapped('m')
        for key in ('deformed_positions','displacements','positions_trajectory'):
            np.testing.assert_allclose(np.asarray(getattr(mm,key))*.001,getattr(m,key))
        for key in ('positions','velocities','nodal_forces','potential_energy','pressure_tensor'):
            self.assertEqual(mm.dynamics_state[key],m.dynamics_state[key])
        np.testing.assert_equal(mm.edge_forces,m.edge_forces)
        self.assertEqual(mm.energy,m.energy)
        self.assertAlmostEqual(mm.max_displacement*.001,m.max_displacement)
        p=backend.prepare_harmonic_beads(graph(reverse=False),**options())
        r=adapter.to_sim_result(p,raw_result(p),RUNTIME,0,True)
        np.testing.assert_equal(r.deformed_positions,mm.deformed_positions)

    def test_signed_force_uses_explicit_equilibrium(self):
        p=backend.prepare_harmonic_beads(graph(),**{**options(),'bonds':{77:backend.HarmonicBond(2.,.002)}})
        r=adapter.to_sim_result(p,raw_result(p),RUNTIME,0)
        self.assertAlmostEqual(r.edge_forces[0],-.0014)
        self.assertAlmostEqual(r.edge_stretches[0],.65)

    def test_detail_and_summary_roundtrip(self):
        r=mapped()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json'
            for detailed in (True,False):
                r.save(p,detailed=detailed);loaded=SimResult.load(p)
                self.assertEqual(loaded.to_dict(detailed),r.to_dict(detailed))
                self.assertEqual(loaded.dynamics_state['availability'],'available' if detailed else 'summary_only')
                if not detailed:
                    self.assertNotIn('positions',loaded.dynamics_state)
                    self.assertNotIn('pressure_tensor',loaded.dynamics_state['final'])
                    self.assertNotIn('velocities',loaded.to_dict(True)['dynamics_state']['final'])
        default=mapped(record=False)
        self.assertIsNone(default.dynamics_state);self.assertEqual(default.positions_trajectory,[])
        self.assertNotIn('dynamics_state',default.to_dict(True))

    def test_state_malformed_rejected(self):
        original=mapped()
        changes=[lambda s:s.update(backend='taichi'),lambda s:s.update(units={}),
            lambda s:s['steps'].__setitem__(1,2),lambda s:s['node_ids'].reverse(),
            lambda s:s['times'].__setitem__(1,.5),lambda s:s['nodal_forces'][0].pop(),
            lambda s:s['pressure_tensor'][0].pop(),lambda s:s['total_energy'].__setitem__(1,5),
            lambda s:s['final'].update(time=5),lambda s:s.update(work_ledger={'value':0}),
            lambda s:s.update(source_identity={'prepared_sha256':'x','graph_sha256':'y'}),
            lambda s:s['final']['pressure_tensor'].__setitem__(0,True),
            lambda s:s['final'].update(step=10**30)]
        for change in changes:
            with self.subTest(change=change):
                r=copy.deepcopy(original);change(r.dynamics_state)
                with self.assertRaises(ValueError):r.to_dict(True)
        r=copy.deepcopy(original);r.deformed_positions*=1000
        with self.assertRaises(ValueError):r.to_dict(True)
        r=copy.deepcopy(original);r.algorithmic_energy_ledger={}
        with self.assertRaises(ValueError):r.to_dict(True)

    def test_recompute_rejected_before_mutation(self):
        r=mapped();before=r.to_dict(True)
        with self.assertRaises(ValueError):r.compute_detailed(graph(),stiffness=8)
        self.assertEqual(before,r.to_dict(True))

    def test_geometry_declarations_and_shapes_rejected(self):
        for change in (lambda r:r.metadata.update(original_graph_length_to_m=1),
                       lambda r:r.metadata['field_units'].update(displacements='m'),
                       lambda r:setattr(r,'deformed_positions',r.deformed_positions[:1]),
                       lambda r:setattr(r,'positions_trajectory',[r.positions_trajectory[-1]]),
                       lambda r:setattr(r,'displacements',np.array([[0,0,0]])),
                       lambda r:setattr(r,'displacements',np.full((2,3),np.nan))):
            r=mapped();change(r)
            with self.assertRaises(ValueError):r.to_dict(True)

    def test_native_axial_schema_unchanged(self):
        state=dict(schema='native_axial_state/1',representation='summary',availability='summary_only',
            units=None,dt=.01,final=dict(step=0,time=0.,potential_energy=0.,kinetic_energy=0.,total_energy=0.,near_cutoff=False),
            work_ledger=dict(status='unavailable',value=None,reason='not_collected'),cutoff_check='sampled_only')
        self.assertEqual(_validate_dynamics_state(state),state)
        r=SimResult(dynamics_state=state)
        self.assertEqual(r.to_dict()['dynamics_state'],state)

    def test_incomplete_and_wrong_identity_rejected(self):
        p=backend.prepare_harmonic_beads(graph(),**options());raw=raw_result(p)
        for changed in (replace(raw,identity='x'),replace(raw,trajectory=raw.trajectory[:-1]),replace(raw,thermo=raw.thermo[:-1])):
            with self.assertRaises(ValueError):adapter.to_sim_result(p,changed,RUNTIME,0)

    def test_capabilities_detached(self):
        a=adapter.source_capabilities();a['constraints'].clear()
        self.assertTrue(adapter.source_capabilities()['constraints'])
        self.assertEqual(adapter.source_capabilities()['runtime_readiness'],'unknown')


class RendererTests(unittest.TestCase):
    def test_pyvista_stable_edge_mapping_without_renderer(self):
        root=Path(__file__).resolve().parents[1]
        def extract(path,names,ns):
            tree=ast.parse((root/path).read_text(encoding='utf-8'))
            body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
            body += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
            module=ast.Module(body=body,type_ignores=[]);ast.fix_missing_locations(module)
            exec(compile(module,str(path),'exec'),ns)
        aligned={'np':np}
        extract('fibernet/viz/render.py',{'_identity_order','_aligned_array','_align_result'},aligned)
        namespace={'np':np}
        extract('fibernet/viz/render3d.py',{'render_structure_graph_3d'},namespace)
        g=graph();g.add_node([2,0,0]);g.add_edge(40,41,n_internal=0)
        d=g.to_dict();d['edges'][1]['id']=15;g=StructureGraph.from_dict(d)
        r=SimResult(deformed_positions=np.array([[0,0,0],[1.1,0,0],[2,0,0]]),edge_forces=np.array([1.,3.]),
            metadata={'node_ids':[10,40,41],'edge_ids':[15,77]})
        colors=[];segments=[]
        class Plotter:
            def __init__(self,**kwargs):pass
            def add_mesh(self,mesh,**kwargs):colors.append(kwargs['color'])
            def add_axes(self):pass
        class Spline:
            def __init__(self,points,**kwargs):segments.append(points.copy())
            def tube(self,**kwargs):return self
        import sys
        modules={'pyvista':types.SimpleNamespace(Plotter=Plotter,Spline=Spline),
            'matplotlib':types.SimpleNamespace(colormaps=types.SimpleNamespace(get_cmap=lambda _:lambda x:(x,0,0,1))),
            'fibernet.viz.render':types.SimpleNamespace(_align_result=aligned['_align_result'])}
        with patch.dict(sys.modules,modules):
            namespace['render_structure_graph_3d'](g,r,color_by='stress',show_nodes=False)
        self.assertEqual(colors,[[255,0,0],[0,0,0]])
        np.testing.assert_array_equal(segments[0],[[0,0,0],[1.1,0,0]])
        np.testing.assert_array_equal(r.edge_forces,[1.,3.])


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.calls=[];owner=self
        class FakeTask:
            def __init__(self,p,r,w,l):
                owner.calls.append('construct');self.result=raw_result(p);self.error='fixture failure';self.state=backend.TaskState.CREATED
            def start(self):owner.calls.append('start');self.state=backend.TaskState.RUNNING;return self
            def wait(self):owner.calls.append('wait');self.state=backend.TaskState.SUCCEEDED;return self.state
            def cancel(self):owner.calls.append('cancel')
        self.Task=FakeTask
        self.args=dict(options(),runtime=RUNTIME,workdir='never-created',limits=backend.ResourceLimits())
        tree=ast.parse((Path(__file__).resolve().parents[1]/'fibernet/easy.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simulate')
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
        ast.fix_missing_locations(module);ns={'np':np};exec(compile(module,'easy.py','exec'),ns);self.simulate=ns['simulate']

    def test_easy_dispatch_success_and_signature(self):
        import inspect
        signature=inspect.signature(self.simulate)
        self.assertEqual(list(signature.parameters),['graph','mode','strain','backend','save_path','kwargs'])
        self.assertEqual(signature.parameters['backend'].default,'spring')
        with patch.object(backend,'LAMMPSTask',self.Task):
            r=self.simulate(graph(),mode='dynamics',backend='lammps',record_state=True,**self.args)
        self.assertIsInstance(r,SimResult);self.assertEqual(self.calls,['construct','start','wait'])

    def test_options_fail_before_constructor(self):
        for changes in ({'arch':'cpu'},{'stiffness':1},{'record_state':1},{'damping':0},{'dt':.01}):
            with patch.object(backend,'LAMMPSTask',self.Task),self.assertRaises((TypeError,ValueError)):
                self.simulate(graph(),mode='dynamics',backend='lammps',**{**self.args,**changes})
        for mode,strain in [('stretch',2),('dynamics',1.1)]:
            with patch.object(backend,'LAMMPSTask',self.Task),self.assertRaises(ValueError):
                self.simulate(graph(),mode=mode,strain=strain,backend='lammps',**self.args)
        for missing in self.args:
            args=dict(self.args);args.pop(missing)
            with patch.object(backend,'LAMMPSTask',self.Task),self.assertRaises(TypeError):
                self.simulate(graph(),mode='dynamics',backend='lammps',**args)
        self.assertEqual(self.calls,[])

    def test_prepare_rejection_before_constructor(self):
        for g,args in [(graph(),{**self.args,'fixed_node_ids':[10],'forces_N':{10:(1,0,0)}}),
                       (graph(),{**self.args,'length_unit':'unknown'})]:
            with patch.object(backend,'LAMMPSTask',self.Task),self.assertRaises(ValueError):
                self.simulate(g,mode='dynamics',backend='lammps',**args)
        for curved in (True,False):
            g=graph()
            if curved:g.edges[77].internal_points=np.array([[.5,.2,0]])
            else:
                d=g.to_dict();d['edges'].append({**d['edges'][0],'id':78});d['next_edge_id']=79;g=StructureGraph.from_dict(d)
            with patch.object(backend,'LAMMPSTask',self.Task),self.assertRaises(ValueError):
                self.simulate(g,mode='dynamics',backend='lammps',**self.args)
        self.assertEqual(self.calls,[])

    def test_failure_cancel_and_reap(self):
        for status in (backend.TaskState.FAILED,backend.TaskState.CANCELLED,backend.TaskState.TIMED_OUT,backend.TaskState.RESOURCE_LIMIT):
            owner=self
            class Failed(self.Task):
                def wait(self):owner.calls.append('wait');return status
            self.calls.clear()
            with patch.object(backend,'LAMMPSTask',Failed),self.assertRaises(backend.LAMMPSContractError):
                self.simulate(graph(),mode='dynamics',backend='lammps',**self.args)
            self.assertEqual(self.calls,['construct','start','wait','cancel','wait'])

    def test_interrupt_timeout_start_exception_cleanup(self):
        for exception in (KeyboardInterrupt,TimeoutError,RuntimeError):
            owner=self
            class Interrupted(self.Task):
                first=True
                def wait(self):
                    owner.calls.append('wait')
                    if self.first:self.first=False;raise exception('fixture')
                    return backend.TaskState.CANCELLED
            self.calls.clear()
            with patch.object(backend,'LAMMPSTask',Interrupted),self.assertRaises(exception):
                self.simulate(graph(),mode='dynamics',backend='lammps',**self.args)
            self.assertEqual(self.calls,['construct','start','wait','cancel','wait'])
        class StartFailed(self.Task):
            def start(self):raise RuntimeError('start')
        self.calls.clear()
        with patch.object(backend,'LAMMPSTask',StartFailed),self.assertRaises(RuntimeError):
            self.simulate(graph(),mode='dynamics',backend='lammps',**self.args)
        self.assertEqual(self.calls,['construct','cancel','wait'])





class SourceEncodingTests(unittest.TestCase):
    def test_dispatch_source_under_cp1252_default(self):
        original=Path.read_text
        def local_default(path,encoding=None,errors=None):
            return original(path,encoding=encoding or 'cp1252',errors=errors)
        with patch.object(Path,'read_text',local_default):
            test=DispatchTests('test_easy_dispatch_success_and_signature')
            test.setUp()
            try:test.test_easy_dispatch_success_and_signature()
            finally:test.tearDown()

    def test_renderer_sources_under_cp1252_default(self):
        original=Path.read_text
        def local_default(path,encoding=None,errors=None):
            return original(path,encoding=encoding or 'cp1252',errors=errors)
        with patch.object(Path,'read_text',local_default):
            RendererTests('test_pyvista_stable_edge_mapping_without_renderer').test_pyvista_stable_edge_mapping_without_renderer()

if __name__=='__main__':unittest.main()
