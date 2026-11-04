"""Caller geometry policy with inert constructors and result assembly.

Facade methods execute with inert constructors. Direct rejection executes the
production method until its early gate. Success provenance executes the actual
post-kernel result assembly suffix with inert arrays/validators, not a solve.
"""
from __future__ import annotations
import ast
import copy
import functools
import inspect
import time
import types
import unittest
from .support import ROOT,H,NP,graph,definition,load_functions,isolated_builtins

class DispatchPolicy(unittest.TestCase):
    def setUp(self):
        self.calls=[]; owner=self
        self.result=types.SimpleNamespace(metadata={'unchanged':True})
        class Engine:
            def __init__(self,**kwargs): owner.calls.append(('construct',kwargs))
            def stretch_test(self,graph,target_stretch=2.,num_steps=5000,*,geometry_policy='reject_internal_points'):
                owner.calls.append(('stretch',graph,locals().copy())); return owner.result
            def dynamics(self,graph,dt=.01,*,geometry_policy='reject_internal_points'):
                owner.calls.append(('dynamics',graph,locals().copy())); return owner.result
        self.Engine=Engine
        self.ns={**H,'TaichiEngine':Engine,'np':NP}
        self.sim=load_functions('fibernet/easy.py',['_taichi_engine_class','simulate'],self.ns)['simulate']

    def test_alias_mode_default_rejects_before_construct(self):
        for alias in ('spring','taichi'):
            for mode in ('stretch','dynamics'):
                with self.subTest(alias=alias,mode=mode),self.assertRaises(H['UnsupportedInternalPointsError']):
                    self.sim(graph(True),backend=alias,mode=mode)
        self.assertEqual(self.calls,[])

    def test_alias_mode_chord_forwards_and_keeps_result_identity(self):
        for alias in ('spring','taichi'):
            for mode in ('stretch','dynamics'):
                self.calls.clear(); g=graph(True)
                self.assertIs(self.sim(g,backend=alias,mode=mode,geometry_policy='endpoint_chord'),self.result)
                self.assertIs(self.calls[-1][1],g)
                self.assertEqual(self.calls[-1][2]['geometry_policy'],'endpoint_chord')
        self.assertEqual(self.result.metadata,{'unchanged':True})

    def test_straight_omission_retains_call_kwargs(self):
        for mode,option in (('stretch',{'num_steps':3}),('dynamics',{'dt':.2})):
            self.calls.clear();g=graph()
            self.assertIs(self.sim(g,mode=mode,**option),self.result)
            self.assertEqual(self.calls[0],('construct',{'arch':'cpu','num_threads':4}))
            for k,v in option.items(): self.assertEqual(self.calls[-1][2][k],v)
            self.assertEqual(self.calls[-1][2]['geometry_policy'],'reject_internal_points')

    def test_invalid_selectors_keywords_policy_before_constructor(self):
        for kw in ({'backend':'bad'}, {'mode':'bad'}, {'arch':'cuda'}, {'num_threads':0},
                   {'geometry_policy':'auto'},{'geometry_policy':True},{'bogus':1}):
            with self.subTest(kw=kw),self.assertRaises((ValueError,TypeError)): self.sim(graph(),**kw)
        # Binding errors keep precedence over the graph preflight.
        with self.assertRaises(TypeError): self.sim(graph(True),bogus=1)
        self.assertEqual(self.calls,[])

    def test_non_taichi_keyword_behavior_stays_unsupported(self):
        owner=self
        class FEM:
            def __init__(self,E=1,nu=.3): owner.calls.append(('fem',))
            def stretch_test(self,g,target_stretch=2,dim=3): return object()
        def lammps(g,mode,strain): return object()
        self.ns.update(BeamFrameFEM=FEM,simulate_lammps=lammps)
        for backend in ('fem','lammps'):
            with self.assertRaises(TypeError): self.sim(graph(True),backend=backend,geometry_policy='endpoint_chord')
        self.assertEqual(self.calls,[])

class DirectPolicy(unittest.TestCase):
    def methods(self,**extra):
        def forbidden(*a,**k): raise AssertionError('owner/provider/array touched')
        ns={**H,'_check_taichi_owner':forbidden,'_graph_to_arrays':forbidden,
            '_legacy_damping':forbidden,'np':NP,**extra}
        load_functions('fibernet/sim/accelerated.py',['dynamics','stretch_test'],ns,class_name='TaichiEngine')
        return ns

    def test_keyword_only_tail_and_old_positional_contract(self):
        expected = {
            'dynamics': (
                ['self', 'graph', 'fixed_nodes', 'displacement_schedule',
                 'external_force', 'stiffness', 'damping', 'dt', 'num_steps',
                 'save_interval', 'spring_k', 'dashpot', 'drag', 'max_nodes'],
                [None, None, None, 1e5, None, 1e-4, 5000, 500, None, 10., 1., 20000]),
            'stretch_test': (
                ['self', 'graph', 'target_stretch', 'stiffness', 'damping',
                 'num_steps', 'save_interval', 'ramp_fraction', 'auto_steps', 'max_nodes'],
                [2., 1e5, None, None, 1000, .2, True, 20000]),
        }
        ns=self.methods()
        for name in ('dynamics','stretch_test'):
            new=definition('fibernet/sim/accelerated.py',name,'TaichiEngine')
            self.assertEqual([arg.arg for arg in new.args.args], expected[name][0])
            self.assertEqual([ast.literal_eval(value) for value in new.args.defaults], expected[name][1])
            self.assertEqual(new.args.posonlyargs, [])
            self.assertIsNone(new.args.vararg)
            self.assertIsNone(new.args.kwarg)
            p=inspect.signature(ns[name]).parameters['geometry_policy']
            self.assertEqual(p.kind,inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(p.default,'reject_internal_points')

    def test_reject_before_all_direct_owner_array_sentinels(self):
        ns=self.methods()
        for name in ('dynamics','stretch_test'):
            with self.subTest(method=name),self.assertRaises(H['UnsupportedInternalPointsError']):
                ns[name](object(),graph(True))
            with self.assertRaises(H['InvalidGeometryPolicyError']):
                ns[name](object(),object(),geometry_policy='bad')

    def test_existing_material_decorator_preserves_policy_signature(self):
        ns=self.methods()
        dec=load_functions('fibernet/sim/_taichi_viscoelastic.py',['material_options'],{'wraps':functools.wraps})['material_options']
        wrapped=dec(ns['dynamics'])
        self.assertEqual(inspect.signature(wrapped),inspect.signature(ns['dynamics']))
        with self.assertRaises(H['UnsupportedInternalPointsError']): wrapped(object(),graph(True))

    def test_stretch_forwards_policy_without_native_work(self):
        class Column:
            def max(self): return 1.
            def min(self): return 0.
        class Pos:
            shape=(3,3)
            def __getitem__(self,key): return Column()
        calls=[]
        fake_np=types.SimpleNamespace(**vars(NP),array=lambda a:list(a),zeros=lambda n:[0]*n)
        ns=self.methods(np=fake_np,_check_taichi_owner=lambda:None,
                        _graph_to_arrays=lambda g:(Pos(),[],[],None),
                        _get_boundary_indices=lambda p:{'left':[0],'right':[2]},nx=object())
        result=object()
        selfobj=types.SimpleNamespace(dynamics=lambda g,**kw:(calls.append((g,kw)),result)[1])
        g=graph(True)
        self.assertIs(ns['stretch_test'](selfobj,g,num_steps=2,auto_steps=False,geometry_policy='endpoint_chord'),result)
        self.assertEqual(calls[0][1]['geometry_policy'],'endpoint_chord')
        self.assertEqual(calls[0][1]['num_steps'],2)

    def test_real_result_assembly_suffix_all_material_state_branches(self):
        node=definition('fibernet/sim/accelerated.py','dynamics','TaichiEngine')
        start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets))
        fn=ast.FunctionDef(name='assemble',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=node.body[start:],decorator_list=[])
        module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        class Vec:
            def __getitem__(self,k): return self
            def __sub__(self,o): return self
            def __truediv__(self,o): return self
            def tolist(self): return [1.,1.,1.]
        class Result(types.SimpleNamespace):
            def compute_detailed(self,g): self.energy=12.;self.metadata['detail_kept']=True
        for material,state in ((False,False),(False,True),(True,False),(True,True)):
            g=graph(True);record=H['preflight_taichi_geometry'](g,geometry_policy='endpoint_chord')
            vals=dict(positions=[],velocities=[],kinetic_energy=0.,potential_energy=0.,total_energy=0.,near_cutoff=False,step=0,time=0.)
            def populate(r,*a): r.energy=7.;r.metadata['spring_coefficient_provenance']={'old':'kept'}
            ns=dict(__builtins__=isolated_builtins(),SimResult=Result,displacements=Vec(),time=time,t0=time.time(),pos_final=Vec(),trajectory=[],
                record_state=state,np=types.SimpleNamespace(max=lambda x:1.,linalg=types.SimpleNamespace(norm=lambda *a,**k:Vec())),
                pos_orig=Vec(),elements=Vec(),rest_lengths_np=Vec(),max_stretch_history=[],state_samples=[vals],
                save_interval=1,node_ids=[8,-3,91],graph=g,stiffness=1.,damping=.3,dt=.01,num_steps=1,
                geometry_record=record,material_mode=material,material_history=types.SimpleNamespace(result=lambda *a:{'existing_material':'kept'}),
                fixed_nodes=[],ext_f=None,drag=1.,_populate_spring_result=populate,stiff_np=Vec(),spring_k=None,spring_scalar=True,
                explicit_mass=True,units={'length':'m'},masses=Vec(),num_nodes=3,_validate_dynamics_state=lambda state,*a:state,
                record_energy_ledger=(state and not material),ledger=types.SimpleNamespace(result=lambda *a:{'old_ledger':'kept'}),ledger_rows=[],dashpot=1.)
            exec(compile(module,'post_kernel_result_assembly_only','exec'),ns)
            r=ns['assemble']()
            self.assertIs(r.metadata['taichi_geometry'],record)
            self.assertEqual(r.metadata['node_ids'],[8,-3,91]);self.assertEqual(r.metadata['edge_ids'],[9,2])
            self.assertIn('native_axial_inputs',r.metadata)
            self.assertIn('native_axial_viscoelastic' if material else 'spring_coefficient_provenance',r.metadata)
            if state and not material: self.assertEqual(r.algorithmic_energy_ledger,{'old_ledger':'kept'})
