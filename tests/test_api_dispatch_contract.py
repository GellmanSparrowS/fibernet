"""Inert dispatch contract checks; no FiberNet/Taichi/LAMMPS import or solve."""
from __future__ import annotations
import ast
import builtins
import inspect
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def extract(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), namespace)
    return namespace[name]


class DispatchContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.graph = types.SimpleNamespace(dimension=3)
        self.result = object()
        owner = self
        class Engine:
            def __init__(self, *, arch, num_threads): owner.calls.append(('init', arch, num_threads))
            def stretch_test(self, graph, target_stretch=2.0, *, num_steps=None):
                owner.calls.append(('stretch', target_stretch)); return owner.result
            def dynamics(self, graph, *, num_steps=5, material=None):
                owner.calls.append(('dynamics', num_steps)); return owner.result
        class FEM:
            def __init__(self, E=1e9, nu=.3): owner.calls.append(('fem_init', E, nu))
            def stretch_test(self, graph, target_stretch=2., dim=2, pct=.1, nonlinear=None, deduplicate=False):
                owner.calls.append(('fem_stretch', target_stretch, dim)); return 'raw'
            def history(self, graph, *, material, times, constraints, length_unit, loads=None, initial_state=None):
                owner.calls.append(('history',)); return owner.result
            def to_sim_result(self, raw, graph): return owner.result
        def adapter(graph, mode, strain, **kwargs):
            owner.calls.append(('lammps', mode, strain)); return owner.result
        fake_modules = {'fibernet.ml.beam_frame_fem': types.SimpleNamespace(BeamFrameFEM=FEM),
                        'fibernet.sim._lammps_result': types.SimpleNamespace(simulate_lammps=adapter)}
        real_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name in fake_modules: return fake_modules[name]
            if name.split('.')[0] in ('fibernet', 'taichi', 'lammps'):
                raise AssertionError('forbidden native/package import ' + name)
            return real_import(name, *args, **kwargs)
        self.guard = patch('builtins.__import__', side_effect=guard)
        self.guard.start(); self.addCleanup(self.guard.stop)
        self.simulate = extract('fibernet/easy.py', 'simulate', {'TaichiEngine': Engine, 'np': np})
        self.namespace = self.simulate.__globals__

    def test_signature_and_omitted_defaults(self):
        sig = inspect.signature(self.simulate)
        self.assertEqual(list(sig.parameters), ['graph','mode','strain','backend','save_path','kwargs'])
        self.assertIsNone(sig.parameters['strain'].default)
        for options in ({}, {'strain': None}):
            self.calls.clear(); self.assertIs(self.simulate(self.graph, **options), self.result)
            self.assertEqual(self.calls[-1], ('stretch', 2.))
        self.simulate(self.graph, backend='fem')
        self.assertEqual(self.calls[-1], ('fem_stretch', 2., 3))
        self.simulate(self.graph, mode='dynamics', backend='lammps')
        self.assertEqual(self.calls[-1], ('lammps', 'dynamics', 2.))

    def test_all_explicit_nonstrain_modes_rejected_before_init(self):
        for backend, mode in [('spring','dynamics'),('taichi','dynamics'),('fem','history'),('lammps','dynamics')]:
            for value in (2., np.float64(2), float('nan'), float('inf'), 0., 1.01, True, False, '2', [], {}):
                with self.subTest(backend=backend, value=value):
                    with self.assertRaisesRegex(ValueError, 'strain'):
                        self.simulate(self.graph, mode=mode, backend=backend, strain=value)
                    self.assertEqual(self.calls, [])

    def test_stretch_numeric_meaning_unchanged(self):
        for value in (1.01, 2.):
            self.simulate(self.graph, strain=value)
            self.assertEqual(self.calls[-1], ('stretch', value))

    def test_none_nonstrain_dispatch(self):
        self.simulate(self.graph, mode='dynamics', strain=None)
        self.assertEqual(self.calls[-1], ('dynamics', 5))
        self.simulate(self.graph, mode='history', backend='fem', strain=None,
                      material=object(), times=[0,1], constraints={}, length_unit='m')
        self.assertEqual(self.calls[-1], ('history',))

    def test_unknown_kwargs_before_initialization(self):
        for backend, mode in [('spring','stretch'),('taichi','dynamics'),('fem','stretch'),('fem','history')]:
            with self.subTest(backend=backend, mode=mode):
                with self.assertRaises(TypeError):
                    self.simulate(self.graph, backend=backend, mode=mode, unknown_option=1)
                self.assertEqual(self.calls, [])
        with self.assertRaises(TypeError): self.simulate(self.graph, target_stretch=1.1)
        self.assertEqual(self.calls, [])

    def test_strict_constructor_errors_propagate(self):
        for error in (ImportError('missing optional backend'), RuntimeError('runtime unavailable')):
            class Broken:
                def __init__(self, *, arch, num_threads): raise error
                def stretch_test(self, graph, target_stretch=2.0):
                    raise AssertionError('must not execute')
            self.namespace['TaichiEngine'] = Broken
            with self.assertRaises(type(error)) as caught: self.simulate(self.graph)
            self.assertIs(caught.exception, error)
            self.assertEqual(self.calls, [])

    def test_unknown_modes_and_invalid_dim_before_init(self):
        for backend in ('spring','taichi','fem'):
            for mode in ('unknown', [], None):
                with self.assertRaises(ValueError): self.simulate(self.graph, backend=backend, mode=mode)
                self.assertEqual(self.calls, [])
        for dim in (True, np.bool_(True), 4, '3'):
            with self.assertRaises(ValueError): self.simulate(self.graph, backend='fem', dim=dim)
            self.assertEqual(self.calls, [])


class FEMContract(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        method=extract('fibernet/ml/beam_frame_fem.py','stretch_test',{'np':np,'warnings':warnings})
        owner=self
        class FEM:
            stretch_test=method
            def graph_to_fem_input(self, graph, *, dim, pct):
                owner.calls.append('convert')
                return dict(boundaries={'left':[0],'right':[1]},x_range=1.,edge_index=None,node_pos=None,radii=None)
            def solve_2d(self, *args, **kwargs): owner.calls.append('linear2'); return {'u':np.zeros((2,3))}
            def solve_2d_nonlinear(self, *args, **kwargs): owner.calls.append('incremental2'); return {'u':np.zeros((2,3))}
            def solve_3d(self, *args, **kwargs): owner.calls.append('linear3'); return {'u':np.zeros((2,6))}
        self.fem=FEM()

    def test_3d_true_refuses_before_conversion(self):
        for nonlinear in (True, np.bool_(True)):
            with self.assertRaisesRegex(NotImplementedError,'linear-only'):
                self.fem.stretch_test(object(),dim=3,nonlinear=nonlinear)
            self.assertEqual(self.calls,[])

    def test_3d_legacy_linear_large_stretch_warns_once(self):
        for nonlinear in (None,False,np.bool_(False)):
            self.calls.clear()
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always');self.fem.stretch_test(object(),dim=3,nonlinear=nonlinear)
            self.assertEqual(self.calls,['convert','linear3'])
            self.assertEqual(len(seen),1);self.assertIn('small-deformation',str(seen[0].message))

    def test_3d_small_stretch_no_warning(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always');self.fem.stretch_test(object(),dim=3,target_stretch=1.01)
        self.assertEqual(seen,[]);self.assertEqual(self.calls,['convert','linear3'])

    def test_2d_valid_dispatch_preserved(self):
        for ratio, nonlinear, method in [(2.,None,'incremental2'),(1.01,None,'linear2'),(2.,False,'linear2'),(1.01,True,'incremental2')]:
            self.calls.clear();self.fem.stretch_test(object(),target_stretch=ratio,nonlinear=nonlinear)
            self.assertEqual(self.calls,['convert',method])

    def test_malformed_controls_fail_before_conversion(self):
        for options in [{'nonlinear':v} for v in (1,0,'false',[],{})]+[{'dim':v} for v in (True,4,'3')]+[{'target_stretch':v} for v in (True,None,'2',float('nan'),float('inf'),[2])]:
            with self.subTest(options=options),self.assertRaises((TypeError,ValueError)):
                self.fem.stretch_test(object(),**options)
            self.assertEqual(self.calls,[])


class LegacyDocsContract(unittest.TestCase):
    def test_legacy_mechanics_clear_error_without_import(self):
        legacy=extract('fibernet/api.py','simulate_mechanics',{})
        for model in ('linear','bilinear','neo_hookean','unknown'):
            with patch('builtins.__import__', side_effect=AssertionError('unexpected import')):
                with self.assertRaisesRegex(NotImplementedError,'not a drop-in replacement'):
                    legacy(object(),model=model)

    def test_examples_and_obsolete_claims(self):
        for relative in ('fibernet/easy.py','fibernet/__init__.py'):
            source=(ROOT/relative).read_text(encoding='utf-8')
            self.assertNotIn('pattern_2d("',source)
        self.assertNotIn('fn.create(', (ROOT/'fibernet/api.py').read_text(encoding='utf-8'))
        self.assertNotIn('All simulation now uses TaichiEngine',(ROOT/'fibernet/sim/fem.py').read_text(encoding='utf-8'))
        self.assertIn('summary by default', (ROOT/'fibernet/easy.py').read_text(encoding='utf-8'))

if __name__=='__main__': unittest.main(verbosity=2)
