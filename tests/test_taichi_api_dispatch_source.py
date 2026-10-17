"""Standalone source-only dispatch tests; run directly, without pytest conftest.

Extract only the wrapper function ASTs. All backend imports and calls are test
objects. No FiberNet, NumPy or Taichi import/native initialization is allowed.
"""
from __future__ import annotations
import ast
import builtins
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE_WRAPPERS = ROOT / 'tests' / 'fixtures' / 'taichi_dispatch_core73.txt'


class FakeInteger(int):
    pass


class FakeBool:
    pass


def extract(relative, name, namespace, base=False):
    tree = ast.parse((BASE_WRAPPERS if base else ROOT / relative).read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(relative), 'exec'), namespace)
    return namespace[name]


class DispatchChecks(unittest.TestCase):
    def setUp(self):
        self.module_names_before = set(sys.modules)
        self.calls = []
        self.result = types.SimpleNamespace(
            metadata={'node_ids': [8, -3, 91], 'edge_ids': [9, 2]},
            dynamics_state={'final': {'velocities': object()}},
            edge_forces=object(), energy=2.0,
        )
        self.result.save = lambda path: self.calls.append(('save', path))
        self.graph = types.SimpleNamespace(dimension=3, nodes={8: object(), -3: object(), 91: object()}, edges={9: (8, -3), 2: (8, -3)})
        owner = self
        class Engine:
            def __init__(self, **kwargs): owner.calls.append(('engine', kwargs))
            def dynamics(self, graph, **kwargs):
                owner.calls.append(('dynamics', graph, kwargs)); return owner.result
            def stretch_test(self, graph, **kwargs):
                owner.calls.append(('stretch', graph, kwargs)); return owner.result
        self.engine = Engine
        self.np = types.SimpleNamespace(integer=FakeInteger, bool_=FakeBool, zeros_like=lambda value: ('zeros_like', value))
        self.namespace = {'TaichiEngine': Engine, 'np': self.np}
        self.simulate = extract('fibernet/easy.py', 'simulate', self.namespace)
        self.real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'numpy' or name == 'taichi' or name.startswith(('numpy.', 'taichi.', 'fibernet')):
                raise AssertionError('Unapproved native/library import: ' + name)
            return self.real_import(name, *args, **kwargs)
        self.guard = patch('builtins.__import__', side_effect=guarded_import)
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_default_stretch_preserved(self):
        self.assertIs(self.simulate(self.graph, strain=1.1, num_steps=3), self.result)
        self.assertEqual(self.calls, [('engine', {'arch': 'cpu', 'num_threads': 4}), ('stretch', self.graph, {'target_stretch': 1.1, 'num_steps': 3})])

    def test_aliases_forward_identical_dynamics_and_return_object(self):
        state = {'length': 'm', 'time': 's', 'mass': 'kg', 'frame': 'world'}
        kw = dict(fixed_nodes=[1], spring_k=[2, 3], nodal_masses_by_id={8: 1, -3: 2, 91: 3}, state_units=state, record_state=True)
        before = copy.deepcopy(kw)
        for backend in ('spring', 'taichi'):
            self.calls.clear()
            self.assertIs(self.simulate(self.graph, mode='dynamics', backend=backend, **kw), self.result)
            self.assertEqual(self.calls, [('engine', {'arch': 'cpu', 'num_threads': 4}), ('dynamics', self.graph, kw)])
            self.assertIs(self.calls[-1][1], self.graph)
            self.assertIs(self.calls[-1][2]['state_units'], state)
        self.assertEqual(kw, before)
        self.assertEqual(list(self.graph.nodes), [8, -3, 91])
        self.assertEqual(self.graph.edges, {9: (8, -3), 2: (8, -3)})
        self.assertEqual(self.result.metadata, {'node_ids': [8, -3, 91], 'edge_ids': [9, 2]})

    def test_alias_stretch(self):
        self.simulate(self.graph, backend='taichi', strain=1.02)
        self.assertEqual(self.calls[-1], ('stretch', self.graph, {'target_stretch': 1.02}))

    def test_runtime_options_consumed_by_constructor(self):
        for backend in ('spring', 'taichi'):
            for arch in ('cpu', 'gpu'):
                self.calls.clear()
                self.simulate(self.graph, backend=backend, mode='dynamics', arch=arch, num_threads=FakeInteger(2), dt=.01)
                self.assertEqual(self.calls, [('engine', {'arch': arch, 'num_threads': 2}), ('dynamics', self.graph, {'dt': .01})])
                self.assertIs(type(self.calls[0][1]['num_threads']), int)

    def test_invalid_backend_rejected_before_constructor(self):
        for value in ('unknown', 'numpy', '', None, [], 1, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'backend'):
                self.simulate(self.graph, backend=value)
        self.assertEqual(self.calls, [])

    def test_invalid_mode_rejected_before_constructor(self):
        for backend in ('spring', 'taichi'):
            for value in ('unknown', '', None, [], 1):
                with self.subTest(backend=backend, value=value), self.assertRaisesRegex(ValueError, 'mode'):
                    self.simulate(self.graph, backend=backend, mode=value)
        self.assertEqual(self.calls, [])

    def test_invalid_arch_rejected_before_constructor(self):
        for value in ('cuda', '', None, [], 1, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'arch'):
                self.simulate(self.graph, arch=value)
        self.assertEqual(self.calls, [])

    def test_invalid_threads_rejected_before_constructor(self):
        for value in (0, -1, True, False, FakeBool(), 1.0, '1', None, [], float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'num_threads'):
                self.simulate(self.graph, num_threads=value)
        self.assertEqual(self.calls, [])

    def test_dependency_and_runtime_errors_propagate_without_fallback(self):
        for exc in (ImportError('Taichi required: pip install taichi==1.7.4'), RuntimeError('unsupported version'), RuntimeError('architecture unavailable'), RuntimeError('wrong owner')):
            def fail(**kwargs): raise exc
            self.namespace['TaichiEngine'] = fail
            for backend in ('spring', 'taichi'):
                with self.subTest(backend=backend, error=str(exc)), self.assertRaises(type(exc)) as caught:
                    self.simulate(self.graph, backend=backend)
                self.assertIs(caught.exception, exc)
        self.assertEqual(self.calls, [])

    def test_no_result_fields_fabricated_or_reinterpreted(self):
        for backend in ('spring', 'taichi'):
            self.result = object()
            self.assertIs(self.simulate(self.graph, backend=backend, mode='dynamics', record_state=False), self.result)

    def test_native_failure_propagates_and_does_not_save(self):
        exc = ValueError('invalid native input')
        class Broken:
            def __init__(self, **kwargs): pass
            def dynamics(self, *a, **kw): raise exc
        self.namespace['TaichiEngine'] = Broken
        with self.assertRaises(ValueError) as caught:
            self.simulate(self.graph, mode='dynamics', save_path='unused.json')
        self.assertIs(caught.exception, exc)
        self.assertEqual(self.calls, [])

    def test_save_uses_same_result(self):
        with patch('builtins.print'):
            self.assertIs(self.simulate(self.graph, mode='dynamics', save_path='saved.json'), self.result)
        self.assertEqual(self.calls[-1], ('save', 'saved.json'))

    def test_fem_rejects_dynamics_before_backend_import(self):
        with self.assertRaisesRegex(ValueError, 'stretch'):
            self.simulate(self.graph, backend='fem', mode='dynamics')
        self.assertEqual(self.calls, [])

    def test_fem_dispatch_matches_original(self):
        owner = self
        class FEM:
            def __init__(self, **kwargs): owner.calls.append(('fem', kwargs))
            def stretch_test(self, graph, **kwargs): owner.calls.append(('fem_stretch', graph, kwargs)); return 'raw'
            def to_sim_result(self, raw, graph): owner.calls.append(('fem_result', raw, graph)); return owner.result
        fake = types.SimpleNamespace(BeamFrameFEM=FEM)
        real = builtins.__import__
        def allow_fem(name, *args, **kwargs):
            return fake if name == 'fibernet.ml.beam_frame_fem' else real(name, *args, **kwargs)
        old = extract('fibernet/easy.py', 'simulate', {'TaichiEngine': self.engine, 'np': self.np}, base=True)
        with patch('builtins.__import__', side_effect=allow_fem):
            for options in ({}, {'E': 10., 'nu': .2, 'dim': 2, 'nonlinear': False}):
                self.calls.clear(); self.assertIs(old(self.graph, backend='fem', strain=1.02, **options), self.result); before = list(self.calls)
                self.calls.clear(); self.assertIs(self.simulate(self.graph, backend='fem', strain=1.02, **options), self.result)
                self.assertEqual(self.calls, before)

    def legacy(self, base=False):
        def builder(graph): self.calls.append(('builder', graph)); return ('edges', 'rest', 'stiffness', 'masses', 'positions')
        def recurrence(*args, **kwargs): self.calls.append(('numpy', args, kwargs)); return self.result
        return extract('fibernet/api.py', 'simulate_dynamics', {'np': self.np, '_build_mass_spring_system': builder, '_numpy_dynamics': recurrence}, base=base)

    def test_legacy_default_and_taichi_fail_before_builder(self):
        legacy = self.legacy()
        for options in ({}, {'backend': 'taichi'}):
            with self.assertRaises(NotImplementedError) as caught: legacy(self.graph, **options)
            for term in ('crosslink', 'EA/L', 'EA', 'mm', 'StructureGraph', 'fibernet', 'Endpoint-only'):
                self.assertIn(term, str(caught.exception))
        self.assertEqual(self.calls, [])

    def test_legacy_unknown_backend_fails_before_builder(self):
        legacy = self.legacy()
        for value in ('typo', 'spring', '', None, [], 1, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'backend'):
                legacy(self.graph, backend=value)
        self.assertEqual(self.calls, [])

    def test_legacy_numpy_arguments_and_return_match_original(self):
        legacy = self.legacy(); old = self.legacy(base=True)
        fake = types.SimpleNamespace(TaichiEngine=self.engine, HAS_TAICHI=False)
        real = builtins.__import__
        def allow_legacy(name, *args, **kwargs):
            return fake if name == 'fibernet.sim.accelerated' else real(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=allow_legacy):
            for options in ({}, {'dt': .1, 'steps': 2, 'damping': .2, 'external_force': 'force', 'fixed_nodes': [2], 'save_interval': 1, 'unused': 3}):
                self.calls.clear(); self.assertIs(old(self.graph, backend='numpy', **options), self.result); before = list(self.calls)
                self.calls.clear(); self.assertIs(legacy(self.graph, backend='numpy', **options), self.result)
                self.assertEqual(self.calls, before)

    def test_no_actual_native_modules_loaded(self):
        self.assertFalse(any(n == 'taichi' or n.startswith('taichi.') or n == 'numpy' or n.startswith('numpy.') or n == 'fibernet' or n.startswith('fibernet.') for n in set(sys.modules) - self.module_names_before))


if __name__ == '__main__':
    assert not any(n == 'numpy' or n == 'taichi' or n == 'fibernet' or n.startswith(('numpy.', 'taichi.', 'fibernet.')) for n in sys.modules)
    unittest.main(verbosity=2)
