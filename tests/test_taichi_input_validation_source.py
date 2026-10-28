"""Exact-source host-control regressions, not a Taichi/backend execution.

Run directly with Python to avoid pytest's provider-importing conftest. Execute
the actual dynamics prefix through input validation and the later NumPy fixed
mask statements. Geometry/ownership checks use inert test adapters; no engine,
native fields, kernels, material model or integration loop is constructed.
"""
from __future__ import annotations

import ast
import builtins
import copy
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'fibernet/sim/accelerated.py'
MODES = {
    'default': {},
    'recorded': {'record_state': True},
    'row_masses': {'nodal_masses': [1., 2., 3.]},
    'id_masses': {'nodal_masses_by_id': {10: 1., 40: 2., 99: 3.}},
    'units': {'state_units': dict(length='m', time='s', mass='kg', frame='world')},
}


def load_control_path():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    helper_names = {'_finite_real_array', '_strict_state_ids', '_state_units',
                    '_validated_fixed_rows', '_resolve_nodal_masses',
                    '_legacy_damping', '_graph_to_arrays'}
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name in helper_names]
    assert {n.name for n in helpers} == helper_names
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == 'TaichiEngine')
    original = next(n for n in engine.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'dynamics')
    stop = next(i for i, n in enumerate(original.body)
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name)
                and n.value.func.id == '_prepare_displacement_schedule')
    prefix = copy.deepcopy(original)
    prefix.decorator_list = []
    prefix.body = prefix.body[:stop] + ast.parse('return locals()').body
    mask_index = next(i for i, n in enumerate(original.body)
                      if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'fixed_arr'
                              for t in n.targets))
    mask = original.body[mask_index:mask_index + 2]
    assert isinstance(mask[1], ast.For)
    assert ast.unparse(mask[1].iter) == 'fixed_nodes or []'
    assert not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                   and n.value.id == 'ti' for n in ast.walk(prefix))
    namespace = dict(np=np, time=time, warnings=warnings, _DAMPING_UNSET=None,
                     preflight_taichi_geometry=lambda *a, **kw: {},
                     _check_taichi_owner=lambda: None)
    module = ast.Module(body=ast.parse('from __future__ import annotations').body
                        + helpers + [prefix], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    mask_code = compile(ast.Module(body=mask, type_ignores=[]), str(SOURCE), 'exec')
    return namespace, mask_code


def graph(ids=(10, 40, 99)):
    positions = {10: [0., 0., 0.], 40: [1., 0., 0.], 99: [2., 0., 0.]}
    return SimpleNamespace(
        num_nodes=3,
        nodes={node: SimpleNamespace(position=positions[node]) for node in ids},
        edges={7: SimpleNamespace(node_i=10, node_j=40),
               8: SimpleNamespace(node_i=40, node_j=99)})


class InputValidationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.mask_code = load_control_path()

    def setUp(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'fibernet', 'taichi', 'lammps', 'exudyn'}:
                raise AssertionError('No provider/package import in source test: ' + name)
            return original_import(name, *args, **kwargs)

        guard = patch('builtins.__import__', side_effect=guarded_import)
        guard.start()
        self.addCleanup(guard.stop)

    def controls(self, mode, supplied_graph=None, **kwargs):
        options = copy.deepcopy(MODES[mode])
        options.update(kwargs)
        return self.namespace['dynamics'](object(), supplied_graph or graph(), **options)

    def assert_rejected(self, cases, parameter):
        for mode in MODES:
            for label, value in cases:
                with self.subTest(mode=mode, parameter=parameter, case=label):
                    with self.assertRaisesRegex(ValueError, parameter):
                        self.controls(mode, **{parameter: value})

    def test_invalid_fixed_rows_in_every_mode(self):
        self.assert_rejected([
            ('negative', [-1]), ('out_of_range', [3]), ('bool', [True]),
            ('numpy_bool', [np.bool_(False)]), ('fractional', [.5]),
            ('integer_float', [1.]), ('string', ['1']), ('scalar', 1),
            ('nested', [[1]]), ('rank_two', np.array([[1]])),
            ('masked', np.ma.array([1], mask=[True])),
            ('numpy_float', np.array([1.])),
        ], 'fixed_nodes')

    def test_invalid_dt_in_every_mode(self):
        self.assert_rejected([
            ('zero', 0), ('negative', -.01), ('nan', np.nan), ('inf', np.inf),
            ('bool', True), ('numpy_bool', np.bool_(False)), ('vector', [.01]),
            ('string', '0.01'), ('complex', .01 + 0j),
            ('masked', np.ma.array(.01, mask=True)),
        ], 'dt')

    def test_invalid_active_damping_in_every_mode(self):
        cases = [('negative', -1.), ('nan', np.nan), ('inf', np.inf),
                 ('bool', True), ('numpy_bool', np.bool_(False)),
                 ('vector', [0.]), ('string', '0'), ('complex', 0j),
                 ('masked', np.ma.array(0., mask=True))]
        for parameter in ('dashpot', 'drag'):
            self.assert_rejected(cases, parameter)

    def test_invalid_loads_in_every_mode(self):
        self.assert_rejected([
            ('wrong_components', np.zeros((3, 2))),
            ('wrong_nodes', np.zeros((2, 3))), ('flat', np.zeros(9)),
            ('scalar', 1.), ('nan', np.full((3, 3), np.nan)),
            ('inf', np.full((3, 3), np.inf)),
            ('complex', np.zeros((3, 3), dtype=complex)),
            ('boolean', np.zeros((3, 3), dtype=bool)),
            ('strings', np.full((3, 3), '0')),
            ('masked', np.ma.array(np.zeros((3, 3)), mask=True)),
        ], 'external_force')

    def test_ragged_loads_in_every_mode(self):
        # NumPy itself may describe the ragged shape before the named validator.
        for mode in MODES:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.controls(mode, external_force=[[0, 0, 0], [0], [0, 0, 0]])

    def test_valid_rows_and_exact_mask_in_every_mode(self):
        cases = [(None, [0, 0, 0]), ([], [0, 0, 0]), ((), [0, 0, 0]),
                 (np.array([], dtype=int), [0, 0, 0]), ([1], [0, 1, 0]),
                 ((0, 2), [1, 0, 1]), ([np.int64(1)], [0, 1, 0]),
                 (np.array([0, 2], dtype=np.int32), [1, 0, 1]),
                 ([1, 1], [0, 1, 0])]
        for mode in MODES:
            for rows, expected in cases:
                with self.subTest(mode=mode, rows=repr(rows)):
                    result = self.controls(mode, fixed_nodes=rows)
                    exec(self.mask_code, dict(self.namespace, **result), result)
                    np.testing.assert_array_equal(result['fixed_arr'], expected)
                    self.assertIsInstance(result['fixed_nodes'], list)

    def test_valid_scalars_loads_and_detachment_in_every_mode(self):
        for mode in MODES:
            for dt, dashpot, drag in ((1, 0, 0), (.01, .2, .3),
                                      (np.float64(.02), np.int64(0), np.float32(.5))):
                with self.subTest(mode=mode, dt=dt, dashpot=dashpot, drag=drag):
                    rows = [0, np.int64(2)]
                    forces = np.arange(9, dtype=np.float32).reshape(3, 3)
                    original = forces.copy()
                    g = graph()
                    before = copy.deepcopy(vars(g))
                    result = self.controls(mode, g, fixed_nodes=rows,
                                           external_force=forces, dt=dt,
                                           dashpot=dashpot, drag=drag)
                    self.assertEqual(vars(g), before)
                    self.assertEqual(rows, [0, 2])
                    np.testing.assert_array_equal(forces, original)
                    self.assertEqual(result['dt'], dt)
                    self.assertEqual(result['dashpot'], dashpot)
                    self.assertEqual(result['drag'], drag)
                    self.assertEqual(result['external_force'].dtype, np.dtype('float64'))
                    self.assertFalse(np.shares_memory(result['external_force'], forces))
                    rows.clear()
                    forces[:] = -99
                    self.assertEqual(result['fixed_nodes'], [0, 2])
                    np.testing.assert_array_equal(result['external_force'], original)

    def test_sparse_ids_and_reordering_do_not_reinterpret_rows(self):
        for ids in ((10, 40, 99), (99, 10, 40)):
            for mode in MODES:
                with self.subTest(ids=ids, mode=mode):
                    g = graph(ids)
                    result = self.controls(mode, g, fixed_nodes=[1])
                    self.assertEqual(result['node_ids'], list(ids))
                    exec(self.mask_code, dict(self.namespace, **result), result)
                    np.testing.assert_array_equal(result['fixed_arr'], [0, 1, 0])
                    self.assertEqual(list(g.nodes), list(ids))
                    with self.assertRaisesRegex(ValueError, 'fixed_nodes'):
                        self.controls(mode, g, fixed_nodes=[40])

    def test_legacy_damping_stays_metadata_only(self):
        for mode in MODES:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                self.assertEqual(self.controls(mode)['damping'], .3)
                self.assertEqual(caught, [])
                result = self.controls(mode, damping=.6, dashpot=0., drag=0.)
                self.assertEqual(result['damping'], .6)
                self.assertEqual(len(caught), 1)
                self.assertIn('metadata-only', str(caught[0].message))
                self.assertEqual(result['dashpot'], 0.)
                self.assertEqual(result['drag'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
