"""Isolated provider-double controls; never imports a real LAMMPS provider.

Run directly with unittest to avoid the repository's native-provider conftest:
    python tests/test_legacy_lammps_fail_closed.py
Only tiny NumPy arrays and one-point synthetic fiber data are used.
"""
import ast
import copy
import hashlib
import json
import ntpath
import os
import posixpath
import builtins
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'fibernet/integrations/lammps_integration.py'


def stable_ast(value):
    """Represent syntax consistently across supported Python 3.9–3.12."""
    if isinstance(value, ast.AST):
        return [type(value).__name__, [
            [name, stable_ast(field)] for name, field in ast.iter_fields(value)
            if not (name == 'type_params' and field == [])]]
    if isinstance(value, list):
        return [stable_ast(item) for item in value]
    return value


def syntax_hash(value):
    return hashlib.sha256(json.dumps(stable_ast(value), ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()


class LegacyFailureControls(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.import_error = ImportError('deliberately missing provider')
        self.provider = None
        self.import_attempts = []
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == 'lammps' or name.startswith('lammps.'):
                self.import_attempts.append(name)
                if self.provider is None:
                    raise self.import_error
                return self.provider
            if name.split('.')[0] in ('taichi', 'mpi4py', 'exudyn'):
                raise AssertionError('native provider import forbidden: ' + name)
            return original_import(name, *args, **kwargs)

        self.contexts.enter_context(patch('builtins.__import__', side_effect=guarded_import))
        network_stub = types.ModuleType('fibernet.core.network')
        network_stub.FiberNetwork = type('FiberNetwork', (), {})
        self.contexts.enter_context(patch.dict(sys.modules, {'fibernet.core.network': network_stub}))
        spec = importlib.util.spec_from_file_location('_legacy_lammps_under_test', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        self.contexts.enter_context(patch.dict(sys.modules, {spec.name: self.module}))
        spec.loader.exec_module(self.module)
        self.network = object()

    def missing_bridge(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            bridge = self.module.LAMMPSBridge(self.network)
        self.assertEqual(len(caught), 1)
        return bridge

    def provider_bridge(self):
        engine = Mock()
        engine.get_natoms.return_value = 2
        engine.gather_atoms.return_value = [1., 2., 3., 4., 5., 6.]
        factory = Mock(return_value=engine)
        self.provider = types.SimpleNamespace(lammps=factory)
        bridge = self.module.LAMMPSBridge(self.network)
        bridge.write_data_file = Mock()
        bridge.write_input_script = Mock()
        return bridge, factory, engine

    def assert_failure(self, expected, operation):
        with self.assertRaises(BaseException) as caught:
            operation()
        self.assertIs(caught.exception, expected)
        return caught.exception

    def test_missing_provider_before_any_write_or_tempdir(self):
        bridge = self.missing_bridge()
        with patch.object(self.module.tempfile, 'mkdtemp', side_effect=AssertionError('tempdir forbidden')), \
             patch.object(builtins, 'open', side_effect=AssertionError('write forbidden')):
            for workdir in (None, '/must/not/write'):
                self.assert_failure(self.import_error, lambda: bridge.run_simulation(workdir=workdir))
        self.assertEqual(self.import_attempts, ['lammps'])

    def test_missing_provider_preserves_cause(self):
        cause = OSError('provider shared library unavailable')
        self.import_error.__cause__ = cause
        bridge = self.missing_bridge()
        error = self.assert_failure(self.import_error, bridge.run_simulation)
        self.assertIs(error.__cause__, cause)

    def test_export_only_still_works(self):
        bridge = self.missing_bridge()
        bridge.network = types.SimpleNamespace(fibers=[types.SimpleNamespace(centerline=np.array([[1., 2., 3.]]))])
        with tempfile.TemporaryDirectory() as folder:
            data = str(Path(folder) / 'fiber.data')
            script = str(Path(folder) / 'in.fiber')
            self.assertEqual(bridge.write_data_file(data, segments_per_fiber=2), data)
            self.assertEqual(bridge.write_input_script(script, data), script)
            self.assertIn('2 atoms', Path(data).read_text())
            self.assertIn('run             10000', Path(script).read_text())

    def test_constructor_failure_no_close(self):
        bridge, factory, engine = self.provider_bridge()
        error = RuntimeError('constructor failed')
        factory.side_effect = error
        self.assert_failure(error, lambda: bridge.run_simulation(workdir='stub'))
        engine.close.assert_not_called()

    def test_execution_failure_closes_once(self):
        bridge, _, engine = self.provider_bridge()
        error = RuntimeError('execution failed')
        engine.file.side_effect = error
        self.assert_failure(error, lambda: bridge.run_simulation(workdir='stub'))
        engine.close.assert_called_once_with()
        engine.get_natoms.assert_not_called()

    def test_each_extraction_failure_closes_once(self):
        for method in ('get_natoms', 'gather_atoms'):
            with self.subTest(method=method):
                bridge, _, engine = self.provider_bridge()
                error = ValueError(method + ' failed')
                getattr(engine, method).side_effect = error
                self.assert_failure(error, lambda: bridge.run_simulation(workdir='stub'))
                engine.close.assert_called_once_with()

    def test_coordinate_conversion_failure_closes_once(self):
        bridge, _, engine = self.provider_bridge()
        engine.gather_atoms.return_value = [1., 2.]
        with self.assertRaises(ValueError):
            bridge.run_simulation(workdir='stub')
        engine.close.assert_called_once_with()

    def test_primary_failure_survives_cleanup_failure(self):
        for stage in ('file', 'get_natoms', 'gather_atoms'):
            with self.subTest(stage=stage):
                bridge, _, engine = self.provider_bridge()
                primary = RuntimeError('primary ' + stage)
                cleanup = OSError('cleanup failed')
                getattr(engine, stage).side_effect = primary
                engine.close.side_effect = cleanup
                error = self.assert_failure(primary, lambda: bridge.run_simulation(workdir='stub'))
                self.assertIs(error.lammps_close_error, cleanup)
                engine.close.assert_called_once_with()

    def test_interrupt_closes_and_remains_primary(self):
        bridge, _, engine = self.provider_bridge()
        primary = KeyboardInterrupt('interrupted')
        engine.file.side_effect = primary
        engine.close.side_effect = SystemExit('cleanup interrupted')
        self.assert_failure(primary, lambda: bridge.run_simulation(workdir='stub'))
        engine.close.assert_called_once_with()

    def test_close_only_failure_is_not_success(self):
        bridge, _, engine = self.provider_bridge()
        error = OSError('close failed')
        engine.close.side_effect = error
        self.assert_failure(error, lambda: bridge.run_simulation(workdir='stub'))
        engine.close.assert_called_once_with()

    def test_success_preserves_coordinates_time_and_provider_arguments(self):
        bridge, factory, engine = self.provider_bridge()
        result = bridge.run_simulation(temperature=321., nsteps=17, workdir='stub')
        factory.assert_called_once_with(cmdargs=['-log', 'none', '-screen', 'none'])
        data_path = os.path.join('stub', 'fiber.data')
        script_path = os.path.join('stub', 'in.fiber')
        bridge.write_data_file.assert_called_once_with(data_path)
        bridge.write_input_script.assert_called_once_with(script_path, data_path, temperature=321., nsteps=17)
        engine.file.assert_called_once_with(script_path)
        engine.gather_atoms.assert_called_once_with('x', 1, 3)
        engine.close.assert_called_once_with()
        np.testing.assert_array_equal(result.final_structure, [[1., 2., 3.], [4., 5., 6.]])
        self.assertEqual(result.simulation_time, 0.017)
        self.assertEqual(result.to_dict(), {'num_frames': 0, 'final_energy': None, 'final_temperature': None, 'simulation_time': 0.017})
        self.assertEqual(self.import_attempts, ['lammps', 'lammps'])

    def test_success_path_arguments_follow_platform_join(self):
        for path_flavor in (posixpath, ntpath):
            with self.subTest(path_flavor=path_flavor.__name__):
                bridge, _, engine = self.provider_bridge()
                data_path = path_flavor.join('stub', 'fiber.data')
                script_path = path_flavor.join('stub', 'in.fiber')
                with patch.object(self.module.os, 'path', path_flavor):
                    bridge.run_simulation(workdir='stub')
                bridge.write_data_file.assert_called_once_with(data_path)
                bridge.write_input_script.assert_called_once_with(
                    script_path, data_path, temperature=300.0, nsteps=1000)
                engine.file.assert_called_once_with(script_path)
                engine.close.assert_called_once_with()

    def test_unavailable_and_measured_zero_are_distinct(self):
        absent = self.module.LAMMPSResult().to_dict()
        zero = self.module.LAMMPSResult(energies=[0.], temperatures=[0.]).to_dict()
        self.assertIsNone(absent['final_energy'])
        self.assertIsNone(absent['final_temperature'])
        self.assertEqual(zero['final_energy'], 0.)
        self.assertEqual(zero['final_temperature'], 0.)
        self.assertIsNotNone(zero['final_energy'])
        self.assertIsNotNone(zero['final_temperature'])

    def test_wrapper_forwards_arguments_and_failure(self):
        bridge = Mock()
        factory = Mock(return_value=bridge)
        with patch.object(self.module, 'LAMMPSBridge', factory):
            result = self.module.run_lammps_md(self.network, temperature=310., nsteps=23, workdir='chosen')
            self.assertIs(result, bridge.run_simulation.return_value)
            factory.assert_called_once_with(self.network)
            bridge.run_simulation.assert_called_once_with(temperature=310., nsteps=23, workdir='chosen')
            error = RuntimeError('wrapper failure')
            bridge.run_simulation.side_effect = error
            self.assert_failure(error, lambda: self.module.run_lammps_md(self.network))

    def test_real_provider_cannot_escape_import_trap(self):
        with self.assertRaises(ImportError) as caught:
            __import__('lammps')
        self.assertIs(caught.exception, self.import_error)
        with self.assertRaises(ImportError):
            __import__('lammps.core')


    def test_invalid_controls_precede_real_writers_and_provider(self):
        factory = Mock(side_effect=AssertionError('provider construction forbidden'))
        self.provider = types.SimpleNamespace(lammps=factory)
        bridge = self.module.LAMMPSBridge(self.network)
        cases = {'nsteps': ['10', True, 1.5, -1],
                 'temperature': ['300', True, float('nan'), float('inf'), -1.]}
        # Keep both production writer methods intact: validation must stop before
        # even the deliberately unusable network is inspected.
        with patch.object(self.module.tempfile, 'mkdtemp', side_effect=AssertionError('tempdir forbidden')), \
             patch.object(builtins, 'open', side_effect=AssertionError('write forbidden')):
            for name, values in cases.items():
                for value in values:
                    with self.subTest(name=name, value=value):
                        with self.assertRaisesRegex(ValueError, name):
                            bridge.run_simulation(**{name: value})
        factory.assert_not_called()

    def test_standalone_writer_rejects_invalid_controls_without_provider(self):
        bridge = self.missing_bridge()
        cases = {'nsteps': ['10', True, 1.5, -1],
                 'dump_freq': ['100', True, 1.5, 0, -1],
                 'temperature': ['300', True, float('nan'), float('inf'), -1.],
                 'timestep': ['0.001', True, float('nan'), float('inf'), 0., -1.]}
        with patch.object(builtins, 'open', side_effect=AssertionError('write forbidden')):
            for name, values in cases.items():
                for value in values:
                    with self.subTest(name=name, value=value):
                        with self.assertRaisesRegex(ValueError, name):
                            bridge.write_input_script('unused.in', 'unused.data', **{name: value})
        self.assertEqual(self.import_attempts, ['lammps'])

    def test_script_preflight_preserves_zero_and_force_field_text(self):
        bridge = self.missing_bridge()
        coefficients = {'harmonic': 'bond_coeff      * 100.0 1.0',
                        'fene': 'bond_coeff      * 30.0 1.5 1.0 1.0',
                        'morse': 'bond_coeff      * 10.0 2.0 1.0'}
        with tempfile.TemporaryDirectory() as folder:
            script = str(Path(folder) / 'in.fiber')
            for force_field, coefficient in coefficients.items():
                with self.subTest(force_field=force_field):
                    bridge.write_input_script(script, 'fiber.data', temperature=0.,
                        timestep=.002, nsteps=0, dump_freq=1, force_field=force_field)
                    text = Path(script).read_text()
                    self.assertIn(coefficient, text)
                    self.assertIn('temp 0.0 0.0 100.0', text)
                    self.assertIn('timestep        0.002', text)
                    self.assertIn('custom 1 dump.lammpstrj id type x y z', text)
                    self.assertIn('run             0', text)
        self.assertEqual(self.import_attempts, ['lammps'])

    def test_generated_physics_and_signatures_unchanged(self):
        # Lock the existing public call signatures and generated-text writers.
        tree = ast.parse(SOURCE.read_text())
        # Inventory direct public definitions, not unrelated nested helpers.
        top_functions = {node.name: node for node in tree.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertEqual(set(top_functions), {'run_lammps_md', '_checked_legacy_controls'})
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertEqual(set(classes), {'LAMMPSResult', 'LAMMPSBridge'})
        definitions = {'run_lammps_md': top_functions['run_lammps_md']}
        expected_methods = {'LAMMPSResult': {'to_dict'},
                            'LAMMPSBridge': {'__init__', 'write_data_file',
                                             'write_input_script', 'run_simulation'}}
        for name, expected in expected_methods.items():
            methods = {node.name: node for node in classes[name].body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            self.assertEqual(set(methods), expected, name)
            definitions.update(methods)
        # Explicitly admit exactly the reviewed private scalar validator, including
        # both nested checks and their bounds; behavior is exercised below too.
        helper = top_functions['_checked_legacy_controls']
        self.assertEqual([node.name for node in helper.body if isinstance(node, ast.FunctionDef)],
                         ['real', 'integer'])
        self.assertEqual(syntax_hash(helper), '917bb57343923d6162d90401bf5b07a9bfad5c55dbbcb347038bd46f4dc11c87')
        expected_signatures = {'run_lammps_md': '5c59b061cffe9cedbd4a86628088586d8bc923385b7c956c0e7b8e49b646faef', 'to_dict': 'd2f53b54bcf64664c681c7039aa2f19defd9a60d393f079097db7394eae7063b', '__init__': '1ddd3cd3684fcd9858d4e948e48dd056bfe2a97886a822327aba3d6ff3ee867d', 'write_data_file': '91883e10c2bdc36ad8df39d3f80e60ab5967d4bc60e2e39c35653f63d82511eb', 'write_input_script': '42357c5a65fcfc0a45fdae9f541821de22c26a9a9a45cf161e0d99f5916cc09f', 'run_simulation': '9bcf9b96d3b0551fa2e46dd816cbcd00aa80ec5d2fbabd35aa2c50d7adc66fc4'}
        expected_writers = {'write_data_file': '9c9b13a29c566c787f3a279c013653704af07e42b6e39481c06b7389f3c737e5', 'write_input_script': 'ecbb9f2fd209888983617d6c0c667cb29495369cca5f36489ff7de7d527a8b0b'}
        self.assertEqual(set(definitions), set(expected_signatures))
        for name, expected in expected_signatures.items():
            actual = syntax_hash(definitions[name].args)
            self.assertEqual(actual, expected, name)
        for name, expected in expected_writers.items():
            writer = copy.deepcopy(definitions[name])
            if name == 'write_input_script':
                approved_preflight = ast.parse(
                    'temperature, timestep, nsteps, dump_freq = '
                    '_checked_legacy_controls(temperature, timestep, nsteps, dump_freq)'
                ).body[0]
                self.assertEqual(stable_ast(writer.body[1]), stable_ast(approved_preflight))
                # Remove only the exact initial validation assignment, never other
                # statements or calls. Keep the historical whole-writer hash below.
                del writer.body[1]
                # Reverse only the four reviewed parameter-documentation edits.
                # Every other docstring byte and all generated physics stay pinned.
                doc = writer.body[0].value.value
                for current, original in (
                    ('Finite nonnegative simulation temperature (K).', 'Simulation temperature (K).'),
                    ('Finite positive time step (ps).', 'Time step (ps).'),
                    ('Nonnegative integer number of simulation steps (not boolean).', 'Number of simulation steps.'),
                    ('Positive integer dump frequency for trajectory (not boolean).', 'Dump frequency for trajectory.'),
                ):
                    self.assertEqual(doc.count(current), 1)
                    doc = doc.replace(current, original)
                writer.body[0].value.value = doc
            actual = syntax_hash(writer)
            self.assertEqual(actual, expected, name)



if __name__ == '__main__':
    unittest.main(verbosity=2)
