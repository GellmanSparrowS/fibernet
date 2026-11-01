"""Regression tests for legacy LAMMPS scalar controls."""
import importlib
import sys
import types
import unittest
from unittest import mock

import numpy as np
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork

legacy = importlib.import_module('fibernet.integrations.lammps_integration')


class LegacyLAMMPSControlPreflightTests(unittest.TestCase):
    def setUp(self):
        self.provider = types.ModuleType('lammps')
        self.engine = mock.Mock()
        self.engine.get_natoms.return_value = 1
        self.engine.gather_atoms.return_value = [0.0, 0.0, 0.0]
        self.provider.lammps = mock.Mock(return_value=self.engine)
        self.modules = mock.patch.dict(sys.modules, {'lammps': self.provider})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.network = FiberNetwork(fibers=[Fiber(centerline=np.array(
            [[0., 0., 0.], [1., 0., 0.]]))])
        self.bridge = legacy.LAMMPSBridge(self.network)

    def assert_run_rejected(self, name, value, convenience=False):
        with mock.patch.object(legacy.tempfile, 'mkdtemp') as mkdir, \
                mock.patch('builtins.open', mock.mock_open()) as opened:
            with self.assertRaisesRegex(ValueError, name):
                if convenience:
                    legacy.run_lammps_md(self.network, **{name: value})
                else:
                    self.bridge.run_simulation(**{name: value})
            mkdir.assert_not_called()
            opened.assert_not_called()
            self.provider.lammps.assert_not_called()

    def test_invalid_run_controls_rejected_before_side_effects(self):
        cases = {
            'nsteps': ['10', '10\nrun 10', True, np.bool_(True), 1.5, -1],
            'temperature': ['300', True, np.bool_(True), float('nan'),
                            float('inf'), -1.0, 10**1000],
        }
        for convenience in (False, True):
            for name, values in cases.items():
                for value in values:
                    with self.subTest(name=name, value=value, convenience=convenience):
                        self.assert_run_rejected(name, value, convenience)

    def test_standalone_script_controls_rejected_before_open(self):
        cases = {
            'nsteps': ['10', True, 1.5, -1],
            'dump_freq': ['100', True, 1.5, 0, -1],
            'temperature': ['300', True, float('nan'), float('inf'), -1.],
            'timestep': ['0.001', True, float('nan'), float('inf'), 0., -1.],
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value), \
                        mock.patch('builtins.open', mock.mock_open()) as opened:
                    with self.assertRaisesRegex(ValueError, name):
                        self.bridge.write_input_script('unused.in', 'unused.data',
                                                       **{name: value})
                    opened.assert_not_called()
                    self.provider.lammps.assert_not_called()

    def test_zero_temperature_and_steps_remain_admitted(self):
        # No disk writes and no real native provider. Actual writer/run methods execute.
        with mock.patch.object(legacy.tempfile, 'mkdtemp', return_value='/fake') as mkdir, \
                mock.patch('builtins.open', mock.mock_open()) as opened:
            result = self.bridge.run_simulation(temperature=0., nsteps=np.int64(0))
        mkdir.assert_called_once_with()
        self.assertEqual(opened.call_count, 2)
        self.provider.lammps.assert_called_once_with(cmdargs=['-log', 'none', '-screen', 'none'])
        self.engine.file.assert_called_once_with('/fake/in.fiber')
        self.engine.close.assert_called_once_with()
        self.assertEqual(result.simulation_time, 0.)
        text = ''.join(call.args[0] for call in opened().write.call_args_list)
        self.assertIn('temp 0.0 0.0', text)
        self.assertIn('run             0', text)

    def test_positive_integral_and_finite_real_controls_remain_admitted(self):
        with mock.patch('builtins.open', mock.mock_open()) as opened:
            self.bridge.write_input_script('unused.in', 'unused.data',
                temperature=np.float64(300.), timestep=np.float64(.001),
                nsteps=np.int64(10), dump_freq=np.int64(1))
        text = opened().write.call_args.args[0]
        self.assertIn('run             10', text)
        self.assertIn('custom 1 dump.lammpstrj', text)

    def test_original_execution_error_and_cleanup_are_preserved(self):
        failure = RuntimeError('fake native failure')
        self.engine.file.side_effect = failure
        with mock.patch('builtins.open', mock.mock_open()):
            with self.assertRaises(RuntimeError) as raised:
                self.bridge.run_simulation(nsteps=1, workdir='/fake')
        self.assertIs(raised.exception, failure)
        self.engine.close.assert_called_once_with()

    def test_missing_provider_error_precedes_control_errors(self):
        # Model an absent provider at the import boundary, not by altering bridge methods.
        with mock.patch.dict(sys.modules, {'lammps': None}):
            with self.assertWarns(UserWarning):
                bridge = legacy.LAMMPSBridge(self.network)
            with mock.patch.object(legacy.tempfile, 'mkdtemp') as mkdir, \
                    mock.patch('builtins.open', mock.mock_open()) as opened:
                with self.assertRaises(ModuleNotFoundError) as raised:
                    bridge.run_simulation(nsteps='10')
                self.assertIs(raised.exception, bridge._lammps_import_error)
                mkdir.assert_not_called()
                opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
