"""Small synthetic plots and config preflight only: no generators/fit/physics."""
import builtins
import contextlib
import pathlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.viz import plot2d
from fibernet.utils.config import ExperimentConfig, run_from_config


@contextlib.contextmanager
def no_computation():
    # Catch attempted entry into generation, fitting, training or solver functions.
    import sys
    old = sys.getprofile()
    def profile(frame, event, arg):
        if event != 'call':
            return
        filename = frame.f_code.co_filename.replace('\\', '/')
        name = frame.f_code.co_name
        if (any(part in filename for part in ('/fibernet/gen/', '/fibernet/sim/')) and not name.startswith('<')) or (
            '/fibernet/' in filename and name in {'fit', 'train', 'solve', 'simulate', 'get_material', 'apply_uniaxial_strain'}):
            raise AssertionError('forbidden computation: ' + filename + ':' + name)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(old)


class OrientationContract(unittest.TestCase):
    def setUp(self):
        self.existing_figures = set(plot2d.plt.get_fignums())
        self.guard = no_computation()
        self.guard.__enter__()
    def tearDown(self):
        for number in set(plot2d.plt.get_fignums()) - self.existing_figures:
            plot2d.plt.close(number)
        self.guard.__exit__(None, None, None)
    def network(self, paths, dim=2):
        return FiberNetwork([Fiber(np.asarray(p, dtype=float)) for p in paths], dimension=dim)
    def test_known_axes_density_centers_width_equal_weight(self):
        net = self.network([[[0,0,0],[1,0,0]], [[0,0,0],[0,9,0]], [[1,0,0],[0,0,0]]])
        fig = plot2d.plot_orientation_distribution(net)
        bars = fig.axes[0].patches
        self.assertEqual(len(bars), 36)
        width = np.pi/36
        np.testing.assert_allclose([p.get_width() for p in bars], width)
        np.testing.assert_allclose([p.get_x()+p.get_width()/2 for p in bars], (np.arange(36)+.5)*width)
        heights = np.array([p.get_height() for p in bars])
        self.assertAlmostEqual(sum(heights)*width, 1)
        self.assertAlmostEqual(heights[0]*width, 2/3)
        self.assertAlmostEqual(heights[18]*width, 1/3)
    def test_curve_and_3d_use_xy_endpoint_projection(self):
        net = self.network([[[0,0,0],[0,20,8],[2,0,4]], [[0,0,0],[2,2,4]]], 3)
        fig = plot2d.plot_orientation_distribution(net)
        heights = np.array([p.get_height() for p in fig.axes[0].patches])
        self.assertEqual(set(np.flatnonzero(heights)), {0,9})
        self.assertIn('XY', fig.axes[0].get_title())
    def test_empty_saves(self):
        with tempfile.TemporaryDirectory() as td:
            dest = pathlib.Path(td)/'empty.png'
            fig = plot2d.plot_orientation_distribution(FiberNetwork(), save_path=str(dest))
            self.assertEqual(fig.axes[0].get_title(), 'No fibers')
            self.assertEqual(len(fig.axes[0].patches), 0)
            self.assertGreater(dest.stat().st_size, 0)
            if os.environ.get('FIBERNET_PLOT_EVIDENCE'):
                shutil.copyfile(dest, pathlib.Path(os.environ['FIBERNET_PLOT_EVIDENCE'])/dest.name)
    def test_nonempty_saves(self):
        with tempfile.TemporaryDirectory() as td:
            dest = pathlib.Path(td)/'axes.png'
            plot2d.plot_orientation_distribution(self.network([[[0,0,0],[1,0,0]]]), save_path=str(dest))
            self.assertGreater(dest.stat().st_size, 0)
            if os.environ.get('FIBERNET_PLOT_EVIDENCE'):
                shutil.copyfile(dest, pathlib.Path(os.environ['FIBERNET_PLOT_EVIDENCE'])/dest.name)
    def test_invalid_chords_fail_before_direction_or_figure(self):
        paths = [ [[0,0,0],[0,0,0]], [[0,0,0],[1,1,0],[0,0,0]], [[0,0,0]],
                  [[0,0,0],[0,0,2]], [[0,0,0],[np.nan,1,0]],
                  [[-1e308,0,0],[1e308,0,0]], [[0,0,0],[1e200,0,0]],
                  [[0,0,0],[1e-13,0,0]] ]
        for path in paths:
            with self.subTest(path=path):
                net = self.network([path])
                with patch.object(net, 'fiber_orientations', side_effect=AssertionError('direction called')), patch.object(plot2d.plt, 'subplots', side_effect=AssertionError('figure created')):
                    with self.assertRaisesRegex(ValueError, 'fiber 0'):
                        plot2d.plot_orientation_distribution(net)
    def test_type_dimension_and_near_threshold(self):
        with self.assertRaises(TypeError):
            plot2d.plot_orientation_distribution(object())
        with self.assertRaises(ValueError):
            plot2d.plot_orientation_distribution(FiberNetwork(dimension=4))
        fig = plot2d.plot_orientation_distribution(self.network([[[0,0,0],[2e-12,0,0]]]))
        self.assertEqual(len(fig.axes[0].patches), 36)
    def test_mixed_valid_invalid_does_not_silently_drop_fiber(self):
        net = self.network([[[0,0,0],[1,0,0]], [[0,0,0],[0,0,4]]], 3)
        with patch.object(plot2d.plt, 'subplots', side_effect=AssertionError('figure created')):
            with self.assertRaisesRegex(ValueError, 'fiber 1'):
                plot2d.plot_orientation_distribution(net)
    def test_missing_matplotlib_message(self):
        with patch.object(plot2d, 'HAS_MATPLOTLIB', False):
            with self.assertRaisesRegex(ImportError, 'matplotlib'):
                plot2d.plot_orientation_distribution(FiberNetwork())


class ConfigPreflight(unittest.TestCase):
    @contextlib.contextmanager
    def no_runtime_import(self):
        original = builtins.__import__
        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if (level and (name in ('sim', 'materials', 'core.material', 'analysis') or 'gen' in fromlist)):
                raise AssertionError('runtime import attempted before preflight: ' + name)
            return original(name, globals, locals, fromlist, level)
        with no_computation(), patch('builtins.__import__', side_effect=guarded):
            yield
    def test_aggregate_before_generation_material_or_output(self):
        config = ExperimentConfig(network={'generator':'never_call'}, material={'type':'never_lookup'},
            simulation={'type':'mechanical'}, analysis={'morphology':True,'stress_strain':True}, output={'directory':'never_write'})
        before = config.to_dict(); h = config.compute_hash()
        with self.no_runtime_import():
            with self.assertRaises(NotImplementedError) as error:
                run_from_config(config)
        for key in ('simulation.type=mechanical','analysis.morphology','analysis.stress_strain','No network was generated'):
            self.assertIn(key, str(error.exception))
        self.assertEqual(config.to_dict(), before); self.assertEqual(config.compute_hash(), h)
    def test_each_legacy_branch(self):
        for kwargs in ({'simulation':{'type':'mechanical'}}, {'analysis':{'morphology':True}}, {'analysis':{'stress_strain':True}}):
            with self.subTest(kwargs=kwargs), self.no_runtime_import(), self.assertRaises(NotImplementedError):
                run_from_config(ExperimentConfig(network={'generator':'never_call'}, **kwargs))
    def test_false_flags_continue_to_validation_not_runtime(self):
        with self.no_runtime_import(), self.assertRaisesRegex(ValueError, "Missing 'network'"):
            run_from_config(ExperimentConfig(analysis={'morphology':False,'stress_strain':False}))
    def test_thermal_keeps_existing_priority_message(self):
        config = ExperimentConfig(simulation={'type':'thermal'},analysis={'morphology':True})
        with self.no_runtime_import(), self.assertRaisesRegex(NotImplementedError, 'Legacy thermal configuration execution is unsupported; no network was generated'):
            run_from_config(config)
    def test_json_file_same_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)/'config.json'
            ExperimentConfig(network={'generator':'never_call'},simulation={'type':'mechanical'}).to_json(p)
            with self.no_runtime_import(), self.assertRaisesRegex(NotImplementedError, 'simulation.type=mechanical'):
                run_from_config(p)

if __name__ == '__main__':
    unittest.main()
