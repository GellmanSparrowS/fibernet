"""NO-FIT plot ownership checks, each isolated from the test collector's backend."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class PlotStateSubprocessContracts(unittest.TestCase):
    def run_scenario(self, name):
        env = dict(os.environ, MPLBACKEND='svg', OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                 '--scenario', name], env=env, capture_output=True,
                                text=True, timeout=45)
        output = result.stdout + result.stderr
        print(output, end='', flush=True)
        self.assertEqual(result.returncode, 0, output)
        self.assertIn('CHILD_FIT_ATTEMPTS=[]', output)

    def test_detached_factory_does_not_import_pyplot(self):
        self.run_scenario("test_detached_factory_does_not_import_pyplot")

    def test_detached_success_edit_export_all_four(self):
        self.run_scenario("test_detached_success_edit_export_all_four")

    def test_invalid_inputs_preserve_user_state(self):
        self.run_scenario("test_invalid_inputs_preserve_user_state")

    def test_save_error_preserves_original_and_user_state_all_four(self):
        self.run_scenario("test_save_error_preserves_original_and_user_state_all_four")

    def test_show_true_uses_pyplot_and_display_all_four(self):
        self.run_scenario("test_show_true_uses_pyplot_and_display_all_four")


def _run_child(scenario):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    import tempfile
    import unittest
    from unittest.mock import patch

    import matplotlib
    matplotlib.use('svg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib._pylab_helpers import Gcf
    import numpy as np

    spec = importlib.util.spec_from_file_location('plot_state_utils', Path(__file__).parents[1] / 'fibernet/ml/utils.py')
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)


    class PlotStateContracts(unittest.TestCase):
        def setUp(self):
            self.user, self.ax = plt.subplots()
            self.line, = self.ax.plot([0, 1], [3, 4])
            self.manager = Gcf.get_active()
            self.backend = matplotlib.get_backend()
            self.numbers = plt.get_fignums()
            self.style = dict(matplotlib.rcParams)
            self.addCleanup(plt.close, self.user)
            self.tmp = tempfile.TemporaryDirectory()
            self.addCleanup(self.tmp.cleanup)
            self.root = Path(self.tmp.name)
            model = patch.object(utils, '_build_model', return_value=SimpleNamespace())
            curve = patch('sklearn.model_selection.learning_curve', return_value=(np.array([2, 4]), np.array([[.2, .4], [.6, .8]]), np.array([[.1, .3], [.5, .7]])))
            model.start(); self.addCleanup(model.stop)
            self.curve = curve.start(); self.addCleanup(curve.stop)

        def calls(self):
            return {
                'predictions': lambda **kw: utils.plot_predictions([1., 2., 3.], [1.1, 1.9, 3.], **kw),
                'importance': lambda **kw: utils.plot_feature_importance(SimpleNamespace(coef_=np.array([1., -2.])), ['a', 'b'], top_k=2, **kw),
                'residuals': lambda **kw: utils.plot_residuals([1., 2., 3.], [1.1, 1.9, 3.], **kw),
                'learning': lambda **kw: utils.plot_learning_curve(np.ones((4, 2)), np.arange(4.), **kw),
            }

        def unchanged(self):
            self.assertEqual(matplotlib.get_backend(), self.backend)
            self.assertEqual(plt.get_fignums(), self.numbers)
            self.assertIs(Gcf.get_active(), self.manager)
            self.assertIs(plt.gcf(), self.user)
            self.assertIs(self.ax.lines[0], self.line)
            self.assertEqual(dict(matplotlib.rcParams), self.style)

        def test_detached_success_edit_export_all_four(self):
            with patch.object(matplotlib, 'use', side_effect=AssertionError('global backend change')), patch.object(plt, 'subplots', side_effect=AssertionError('pyplot allocation')):
                for name, call in self.calls().items():
                    with self.subTest(helper=name):
                        fig = call(save_path=self.root / (name + '.png'))
                        self.assertIsInstance(fig, Figure)
                        self.assertIsInstance(fig.canvas, FigureCanvasAgg)
                        self.assertEqual(len(fig.axes), 2 if name == 'residuals' else 1)
                        self.assertEqual(fig.get_facecolor(), matplotlib.colors.to_rgba('#0a0a0f'))
                        self.assertTrue((self.root / (name + '.png')).stat().st_size > 0)
                        self.assertFalse(any(m.canvas.figure is fig for m in Gcf.get_all_fig_managers()))
                        fig.axes[0].set_title('caller edit')
                        fig.savefig(self.root / (name + '-edited.svg'))
                        self.assertEqual(fig.axes[0].get_title(), 'caller edit')
                        self.assertIn('caller edit', (self.root / (name + '-edited.svg')).read_text())
                        self.unchanged()
            self.curve.assert_called_once()

        def test_save_error_preserves_original_and_user_state_all_four(self):
            for name, call in self.calls().items():
                with self.subTest(helper=name):
                    error = OSError('injected save failure')
                    with patch.object(Figure, 'savefig', side_effect=error):
                        with self.assertRaises(OSError) as caught:
                            call(save_path=self.root / 'not-written.png')
                    self.assertIs(caught.exception, error)
                    self.unchanged()

        def test_invalid_inputs_preserve_user_state(self):
            calls = [lambda: utils.plot_predictions([], []),
                     lambda: utils.plot_feature_importance(SimpleNamespace(coef_=[1., 2.]), ['wrong']),
                     lambda: utils.plot_residuals([1., 2.], [1., 2., 3.])]
            for call in calls:
                with self.assertRaises(ValueError):
                    call()
                self.unchanged()
            error = ValueError('injected learning-curve input rejection')
            self.curve.side_effect = error
            with self.assertRaises(ValueError) as caught:
                self.calls()['learning']()
            self.assertIs(caught.exception, error)
            self.unchanged()

        def test_show_true_uses_pyplot_and_display_all_four(self):
            for name, call in self.calls().items():
                with self.subTest(helper=name):
                    with patch.object(plt, 'subplots', wraps=plt.subplots) as allocate, patch.object(plt, 'show') as display, patch.object(matplotlib, 'use', side_effect=AssertionError('backend switch')):
                        fig = call(show=True)
                        allocate.assert_called_once()
                        display.assert_called_once_with()
                    self.assertTrue(any(m.canvas.figure is fig for m in Gcf.get_all_fig_managers()))
                    self.assertEqual(matplotlib.get_backend(), self.backend)
                    plt.close(fig)
                    self.unchanged()

        def test_detached_factory_does_not_import_pyplot(self):
            import builtins
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name == 'matplotlib.pyplot':
                    raise AssertionError('detached path imported pyplot')
                return original(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=guarded):
                for call in self.calls().values():
                    fig = call()
                    self.assertIsInstance(fig, Figure)
                    self.unchanged()

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append('fit')
        raise AssertionError('child fitting forbidden')
    patchers = [patch.object(cls, 'fit', side_effect=forbidden) for cls in
                (RandomForestRegressor, GradientBoostingRegressor, Ridge,
                 LinearRegression, MLPRegressor, Pipeline, StandardScaler, SVR)]
    for item in patchers:
        item.start()
    try:
        suite = unittest.TestSuite([PlotStateContracts(scenario)])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print('CHILD_FIT_ATTEMPTS=' + repr(attempts), flush=True)
        return result.wasSuccessful() and result.testsRun == 1 and not attempts
    finally:
        for item in reversed(patchers):
            item.stop()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--scenario':
        raise SystemExit(0 if _run_child(sys.argv[2]) else 1)
    unittest.main()
