"""Small RL plotting contracts, isolated from the test collector's backend."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestRLPlotState(unittest.TestCase):
    def run_scenario(self, name):
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, MPLBACKEND='TkAgg' if name == 'fresh_headless' else 'svg',
                       MPLCONFIGDIR=temporary, XDG_CACHE_HOME=temporary,
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            result = subprocess.run(
                [sys.executable, '-B', str(Path(__file__).resolve()), '--scenario', name],
                env=env, capture_output=True, text=True, timeout=45)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn('RL_PLOT_SCENARIO_PASSED=' + name, output)

    def test_fresh_headless_without_pyplot(self):
        self.run_scenario('fresh_headless')

    def test_headless_save_edit_and_registry(self):
        self.run_scenario('headless_save')

    def test_detached_calls_do_not_access_pyplot(self):
        self.run_scenario('no_pyplot_access')

    def test_invalid_input_preserves_caller_state(self):
        self.run_scenario('invalid_input')

    def test_save_error_preserves_caller_state(self):
        self.run_scenario('save_error')

    def test_show_true_preserves_backend_and_displays(self):
        self.run_scenario('show_true')

    def test_plot_values_and_action_grid(self):
        self.run_scenario('plot_values')

    def test_public_exports_and_signatures(self):
        self.run_scenario('public_exports')


def _run_scenario(name):
    import builtins
    import inspect
    import io
    from unittest.mock import patch

    # Normal package imports exercise public exports. No training, environment
    # step, native runtime initialization or physical solve is requested.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import fibernet
    import fibernet.rl as rl
    from fibernet.rl import utils
    import matplotlib
    import numpy as np

    check = unittest.TestCase()
    names = ('plot_reward_curve', 'plot_convergence', 'plot_action_distribution')
    calls = {
        'reward': lambda **kw: utils.plot_reward_curve([1., 2., 3.], window=2, **kw),
        'convergence': lambda **kw: utils.plot_convergence([3., 2., 1.], **kw),
        'actions': lambda **kw: utils.plot_action_distribution([1., 2., 3.], **kw),
        'actions_grid': lambda **kw: utils.plot_action_distribution(
            [{'a':1., 'b':2., 'c':3., 'd':4.}, {'a':2., 'b':3., 'c':4., 'd':5.}], **kw),
    }
    if name == 'public_exports':
        expected = {
            'plot_reward_curve': [('rewards', inspect.Parameter.empty), ('window', 20),
                                 ('title', 'Training Progress'), ('save_path', None), ('show', False)],
            'plot_convergence': [('values', inspect.Parameter.empty), ('minimize', True),
                                 ('title', 'Optimization Convergence'), ('save_path', None), ('show', False)],
            'plot_action_distribution': [('actions', inspect.Parameter.empty), ('action_names', None),
                                         ('title', 'Action Distribution'), ('save_path', None), ('show', False)],
        }
        for symbol in names:
            owner = getattr(utils, symbol)
            check.assertIs(getattr(rl, symbol), owner)
            check.assertIs(getattr(fibernet, symbol), owner)
            parameters = list(inspect.signature(owner).parameters.values())
            check.assertEqual([(p.name, p.default) for p in parameters], expected[symbol])
            check.assertEqual(parameters[0].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            check.assertTrue(all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in parameters[1:]))
        return

    if name == 'fresh_headless':
        check.assertNotIn('matplotlib.pyplot', sys.modules)
        backend = matplotlib.get_backend()
        check.assertEqual(backend.lower(), 'tkagg')
        for call in calls.values():
            fig = call()
            stream = io.BytesIO()
            fig.savefig(stream, format='pdf')
            check.assertTrue(stream.getvalue().startswith(b'%PDF-'))
            check.assertNotIn('matplotlib.pyplot', sys.modules)
            check.assertEqual(matplotlib.get_backend(), backend)
        return

    import matplotlib.pyplot as plt
    from matplotlib._pylab_helpers import Gcf
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    user, ax = plt.subplots(num=73)
    line, = ax.plot([0., 1.], [2., 3.])
    manager = Gcf.get_active()
    backend, fignums, params = matplotlib.get_backend(), plt.get_fignums(), dict(matplotlib.rcParams)

    def unchanged():
        check.assertEqual(matplotlib.get_backend(), backend)
        check.assertEqual(plt.get_fignums(), fignums)
        check.assertIs(Gcf.get_active(), manager)
        check.assertIs(plt.gcf(), user)
        check.assertIs(ax.lines[0], line)
        check.assertEqual(dict(matplotlib.rcParams), params)

    try:
        if name == 'headless_save':
            with tempfile.TemporaryDirectory() as temporary:
                for label, call in calls.items():
                    path = Path(temporary) / (label + '.png')
                    with patch.object(matplotlib, 'use', side_effect=AssertionError('global backend switch')):
                        fig = call(save_path=path)
                    check.assertIsInstance(fig, Figure)
                    check.assertIsInstance(fig.canvas, FigureCanvasAgg)
                    check.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                    check.assertGreater(path.stat().st_size, 1000)
                    check.assertFalse(any(m.canvas.figure is fig for m in Gcf.get_all_fig_managers()))
                    fig.axes[0].set_title('caller edited title')
                    svg = path.with_suffix('.svg')
                    fig.savefig(svg)
                    check.assertIn('caller edited title', svg.read_text(encoding='utf-8'))
                    unchanged()
        elif name == 'no_pyplot_access':
            original = builtins.__import__
            def guarded(module, *args, **kwargs):
                if module == 'matplotlib.pyplot':
                    raise AssertionError('detached branch imported pyplot')
                return original(module, *args, **kwargs)
            with patch('builtins.__import__', side_effect=guarded):
                for call in calls.values():
                    check.assertIsInstance(call(), Figure)
                    unchanged()
        elif name == 'invalid_input':
            bad_calls = [lambda: utils.plot_action_distribution([]),
                         lambda: utils.plot_convergence([]),
                         lambda: utils.plot_reward_curve([1., 2.], window=0)]
            for call in bad_calls:
                with check.assertRaises((ValueError, IndexError)):
                    call()
                unchanged()
        elif name == 'save_error':
            for call in calls.values():
                error = OSError('injected save failure')
                with patch.object(Figure, 'savefig', side_effect=error):
                    with check.assertRaises(OSError) as caught:
                        call(save_path='not-written.png')
                check.assertIs(caught.exception, error)
                unchanged()
        elif name == 'show_true':
            for call in calls.values():
                with patch.object(matplotlib, 'use', side_effect=AssertionError('global backend switch')), \
                     patch.object(plt, 'show') as display:
                    fig = call(show=True)
                display.assert_called_once_with()
                check.assertEqual(matplotlib.get_backend(), backend)
                check.assertTrue(any(m.canvas.figure is fig for m in Gcf.get_all_fig_managers()))
                plt.close(fig)
                unchanged()
        elif name == 'plot_values':
            reward = calls['reward']()
            np.testing.assert_allclose(reward.axes[0].lines[0].get_ydata(), [1.5, 2.5])
            np.testing.assert_allclose(reward.axes[0].lines[1].get_ydata(), [1., 2., 3.])
            convergence = calls['convergence']()
            np.testing.assert_allclose(convergence.axes[0].lines[0].get_ydata(), [3., 2., 1.])
            maximum = utils.plot_convergence([3., 2., 4.], minimize=False)
            np.testing.assert_allclose(maximum.axes[0].lines[0].get_ydata(), [3., 3., 4.])
            grid = calls['actions_grid']()
            check.assertEqual(len(grid.axes), 6)
            check.assertEqual([a.get_visible() for a in grid.axes], [True]*4 + [False]*2)
            check.assertEqual([a.get_title() for a in grid.axes[:4]], ['a', 'b', 'c', 'd'])
            unchanged()
        else:
            raise ValueError('Unknown scenario: ' + name)
    finally:
        plt.close('all')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--scenario':
        _run_scenario(sys.argv[2])
        print('RL_PLOT_SCENARIO_PASSED=' + sys.argv[2])
    else:
        unittest.main()
