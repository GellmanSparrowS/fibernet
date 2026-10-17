"""Fresh-process regressions for the normal public import namespace."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_python(source):
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys; sys.path.insert(0, " + repr(str(ROOT / "tests")) + "); "
         "from _source_check_support import install_import_process_audit\n"
         + textwrap.dedent(source)],
        cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if result.stdout:
        print(result.stdout, end='')


class TestLazyPlotImports(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('gymnasium'), 'Gymnasium not installed')
    def test_core_and_real_rl_do_not_import_plotting(self):
        run_python('''
            import importlib.abc
            import sys

            initial_modules = set(sys.modules)
            attempts = []
            class BlockPlotting(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'matplotlib', 'mpl_toolkits', 'plotly', 'pyvista'}:
                        attempts.append(fullname)
                        raise ModuleNotFoundError('blocked: ' + fullname, name=fullname)
            sys.meta_path.insert(0, BlockPlotting())
            probes = install_import_process_audit()

            import fibernet as fn
            from fibernet.core.structure_graph import StructureGraph
            import fibernet.rl.env as env
            from fibernet.easy import show
            import fibernet.viz as viz
            assert callable(show)
            assert fn.StructureGraph is StructureGraph
            assert 'render_graph' in dir(fn)
            assert 'render_network_2d' in dir(viz)
            assert env.HAS_GYM, 'real Gymnasium is required for this check'
            import gymnasium
            assert env.gym is gymnasium
            assert issubclass(env.FiberStructureEnv, gymnasium.Env)
            assert not attempts, attempts
            assert len(probes) <= 2, probes
            assert not any(name.split('.')[0] in {'matplotlib', 'mpl_toolkits', 'plotly', 'pyvista', 'torch', 'sklearn'} for name in set(sys.modules) - initial_modules)
            assert 'fibernet.viz.render' not in sys.modules
            assert 'fibernet.viz.renderer' not in sys.modules
            assert 'fibernet.viz.graph_plot' not in sys.modules
        ''')

    def test_core_imports_without_gym_or_plotting(self):
        run_python('''
            import importlib.abc
            import sys
            plotting_attempts = []
            class BlockOptional(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    root = fullname.split('.')[0]
                    if root in {'matplotlib', 'mpl_toolkits', 'plotly', 'pyvista'}:
                        plotting_attempts.append(fullname)
                        raise ModuleNotFoundError('blocked: ' + fullname, name=fullname)
                    if root in {'gymnasium', 'gym'}:
                        raise ModuleNotFoundError('blocked: ' + fullname, name=fullname)
            sys.meta_path.insert(0, BlockOptional())
            probes = install_import_process_audit()
            import fibernet as fn
            import fibernet.rl.env as env
            import fibernet.easy as easy
            assert not env.HAS_GYM
            assert callable(fn.StructureGraph)
            assert callable(easy.show)
            assert 'render_graph' in easy.__all__
            assert 'render_deformation' in easy.__all__
            assert 'render_graph' not in vars(easy)
            assert 'render_deformation' not in vars(easy)
            assert 'fibernet.viz.render' not in sys.modules
            assert not plotting_attempts, plotting_attempts
        ''')

    def test_missing_matplotlib_is_deferred_and_not_cached(self):
        run_python('''
            import importlib.abc
            import sys
            class BlockMatplotlib(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] == 'matplotlib':
                        raise ModuleNotFoundError('blocked matplotlib', name='matplotlib')
            blocker = BlockMatplotlib()
            sys.meta_path.insert(0, blocker)
            import fibernet as fn
            import fibernet.viz as viz
            for package, name in ((fn, 'render_graph'), (viz, 'render_network_2d')):
                for _ in range(2):
                    try:
                        getattr(package, name)
                    except ModuleNotFoundError as exc:
                        assert exc.name == 'matplotlib'
                    else:
                        raise AssertionError('missing dependency was hidden')
                    assert name not in vars(package)
            try:
                viz.plot_graph(None)
            except ImportError as exc:
                assert str(exc) == 'matplotlib is required: pip install matplotlib'
            else:
                raise AssertionError('graph plotting error semantics changed')
            for package in (fn, viz):
                try:
                    package.missing_public_name
                except AttributeError:
                    pass
                else:
                    raise AssertionError('unknown name must raise AttributeError')
            assert callable(fn.pattern_2d)
        ''')

    def test_public_import_styles_keep_original_objects(self):
        run_python('''
            import inspect
            import fibernet as fn
            import fibernet.viz as viz
            from fibernet import render_graph, THEMES
            from fibernet.viz import render_network_2d, render_network_3d, plot_graph
            from fibernet.viz.render import render_graph as direct_graph
            from fibernet.viz.renderer import render_network_2d as direct_network
            from fibernet.viz.renderer import render_network_3d as direct_3d
            from fibernet.viz.graph_plot import plot_graph as direct_plot
            assert render_graph is fn.render_graph is direct_graph
            assert render_network_2d is viz.render_network_2d is direct_network
            assert render_network_3d is direct_3d
            assert plot_graph is direct_plot
            assert THEMES is fn.THEMES
            assert inspect.signature(render_graph) == inspect.signature(direct_graph)
            top = {}; exec('from fibernet import *', top)
            namespace = {}; exec('from fibernet.viz import *', namespace)
            assert set(fn.__all__).issubset(top)
            assert set(viz.__all__).issubset(namespace)
            for name in fn.__all__:
                assert top[name] is getattr(fn, name)
            for name in viz.__all__:
                assert namespace[name] is getattr(viz, name)
            assert namespace['render_network_3d'] is direct_3d
            assert callable(viz.visualize_deformation)
            import fibernet.easy as easy
            # All public bindings in the original easy.py, including incidental
            # typing/module imports that wildcard imports historically exposed.
            original_easy_names = {
                'annotations', 'json', 'time', 'Path', 'Any', 'Dict', 'List',
                'Optional', 'Union', 'np', 'StructureGraph', 'pattern_2d',
                'pattern_3d', 'TaichiEngine', 'SimResult', 'render_graph',
                'render_deformation', 'show', 'simulate', 'batch_simulate',
                'train_model', 'predict_named', 'train_rl',
                'batch_simulate_from_json',
            }
            exports = {}; exec('from fibernet.easy import *', exports)
            assert set(easy.__all__) == original_easy_names
            assert set(exports) - {'__builtins__'} == original_easy_names
            for name in original_easy_names:
                assert exports[name] is getattr(easy, name)
            assert exports['render_graph'] is direct_graph
            from fibernet.viz.render import render_deformation
            assert exports['render_deformation'] is render_deformation
            print('easy_original_public_names_preserved=' + str(len(original_easy_names)))
        ''')

    def test_tiny_graph_render_matches_direct_renderer(self):
        run_python('''
            import numpy as np
            import fibernet as fn
            from fibernet.viz.render import render_graph
            import matplotlib.pyplot as plt
            graph = fn.StructureGraph(dimension=2)
            graph.add_node([0., 0., 0.])
            graph.add_node([1., 0.5, 0.])
            graph.add_edge(0, 1, radius=0.05)
            figures = [fn.render_graph(graph, theme='publication', figsize=(2, 2)),
                       render_graph(graph, theme='publication', figsize=(2, 2)),
                       fn.show(graph, theme='publication', figsize=(2, 2))]
            pixels = []
            for figure in figures:
                figure.canvas.draw()
                pixels.append(np.asarray(figure.canvas.buffer_rgba()).copy())
            assert pixels[0].shape == pixels[1].shape == pixels[2].shape
            assert np.array_equal(pixels[0], pixels[1])
            assert np.array_equal(pixels[0], pixels[2])
            import hashlib
            print('tiny_rgba_sha256=' + hashlib.sha256(pixels[0].tobytes()).hexdigest())
            plt.close('all')
        ''')


if __name__ == '__main__':
    unittest.main()
