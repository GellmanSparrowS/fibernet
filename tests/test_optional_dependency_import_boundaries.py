"""Public utilities stay independent of unrelated optional dependencies."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


_PRELUDE = r'''
import importlib
import importlib.abc
import inspect
from pathlib import Path
import sys
import typing

PROVIDERS = {
    'taichi', 'gymnasium', 'gym', 'pandas', 'torch', 'torch_geometric',
    'sklearn', 'pyvista', 'plotly', 'stable_baselines3', 'skopt',
    'lammps', 'exudyn', 'ovito', 'MDAnalysis', 'manifold3d', 'h5py',
}
attempts = []
sentinels = {}
missing = False

class ProviderError(RuntimeError):
    pass

class BlockProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in PROVIDERS:
            attempts.append(fullname)
            if missing:
                raise ModuleNotFoundError(
                    'injected missing optional dependency: ' + fullname,
                    name=fullname,
                )
            error = sentinels.setdefault(
                fullname, ProviderError('injected broken optional dependency: ' + fullname)
            )
            raise error

assert not any(name.split('.')[0] in PROVIDERS for name in sys.modules)
sys.meta_path.insert(0, BlockProviders())

# Check the import hook without creating fake provider modules.
for name in sorted(PROVIDERS):
    try:
        importlib.import_module(name)
    except ProviderError as error:
        assert error is sentinels[name]
    else:
        raise AssertionError('optional dependency was not blocked: ' + name)
attempts.clear()
sentinels.clear()

def assert_no_provider_loaded():
    assert not any(name.split('.')[0] in PROVIDERS for name in sys.modules)

def tiny_graph(fn):
    graph = fn.StructureGraph(dimension=2)
    left = graph.add_node([0., 0., 0.], merge=False)
    right = graph.add_node([1., 0., 0.], merge=False)
    graph.add_edge(left, right, radius=0.05, segments=1)
    return graph
'''


def _run_python(source, tmp_path):
    """Use the same selected FiberNet package in a fresh interpreter."""
    spec = importlib.util.find_spec('fibernet')
    assert spec is not None and spec.origin is not None
    package_parent = str(Path(spec.origin).resolve().parent.parent)
    env = dict(os.environ)
    paths = [package_parent]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run(
        [sys.executable, '-B', '-c', _PRELUDE + '\n' + textwrap.dedent(source)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_utilities_and_graph_io_do_not_probe_optional_dependencies(tmp_path):
    _run_python('''
        import fibernet as fn
        import numpy as np

        # Normal public and direct imports return the original callables.
        for name in ('show', 'train_model', 'predict_named', 'plot_reward_curve'):
            value = getattr(fn, name)
            owner = importlib.import_module(
                'fibernet.rl.utils' if name == 'plot_reward_curve' else 'fibernet.easy'
            )
            assert value is getattr(owner, name)
            assert callable(value)
            assert attempts == []

        easy = importlib.import_module('fibernet.easy')
        expected_easy = [
            'annotations', 'json', 'time', 'Path', 'Any', 'Dict', 'List',
            'Optional', 'Union', 'np', 'StructureGraph', 'pattern_2d',
            'pattern_3d', 'TaichiEngine', 'SimResult', 'show', 'simulate',
            'batch_simulate', 'train_model', 'predict_named', 'train_rl',
            'batch_simulate_from_json', 'render_graph', 'render_deformation', 'train_policy',
        ]
        assert easy.__all__ == expected_easy
        assert {'TaichiEngine', 'SimResult', 'render_graph', 'render_deformation'} <= set(dir(easy))

        signatures = {
            'show': "(graph, *, theme='dark', save_path=None, title='', **kwargs)",
            'simulate': "(graph, mode='stretch', strain=None, backend='spring', save_path=None, *, stretch_ratio=None, **kwargs)",
            'batch_simulate': "(configs, output='results.csv', mode='stretch', strain=2.0, *, report_path=None)",
            'train_model': "(csv_path, target='max_force', features=None, model_type='rf', test_size=0.2, save_path=None, *, split=None, groups=None, metadata_keys=None)",
            'predict_named': "(artifact, data, *, extra_columns='error')",
            'train_rl': '(env_config=None, n_episodes=100, save_path=None)',
            'batch_simulate_from_json': "(json_dir, output_dir='simulation_results', mode='stretch', target_stretch=1.5, stiffness=100000.0, damping=0.3, num_steps=1000, save_interval=200, backend='spring', *, geometry_policy='reject_internal_points')",
        }
        for name, expected in signatures.items():
            signature = inspect.signature(getattr(easy, name))
            parameters = [p.replace(annotation=inspect.Signature.empty)
                          for p in signature.parameters.values()]
            actual = signature.replace(
                parameters=parameters, return_annotation=inspect.Signature.empty
            )
            assert str(actual) == expected, (name, actual)

        # An explicit facade override is still honored; reload drops that binding.
        marker = object()
        easy.TaichiEngine = marker
        assert easy._taichi_engine_class() is marker
        importlib.reload(easy)
        assert 'TaichiEngine' not in vars(easy)
        assert easy.__all__ == expected_easy

        utils = importlib.import_module('fibernet.rl.utils')
        rl = importlib.import_module('fibernet.rl')
        assert callable(utils.plot_reward_curve)
        assert rl.__all__ == [
            'train_policy',
            'plot_reward_curve', 'plot_convergence', 'plot_action_distribution',
            'evaluate_agent', 'save_agent', 'load_agent', 'run_bayesian_optimization',
            'ParametricStructureEnv', 'create_rl_environment',
        ]
        assert {'ParametricStructureEnv', 'create_rl_environment'} <= set(dir(rl))

        io = importlib.import_module('fibernet.io')
        assert io.__all__ == [
            'to_lammps', 'from_lammps', 'to_vtk', 'to_xyz', 'to_pdb', 'from_pdb',
            'to_gmsh', 'to_dataframe', 'from_dataframe', 'network_summary',
            'parametric_to_dataframe', 'export_stl', 'export_obj', 'export_ply',
            'export_to_abaqus', 'export_to_ansys', 'export_to_gmsh_fea', 'export_to_lammps_fea',
        ]
        assert {'to_dataframe', 'from_dataframe', 'network_summary',
                'parametric_to_dataframe'} <= set(dir(io))

        # These are real tiny I/O operations, not mocked implementations.
        fiber = fn.Fiber([[0., 0., 0.], [1., 0., 0.]], radius=.05)
        network = fn.FiberNetwork(fibers=[fiber])
        destination = Path('tiny.xyz')
        io.to_xyz(network, str(destination), bead_spacing=2., comment='optional imports')
        from fibernet.io.xyz import from_xyz
        assert np.array_equal(from_xyz(str(destination)), fiber.centerline)

        graph = tiny_graph(fn)
        restored = fn.StructureGraph.from_networkx_lossless(graph.to_networkx_lossless())
        assert restored.to_dict() == graph.to_dict()
        assert attempts == []
        assert_no_provider_loaded()
    ''', tmp_path)


@pytest.mark.parametrize(('module_name', 'attribute', 'provider'), [
    ('fibernet.easy', 'TaichiEngine', 'taichi'),
    ('fibernet.easy', 'SimResult', 'taichi'),
    ('fibernet.rl', 'ParametricStructureEnv', 'gymnasium'),
    ('fibernet.io', 'to_dataframe', 'pandas'),
])
def test_explicit_alias_rethrows_owner_failure_without_caching(
    tmp_path, module_name, attribute, provider
):
    _run_python('''
        module = importlib.import_module(MODULE_NAME)
        assert attempts == []
        for _ in range(2):
            try:
                getattr(module, ATTRIBUTE)
            except ProviderError as error:
                assert error is sentinels[PROVIDER]
            else:
                raise AssertionError('owning module failure was suppressed')
            assert ATTRIBUTE not in vars(module)
        assert attempts == [PROVIDER, PROVIDER]
        assert_no_provider_loaded()
    '''.replace('MODULE_NAME', repr(module_name))
       .replace('ATTRIBUTE', repr(attribute))
       .replace('PROVIDER', repr(provider)), tmp_path)


def test_spring_selection_preserves_owner_failure(tmp_path):
    _run_python('''
        import fibernet as fn
        simulate = fn.simulate
        assert attempts == []
        try:
            simulate(tiny_graph(fn), backend='spring', mode='stretch', num_steps=1)
        except ProviderError as error:
            assert error is sentinels['taichi']
        else:
            raise AssertionError('spring selection hid its owning module failure')
        assert attempts == ['taichi']
        assert_no_provider_loaded()
    ''', tmp_path)


def test_type_hint_resolution_preserves_owner_failure(tmp_path):
    _run_python('''
        easy = importlib.import_module('fibernet.easy')
        assert attempts == []
        try:
            typing.get_type_hints(easy.simulate)
        except ProviderError as error:
            assert error is sentinels['taichi']
        else:
            raise AssertionError('type-hint resolution hid its owning module failure')
        assert attempts == ['taichi']
        assert_no_provider_loaded()
    ''', tmp_path)


def test_missing_taichi_retains_real_aliases_type_hints_and_install_message(tmp_path):
    _run_python('''
        missing = True
        easy = importlib.import_module('fibernet.easy')
        assert attempts == []
        hints = typing.get_type_hints(easy.simulate)
        assert attempts == ['taichi']
        accelerated = importlib.import_module('fibernet.sim.accelerated')
        engine_type = easy.TaichiEngine
        assert engine_type is accelerated.TaichiEngine
        assert easy.SimResult is accelerated.SimResult
        from fibernet.sim._viscoelastic_fem import ViscoelasticResult
        assert hints['return'] == typing.Union[accelerated.SimResult, ViscoelasticResult]
        try:
            engine_type(arch='cpu', num_threads=1)
        except ImportError as error:
            assert str(error) == 'Taichi required: pip install taichi==1.7.4'
        else:
            raise AssertionError('missing Taichi error was lost')
        assert_no_provider_loaded()
    ''', tmp_path)


def test_missing_pandas_retains_real_alias_and_install_message(tmp_path):
    _run_python('''
        missing = True
        io = importlib.import_module('fibernet.io')
        assert attempts == []
        call = io.to_dataframe
        owner = importlib.import_module('fibernet.io.pandas_io')
        assert call is owner.to_dataframe
        try:
            call(None)
        except ImportError as error:
            assert str(error) == 'pandas is required. Install with: pip install pandas'
        else:
            raise AssertionError('missing pandas error was lost')
        assert attempts == ['pandas']
        assert_no_provider_loaded()
    ''', tmp_path)
