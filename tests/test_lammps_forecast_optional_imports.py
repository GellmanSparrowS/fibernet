"""Exercise a cold import and a complete forecast with optional providers blocked."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


# This hook is installed only in the child interpreter, never at collection.
_PROVIDER_FREE_FORECAST = textwrap.dedent('''
    import importlib
    import importlib.abc
    import json
    import sys

    blocked = {
        'lammps', 'taichi', 'torch', 'torch_geometric', 'exudyn', 'mujoco',
        'pybullet', 'gym', 'gymnasium', 'pyvista', 'vtk', 'cv2', 'sklearn',
        'pandas', 'stable_baselines3', 'skopt', 'matplotlib', 'mpl_toolkits',
        'plotly', 'MDAnalysis', 'ovito',
    }
    attempts = []
    preloaded = [name for name in sys.modules if name.split('.')[0] in blocked]
    if preloaded:
        raise RuntimeError('Provider already loaded before guard: ' + repr(preloaded))

    class BlockProviders(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in blocked:
                attempts.append(fullname)
                raise ModuleNotFoundError('Provider blocked by forecast test', name=fullname)

    sys.meta_path.insert(0, BlockProviders())
    controls = []
    for name in sorted(blocked):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name or attempts[-1:] != [name]:
                raise RuntimeError('Negative control missed guard: ' + name)
            controls.append(name)
        else:
            raise RuntimeError('Provider unexpectedly imported: ' + name)
    attempts.clear()

    # Import the public API and the real graph implementation, not stand-ins.
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.integrations import ResourceLimits, forecast_cg_fiber_only, prepare_cg
    from _lammps_forecast_test_support import (
        build_branched_case, build_crosslinked_case, capped_case, inert_runtime,
    )
    from pathlib import Path
    from unittest.mock import patch
    from contextlib import ExitStack
    import builtins
    import io
    import os
    import subprocess
    import threading
    from fibernet.integrations import lammps_backend
    from fibernet.integrations import RuntimeSpec

    runtime = inert_runtime(Path.cwd())
    limits = ResourceLimits(5, 10., 512 * 1024**2, 16 * 1024**2, 16 * 1024**2)
    cases = []
    graph, request = build_branched_case(steps=200, timestep_s=5e-8)
    cases.append((graph, request))
    _, graph, request = build_crosslinked_case(steps=1000, timestep_s=5e-8)
    cases.append((graph, request))
    cases.append(capped_case())
    cases.append(capped_case(n=2, edge_count=0, angle_count=0, steps=0))
    prepared = []
    unchanged = []
    for graph, request in cases:
        before = graph.to_dict()
        prepared.append(prepare_cg(graph, request))
        unchanged.append(graph.to_dict() == before)

    targets = [
        (RuntimeSpec, 'checked'), (lammps_backend.LAMMPSTask, '__init__'),
        (subprocess, 'Popen'), (threading.Thread, 'start'),
        (builtins, 'open'), (io, 'open'), (os, 'open'),
        (Path, 'open'), (Path, 'resolve'), (Path, 'mkdir'),
        (Path, 'stat'), (Path, 'is_file'), (Path, 'is_dir'),
    ]
    with ExitStack() as stack:
        for owner, name in targets:
            stack.enter_context(patch.object(
                owner, name, side_effect=AssertionError('Side effect forbidden: ' + name)
            ))
        results = [forecast_cg_fiber_only(
            value, runtime, limits, export_budget_bytes=64 * 1024**2,
            record_state=False,
        ) for value in prepared]

    print(json.dumps({
        'negative_controls': controls,
        'expected_controls': sorted(blocked),
        'provider_attempts': attempts,
        'loaded_providers': [name for name in sys.modules if name.split('.')[0] in blocked],
        'source_unchanged': unchanged,
        'counts': [[result[key] for key in ('nodes', 'edges', 'angles', 'native_frames')]
                   for result in results],
        'readiness': [result['runtime_readiness'] for result in results],
        'execution_admission': [result['execution_admission'] for result in results],
        'record_state': [result['record_state'] for result in results],
        'capped_trajectory_bytes': results[2]['trajectory_bytes'],
        'zero_fits': results[3]['fits_declared_budgets'],
    }, sort_keys=True))
''')


class TestLammpsForecastOptionalImports(unittest.TestCase):
    def test_cold_import_and_full_forecast_need_no_optional_provider(self):
        environment = os.environ.copy()
        for name in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS',
            'BLIS_NUM_THREADS',
        ):
            environment[name] = '1'
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        root = Path(__file__).resolve().parents[1]
        paths = [str(root), str(root / 'tests')]
        if environment.get('PYTHONPATH'):
            paths.append(environment['PYTHONPATH'])
        environment['PYTHONPATH'] = os.pathsep.join(paths)
        with tempfile.TemporaryDirectory(prefix='pairfree-cold-import-') as directory:
            completed = subprocess.run(
                [sys.executable, '-c', _PROVIDER_FREE_FORECAST],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=30, check=False,
            )
            self.assertEqual(
                completed.returncode, 0,
                msg='Isolated forecast failed:\n' + completed.stderr + completed.stdout,
            )
            self.assertEqual(list(Path(directory).iterdir()), [])
        result = json.loads(completed.stdout)
        self.assertEqual(result['negative_controls'], result['expected_controls'])
        for provider in ('lammps', 'taichi', 'torch', 'exudyn'):
            self.assertIn(provider, result['negative_controls'])
        self.assertEqual(result['provider_attempts'], [])
        self.assertEqual(result['loaded_providers'], [])
        self.assertEqual(result['source_unchanged'], [True] * 4)
        self.assertEqual(result['counts'], [
            [7, 6, 5, 201], [48, 104, 64, 1001], [64, 128, 128, 1001], [2, 0, 0, 1],
        ])
        self.assertEqual(result['readiness'], ['not_checked'] * 4)
        self.assertEqual(result['execution_admission'], [False] * 4)
        self.assertEqual(result['record_state'], [False] * 4)
        self.assertEqual(result['capped_trajectory_bytes'], 16336352)
        self.assertTrue(result['zero_fits'])


if __name__ == '__main__':
    unittest.main()
