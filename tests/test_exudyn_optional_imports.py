"""Check provider-free preparation in an isolated, bounded Python process."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


# The import hook exists only in the child process, never during collection or
# in the interpreter running the rest of the test suite.
_PROVIDER_FREE_PREPARATION = textwrap.dedent('''
    import importlib
    import importlib.abc
    import json
    import sys

    blocked = {
        'exudyn', 'taichi', 'lammps', 'mujoco', 'pybullet', 'torch',
        'torch_geometric', 'gym', 'gymnasium', 'pyvista', 'vtk', 'cv2',
        'sklearn', 'pandas', 'stable_baselines3', 'skopt', 'matplotlib',
        'mpl_toolkits', 'plotly', 'MDAnalysis', 'ovito',
    }
    attempts = []
    preloaded = [name for name in sys.modules
                 if name.split('.')[0] in blocked]
    if preloaded:
        raise RuntimeError('Provider preloaded before guard: ' + repr(preloaded))

    class BlockProviders(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in blocked:
                attempts.append(fullname)
                raise ModuleNotFoundError(
                    'Provider blocked by preparation test', name=fullname
                )

    sys.meta_path.insert(0, BlockProviders())

    # Negative controls demonstrate that the hook rejects real provider names,
    # including Torch, before importing any FiberNet or numerical source.
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

    from fibernet import Material, StructureGraph
    from fibernet.sim._exudyn_contract import (
        ChordContactSpec, ExudynContactRequest, SphereSpec, StepPhase,
        prepare_exudyn_contact, source_capabilities, validate_prepared,
    )
    from fibernet.sim._exudyn_mapping import compile_straight_chain

    graph = StructureGraph(dimension=2)
    left = graph.add_node([-0.001, 0.0, 0.0], merge=False)
    right = graph.add_node([0.001, 0.0, 0.0], merge=False)
    graph.add_edge(
        left, right, radius=50e-6, segments=1,
        material=Material(
            name='synthetic preparation test material',
            youngs_modulus=1e6, poissons_ratio=0.3, density=1200.0,
        ),
    )
    request = ExudynContactRequest(
        profile='straight-shared-chain/1', length_unit='m',
        clamped_node_ids=(left, right), shear_correction=(1.0, 1.0),
        sphere=SphereSpec(350e-6, 1e-7, (0.0, 0.00062, 0.0), (0.0, -0.1, 0.0)),
        contact=ChordContactSpec(100.0, 2e-6, 2e-4),
        phases=(StepPhase(0.0, 1e-6, 1),), geometry_energy_budget_J=1e-11,
        gravity_m_s2=(0.0, 0.0, 0.0), external_forces_N=(),
        initial_fiber_state='reference_rest',
    )
    before = graph.to_dict()
    prepared = prepare_exudyn_contact(graph, request)
    mapping = prepared.mapping()
    print(json.dumps({
        'negative_controls': controls,
        'expected_controls': sorted(blocked),
        'provider_attempts': attempts,
        'loaded_providers': [name for name in sys.modules
                             if name.split('.')[0] in blocked],
        'validated': validate_prepared(prepared),
        'source_unchanged': before == graph.to_dict(),
        'compiler_matches': compile_straight_chain(graph, request) == mapping,
        'counts': mapping['counts'],
        'capabilities': source_capabilities(),
        'manifest': prepared.manifest(),
    }, sort_keys=True))
''')


class TestExudynOptionalImports(unittest.TestCase):
    def test_full_preparation_never_imports_optional_providers(self):
        environment = os.environ.copy()
        for variable in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'NUMEXPR_MAX_THREADS', 'BLIS_NUM_THREADS',
        ):
            environment[variable] = '1'
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        root = str(Path(__file__).resolve().parents[1])
        paths = [root]
        if environment.get('PYTHONPATH'):
            paths.append(environment['PYTHONPATH'])
        environment['PYTHONPATH'] = os.pathsep.join(paths)

        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [sys.executable, '-c', _PROVIDER_FREE_PREPARATION],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0,
                msg='Isolated preparation failed:\n' + result.stderr,
            )
            self.assertEqual(list(Path(working_directory).iterdir()), [])

        report = json.loads(result.stdout)
        self.assertEqual(report['negative_controls'], report['expected_controls'])
        self.assertIn('exudyn', report['negative_controls'])
        self.assertIn('torch', report['negative_controls'])
        self.assertEqual(report['provider_attempts'], [])
        self.assertEqual(report['loaded_providers'], [])
        self.assertTrue(report['validated'])
        self.assertTrue(report['source_unchanged'])
        self.assertTrue(report['compiler_matches'])
        self.assertEqual(report['counts'], {
            'physical_nodes': 2, 'physical_edges': 1, 'beam_nodes': 2,
            'sphere_nodes': 1, 'ODE2': 15, 'AE': 12, 'welds': 0,
        })
        self.assertTrue(report['capabilities']['preparation_available'])
        self.assertFalse(report['capabilities']['execution_available'])
        self.assertEqual(report['capabilities']['runtime_readiness'], 'not_probed')
        self.assertFalse(report['manifest']['execution_available'])
        self.assertFalse(report['manifest']['restart_supported'])


if __name__ == '__main__':
    unittest.main()
