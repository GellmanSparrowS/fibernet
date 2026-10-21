"""Cold normal-package imports must not require a learning or simulation backend."""
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


OPTIONAL_PREFIXES = (
    'torch', 'torch_geometric', 'torch_scatter', 'torch_sparse', 'torch_cluster',
    'torch_spline_conv', 'sklearn', 'tensorflow', 'jax', 'taichi', 'exudyn',
    'lammps', 'ngsolve', 'dolfin', 'dolfinx', 'pyvista', 'trimesh', 'manifold3d',
    'skimage', 'pandas', 'tqdm', 'xgboost', 'lightgbm', 'catboost', 'optuna', 'gymnasium',
    'stable_baselines3', 'skopt', 'ray', 'h5py', 'fibernet.sim', 'fibernet.rl',
    'fibernet.integrations',
)
ALLOWED_ML = {'fibernet.ml', 'fibernet.ml.utils', 'fibernet.ml.dataset',
              'fibernet.ml.graph_data', 'fibernet.ml.graph_training_contract'}
ORDERS = ('contract_first', 'root_first', 'ml_first')


def is_optional(name):
    return (any(name == prefix or name.startswith(prefix + '.') for prefix in OPTIONAL_PREFIXES)
            or (name.startswith('fibernet.ml.') and name not in ALLOWED_ML))


class RejectOptionalImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if is_optional(fullname):
            self.attempts.append(fullname)
            raise ImportError('optional backend not required by graph metadata: ' + fullname)
        return None


def probe(order, root):
    """Check one cold import order using the real package at the source root."""
    if order not in ORDERS:
        raise ValueError('unknown import order')
    root = Path(root).resolve()
    if any(name == 'fibernet' or name.startswith('fibernet.') or is_optional(name)
           for name in sys.modules):
        raise AssertionError('probe requires a cold interpreter')
    sys.path.insert(0, str(root))
    guard = RejectOptionalImports()
    sys.meta_path.insert(0, guard)
    try:
        if order == 'root_first':
            importlib.import_module('fibernet')
            importlib.import_module('fibernet.ml.graph_data')
        elif order == 'ml_first':
            importlib.import_module('fibernet.ml')
        contract = importlib.import_module('fibernet.ml.graph_training_contract')
        data = importlib.import_module('fibernet.ml.graph_data')
        core = importlib.import_module('fibernet.core.structure_graph')
        package = importlib.import_module('fibernet')
        ml = importlib.import_module('fibernet.ml')
        for module, relative in (
            (package, 'fibernet/__init__.py'), (ml, 'fibernet/ml/__init__.py'),
            (data, 'fibernet/ml/graph_data.py'),
            (contract, 'fibernet/ml/graph_training_contract.py'),
            (core, 'fibernet/core/structure_graph.py'),
        ):
            if Path(module.__file__).resolve() != root / relative:
                raise AssertionError('wrong package source: ' + module.__name__)
        assert contract.GraphBatch is data.GraphBatch is ml.GraphBatch
        assert contract.GraphBatchProvenance is data.GraphBatchProvenance is ml.GraphBatchProvenance
        assert package.StructureGraph is core.StructureGraph
        example = importlib.import_module('examples.graph_training_contract')
        if Path(example.__file__).resolve() != root / 'examples/graph_training_contract.py':
            raise AssertionError('wrong example source')
        inputs, split, degree, admission = example.build_records()
        assert degree.as_dict()['summary']['histogram'] == [1, 0, 2]
        assert degree.as_dict()['summary']['node_population'] == 3
        assert degree.as_dict()['summary']['incoming_messages'] == 4
        assert split.as_dict()['roles']['train'] == ['train-a']
        assert len(inputs.as_dict()['samples']) == 3
        assert admission.as_dict()['statistics_hash'] == degree.digest
        loaded = sorted(name for name in sys.modules if is_optional(name))
        if guard.attempts or loaded:
            raise AssertionError({'attempted_optional_imports': guard.attempts,
                                  'loaded_optional_modules': loaded})
        return {'order': order, 'ordinary_package_identity': True,
                'optional_import_attempts': [], 'histogram': [1, 0, 2]}
    finally:
        sys.meta_path.remove(guard)
        sys.path.remove(str(root))


class TestGraphTrainingImportBoundary(unittest.TestCase):
    def check_order(self, order):
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[name] = '1'
        script = Path(__file__).resolve()
        result = subprocess.run(
            [sys.executable, '-I', '-B', str(script), '--probe', order, str(script.parents[1])],
            check=True, capture_output=True, text=True, timeout=60, env=env,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report['order'], order)
        self.assertEqual(report['optional_import_attempts'], [])
        self.assertTrue(report['ordinary_package_identity'])

    def test_contract_first(self):
        self.check_order('contract_first')

    def test_root_first(self):
        self.check_order('root_first')

    def test_ml_first(self):
        self.check_order('ml_first')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--probe':
        print(json.dumps(probe(sys.argv[2], sys.argv[3]), sort_keys=True))
    else:
        unittest.main()
