"""Tiny synthetic validation-split checks, without physics or sklearn training."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

surrogate = load('_physical_surrogate_explicit_validation', ROOT / 'fibernet/ml/physical_surrogate.py')
# Load directly without globally replacing any installed FiberNet modules.
physical = load('_physical_explicit_validation', ROOT / 'fibernet/ml/physical_learning.py')

class Estimator:
    def __init__(self, *_): self.fit_rows = None
    def fit(self, X, Y): self.fit_rows = np.array(X, copy=True); return self
    def predict(self, X): return np.zeros((len(X), 3))

physical.MODEL_PLUGINS['ridge'] = Estimator

class ValidationTests(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        fibernet = types.ModuleType('fibernet'); fibernet.__path__ = []
        ml = types.ModuleType('fibernet.ml'); ml.__path__ = []
        patcher = mock.patch.dict(sys.modules, {'fibernet': fibernet, 'fibernet.ml': ml,
            'fibernet.ml.physical_surrogate': surrogate})
        patcher.start(); self.addCleanup(patcher.stop)
        # 4 paired sources; the validation source contains high values to expose
        # leakage if normalization is fitted on all rows rather than train only.
        self.X = np.array([[0,1],[1,2],[2,3],[3,4],[1000,1001],[1100,1101],[6,7],[7,8]], float)
        self.Y = np.column_stack((self.X[:, 0], self.X[:, 0]+2, self.X[:, 0]+4))
        self.val = np.array([5, 4], dtype=np.int64)
        self.train = np.array([0, 1, 2, 3, 6, 7])

    def test_explicit_source_holdout_and_train_only_scaling_both_branches(self):
        trained = []
        for key in ('mlp', 'ridge'):
            with self.subTest(key=key):
                model = physical.Regressor(key=key, seed=13)
                indices = self.val.copy()
                model.train(self.X, self.Y, epochs=1, batch=3, validation_indices=indices)
                indices[0] = 0
                np.testing.assert_array_equal(model.val_indices, self.val)
                np.testing.assert_array_equal(model.train_indices, self.train)
                np.testing.assert_allclose(model.x_mean, self.X[self.train].mean(0))
                np.testing.assert_allclose(model.y_mean, self.Y[self.train].mean(0))
                np.testing.assert_allclose(model.x_std, self.X[self.train].std(0))
                np.testing.assert_allclose(model.y_std, self.Y[self.train].std(0))
                self.assertFalse({i // 2 for i in model.train_indices} & {i // 2 for i in model.val_indices})
                trained.append(model)
        np.testing.assert_array_equal(trained[0].train_indices, trained[1].train_indices)
        np.testing.assert_array_equal(trained[0].val_indices, trained[1].val_indices)
        np.testing.assert_allclose(trained[1].estimator.fit_rows,
                                   (self.X[self.train]-trained[1].x_mean)/trained[1].x_std)

    def test_legacy_default_indices_unchanged(self):
        expected = np.random.default_rng(13).permutation(len(self.X))
        for key in ('mlp', 'ridge'):
            model = physical.Regressor(key=key, seed=13)
            model.train(self.X, self.Y, epochs=1, batch=3, val_frac=.25)
            np.testing.assert_array_equal(model.val_indices, expected[:2])
            np.testing.assert_array_equal(model.train_indices, expected[2:])

    def test_invalid_indices_rejected_before_publication_or_rng_change(self):
        bad = [[], [0], [0,0], [-1,1], [0,8], [False,1], [0.,1.],
               [0,1,2,3,4,5,6], [[0,1]], np.array([[0,1]]),
               np.ma.array([0,1], mask=[False,False]), '01',
               [np.uint64(2**64-1), 1]]
        for value in bad:
            for key in ('mlp','ridge'):
                with self.subTest(value=str(value), key=key):
                    model = physical.Regressor(key=key, seed=13)
                    marker = object(); model.estimator = marker
                    with self.assertRaises(ValueError):
                        model.train(self.X,self.Y,epochs=1,validation_indices=value)
                    self.assertIs(model.estimator, marker)
                    self.assertFalse(hasattr(model,'train_indices'))
        with self.assertRaises(ValueError):
            physical.Regressor(key='mlp').train(self.X[:4], self.Y[:4], epochs=1,
                                               validation_indices=[0,1])
        direct = surrogate.MLP(seed=13)
        before = copy.deepcopy(direct.rng.bit_generator.state)
        with self.assertRaises(ValueError):
            direct.train(self.X,self.Y,epochs=1,validation_indices=[0,0])
        self.assertEqual(direct.rng.bit_generator.state, before)
        self.assertIsNone(direct.W1)

if __name__ == '__main__': unittest.main()
