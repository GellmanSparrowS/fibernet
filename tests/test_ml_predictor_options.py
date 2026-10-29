"""Small estimator-option/seed regressions; no native simulation or large training."""
import unittest
from unittest.mock import patch
import numpy as np
from fibernet.ml import utils


class DummyEstimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class PredictorOptionsTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(72, dtype=float).reshape(24, 3) / 72
        self.y = self.X[:, 0] + 2 * self.X[:, 1]

    def test_ridge_options_forwarded(self):
        model = utils._build_model('ridge', alpha=.25, fit_intercept=False,
                                   solver='lsqr', tol=1e-5)
        self.assertEqual(list(model.named_steps), ['scaler', 'ridge'])
        params = model.named_steps['ridge'].get_params()
        self.assertEqual((params['alpha'], params['fit_intercept'], params['solver'], params['tol']),
                         (.25, False, 'lsqr', 1e-5))

    def test_svr_options_forwarded(self):
        model = utils._build_model('svm', epsilon=.7, gamma='auto', degree=2)
        params = model.named_steps['svm'].get_params()
        self.assertEqual((params['epsilon'], params['gamma'], params['degree']), (.7, 'auto', 2))

    def test_mlp_options_and_legacy_alias(self):
        model = utils._build_model('mlp', hidden_layer_sizes=(3,), learning_rate_init=.2,
                                   activation='tanh', random_state=9)
        params = model.named_steps['mlp'].get_params()
        self.assertEqual((params['hidden_layer_sizes'], params['learning_rate_init'],
                          params['activation'], params['random_state']), ((3,), .2, 'tanh', 9))
        legacy = utils._build_model('mlp', hidden=(4, 2))
        self.assertEqual(legacy.named_steps['mlp'].hidden_layer_sizes, (4, 2))
        with self.assertRaisesRegex(ValueError, 'only one.*hidden.*hidden_layer_sizes'):
            utils._build_model('mlp', hidden=(3,), hidden_layer_sizes=(3,))

    def test_unknown_options_are_never_silently_dropped(self):
        for name in ('rf', 'gb', 'ridge', 'svm', 'mlp'):
            with self.subTest(model=name), self.assertRaises(TypeError):
                utils._build_model(name, misspelled_parameter=True)

    def test_existing_defaults_and_stage_names(self):
        self.assertEqual(utils._build_model('rf').get_params()['random_state'], 42)
        self.assertEqual(utils._build_model('gb').get_params()['n_estimators'], 100)
        self.assertEqual(utils._build_model('ridge').named_steps['ridge'].alpha, 1.)
        self.assertEqual(utils._build_model('svm').named_steps['svm'].kernel, 'rbf')
        mlp = utils._build_model('mlp').named_steps['mlp']
        self.assertEqual((mlp.hidden_layer_sizes, mlp.max_iter, mlp.random_state), ((64, 32), 500, 42))

    def test_train_seed_reaches_split_and_estimator_without_duplicate_kwargs(self):
        from sklearn.model_selection import train_test_split
        for name in ('rf', 'gb', 'ridge', 'mlp', 'svm'):
            with self.subTest(model=name):
                with patch.object(utils, '_build_model', return_value=DummyEstimator()) as factory, \
                     patch('sklearn.model_selection.train_test_split', wraps=train_test_split) as split:
                    utils.train_predictor(self.X, self.y, model_type=name, random_state=7)
                self.assertEqual(split.call_args.kwargs['random_state'], 7)
                self.assertEqual(factory.call_args.kwargs, {} if name == 'svm' else {'random_state': 7})

    def test_cv_seed_reaches_every_fold_and_kfold(self):
        from sklearn.model_selection import KFold
        with patch.object(utils, '_build_model', return_value=DummyEstimator()) as factory, \
             patch('sklearn.model_selection.KFold', wraps=KFold) as fold:
            result = utils.cross_validate(self.X, self.y, model_type='mlp', cv=3,
                                          random_state=11, hidden=(3,))
        self.assertEqual(fold.call_args.kwargs['random_state'], 11)
        self.assertEqual(factory.call_count, 3)
        self.assertTrue(all(call.kwargs == {'random_state': 11, 'hidden': (3,)}
                            for call in factory.call_args_list))
        self.assertEqual(len(result['fold_r2s']), 3)

    def test_compare_models_seed_forwarding_and_svr_exclusion(self):
        with patch.object(utils, '_build_model', return_value=DummyEstimator()) as factory:
            result = utils.compare_models(self.X, self.y,
                model_types=['rf', 'ridge', 'gb', 'svm', 'mlp'], random_state=13)
        self.assertTrue(all('error' not in row for row in result.values()))
        for call in factory.call_args_list:
            self.assertEqual(call.kwargs, {} if call.args[0] == 'svm' else {'random_state': 13})

    def test_bounded_real_random_forest_fit_is_repeatable(self):
        kwargs = dict(model_type='rf', random_state=7, n_estimators=3, max_depth=2,
                      n_jobs=1, test_size=.25)
        first, metrics = utils.train_predictor(self.X, self.y, **kwargs)
        second, again = utils.train_predictor(self.X, self.y, **kwargs)
        self.assertEqual(first.random_state, 7)
        np.testing.assert_array_equal(first.predict(self.X), second.predict(self.X))
        self.assertEqual(metrics, again)
        self.assertEqual((metrics['n_train'], metrics['n_test']), (18, 6))

    def test_bounded_real_ridge_fit_honors_options(self):
        model, metrics = utils.train_predictor(self.X, self.y, model_type='ridge',
            random_state=17, fit_intercept=False, alpha=.5, solver='lsqr')
        ridge = model.named_steps['ridge']
        self.assertEqual((ridge.fit_intercept, ridge.alpha, ridge.solver, ridge.random_state),
                         (False, .5, 'lsqr', 17))
        self.assertTrue(np.isfinite(metrics['rmse']))


if __name__ == '__main__':
    unittest.main()
