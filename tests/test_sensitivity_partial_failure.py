import unittest
import numpy as np
from fibernet.utils.parametric import sensitivity_analysis

class SensitivityPartialFailure(unittest.TestCase):
    def test_later_success_retains_metrics_and_alignment(self):
        def generate(size):
            if size == 1:
                raise RuntimeError('case failed')
            return size
        result = sensitivity_analysis('size', [1, 2, 3], generate,
                                      lambda n: {'score': n * 2}, num_samples=2,
                                      show_progress=False)
        np.testing.assert_array_equal(result['param_values'], [1, 2, 3])
        np.testing.assert_allclose(result['metrics_mean']['score'], [np.nan, 4, 6], equal_nan=True)
        np.testing.assert_allclose(result['metrics_std']['score'], [np.nan, 0, 0], equal_nan=True)

    def test_successful_and_middle_failure_cases_preserve_statistics(self):
        calls = {}
        def generate(size):
            if size == 2:
                raise RuntimeError('case failed')
            calls[size] = calls.get(size, 0) + 1
            return size + calls[size]
        result = sensitivity_analysis('size', [1, 2, 3], generate,
                                      lambda n: {'score': n}, num_samples=2,
                                      show_progress=False)
        np.testing.assert_allclose(result['metrics_mean']['score'], [2.5, np.nan, 4.5], equal_nan=True)
        np.testing.assert_allclose(result['metrics_std']['score'], [0.5, np.nan, 0.5], equal_nan=True)

    def test_inconsistent_metric_keys_raise_instead_of_losing_data(self):
        for metrics in (({'a': 1}, {'a': 2, 'b': 3}),
                        ({'a': 1, 'b': 2}, {'a': 3})):
            with self.subTest(metrics=metrics):
                with self.assertRaisesRegex(ValueError, 'same metric keys'):
                    sensitivity_analysis('size', [0, 1], lambda size: size,
                                         lambda n: metrics[n], show_progress=False)

    def test_empty_and_all_failed_cases_are_empty(self):
        def fail(**kwargs): raise RuntimeError('failed')
        for values in ([], [1, 2]):
            result = sensitivity_analysis('size', values, fail, lambda n: {'score': n}, show_progress=False)
            self.assertEqual(result['metrics_mean'], {})
            self.assertEqual(result['metrics_std'], {})
            self.assertEqual(result['param_values'].tolist(), values)

if __name__ == '__main__': unittest.main()
