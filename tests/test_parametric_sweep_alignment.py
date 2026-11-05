import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from fibernet.utils.parametric import parametric_sweep

class SweepAlignment(unittest.TestCase):
    def test_failed_generation_or_analysis_has_no_orphan_parameter(self):
        def generate(x):
            if x == 0: raise RuntimeError('generation failed')
            return x
        class BrokenMetrics:
            def items(self):
                yield ('a', 999)
                raise RuntimeError('analysis result failed while iterating')
        def analyze(x):
            if x == 1: return BrokenMetrics()
            if x == 3: raise RuntimeError('analysis failed')
            return {'a':x*10, 'b':x+1}
        params, metrics=parametric_sweep({'x':[0,1,2,3,4]},generate,analyze,show_progress=False)
        np.testing.assert_array_equal(params['x'],[2,4])
        np.testing.assert_array_equal(metrics['a'],[20,40])
        np.testing.assert_array_equal(metrics['b'],[3,5])

    def test_successful_samples_must_have_identical_keys(self):
        for records in (({'a':1},{'a':2,'b':3}), ({'a':1,'b':2},{'a':3})):
            with self.subTest(records=records):
                with self.assertRaisesRegex(ValueError,'same metric keys'):
                    parametric_sweep({'x':[0,1]},lambda x:x,lambda x:records[x],show_progress=False)

    def test_default_progress_without_tqdm_skips_failed_case(self):
        import fibernet.utils.parametric as module
        def generate(x):
            if x == 1: raise RuntimeError('case failed')
            return x
        with patch.object(module, 'HAS_TQDM', False), patch.object(module, 'tqdm', lambda values: values):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                params, metrics=module.parametric_sweep({'x':[1,2]},generate,lambda x:{'a':x})
        np.testing.assert_array_equal(params['x'],[2])
        np.testing.assert_array_equal(metrics['a'],[2])
        self.assertIn('case failed',output.getvalue())

    def test_none_and_nonmapping_are_skipped_but_none_metric_is_retained(self):
        def analyze(x):
            return None if x == 0 else [1, 2] if x == 1 else {'a': None}
        params,metrics=parametric_sweep({'x':[0,1,2]},lambda x:x,analyze,show_progress=False)
        np.testing.assert_array_equal(params['x'],[2])
        self.assertEqual(metrics['a'].tolist(),[None])

    def test_empty_metric_keys_empty_parameters_and_all_failures(self):
        params,metrics=parametric_sweep({'x':[1,2]},lambda x:x,lambda x:{},show_progress=False)
        np.testing.assert_array_equal(params['x'],[1,2]);self.assertEqual(metrics,{})
        with self.assertRaisesRegex(ValueError,'same metric keys'):
            parametric_sweep({'x':[0,1]},lambda x:x,lambda x:{} if x==0 else {'a':1},show_progress=False)
        params,metrics=parametric_sweep({},lambda:3,lambda x:{'a':x},show_progress=False)
        self.assertEqual(params,{});np.testing.assert_array_equal(metrics['a'],[3])
        def fail(**kwargs): raise RuntimeError('failed')
        params,metrics=parametric_sweep({'x':[1,2]},fail,lambda x:{'a':x},show_progress=False)
        self.assertEqual(params['x'].size,0);self.assertEqual(metrics,{})

    def test_order_variation_preserves_metric_alignment(self):
        def analyze(x):
            return {'a':x, 'b':-x} if x == 1 else {'b':-x, 'a':x}
        params,metrics=parametric_sweep({'x':[1,2]},lambda x:x,analyze,show_progress=False)
        np.testing.assert_array_equal(params['x'],[1,2])
        np.testing.assert_array_equal(metrics['a'],[1,2])
        np.testing.assert_array_equal(metrics['b'],[-1,-2])

    def test_first_failure_and_empty_results_preserve_existing_behavior(self):
        def generate(x):
            if x==0: raise ValueError('failure')
            return x
        params, metrics=parametric_sweep({'x':[0,1,2]},generate,lambda x:{'a':x},show_progress=False)
        np.testing.assert_array_equal(params['x'],[1,2]);np.testing.assert_array_equal(metrics['a'],[1,2])
        params,metrics=parametric_sweep({'x':[]},generate,lambda x:{'a':x},show_progress=False)
        self.assertEqual(params['x'].size,0);self.assertEqual(metrics,{})

if __name__ == '__main__': unittest.main()
