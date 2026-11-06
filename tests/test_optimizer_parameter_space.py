"""Parameter-space dispatch regressions, collectable by pytest or unittest.

Compile the actual optimizer class from source without optional providers.
These tests exercise dispatch, not Optuna sampling or optimization results.
"""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'fibernet/ml/optimization.py'
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiObjectiveOptimizer']
assert len(classes) == 1
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), classes[0]], type_ignores=[])
namespace = {}
exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
Optimizer = namespace['MultiObjectiveOptimizer']

class Trial:
    def __init__(self):
        self.calls = []
    def suggest_categorical(self, name, choices):
        self.calls.append(('categorical', name, choices))
        return choices[0]
    def suggest_int(self, name, low, high):
        self.calls.append(('int', name, low, high))
        return (low + high) // 2
    def suggest_float(self, name, low, high):
        self.calls.append(('float', name, low, high))
        return (low + high) / 2

class TestOptimizerParameterSpace(unittest.TestCase):
    def call(self, spec):
        trial = Trial()
        optimizer = Optimizer(lambda p: (0., 0.), {'design': spec})
        result = optimizer._suggest_params(trial)
        return result, trial.calls
    def check_category(self, spec):
        result, calls = self.call(spec)
        self.assertEqual(calls, [('categorical', 'design', spec)])
        self.assertEqual(result, {'design': spec[0]})
    def test_two_string_categories(self):
        self.check_category(['square', 'hex'])
    def test_two_integer_categories_do_not_invent_middle_choice(self):
        self.check_category([2, 4])
    def test_two_float_categories_do_not_become_continuous(self):
        self.check_category([0.1, 0.9])
    def test_two_bool_categories_remain_categories(self):
        self.check_category([False, True])
    def test_three_categories_unchanged(self):
        self.check_category(['square', 'hex', 'triangular'])
    def test_tuple_integer_range_unchanged(self):
        self.assertEqual(self.call((2, 4)), ({'design': 3}, [('int', 'design', 2, 4)]))
    def test_tuple_float_range_unchanged(self):
        self.assertEqual(self.call((0.1, 0.9)), ({'design': 0.5}, [('float', 'design', 0.1, 0.9)]))
    def test_advanced_category_dict_unchanged(self):
        self.assertEqual(self.call({'type': 'categorical', 'choices': ['a', 'b']}), ({'design': 'a'}, [('categorical', 'design', ['a', 'b'])]))
    def test_invalid_outer_specs_keep_valueerror(self):
        for spec in (None, 7, 'bad', (), (1,), (1, 2, 3)):
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, 'Invalid spec for design'):
                self.call(spec)

if __name__ == '__main__':
    unittest.main(verbosity=2)
