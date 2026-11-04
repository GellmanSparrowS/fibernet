import unittest
import numpy as np
from fibernet.utils.validation import ValidationError, validate_array

class ArrayLengthContract(unittest.TestCase):
    def test_scalar_with_length_bound_raises_validation_error(self):
        for value in (3, 3.5, np.array(3), np.float64(3)):
            for bounds in ({'min_length': 1}, {'max_length': 2}, {'min_length': 0, 'max_length': 2}):
                with self.subTest(value=value, bounds=bounds):
                    with self.assertRaisesRegex(ValidationError, 'length'):
                        validate_array(value, 'points', **bounds)

    def test_scalars_without_bounds_and_arrays_preserve_behavior(self):
        self.assertEqual(validate_array(3, 'points').shape, ())
        np.testing.assert_array_equal(validate_array([1, 2], 'points', min_length=2, max_length=2), [1, 2])
        self.assertEqual(validate_array([[1, 2], [3, 4]], 'matrix', min_length=2).shape, (2, 2))
        for bounds in ({'min_length': 3}, {'max_length': 1}):
            with self.assertRaises(ValidationError):
                validate_array([1, 2], 'points', **bounds)

if __name__ == '__main__': unittest.main()
