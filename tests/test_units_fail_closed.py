"""Explicit failure instead of a fictitious network unit conversion."""
import pickle
import unittest
import numpy as np
from fibernet import gen
from fibernet.units import scale_network_properties, convert_length, parse_unit_string


class ScaleNetworkPropertiesContractTests(unittest.TestCase):
    def test_legacy_none_factor_queries_unchanged(self):
        for unit, expected in (("m", 1.), ("mm", 1e-3), ("um", 1e-6), ("nm", 1e-9)):
            self.assertEqual(scale_network_properties(None, unit), expected)
        with self.assertRaisesRegex(ValueError, "Unknown unit"):
            scale_network_properties(None, "invalid")

    def test_real_network_is_untouched_and_rejected(self):
        network = gen.square_lattice_2d(grid_size=(1, 1))
        before = pickle.dumps(network)
        with self.assertRaisesRegex(NotImplementedError, "no data was changed"):
            scale_network_properties(network, "mm")
        self.assertEqual(pickle.dumps(network), before)

    def test_any_network_argument_fails_before_inspection(self):
        class Unreadable:
            def __getattribute__(self, name):
                raise AssertionError("network inspected")
        with self.assertRaisesRegex(NotImplementedError, "convert_length"):
            scale_network_properties(Unreadable(), "mm")

    def test_explicit_migration_helpers_keep_direction_and_arrays(self):
        values = np.array([1., 2.])
        np.testing.assert_array_equal(convert_length(values, "m", "mm"), [1000., 2000.])
        np.testing.assert_array_equal(values, [1., 2.])
        self.assertEqual(parse_unit_string("mm"), (1e-3, "length"))


if __name__ == "__main__":
    unittest.main()
