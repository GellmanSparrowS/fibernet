"""Focused DataFrame import workflow regressions."""
import unittest

try:
    import pandas as pd
except ImportError:
    pd = None

from fibernet.core.material import Material
from fibernet.io.pandas_io import from_dataframe


@unittest.skipIf(pd is None, 'pandas is optional')
class DataFrameImportContract(unittest.TestCase):
    def test_planar_rows_import_in_point_order(self):
        df = pd.DataFrame({'fiber_id': [4, 4], 'point_idx': [1, 0],
                           'x': [2., 0.], 'y': [3., 1.]})
        network = from_dataframe(df)
        self.assertEqual(network.dimension, 2)
        self.assertEqual(network.fibers[0].material.name, 'default')
        self.assertEqual(network.fibers[0].centerline.tolist(),
                         [[0., 1., 0.], [2., 3., 0.]])

    def test_explicit_material_wins_over_table_columns(self):
        df = pd.DataFrame({'fiber_id': [0, 0, 1, 1], 'x': [0., 1., 0., 1.],
                           'y': [0., 0., 1., 1.], 'z': [4., 4., 5., 5.],
                           'youngs_modulus': [1., 1., 2., 2.],
                           'density': [10., 10., 20., 20.],
                           'material_name': ['one', 'one', 'two', 'two']})
        material = Material(name='override', youngs_modulus=17., extra={'tag': 'kept'})
        result = from_dataframe(df, material=material)
        self.assertEqual(result.dimension, 3)
        self.assertEqual(result.fibers[0].centerline[:, 2].tolist(), [4., 4.])
        self.assertTrue(all(fiber.material is material for fiber in result.fibers))
        default = from_dataframe(df)
        self.assertEqual([f.material.name for f in default.fibers], ['one', 'two'])
        self.assertEqual([f.material.youngs_modulus for f in default.fibers], [1., 2.])


if __name__ == '__main__':
    unittest.main()
