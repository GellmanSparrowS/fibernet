"""Focused DataFrame import workflow regressions."""
import unittest

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from fibernet.core.material import Material
from fibernet import Crosslink, Fiber, FiberNetwork
from fibernet.io import from_dataframe, to_dataframe


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


    def test_empty_export_retains_schema_and_can_be_imported(self):
        frame = to_dataframe(FiberNetwork())
        self.assertTrue(frame.empty)
        self.assertEqual(frame.columns.tolist(), [
            'fiber_id', 'point_idx', 'x', 'y', 'z', 'radius',
            'material_name', 'youngs_modulus', 'density',
        ])
        restored = from_dataframe(frame)
        self.assertEqual(restored.num_fibers, 0)
        self.assertEqual(restored.dimension, 3)

    def test_filtered_categories_keep_only_present_fibers_in_point_order(self):
        frame = pd.DataFrame({
            'fiber_id': pd.Categorical([81, 17, 81, 2, 17, 81, 2, 17],
                                       categories=[81, 17, 2, 300], ordered=True),
            'point_idx': [2, 1, 0, 0, 2, 1, 1, 0],
            'x': [2., 1., 0., 0., 3., .5, 5., 0.],
            'y': [0., 1., 0., 5., 2., .5, 5., 2.],
            'z': [1., 3., 0., 0., 5., 0., 0., 2.],
            'radius': [.1, .2, .1, .3, .2, .1, .3, .2],
            'youngs_modulus': [3., 4., 3., 5., 4., 3., 5., 4.],
            'material_name': ['one', 'two', 'one', 'three', 'two', 'one', 'three', 'two'],
        })
        filtered = frame.loc[frame['fiber_id'] != 2]
        before = filtered.copy(deep=True)
        restored = from_dataframe(filtered)
        self.assertEqual([fiber.fiber_id for fiber in restored.fibers], [0, 1])
        self.assertEqual([fiber.material.name for fiber in restored.fibers], ['one', 'two'])
        self.assertEqual([fiber.radius for fiber in restored.fibers], [.1, .2])
        np.testing.assert_array_equal(restored.fibers[0].centerline,
                                      [[0., 0., 0.], [.5, .5, 0.], [2., 0., 1.]])
        np.testing.assert_array_equal(restored.fibers[1].centerline,
                                      [[0., 2., 2.], [1., 1., 3.], [3., 2., 5.]])
        pd.testing.assert_frame_equal(filtered, before)
        self.assertEqual(from_dataframe(filtered.iloc[:0]).num_fibers, 0)

    def test_unordered_categories_keep_category_order_with_duplicate_row_indices(self):
        for categories in ([31, 5, 17], [31, 99, 5, 17]):
            with self.subTest(categories=categories):
                frame = pd.DataFrame({
                    'fiber_id': pd.Categorical([5, 17, 31, 5, 17, 31],
                                               categories=categories, ordered=False),
                    'point_idx': [1, 1, 1, 0, 0, 0],
                    'x': [15., 27., 41., 5., 17., 31.],
                    'y': [1., 1., 1., 0., 0., 0.],
                    'radius': [.1] * 6,
                }, index=[4, 2, 4, 2, 4, 2])
                before = frame.copy(deep=True)
                restored = from_dataframe(frame)
                self.assertEqual([fiber.fiber_id for fiber in restored.fibers], [0, 1, 2])
                for fiber, label in zip(restored.fibers, (31, 5, 17)):
                    np.testing.assert_array_equal(fiber.centerline,
                                                  [[label, 0., 0.], [label + 10., 1., 0.]])
                pd.testing.assert_frame_equal(frame, before)

    def test_named_fiber_index_and_multiindex_levels_remain_supported(self):
        index = pd.Index([19, 7, 19, 7], name='fiber_id')
        multi_index = pd.MultiIndex.from_arrays([['a', 'b', 'a', 'b'], index],
                                                names=['sample', 'fiber_id'])
        for labels in (index, multi_index):
            with self.subTest(index_type=type(labels).__name__):
                frame = pd.DataFrame({'point_idx': [1, 1, 0, 0],
                                      'x': [20., 8., 19., 7.], 'y': [0.] * 4}, index=labels)
                before = frame.copy(deep=True)
                restored = from_dataframe(frame)
                self.assertEqual([fiber.fiber_id for fiber in restored.fibers], [0, 1])
                self.assertEqual([fiber.centerline[:, 0].tolist() for fiber in restored.fibers],
                                 [[7., 8.], [19., 20.]])
                pd.testing.assert_frame_equal(frame, before)

    def test_unordered_categorical_index_uses_observed_category_order(self):
        index = pd.CategoricalIndex([5, 17, 31, 5, 17, 31], categories=[31, 99, 5, 17],
                                    ordered=False, name='fiber_id')
        multi_index = pd.MultiIndex.from_arrays([[0, 1, 0, 1, 0, 1], index],
                                                names=['sample', 'fiber_id'])
        for labels in (index, multi_index):
            with self.subTest(index_type=type(labels).__name__):
                frame = pd.DataFrame({'point_idx': [1, 1, 1, 0, 0, 0],
                                      'x': [15., 27., 41., 5., 17., 31.],
                                      'y': [0.] * 6}, index=labels)
                before = frame.copy(deep=True)
                restored = from_dataframe(frame)
                self.assertEqual([fiber.centerline[:, 0].tolist() for fiber in restored.fibers],
                                 [[31., 41.], [5., 15.], [17., 27.]])
                pd.testing.assert_frame_equal(frame, before)

    def test_missing_fiber_index_labels_raise(self):
        index = pd.Index([7, pd.NA, 7], name='fiber_id')
        frame = pd.DataFrame({'x': [7., 8., 9.], 'y': [0.] * 3}, index=index)
        before = frame.copy(deep=True)
        with self.assertRaisesRegex(ValueError, 'fiber_id'):
            from_dataframe(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_same_fiber_column_and_index_level_remain_ambiguous(self):
        frame = pd.DataFrame({'fiber_id': pd.Categorical([5, 5]),
                              'x': [0., 1.], 'y': [0., 0.]},
                             index=pd.Index([7, 7], name='fiber_id'))
        with self.assertRaisesRegex(ValueError, 'both an index level and a column label'):
            from_dataframe(frame)

    def test_missing_fiber_labels_raise_without_silently_losing_points(self):
        for missing in (pd.NA, np.nan, None):
            with self.subTest(missing=missing):
                frame = pd.DataFrame({'fiber_id': [81, missing, 81], 'point_idx': [0, 1, 2],
                                      'x': [0., .5, 1.], 'y': [0., .5, 0.]})
                before = frame.copy(deep=True)
                with self.assertRaisesRegex(ValueError, 'fiber_id'):
                    from_dataframe(frame)
                pd.testing.assert_frame_equal(frame, before)

    def test_crosslink_flags_use_fiber_positions_with_sparse_labels(self):
        fibers = [
            Fiber([[0., 0., 0.], [.5, .25, .1], [1., 0., 0.]], fiber_id=81),
            Fiber([[0., 0., 1.], [0., .5, 1.], [0., 1., 1.]], fiber_id=2),
            Fiber([[1., 0., 0.], [1.5, .5, .2], [2., 1., 0.]], fiber_id=17),
        ]
        network = FiberNetwork(fibers=fibers, crosslinks=[
            Crosslink(2, 0, 0., 1., np.array([1., 0., 0.])),
        ])
        before = network.to_dict()
        frame = to_dataframe(network)
        independently_linked = network.connectivity_matrix().any(axis=1)
        for index, fiber in enumerate(fibers):
            rows = frame.loc[frame['fiber_id'] == fiber.fiber_id]
            self.assertEqual(rows['has_crosslink'].tolist(), [bool(independently_linked[index])] * 3)
            self.assertEqual(rows['point_idx'].tolist(), [0, 1, 2])
            np.testing.assert_array_equal(rows[['x', 'y', 'z']].to_numpy(), fiber.centerline)
        self.assertNotIn('has_crosslink', to_dataframe(network, include_crosslinks=False))
        self.assertEqual(network.to_dict(), before)

    def test_crosslink_flags_and_layout_for_positional_labels_unchanged(self):
        network = FiberNetwork()
        for y in (0., 1., 0.):
            network.add_fiber(Fiber([[0., y, 0.], [1., y, 0.]]))
        self.assertNotIn('has_crosslink', to_dataframe(network))
        network.add_crosslink(Crosslink(0, 2, 0., 0., np.zeros(3)))
        frame = to_dataframe(network)
        self.assertEqual(frame['fiber_id'].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(frame['has_crosslink'].tolist(), [True, True, False, False, True, True])


if __name__ == '__main__':
    unittest.main()
