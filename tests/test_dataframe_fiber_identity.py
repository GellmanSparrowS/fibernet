"""Round-trip coverage for DataFrame fiber identity."""

from io import StringIO
import unittest

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from fibernet import Fiber, FiberNetwork, gen
from fibernet.core.material import Material
from fibernet.io import from_dataframe, to_dataframe


POSITION = 'fibernet_fiber_index'
STANDARD_COLUMNS = [
    'fiber_id', 'point_idx', 'x', 'y', 'z', 'radius',
    'material_name', 'youngs_modulus', 'density',
]


@unittest.skipIf(pd is None, 'pandas is optional')
class DataFrameFiberIdentity(unittest.TestCase):
    def assert_geometry_and_labels(self, restored, source_fibers):
        self.assertEqual(restored.num_fibers, len(source_fibers))
        self.assertEqual(restored.total_length, sum(fiber.length for fiber in source_fibers))
        for position, (actual, source) in enumerate(zip(restored.fibers, source_fibers)):
            self.assertEqual(actual.fiber_id, position)
            np.testing.assert_array_equal(actual.centerline, source.centerline)
            self.assertEqual(actual.radius, source.radius)
            self.assertEqual(actual.material.name, source.material.name)
            self.assertEqual(actual.material.youngs_modulus, source.material.youngs_modulus)
            self.assertEqual(actual.material.density, source.material.density)

    def test_generated_curves_keep_separate_centerlines_in_memory_and_csv(self):
        source = gen.curved_random_2d(num_fibers=96, seed=23)
        self.assertEqual([fiber.fiber_id for fiber in source.fibers], [0] * 96)
        for position in (3, 20, 83):
            source.fibers[position].radius = position / 50
            source.fibers[position].material = Material(
                name=f'material-{position}', youngs_modulus=position, density=position + 100.
            )
        snapshot = source.to_dict()
        frame = to_dataframe(source, include_crosslinks=False)
        self.assertEqual(frame.columns.tolist(), STANDARD_COLUMNS + [POSITION])
        self.assertEqual(frame['fiber_id'].tolist(), [0] * 1920)
        self.assertEqual(frame[POSITION].tolist(), np.repeat(np.arange(96), 20).tolist())

        shuffled = frame.sample(frac=1., random_state=5)
        before = shuffled.copy(deep=True)
        self.assert_geometry_and_labels(from_dataframe(shuffled), source.fibers)
        pd.testing.assert_frame_equal(shuffled, before)
        csv = pd.read_csv(StringIO(shuffled.to_csv(index=False)), float_precision='round_trip')
        self.assert_geometry_and_labels(from_dataframe(csv), source.fibers)
        subset = shuffled.loc[shuffled[POSITION].isin([83, 3, 20])]
        self.assert_geometry_and_labels(
            from_dataframe(subset), [source.fibers[i] for i in (3, 20, 83)]
        )
        self.assertEqual(source.to_dict(), snapshot)

    def test_hierarchical_export_appends_positions_after_crosslink_flags(self):
        source = gen.hierarchical_lattice(levels=2)
        self.assertEqual(source.num_fibers, 39)
        self.assertTrue(source.crosslinks)
        frame = to_dataframe(source)
        self.assertEqual(frame.columns.tolist(), STANDARD_COLUMNS + ['has_crosslink', POSITION])
        self.assertEqual(frame[POSITION].nunique(), 39)
        self.assert_geometry_and_labels(from_dataframe(frame), source.fibers)

    def test_canonical_and_empty_export_keep_existing_column_order(self):
        self.assertEqual(to_dataframe(FiberNetwork()).columns.tolist(), STANDARD_COLUMNS)
        try:
            source = gen.ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network()
        except ImportError:
            self.skipTest('networkx is optional')
        self.assertEqual(source.num_fibers, 45)
        frame = to_dataframe(source)
        self.assertEqual(frame.columns.tolist(), STANDARD_COLUMNS + ['has_crosslink'])
        self.assert_geometry_and_labels(from_dataframe(frame), source.fibers)

    def test_unique_nonpositional_labels_preserve_source_order_and_values(self):
        source = FiberNetwork(fibers=[
            Fiber([[0., y, 0.], [1., y, 0.]], fiber_id=label)
            for y, label in enumerate([91, -7, 12])
        ])
        frame = to_dataframe(source)
        self.assertEqual(frame['fiber_id'].tolist(), [91, 91, -7, -7, 12, 12])
        self.assertEqual(frame[POSITION].tolist(), [0, 0, 1, 1, 2, 2])
        self.assert_geometry_and_labels(from_dataframe(frame), source.fibers)

    def test_numeric_position_order_overrides_category_order_and_handles_named_levels(self):
        frame = pd.DataFrame({
            POSITION: pd.Categorical([11, 2, 11, 2], categories=[11, 77, 2]),
            'fiber_id': [19, 19, 19, 19], 'point_idx': [1, 1, 0, 0],
            'x': [12., 3., 11., 2.], 'y': [0.] * 4,
        }, index=[4, 2, 4, 2])
        variants = (
            frame,
            frame.set_index(POSITION),
            frame.set_index([POSITION, 'fiber_id']),
            frame.set_index([POSITION, 'point_idx']),
        )
        for table in variants:
            with self.subTest(index_levels=table.index.names):
                before = table.copy(deep=True)
                restored = from_dataframe(table)
                self.assertEqual([f.centerline[:, 0].tolist() for f in restored.fibers],
                                 [[2., 3.], [11., 12.]])
                pd.testing.assert_frame_equal(table, before)

        no_point_index = from_dataframe(frame.drop(columns='point_idx'))
        self.assertEqual([f.centerline[:, 0].tolist() for f in no_point_index.fibers],
                         [[3., 2.], [12., 11.]])

    def test_sample_order_on_a_generated_twenty_point_centerline(self):
        source = gen.curved_random_2d(num_fibers=2, seed=23)
        frame = to_dataframe(source)
        frame = frame.loc[frame[POSITION] == 0].sample(frac=1., random_state=5)
        self.assertEqual(len(frame), 20)
        categorical = frame.assign(
            point_idx=pd.Categorical(frame['point_idx'], categories=frame['point_idx'].tolist())
        )
        self.assert_geometry_and_labels(from_dataframe(categorical), source.fibers[:1])
        self.assert_geometry_and_labels(
            from_dataframe(categorical.set_index([POSITION, 'point_idx'])), source.fibers[:1]
        )

        strings = frame.assign(point_idx=frame['point_idx'].astype(str))
        with self.assertRaisesRegex(ValueError, 'point_idx.*nonnegative integer'):
            from_dataframe(strings)
        # Without the namespace the existing external-table sort still applies.
        self.assertEqual(from_dataframe(strings.drop(columns=POSITION)).num_fibers, 1)

    def test_unnamespaced_fiber_index_does_not_change_legacy_grouping(self):
        frame = pd.DataFrame({
            'fiber_id': [17, 5, 17, 5], 'fiber_index': [0, 0, 0, -3],
            'point_idx': [1, 1, 0, 0], 'x': [18., 6., 17., 5.], 'y': [0.] * 4,
        })
        restored = from_dataframe(frame)
        self.assertEqual([fiber.centerline[:, 0].tolist() for fiber in restored.fibers],
                         [[5., 6.], [17., 18.]])

    def test_rejects_invalid_positions_in_column_and_named_index(self):
        for invalid in (-1, 1., np.inf, True, np.bool_(False), '1', None, pd.NA):
            with self.subTest(invalid=invalid):
                frame = pd.DataFrame({
                    POSITION: pd.Series([0, invalid], dtype=object),
                    'fiber_id': [7, 7], 'x': [0., 1.], 'y': [0., 0.],
                })
                with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                    from_dataframe(frame)
                indexed = frame.set_index(pd.Index(frame[POSITION], dtype=object)).drop(columns=POSITION)
                with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                    from_dataframe(indexed)

    def test_rejects_conflicting_labels_duplicate_points_and_ambiguous_position_names(self):
        frame = pd.DataFrame({
            POSITION: [0, 0], 'fiber_id': [7, 7], 'point_idx': [0, 1],
            'x': [0., 1.], 'y': [0., 0.],
        })
        with self.assertRaisesRegex(ValueError, 'fiber_id must be present'):
            from_dataframe(frame.assign(fiber_id=[7, None]))
        with self.assertRaisesRegex(ValueError, 'conflicting fiber_id'):
            from_dataframe(frame.assign(fiber_id=[7, 9]))
        duplicate = frame.assign(point_idx=[0, 0])
        with self.assertRaisesRegex(ValueError, 'duplicate point_idx'):
            from_dataframe(duplicate)
        with self.assertRaisesRegex(ValueError, 'duplicate point_idx'):
            from_dataframe(duplicate.set_index([POSITION, 'point_idx']))
        for invalid in (pd.NA, np.nan, np.inf, -1, True, 0.5):
            with self.subTest(invalid_point=invalid):
                points = pd.Series([0, invalid], dtype=object)
                invalid_frame = frame.assign(point_idx=points)
                with self.assertRaisesRegex(ValueError, 'point_idx.*nonnegative integer'):
                    from_dataframe(invalid_frame)
                with self.assertRaisesRegex(ValueError, 'point_idx.*nonnegative integer'):
                    from_dataframe(invalid_frame.set_index([POSITION, 'point_idx']))
        with self.assertRaisesRegex(ValueError, 'both an index level and a column label'):
            from_dataframe(frame.set_index(POSITION, drop=False))
        with self.assertRaisesRegex(ValueError, 'single column or index level'):
            from_dataframe(pd.concat([frame, frame[[POSITION]]], axis=1))
