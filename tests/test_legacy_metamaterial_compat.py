"""Provider-free contracts for the legacy factory fallback, not material physics."""
import unittest
from unittest.mock import patch

import numpy as np

from fibernet import Material, api, gen
from fibernet.core import transform


CASES = (
    ("reentrant_honeycomb_2d", dict(cell_height=7, cell_width=9, reentrant_angle=135),
     "vertical_thickness", 1.0, 2, 9, 28),
    ("chiral_honeycomb_2d", dict(node_radius=2, ligament_length=5),
     "num_node_points", 6, 8, 6, 19),
    ("star_honeycomb_2d", dict(cell_size=7, star_angle=75, num_arms=5),
     "star_inner_angle", 30.0, 45, 10, 37),
    ("arrowhead_auxetic_2d", dict(cell_size=7, arrow_angle=75),
     "arm_angle", None, 40, 5, 25),
    ("hierarchical_lattice_2d", dict(cell_size=7, levels=1),
     "base_type", "triangular", "square", 5, 16),
    ("missing_rib_auxetic_2d", dict(cell_size=7), "rib_angle", 45.0, 60, 5, 27),
)


def tiled_reference(cell, tiles, threshold):
    lower, upper = cell.bounding_box()
    space = upper - lower + .1
    expected = transform.tile(cell, repeats=(*tiles, 1), spacing=space)
    expected.crosslinks = []
    expected.auto_crosslink(threshold=threshold, crosslink_type="welded")
    return expected, space


class LegacyMetamaterialCompatTests(unittest.TestCase):
    def test_actual_default_equals_explicit_original_factory(self):
        props = dict(grid_size=(1, 1), cell_height=7, cell_width=9, reentrant_angle=135)
        implicit = api.create_metamaterial(array_size=(1, 1), **props)
        explicit = api.create_metamaterial("reentrant_honeycomb_2d", array_size=(1, 1), **props)
        self.assertEqual(implicit.to_dict(), explicit.to_dict())
        source = gen.reentrant_honeycomb_2d(**props)
        self.assertEqual(implicit.num_fibers, source.num_fibers)
        for observed, original in zip(implicit.fibers, source.fibers):
            # Tiling owns frame/placement; the intrinsic geometry must be unchanged.
            np.testing.assert_allclose(np.diff(observed.centerline, axis=0),
                                       np.diff(original.centerline, axis=0), atol=1e-13, rtol=0)

    def test_documented_names_reuse_six_original_geometries_with_outer_welding(self):
        for name, options, opt, default, _, fibers, welds in CASES:
            with self.subTest(name=name):
                params = dict(options, grid_size=(1, 1), radius=.30,
                              material=Material(name="two-tile test", youngs_modulus=1234))
                source = getattr(gen, name)(**params)
                expected, spacing = tiled_reference(source, (2, 1), .15)
                actual = api.create_metamaterial(name, array_size=(2, 1), weld_threshold=.15, **params)
                explicit = api.create_metamaterial(
                    name, array_size=(2, 1), weld_threshold=.15, **params, **{opt: default}
                )
                self.assertEqual((source.num_fibers, actual.num_fibers), (fibers, 2 * fibers))
                self.assertEqual((actual.dimension, actual.num_crosslinks), (2, welds))
                for k in ("fibers", "crosslinks", "box_size"):
                    self.assertEqual(actual.to_dict()[k], expected.to_dict()[k])
                    self.assertEqual(actual.to_dict()[k], explicit.to_dict()[k])
                for index, member in enumerate(source.fibers):
                    a, b = actual.fibers[index], actual.fibers[index + fibers]
                    np.testing.assert_allclose(b.centerline - a.centerline,
                                               np.broadcast_to([spacing[0], 0, 0], a.centerline.shape),
                                               rtol=0, atol=1e-13)
                    self.assertEqual(a.radius, member.radius)
                    self.assertEqual(a.material.to_dict(), member.material.to_dict())
                self.assertTrue(all(c.crosslink_type == "welded" for c in actual.crosslinks))
                self.assertEqual(actual.metadata["cell_params"], params)

    def test_registered_implementations_keep_precedence_and_no_new_registry_names(self):
        for name, *_ in CASES:
            self.assertNotIn(name, api.list_generators())
        with self.assertRaisesRegex(ValueError, "Unknown generator"):
            api.create("reentrant_honeycomb_2d")
        props = dict(mode="star", grid_size=(1, 1), cell_size=3, angle=115, seed=3)
        existing = api.create_metamaterial("metamaterial_2d", array_size=(1, 1), **props)
        expected, _ = tiled_reference(gen.metamaterial_2d(**props), (1, 1), .5)
        self.assertEqual(existing.to_dict()["fibers"], expected.to_dict()["fibers"])
        self.assertEqual(existing.to_dict()["crosslinks"], expected.to_dict()["crosslinks"])
        with patch.dict(api._registry.generators):
            @api.register_generator("reentrant_honeycomb_2d")
            def provided(custom_scale, vertical_thickness):
                return gen.reentrant_honeycomb_2d(grid_size=(1, 1), cell_height=custom_scale)

            # A user's implementation may accept parameters unused by the built-in one.
            custom = api.create_metamaterial(array_size=(1, 1), custom_scale=4, vertical_thickness=3)
            provided_only = api.create("reentrant_honeycomb_2d", custom_scale=4, vertical_thickness=3)
            expected, _ = tiled_reference(provided_only, (1, 1), .5)
            self.assertEqual(custom.to_dict()["fibers"], expected.to_dict()["fibers"])
        self.assertNotIn("reentrant_honeycomb_2d", api.list_generators())

    def test_fallback_does_not_accept_unused_options_or_unknowns(self):
        for name, options, option, _, bad, *_ in CASES:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, option):
                api.create_metamaterial(name, grid_size=(1, 1), **options, **{option: bad})
        with self.assertRaisesRegex(TypeError, "unknown_option"):
            api.create_metamaterial(unknown_option=3)
        with self.assertRaisesRegex(ValueError, "Unknown unit cell"):
            api.create_metamaterial("undocumented")
