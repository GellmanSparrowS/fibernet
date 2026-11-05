"""Gyroid factory routing and bounded real-backend equivalence, not physics."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from fibernet import api, gen
from fibernet.core.material import Material
from fibernet.gen import _fixes, metamaterials, unified


GYROID_SIGNATURE = inspect.signature(metamaterials.gyroid_lattice_3d)
FACTORIES = (gen.lattice_3d, lambda **kw: api.create("lattice_3d", **kw))


class TestGyroidFactoryContract(unittest.TestCase):
    def capture(self, calls, result):
        def backend(*args, **kwargs):
            bound = GYROID_SIGNATURE.bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append(dict(bound.arguments))
            return result

        return backend

    def test_public_defaults_reach_backend(self):
        for factory in FACTORIES:
            with self.subTest(factory=factory):
                calls, result = [], object()
                with patch.object(metamaterials, "gyroid_lattice_3d", self.capture(calls, result)):
                    self.assertIs(factory(topology="gyroid"), result)
                self.assertEqual(calls, [dict(cell_size=10.0, grid_size=(3, 3, 3),
                                             radius=0.1, resolution=8,
                                             material=None, threshold=0.0)])

    def test_each_shared_override_and_backend_options_reach_backend(self):
        material = Material(name="factory_contract", youngs_modulus=1234.0)
        cases = [dict(grid_size=(1, 2, 3)), dict(radius=0.23),
                 dict(material=material),
                 dict(cell_size=0.75, grid_size=(1, 2, 3), radius=0.23,
                      material=material, resolution=4, threshold=0.2)]
        for factory in FACTORIES:
            for options in cases:
                with self.subTest(factory=factory, options=options):
                    calls, result = [], object()
                    expected = dict(cell_size=10.0, grid_size=(3, 3, 3), radius=0.1,
                                    resolution=8, material=None, threshold=0.0)
                    expected.update(options)
                    before = options.copy()
                    with patch.object(metamaterials, "gyroid_lattice_3d", self.capture(calls, result)):
                        self.assertIs(factory(topology="GYROID", seed=17,
                                              perturbation=0.2, **options), result)
                    self.assertEqual(calls, [expected])
                    self.assertIs(calls[0]["material"], expected["material"])
                    self.assertEqual(options, before)

    def test_unknown_backend_options_still_reject(self):
        for factory in FACTORIES:
            for option in ("raduis", "resoluton", "spacing"):
                with self.subTest(factory=factory, option=option):
                    calls = []
                    with patch.object(metamaterials, "gyroid_lattice_3d", self.capture(calls, object())):
                        with self.assertRaisesRegex(TypeError, option):
                            factory(topology="gyroid", **{option: 1})
                    self.assertEqual(calls, [])

    def test_public_and_direct_backend_defaults_are_unchanged(self):
        expected = [("topology", "octet"), ("cell_size", 10.0),
                    ("grid_size", (3, 3, 3)), ("perturbation", 0.0),
                    ("radius", 0.1), ("material", None), ("seed", None)]
        for factory in (unified.lattice_3d, _fixes.lattice_3d_fixed):
            parameters = list(inspect.signature(factory).parameters.values())
            self.assertEqual([(p.name, p.default) for p in parameters[:-1]], expected)
            self.assertEqual(parameters[-1].kind, inspect.Parameter.VAR_KEYWORD)
        self.assertEqual(GYROID_SIGNATURE.parameters["grid_size"].default, (2, 2, 2))
        self.assertEqual(GYROID_SIGNATURE.parameters["radius"].default, 0.15)
        self.assertIs(api._registry.generators["lattice_3d"], unified.lattice_3d)
        self.assertIs(gen.lattice_3d, unified.lattice_3d)

    def test_small_real_factory_matches_direct_backend(self):
        material = Material(name="small_gyroid", youngs_modulus=1234.0,
                            poissons_ratio=0.25)
        options = dict(cell_size=0.75, grid_size=(1, 1, 1), radius=0.23,
                       material=material, resolution=3)
        expected = metamaterials.gyroid_lattice_3d(**options)
        actual = api.create("lattice_3d", topology="gyroid", **options)
        self.assertGreater(expected.num_fibers, 0)
        self.assertEqual(actual.num_fibers, expected.num_fibers)
        self.assertEqual(actual.num_crosslinks, expected.num_crosslinks)
        np.testing.assert_array_equal(actual.box_size, expected.box_size)
        self.assertEqual(actual.metadata, expected.metadata)
        for left, right in zip(actual.fibers, expected.fibers):
            self.assertEqual(left.radius, 0.23)
            self.assertIs(left.material, material)
            self.assertEqual(left.fiber_id, right.fiber_id)
            np.testing.assert_array_equal(left.centerline, right.centerline)
            self.assertTrue(np.isfinite(left.centerline).all())
        for left, right in zip(actual.crosslinks, expected.crosslinks):
            for field in ("fiber_i", "fiber_j", "param_i", "param_j", "crosslink_type"):
                self.assertEqual(getattr(left, field), getattr(right, field))
            np.testing.assert_array_equal(left.position, right.position)


if __name__ == "__main__":
    unittest.main()
