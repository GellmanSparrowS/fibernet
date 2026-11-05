"""Real generator/native JSON contracts, not optional-exporter safety."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fibernet import gen
import fibernet.api as legacy
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material
from fibernet.core.network import FiberNetwork


class FalseyMaterial(Material):
    def __bool__(self):
        return False


class UnifiedDefaultMaterialTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fibernet-unified-material-")
        self.addCleanup(temporary.cleanup)
        self._temporary_directory = temporary
        self.output_directory = Path(temporary.name)
        self._output_number = 0

    def output_path(self, label):
        self._output_number += 1
        return self.output_directory / ("{:03d}-{}.json".format(self._output_number, label))

    @staticmethod
    def cases():
        return (("lattice_2d", {"topology": "honeycomb", "grid_size": (2, 2)}),
                ("metamaterial_2d", {"mode": "reentrant", "grid_size": (2, 2)}),
                ("metamaterial_2d", {"mode": "chiral", "grid_size": (2, 2)}))

    @staticmethod
    def generate(route, generator, parameters, material_arguments):
        kwargs = dict(parameters)
        kwargs.update(material_arguments)
        if route == "gen":
            return getattr(gen, generator)(**kwargs)
        if route == "legacy":
            return legacy.create(generator, **kwargs)
        raise AssertionError("unknown test route")

    def assert_usable_native_network(self, network):
        self.assertGreater(network.num_fibers, 1)
        self.assertIn("FiberNetwork", network.summary())
        self.assertIn("Material:", network.describe())
        for fiber in network.fibers:
            self.assertIn("Fiber(", repr(fiber))
        record = network.to_dict()
        original = deepcopy(record)
        expected = deepcopy(original)
        # Existing reader behavior assigns each loaded Fiber its list index.
        # No other field is normalized, and stable legacy IDs are not claimed.
        for index, fiber_record in enumerate(expected["fibers"]):
            fiber_record["fiber_id"] = index
        def assert_roundtrip(restored):
            self.assertEqual(restored.to_dict(), expected)
            self.assertEqual(record, original)
            self.assertEqual(network.to_dict(), original)
        assert_roundtrip(FiberNetwork.from_dict(record))
        native_path = self.output_path("native")
        network.save_json(str(native_path))
        with native_path.open("r", encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), original)
        assert_roundtrip(FiberNetwork.load_json(str(native_path)))
        exported_path = self.output_path("export")
        legacy.export(network, str(exported_path))
        with exported_path.open("r", encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), original)
        assert_roundtrip(FiberNetwork.load_json(str(exported_path)))

    def test_omitted_and_explicit_none_use_independent_generic_materials(self):
        expected_material = Material().to_dict()
        self.assertEqual(expected_material["name"], "generic")
        self.assertEqual(expected_material["density"], 1000.)
        self.assertEqual(expected_material["youngs_modulus"], 1e9)
        self.assertEqual(expected_material["poissons_ratio"], .3)
        for route in ("gen", "legacy"):
            for generator, parameters in self.cases():
                for argument_name, material_arguments in (("omitted", {}),
                        ("explicit_none", {"material": None})):
                    with self.subTest(route=route, generator=generator,
                            parameters=parameters, material_argument=argument_name):
                        network = self.generate(route, generator, parameters, material_arguments)
                        self.assertGreater(network.num_fibers, 1)
                        materials = [fiber.material for fiber in network.fibers]
                        for material in materials:
                            self.assertIsInstance(material, Material)
                            self.assertEqual(material.to_dict(), expected_material)
                        self.assertEqual(len({id(material) for material in materials}), len(materials))
                        self.assert_usable_native_network(network)
                        materials[0].extra["only_first_fiber"] = True
                        for material in materials[1:]:
                            self.assertNotIn("only_first_fiber", material.extra)

    def test_supplied_material_identity_including_falsey_material(self):
        for route in ("gen", "legacy"):
            for generator, parameters in self.cases():
                for material_class in (Material, FalseyMaterial):
                    with self.subTest(route=route, generator=generator,
                            parameters=parameters, material_class=material_class.__name__):
                        supplied = material_class(name="supplied", density=1234.,
                            youngs_modulus=2.5e9, poissons_ratio=.25, extra={"provenance": "test"})
                        if material_class is FalseyMaterial:
                            self.assertFalse(supplied)
                        network = self.generate(route, generator, parameters, {"material": supplied})
                        self.assertGreater(network.num_fibers, 1)
                        for fiber in network.fibers:
                            self.assertIs(fiber.material, supplied)
                        self.assert_usable_native_network(network)
                        for fiber_record in network.to_dict()["fibers"]:
                            self.assertEqual(fiber_record["material"], supplied.to_dict())

    def test_standalone_fiber_explicit_none_semantics_are_unchanged(self):
        centerline = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=float)
        omitted = Fiber(centerline=centerline.copy())
        another = Fiber(centerline=centerline.copy())
        self.assertIsInstance(omitted.material, Material)
        self.assertIsNot(omitted.material, another.material)
        self.assertEqual(omitted.material.to_dict(), Material().to_dict())
        explicit_none = Fiber(centerline=centerline.copy(), material=None)
        self.assertIsNone(explicit_none.material)
        with self.assertRaisesRegex(AttributeError, "to_dict"):
            explicit_none.to_dict()
        with self.assertRaisesRegex(AttributeError, "name"):
            repr(explicit_none)


if __name__ == "__main__":
    unittest.main()
