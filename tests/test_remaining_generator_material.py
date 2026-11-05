"""Small native-I/O contracts for four non-specialty generator families."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from fibernet import gen
import fibernet.api as legacy
from fibernet.core.material import Material
from fibernet.core.network import FiberNetwork


CASES = (
    ('lattice_3d', {'topology': 'cubic', 'grid_size': (1, 1, 1)}),
    ('lattice_3d', {'topology': 'octet', 'grid_size': (1, 1, 1)}),
    ('lattice_3d', {'topology': 'diamond', 'grid_size': (1, 1, 1)}),
    ('curved_random_2d', {'num_fibers': 2, 'num_points': 4, 'seed': 7}),
    ('entangled_3d', {'num_fibers': 2, 'num_points': 4, 'seed': 7}),
    ('hierarchical_lattice', {'levels': 1}),
)


class FalseyMaterial(Material):
    def __bool__(self):
        return False


def generate(route, name, kwargs):
    if route == 'gen':
        return getattr(gen, name)(**kwargs)
    return legacy.create(name, **kwargs)


class RemainingGeneratorMaterialTests(unittest.TestCase):
    def test_default_material_native_usage_and_json_roundtrip(self):
        expected_material = Material().to_dict()
        with tempfile.TemporaryDirectory(prefix='fibernet-material-test-') as directory:
            sequence = 0
            for route in ('gen', 'legacy'):
                for name, parameters in CASES:
                    for explicit_none in (False, True):
                        with self.subTest(route=route, name=name, parameters=parameters,
                                          explicit_none=explicit_none):
                            kwargs = dict(parameters)
                            if explicit_none:
                                kwargs['material'] = None
                            net = generate(route, name, kwargs)
                            self.assertGreater(net.num_fibers, 1)
                            materials = [f.material for f in net.fibers]
                            self.assertEqual(len({id(m) for m in materials}), len(materials))
                            for material in materials:
                                self.assertIsInstance(material, Material)
                                self.assertEqual(material.to_dict(), expected_material)
                            self.assertIn('Material: generic', net.describe())
                            for fiber in net.fibers:
                                self.assertIn("mat='generic'", repr(fiber))
                            record = net.to_dict()
                            preserved = deepcopy(record)
                            expected_restored = deepcopy(record)
                            # Existing native reader assigns list-index IDs.
                            for i, fiber_record in enumerate(expected_restored['fibers']):
                                fiber_record['fiber_id'] = i
                            self.assertEqual(FiberNetwork.from_dict(record).to_dict(), expected_restored)
                            sequence += 1
                            path = Path(directory) / f'{sequence}.json'
                            net.save_json(str(path))
                            self.assertEqual(json.loads(path.read_text()), preserved)
                            self.assertEqual(FiberNetwork.load_json(str(path)).to_dict(), expected_restored)
                            self.assertEqual(record, preserved)
                            self.assertEqual(net.to_dict(), preserved)
                            materials[0].extra['only_first'] = True
                            self.assertTrue(all('only_first' not in m.extra for m in materials[1:]))

    def test_supplied_material_identity_including_falsey(self):
        for route in ('gen', 'legacy'):
            for name, parameters in CASES:
                for material_type in (Material, FalseyMaterial):
                    with self.subTest(route=route, name=name, parameters=parameters,
                                      material_type=material_type.__name__):
                        supplied = material_type(name='explicit', youngs_modulus=2e9,
                                                 extra={'origin': 'caller'})
                        expected = supplied.to_dict()
                        net = generate(route, name, dict(parameters, material=supplied))
                        self.assertGreater(net.num_fibers, 1)
                        for fiber in net.fibers:
                            self.assertIs(fiber.material, supplied)
                        for record in net.to_dict()['fibers']:
                            self.assertEqual(record['material'], expected)
                        self.assertEqual(supplied.to_dict(), expected)


if __name__ == '__main__':
    unittest.main()
