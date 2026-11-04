"""Synthetic generator-to-storage usage checks; no solver/physics acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from fibernet import gen
from fibernet.core.material import Material
from fibernet.utils.units import convert_network_units


class GeneratorUnitsHdf5UsageTests(unittest.TestCase):
    def test_default_material_conversion_and_hdf5_schema(self):
        cases = (
            ('lattice_2d', {'topology': 'honeycomb', 'grid_size': (2, 2)}),
            ('lattice_3d', {'topology': 'cubic', 'grid_size': (1, 1, 1)}),
        )
        with tempfile.TemporaryDirectory(prefix='fibernet-joint-usage-') as folder:
            for name, parameters in cases:
                with self.subTest(generator=name):
                    original = getattr(gen, name)(cell_size=.01, radius=.0001,
                                                  material=None, **parameters)
                    before = deepcopy(original.to_dict())
                    self.assertGreater(len(original.fibers), 1)
                    self.assertGreater(len(original.crosslinks), 0)
                    for fiber in original.fibers:
                        self.assertIs(type(fiber.material), Material)
                    converted = convert_network_units(original, 'SI', 'CGS')
                    self.assertEqual(original.to_dict(), before)
                    self.assertEqual(converted.metadata['unit_system'], 'CGS')
                    for old, new in zip(original.fibers, converted.fibers):
                        self.assertIsNot(old.material, new.material)
                        np.testing.assert_allclose(new.centerline, old.centerline * 100.,
                                                   rtol=1e-14, atol=0.)
                        self.assertAlmostEqual(new.material.density, 1.)
                        self.assertAlmostEqual(new.material.youngs_modulus / 1e9, 10.)
                    path = Path(folder) / (name + '.h5')
                    converted_before = deepcopy(converted.to_dict())
                    converted.save_hdf5(path)
                    self.assertEqual(converted.to_dict(), converted_before)
                    self.assertEqual(original.to_dict(), before)
                    with h5py.File(path, 'r') as saved:
                        self.assertEqual(saved.attrs['version'], '0.1.0')
                        self.assertEqual(json.loads(saved.attrs['metadata']), converted.metadata)
                        self.assertEqual(len(saved['fibers']), len(converted.fibers))
                        self.assertEqual(len(saved['crosslinks']), len(converted.crosslinks))
                        for index, fiber in enumerate(converted.fibers):
                            record = saved['fibers'][f'fiber_{index}']
                            np.testing.assert_array_equal(record['centerline'][...], fiber.centerline)
                            self.assertEqual(record.attrs['radius'], fiber.radius)
                            self.assertEqual(json.loads(record.attrs['material']), fiber.material.to_dict())
                        for index, (old, new) in enumerate(zip(original.crosslinks, converted.crosslinks)):
                            record = saved['crosslinks'][f'cl_{index}']
                            np.testing.assert_allclose(record['position'][...], old.position * 100.,
                                                       rtol=1e-14, atol=0.)
                            for field in ('fiber_i', 'fiber_j', 'param_i', 'param_j'):
                                self.assertEqual(record.attrs[field], getattr(new, field))
                            # Existing schema omits these; this test does not claim a lossless roundtrip.
                            self.assertNotIn('strength', record.attrs)
                            self.assertNotIn('stiffness', record.attrs)
                    self.assertEqual(list(Path(folder).glob('.fibernet-*.h5.tmp')), [])
