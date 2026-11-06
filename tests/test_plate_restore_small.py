"""Bounded factory regression; exactly one actual geometry construction."""
import unittest
from unittest.mock import patch
import numpy as np
from fibernet import gen
from fibernet.api import create
from fibernet.core.material import Material

class PlateRestore(unittest.TestCase):
    def test_dispatch(self):
        marker, material = object(), object()
        with patch('fibernet.gen.metamaterials.plate_lattice_3d', return_value=marker) as call:
            self.assertIs(gen.lattice_3d(topology='plate', cell_size=2.5, grid_size=(1,2,3), radius=.2, material=material, plate_thickness=.7), marker)
            call.assert_called_once_with(spacing=2.5, grid_size=(1,2,3), radius=.2, material=material, plate_thickness=.7)
        with patch('fibernet.gen.metamaterials.plate_lattice_3d') as call:
            with self.assertRaisesRegex(TypeError, 'uses cell_size'):
                create('lattice_3d', topology='plate', spacing=2.)
            call.assert_not_called()
    def test_single_real_plate(self):
        material=Material(name='single_plate_fixture', youngs_modulus=1234., poissons_ratio=.25)
        net=create('lattice_3d', topology='plate', cell_size=.75, grid_size=(1,1,1), radius=.2, material=material)
        self.assertEqual(net.num_fibers,24)
        self.assertEqual(net.num_crosslinks,120)
        np.testing.assert_allclose(net.box_size,[.75,.75,.75])
        self.assertEqual(net.metadata['spacing'],.75)
        for f in net.fibers:
            self.assertEqual(f.radius,.2)
            self.assertIs(f.material,material)
            self.assertTrue(np.isfinite(f.centerline).all())

if __name__ == '__main__': unittest.main()
