import tempfile
import unittest
from pathlib import Path
import numpy as np
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.io.pdb import to_pdb, from_pdb

class PdbFixedConect(unittest.TestCase):
    def test_writer_format_with_adjacent_five_digit_serials(self):
        network = FiberNetwork()
        network.add_fiber(Fiber.straight([0, 0, 0], [1, 0, 0], radius=.1, segments=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pdb'
            to_pdb(network, str(path), bead_spacing=2.)
            lines = path.read_text().splitlines(keepends=True)
            # The exact PDB fixed-width layout produced by the writer at larger
            # serial values, without generating ten thousand atoms.
            atom = 0
            for i, line in enumerate(lines):
                if line.startswith('ATOM'):
                    atom += 1
                    lines[i] = line[:6] + f'{9999 + atom:5d}' + line[11:]
                elif line.startswith('CONECT'):
                    lines[i] = 'CONECT' + f'{10000:5d}{10001:5d}' + '\n'
            path.write_text(''.join(lines))
            self.assertIn('CONECT1000010001', path.read_text())
            restored = from_pdb(str(path))
        self.assertEqual(restored.num_fibers, 1)
        np.testing.assert_allclose(restored.fibers[0].centerline, [[0,0,0],[1,0,0]])

    def test_free_form_uneven_width_serials(self):
        for serials, record in (((1234, 12345), 'CONECT 1234 12345'),
                                ((12345, 67890), 'CONECT12345 67890')):
            with self.subTest(record=record), tempfile.TemporaryDirectory() as directory:
                network = FiberNetwork()
                network.add_fiber(Fiber.straight([0, 0, 0], [1, 0, 0], radius=.1, segments=1))
                path = Path(directory) / 'fixture.pdb'
                to_pdb(network, str(path), bead_spacing=2.)
                lines = path.read_text().splitlines(keepends=True)
                atom = 0
                for i, line in enumerate(lines):
                    if line.startswith('ATOM'):
                        lines[i] = line[:6] + f'{serials[atom]:5d}' + line[11:]
                        atom += 1
                    elif line.startswith('CONECT'):
                        lines[i] = record + '\n'
                path.write_text(''.join(lines))
                result = from_pdb(str(path))
                self.assertEqual(result.num_fibers, 1)
                np.testing.assert_allclose(result.fibers[0].centerline, [[0,0,0],[1,0,0]])

    def test_fixed_columns_with_adjacent_and_extra_neighbors(self):
        cases = (((9999, 10000), 'CONECT 999910000'),
                 ((1, 2), 'CONECT    1    210000'))
        for serials, record in cases:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as directory:
                network = FiberNetwork()
                network.add_fiber(Fiber.straight([0, 0, 0], [1, 0, 0], radius=.1, segments=1))
                path = Path(directory) / 'fixture.pdb'
                to_pdb(network, str(path), bead_spacing=2.)
                lines = path.read_text().splitlines(keepends=True)
                atom = 0
                for i, line in enumerate(lines):
                    if line.startswith('ATOM'):
                        lines[i] = line[:6] + f'{serials[atom]:5d}' + line[11:]
                        atom += 1
                    elif line.startswith('CONECT'):
                        lines[i] = record + '\n'
                path.write_text(''.join(lines))
                result = from_pdb(str(path))
                self.assertEqual(result.num_fibers, 1)
                np.testing.assert_allclose(result.fibers[0].centerline, [[0,0,0],[1,0,0]])

    def test_public_generator_export_and_fixed_width_import(self):
        from fibernet.gen import random_straight_2d
        from fibernet.api import export
        network = random_straight_2d(num_fibers=1, fiber_length=1., box_size=10.,
                                     radius=.1, seed=7, ensure_connected=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'generated.pdb'
            export(network, str(path), bead_spacing=2.)
            lines = path.read_text().splitlines(keepends=True)
            serial = 10000
            for i, line in enumerate(lines):
                if line.startswith('ATOM'):
                    lines[i] = line[:6] + f'{serial:5d}' + line[11:]
                    serial += 1
                elif line.startswith('CONECT'):
                    lines[i] = f'CONECT{10000:5d}{10001:5d}\n'
            path.write_text(''.join(lines))
            result = from_pdb(str(path))
        self.assertEqual(result.num_fibers, 1)
        expected = np.vstack([network.fibers[0].start_point, network.fibers[0].end_point])
        np.testing.assert_allclose(result.fibers[0].centerline, expected, atol=0.00005, rtol=0)

    def test_existing_space_delimited_records_remain_supported(self):
        network = FiberNetwork()
        network.add_fiber(Fiber.straight([0, 0, 0], [1, 0, 0], radius=.1, segments=1))
        for free_form in (False, True):
            with self.subTest(free_form=free_form), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'fixture.pdb'
                to_pdb(network, str(path), bead_spacing=2.)
                if free_form:
                    # Preserve the old nonstandard whitespace parser too.
                    path.write_text(path.read_text().replace('CONECT    1    2', 'CONECT 1 2'))
                result = from_pdb(str(path))
            self.assertEqual(result.num_fibers, 1)
            np.testing.assert_allclose(result.fibers[0].centerline, [[0,0,0],[1,0,0]])

if __name__ == '__main__': unittest.main()
