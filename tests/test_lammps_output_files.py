"""Offline file parsing uses synthetic protocol data, not native evidence."""
from pathlib import Path
import os
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import (HarmonicBond, RunSpec, LAMMPSContractError,
                                   parse_output_files, parse_outputs,
                                   prepare_harmonic_beads)


def fixture():
    graph = StructureGraph(dimension=3)
    a = graph.add_node([0, 0, 0]); b = graph.add_node([1, 0, 0])
    edge = graph.add_edge(a, b)
    prepared = prepare_harmonic_beads(
        graph, length_unit='mm', maximum_bond_length_m=.01,
        masses_kg={a: 1, b: 1}, bonds={edge: HarmonicBond(2, .001)},
        run=RunSpec(1e-6, 0, 1, ((-.01, .01),) * 3))
    dump = ('ITEM: UNITS\nsi\nITEM: TIME\n0\nITEM: TIMESTEP\n0\n'
            'ITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS ff ff ff\n'
            '-0.01 0.01\n')
    dump += ('-0.01 0.01\n-0.01 0.01\n'
             'ITEM: ATOMS id type x y z vx vy vz fx fy fz\n'
             '1 1 0 0 0 0 0 0 0 0 0\n'
             '2 1 0.001 0 0 0 0 0 0 0 0\n')
    log = ('LAMMPS (synthetic)\nStep Atoms PotEng KinEng TotEng Pxx Pyy Pzz Pxy Pxz Pyz\n'
           '0 2 0 0 0 0 0 0 0 0 0\nLoop time\n# ε\n' + prepared.completion_marker + '\n')
    return prepared, dump, log


class OutputFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.prepared, self.dump, self.log = fixture()
        self.dump_path = self.path / 'trajectory.dump'; self.log_path = self.path / 'log.lammps'
        self.dump_path.write_text(self.dump, encoding='utf-8')
        self.log_path.write_text(self.log, encoding='utf-8')

    def test_matches_existing_parser_without_native_execution(self):
        before = (self.dump_path.read_bytes(), self.log_path.read_bytes())
        with patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
            result = parse_output_files(self.prepared, self.dump_path, self.log_path,
                                        expected_banner='LAMMPS (synthetic)')
        self.assertEqual(result, parse_outputs(self.prepared, self.dump, self.log))
        self.assertEqual(result.workdir, '')
        self.assertEqual((self.dump_path.read_bytes(), self.log_path.read_bytes()), before)

    def test_combined_byte_cap_and_option_validation(self):
        exact = len(self.dump.encode()) + len(self.log.encode())
        self.assertEqual(parse_output_files(self.prepared, self.dump_path, self.log_path,
                                            max_output_bytes=exact).identity, self.prepared.identity)
        for limit in (exact - 1, 0, True, 64 * 1024 * 1024 + 1):
            with self.subTest(limit=limit), self.assertRaises(LAMMPSContractError):
                parse_output_files(self.prepared, self.dump_path, self.log_path, max_output_bytes=limit)
        real_fstat = os.fstat
        def stale_size(fd):
            info = real_fstat(fd)
            return SimpleNamespace(st_mode=info.st_mode, st_size=0)
        with patch('fibernet.integrations.lammps_backend.os.fstat', side_effect=stale_size):
            with self.assertRaisesRegex(LAMMPSContractError, 'byte budget'):
                parse_output_files(self.prepared, self.dump_path, self.log_path,
                                   max_output_bytes=len(self.dump.encode()) - 1)
        for banner in ('', 'line\nnext', 5):
            with self.subTest(banner=banner), self.assertRaises(LAMMPSContractError):
                parse_output_files(self.prepared, self.dump_path, self.log_path, expected_banner=banner)

    def test_invalid_content_and_banner_fail_closed(self):
        with self.assertRaisesRegex(LAMMPSContractError, 'banner'):
            parse_output_files(self.prepared, self.dump_path, self.log_path, expected_banner='wrong')
        with self.assertRaisesRegex(LAMMPSContractError, 'banner'):
            parse_output_files(self.prepared, self.dump_path, self.log_path, expected_banner='LAMMPS')
        self.log_path.write_text(self.log.replace(self.prepared.completion_marker, 'missing'), encoding='utf-8')
        with self.assertRaisesRegex(LAMMPSContractError, 'marker'):
            parse_output_files(self.prepared, self.dump_path, self.log_path)
        self.log_path.write_bytes(b'\xff')
        with self.assertRaisesRegex(LAMMPSContractError, 'UTF-8'):
            parse_output_files(self.prepared, self.dump_path, self.log_path)

    def test_nonregular_files_are_rejected(self):
        with self.assertRaises(LAMMPSContractError):
            parse_output_files(self.prepared, self.path, self.log_path)
        if hasattr(os, 'mkfifo') and hasattr(os, 'O_NONBLOCK'):
            fifo = self.path / 'fifo'; os.mkfifo(fifo)
            with self.assertRaisesRegex(LAMMPSContractError, 'regular file'):
                parse_output_files(self.prepared, fifo, self.log_path)


if __name__ == '__main__': unittest.main()
