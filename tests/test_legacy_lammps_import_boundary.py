"""Legacy reader boundary only; no LAMMPS/native execution or model conversion."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fibernet import io
from fibernet.io.lammps import from_lammps
from fibernet.api import load


class LegacyLammpsImportBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def assert_reject_before_io(self, operation, suffix='.data'):
        for exists in (False, True):
            with self.subTest(exists=exists, suffix=suffix):
                path = self.root / ('existing' + suffix if exists else 'absent' + suffix)
                original = b'LAMMPS original\nKEEP\x00UNCHANGED\n'
                if exists:
                    path.write_bytes(original)
                with patch('builtins.open', side_effect=AssertionError('must not open')) as opened:
                    with patch.object(Path, 'open', side_effect=AssertionError('must not open')) as path_opened:
                        with self.assertRaisesRegex(NotImplementedError, 'legacy from_lammps import is unsupported'):
                            operation(path)
                opened.assert_not_called()
                path_opened.assert_not_called()
                if exists:
                    self.assertEqual(path.read_bytes(), original)
                else:
                    self.assertFalse(path.exists())

    def test_direct_rejects_before_io(self):
        self.assert_reject_before_io(from_lammps)

    def test_public_io_alias_rejects_before_io(self):
        self.assertIs(io.from_lammps, from_lammps)
        self.assert_reject_before_io(io.from_lammps)

    def test_api_inferred_extensions_reject_before_io(self):
        for suffix in ('.data', '.lammps'):
            self.assert_reject_before_io(load, suffix)

    def test_api_explicit_format_rejects_before_io(self):
        self.assert_reject_before_io(lambda path: load(path, format='lammps'), '.custom')

    def test_spacing_cannot_bypass_rejection(self):
        for spacing in (0, -1, 1, float('nan'), object()):
            self.assert_reject_before_io(lambda path: from_lammps(path, bead_spacing=spacing))

    def test_input_is_not_inspected(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError('input must not be evaluated')
        with self.assertRaises(NotImplementedError):
            from_lammps(Untouchable(), bead_spacing=Untouchable())

    def test_message_preserves_original_not_fictional_recovery(self):
        with self.assertRaises(NotImplementedError) as caught:
            from_lammps(self.root / 'original.data')
        for text in ('Retain the original LAMMPS file', 'If an original FiberNet graph exists',
                     'JSON cannot recover', 'no automatic replacement importer'):
            self.assertIn(text, str(caught.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
