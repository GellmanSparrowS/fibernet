import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fibernet.core.network import FiberNetwork


class InjectedH5File:
    def __init__(self, write_error=None, close_error=None):
        self.attrs = {}
        self.write_error = write_error
        self.close_error = close_error
        self.close_calls = 0

    def create_group(self, name):
        if self.write_error is not None:
            raise self.write_error
        return self

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@unittest.skipUnless(importlib.util.find_spec('h5py') is not None, 'h5py unavailable: unverified')
class Hdf5FailureDiagnosticsTests(unittest.TestCase):
    def test_primary_write_error_survives_close_and_cleanup_combinations(self):
        for fail_close in (False, True):
            for fail_cleanup in (False, True):
                with self.subTest(close=fail_close, cleanup=fail_cleanup):
                    with tempfile.TemporaryDirectory() as directory:
                        target = Path(directory) / 'network.h5'
                        original = b'existing output must survive'
                        target.write_bytes(original)
                        cause = ValueError('original cause')
                        primary = RuntimeError('write failure')
                        primary.__cause__ = cause
                        primary.__suppress_context__ = True
                        close_error = OSError('close failure')
                        cleanup_error = PermissionError('cleanup failure')
                        handle = InjectedH5File(primary, close_error if fail_close else None)
                        import os
                        real_unlink = os.unlink

                        def controlled_unlink(path, *args, **kwargs):
                            if fail_cleanup:
                                raise cleanup_error
                            return real_unlink(path, *args, **kwargs)

                        with patch('h5py.File', return_value=handle), patch('os.unlink', side_effect=controlled_unlink):
                            with self.assertRaises(RuntimeError) as caught:
                                FiberNetwork().save_hdf5(target)
                        self.assertIs(caught.exception, primary)
                        self.assertIs(primary.__cause__, cause)
                        self.assertTrue(primary.__suppress_context__)
                        self.assertEqual(handle.close_calls, 1)
                        self.assertEqual(target.read_bytes(), original)
                        if fail_close:
                            self.assertIs(primary.hdf5_close_error, close_error)
                        else:
                            self.assertFalse(hasattr(primary, 'hdf5_close_error'))
                        if fail_cleanup:
                            self.assertIs(primary.hdf5_cleanup_error, cleanup_error)
                            self.assertEqual(len(list(Path(directory).glob('.fibernet-*.h5.tmp'))), 1)
                        else:
                            self.assertFalse(hasattr(primary, 'hdf5_cleanup_error'))
                            self.assertEqual(list(Path(directory).glob('.fibernet-*.h5.tmp')), [])

    def test_close_only_failure_prevents_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'network.h5'
            target.write_bytes(b'old')
            close_error = OSError('close failure')
            handle = InjectedH5File(close_error=close_error)
            with patch('h5py.File', return_value=handle), patch('os.replace') as replace:
                with self.assertRaises(OSError) as caught:
                    FiberNetwork().save_hdf5(target)
            self.assertIs(caught.exception, close_error)
            self.assertEqual(handle.close_calls, 1)
            replace.assert_not_called()
            self.assertEqual(target.read_bytes(), b'old')
            self.assertEqual(list(Path(directory).glob('.fibernet-*.h5.tmp')), [])

    def test_descriptor_close_failure_is_not_retried(self):
        import os
        import h5py
        real_close = os.close
        failure = OSError('injected report after actual close')
        calls = []

        def close_then_report(fd):
            calls.append(fd)
            real_close(fd)
            raise failure

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'network.h5'
            target.write_bytes(b'old')
            with patch.object(h5py, 'File') as h5file, patch('os.close', side_effect=close_then_report):
                with self.assertRaises(OSError) as caught:
                    FiberNetwork().save_hdf5(target)
            self.assertIs(caught.exception, failure)
            self.assertTrue(failure.hdf5_descriptor_close_uncertain)
            self.assertEqual(len(calls), 1)
            h5file.assert_not_called()
            self.assertEqual(target.read_bytes(), b'old')
            self.assertEqual(list(Path(directory).glob('.fibernet-*.h5.tmp')), [])

    def test_bytes_path_is_rejected_before_temp_creation(self):
        with patch('tempfile.mkstemp') as create_temp:
            with self.assertRaisesRegex(TypeError, 'PathLike'):
                FiberNetwork().save_hdf5(b'network.h5')
        create_temp.assert_not_called()


if __name__ == '__main__':
    unittest.main()
