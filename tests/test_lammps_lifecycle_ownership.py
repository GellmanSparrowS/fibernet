"""Owned-child interruption controls; mocked processes and signals only."""
import inspect
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, call

from test_lammps_backend_contract import prepared
from fibernet.integrations.lammps_backend import LAMMPSTask, RuntimeSpec, TaskState


@unittest.skipUnless(os.name == 'posix', 'process-group lifecycle requires POSIX')
class LifecycleOwnershipTests(unittest.TestCase):
    def make_task(self, root, mode):
        return LAMMPSTask(prepared(), RuntimeSpec(mode, sys.executable, '/fake/native',
                         expected_banner='LAMMPS (test fixture)'), Path(root)/'run')

    def test_interrupt_after_stored_handle_before_running(self):
        lines, start = inspect.getsourcelines(LAMMPSTask.start)
        target = start + next(i for i, text in enumerate(lines) if text.strip() == 'self.state = TaskState.RUNNING')
        for mode in ('cli', 'cli_serial_static'):
            for interruption in (KeyboardInterrupt, SystemExit):
                with self.subTest(mode=mode, interruption=interruption), tempfile.TemporaryDirectory() as root:
                    process = Mock(pid=123456789); process.wait.return_value = 0
                    with patch.object(RuntimeSpec, 'checked', lambda self:self), patch('subprocess.Popen', return_value=process), patch('os.killpg') as kill:
                        task = self.make_task(root, mode)
                        original = interruption('stored handle interruption')
                        def trace(frame, event, arg):
                            if frame.f_code is LAMMPSTask.start.__code__ and event == 'line' and frame.f_lineno == target:
                                sys.settrace(None)
                                raise original
                            return trace
                        previous = sys.gettrace(); sys.settrace(trace)
                        try:
                            with self.assertRaises(interruption) as caught: task.start()
                        finally: sys.settrace(previous)
                        self.assertIs(caught.exception, original)
                        self.assertEqual(task.wait(.1), TaskState.FAILED)
                        self.assertTrue(task._stream.closed)
                        self.assertEqual(kill.call_count, 2)
                        self.assertEqual(process.wait.call_args_list, [call(timeout=.25), call()])
                        task.cancel(); self.assertEqual(kill.call_count, 2)

    def test_interrupt_thread_setup_preserves_primary_exception(self):
        for mode in ('cli', 'cli_serial_static'):
            for target in ('threading.Thread', 'threading.Thread.start'):
                for interruption in (KeyboardInterrupt, SystemExit):
                    with self.subTest(mode=mode, target=target, interruption=interruption), tempfile.TemporaryDirectory() as root:
                        process=Mock(pid=123456789); process.wait.return_value=0
                        original=interruption('thread setup interruption')
                        with patch.object(RuntimeSpec, 'checked', lambda self:self), patch('subprocess.Popen', return_value=process), patch(target, side_effect=original), patch('os.killpg') as kill:
                            task=self.make_task(root,mode)
                            with self.assertRaises(interruption) as caught:task.start()
                            self.assertIs(caught.exception,original)
                            self.assertEqual(task.wait(.1),TaskState.FAILED)
                            self.assertTrue(task._stream.closed)
                            self.assertEqual(kill.call_count,2);self.assertEqual(process.wait.call_args_list, [call(timeout=.25), call()])

    def test_created_owned_process_cancel_closes_and_completes(self):
        for mode in ('cli','cli_serial_static'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root, patch.object(RuntimeSpec,'checked',lambda self:self), patch('os.killpg') as kill:
                task=self.make_task(root,mode);task._process=Mock(pid=123456789);task._process.wait.return_value=0;task._stream=io.BytesIO()
                task.cancel();self.assertEqual(task.wait(.1),TaskState.CANCELLED)
                self.assertTrue(task._stream.closed);self.assertEqual(kill.call_count,2)
                self.assertEqual(task._process.wait.call_args_list, [call(timeout=.25), call()]);task.cancel();self.assertEqual(kill.call_count,2)

    def test_start_cleanup_errors_preserve_primary_and_retriable_ownership(self):
        for mode in ('cli','cli_serial_static'):
            for cleanup_site in ('kill','wait'):
                with self.subTest(mode=mode,cleanup_site=cleanup_site),tempfile.TemporaryDirectory() as root:
                    original=KeyboardInterrupt('primary');cleanup=OSError('cleanup failure')
                    process=Mock(pid=123456789);process.wait.side_effect=cleanup if cleanup_site=='wait' else None
                    with patch.object(RuntimeSpec,'checked',lambda self:self),patch('subprocess.Popen',return_value=process),patch('threading.Thread.start',side_effect=original),patch('os.killpg',side_effect=cleanup if cleanup_site=='kill' else None):
                        task=self.make_task(root,mode)
                        with self.assertRaises(KeyboardInterrupt) as caught:task.start()
                    self.assertIs(caught.exception,original);self.assertIs(caught.exception.__cause__,cleanup)
                    self.assertEqual(task.state,TaskState.RUNNING);self.assertFalse(task._done.is_set())
                    self.assertIn('cleanup incomplete',task.error)
                    with self.assertRaisesRegex(TimeoutError,'no active watchdog'):task.wait(.001)
                    process.wait.side_effect=None;process.wait.return_value=0
                    with patch('os.killpg') as kill:task.cancel()
                    self.assertEqual(task.wait(.1),TaskState.CANCELLED);self.assertTrue(task._stream.closed);self.assertEqual(kill.call_count,2)

    def test_interrupted_cancel_can_be_retried(self):
        with tempfile.TemporaryDirectory() as root,patch.object(RuntimeSpec,'checked',lambda self:self):
            task=self.make_task(root,'cli_serial_static');task._process=Mock(pid=123456789);task._stream=io.BytesIO()
            with patch('os.killpg',side_effect=KeyboardInterrupt('cancel interrupted')):
                with self.assertRaises(KeyboardInterrupt):task.cancel()
            self.assertEqual(task.state,TaskState.RUNNING);self.assertFalse(task._done.is_set())
            with patch('os.killpg') as kill:task.cancel()
            self.assertEqual(task.wait(.1),TaskState.CANCELLED);self.assertTrue(task._stream.closed);self.assertEqual(kill.call_count,2)

    def test_close_error_does_not_mask_primary_after_reap(self):
        with tempfile.TemporaryDirectory() as root,patch.object(RuntimeSpec,'checked',lambda self:self):
            task=self.make_task(root,'cli_serial_static');original=SystemExit('primary');cleanup=OSError('close failed')
            process=Mock(pid=123456789);process.wait.return_value=0
            def interrupt():
                stream=task._stream;stream.close();task._stream=Mock();task._stream.close.side_effect=cleanup
                raise original
            with patch('subprocess.Popen',return_value=process),patch('threading.Thread.start',side_effect=interrupt),patch('os.killpg') as kill:
                with self.assertRaises(SystemExit) as caught:task.start()
            self.assertIs(caught.exception,original);self.assertIs(caught.exception.__cause__,cleanup)
            self.assertEqual(process.wait.call_args_list, [call(timeout=.25), call()]);self.assertEqual(kill.call_count,2)
            self.assertEqual(task.wait(.1),TaskState.FAILED);self.assertIn('cleanup incomplete',task.error)

if __name__ == '__main__':unittest.main()
