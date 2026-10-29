"""Executor ownership on batch success and exceptional exits; no simulations."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor as RealExecutor, Future as RealFuture
from unittest.mock import patch

import importlib
import sys
batch = importlib.import_module('fibernet.utils.batch')


class Future:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error
    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class Executor:
    def __init__(self, futures, submit_error=None, cleanup_error=None):
        self.futures = iter(futures)
        self.submit_error = submit_error
        self.cleanup_error = cleanup_error
        self.calls = []
    def submit(self, fn, value):
        self.calls.append(('submit', value))
        if self.submit_error is not None and value == 1:
            raise self.submit_error
        return next(self.futures)
    def shutdown(self, wait=True, *, cancel_futures=False):
        self.calls.append(('shutdown', wait, cancel_futures))
        if self.cleanup_error is not None:
            raise self.cleanup_error


class Lifecycle(unittest.TestCase):
    def invoke(self, executor, networks=(0, 1), iterator=None, **kwargs):
        with patch.object(batch, 'ThreadPoolExecutor', return_value=executor), patch.object(batch, 'as_completed', iterator or (lambda fs: reversed(list(fs)))):
            return batch.batch_simulate(networks, lambda x: x, parallel=True, show_progress=False, **kwargs)

    def test_success_waits_and_preserves_input_order(self):
        e = Executor([Future('a'), Future('b')])
        self.assertEqual(self.invoke(e).results, ['a', 'b'])
        self.assertEqual(e.calls[-1], ('shutdown', True, False))

    def test_collected_error_still_waits(self):
        e = Executor([Future('a'), Future(error=ValueError('bad'))])
        r = self.invoke(e)
        self.assertEqual(r.results, ['a'])
        self.assertEqual(r.errors[0]['index'], 1)
        self.assertEqual(e.calls[-1], ('shutdown', True, False))

    def test_task_error_cancels_without_waiting(self):
        error = ValueError('task')
        e = Executor([Future(), Future(error=error)])
        with self.assertRaises(ValueError) as caught:
            self.invoke(e, raise_on_error=True)
        self.assertIs(caught.exception, error)
        self.assertEqual(e.calls[-1], ('shutdown', False, True))

    def test_partial_submission_cleans_up(self):
        error = RuntimeError('submit')
        e = Executor([Future()], submit_error=error)
        with self.assertRaises(RuntimeError) as caught:
            self.invoke(e)
        self.assertIs(caught.exception, error)
        self.assertEqual(e.calls[-1], ('shutdown', False, True))

    def test_iterator_failure_cleans_up(self):
        error = RuntimeError('iterator')
        def broken(fs):
            yield next(iter(fs))
            raise error
        e = Executor([Future(), Future()])
        with self.assertRaises(RuntimeError) as caught:
            self.invoke(e, iterator=broken)
        self.assertIs(caught.exception, error)
        self.assertEqual(e.calls[-1], ('shutdown', False, True))

    def test_input_iteration_failure_cleans_up(self):
        error = RuntimeError('input iteration')
        def inputs():
            yield 0
            raise error
        e = Executor([Future()])
        with self.assertRaises(RuntimeError) as caught:
            self.invoke(e, networks=inputs())
        self.assertIs(caught.exception, error)
        self.assertEqual(e.calls[-1], ('shutdown', False, True))

    def test_progress_construction_and_iteration_failure(self):
        from types import SimpleNamespace
        for phase in ('construct', 'iterate'):
            with self.subTest(phase=phase):
                error = RuntimeError('progress ' + phase)
                def wrap(*args, **kwargs):
                    if phase == 'construct':
                        raise error
                    def broken():
                        raise error
                        yield
                    return broken()
                e = Executor([Future()])
                with patch.object(batch, 'ThreadPoolExecutor', return_value=e), patch.dict(sys.modules, {'tqdm': SimpleNamespace(tqdm=wrap)}):
                    with self.assertRaises(RuntimeError) as caught:
                        batch.batch_simulate([0], lambda x: x, parallel=True, show_progress=True)
                self.assertIs(caught.exception, error)
                self.assertEqual(e.calls[-1], ('shutdown', False, True))

    def test_baseexceptions_and_cleanup_errors_preserve_primary(self):
        for error in [KeyboardInterrupt(), SystemExit(4), ValueError('original')]:
            with self.subTest(kind=type(error).__name__):
                e = Executor([Future(error=error)], cleanup_error=RuntimeError('cleanup'))
                with self.assertRaises(type(error)) as caught:
                    self.invoke(e, networks=(0,), raise_on_error=True)
                self.assertIs(caught.exception, error)
                self.assertEqual(e.calls[-1], ('shutdown', False, True))

    def test_success_cleanup_error_is_not_hidden(self):
        error = RuntimeError('cleanup')
        e = Executor([Future()], cleanup_error=error)
        with self.assertRaises(RuntimeError) as caught:
            self.invoke(e, networks=(0,))
        self.assertIs(caught.exception, error)

    def test_empty_parallel_batch_closes_executor(self):
        e = Executor([])
        self.assertEqual(self.invoke(e, networks=()).total_count, 0)
        self.assertEqual(e.calls[-1], ('shutdown', True, False))

    def test_running_task_not_waited_and_pending_task_cancelled(self):
        self.check_running_and_pending(False)

    def test_failed_future_does_not_wait_for_other_running_task(self):
        self.check_running_and_pending(True)

    def check_running_and_pending(self, task_failure):
        started, release, returned = threading.Event(), threading.Event(), threading.Event()
        futures, errors = [], []
        e = RealExecutor(max_workers=1)
        original_submit = e.submit
        def submit(fn, value):
            if value == 2:
                if not started.wait(2):
                    raise RuntimeError('test worker start timeout before failure')
                f = RealFuture()
                f.set_exception(primary)
            else:
                f = original_submit(fn, value)
            futures.append(f)
            return f
        def task(value):
            if value == 0:
                started.set()
                if not release.wait(5):
                    raise RuntimeError('test release timeout')
            return value
        primary = RuntimeError('failed future' if task_failure else 'stop iterator')
        real_as_completed = batch.as_completed
        def broken(fs):
            if not started.wait(2):
                raise RuntimeError('test worker start timeout')
            raise primary
        def invoke():
            try:
                with patch.object(batch, 'ThreadPoolExecutor', return_value=e), patch.object(e, 'submit', submit), patch.object(batch, 'as_completed', real_as_completed if task_failure else broken):
                    batch.batch_simulate([0, 1, 2] if task_failure else [0, 1], task, parallel=True, show_progress=False, raise_on_error=True)
            except BaseException as exc:
                errors.append(exc)
            finally:
                returned.set()
        caller = threading.Thread(target=invoke)
        try:
            caller.start()
            self.assertTrue(returned.wait(2), 'exception path waited for running task')
            self.assertEqual(errors, [primary])
            self.assertTrue(futures[1].cancelled(), 'pending task was not cancelled')
            self.assertFalse(futures[0].done(), 'running task must remain caller-controlled')
        finally:
            release.set()
            caller.join(6)
            e.shutdown(wait=True, cancel_futures=True)
        self.assertFalse(caller.is_alive())


if __name__ == '__main__':
    unittest.main()
