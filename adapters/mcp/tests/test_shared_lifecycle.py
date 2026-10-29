"""Controlled shared-lifecycle regressions, not protocol or numerical evidence.

Run directly with unittest. Production Supervisor/ManagedStore are exercised
with mocked subprocess.Popen and os.killpg: no worker, MCP client, native
FiberNet API, solver, or actual process signal is launched by these tests.
"""
import copy
from pathlib import Path
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anyio

from fibernet_mcp import server
from fibernet_mcp.safety import ContractError, canonical, read_at, sha
from test_preparation import PROVENANCE, bundle, managed_store


LEGACY_OPERATIONS = ('validate', 'topology', 'linear_transport')


def lifecycle_request(operation):
    """Opaque storage-lifecycle request, never submitted to a native solver."""
    if operation != 'linear_transport':
        return None
    return {'length_unit': 'm',
            'conductance_S': [{'edge_id': 9, 'value_S': 1.0}],
            'dirichlet_voltage_V': [{'node_id': 42, 'value_V': 0.0}]}


class ControlledProcess:
    """Process-shaped stub; its synthetic PID must never reach an OS signal."""
    pid = 987654321

    def __init__(self, hard_kill=False):
        self.returncode = None
        self.hard_kill = hard_kill
        self.terminating = False
        self.polls_after_term = 0
        self.wait_calls = 0
        self.shielded_observations = []
        self.signals = []

    def poll(self):
        if self.terminating and self.returncode is None and not self.hard_kill:
            self.polls_after_term += 1
            # Require one asynchronous cleanup checkpoint before soft exit.
            if self.polls_after_term >= 2:
                self.returncode = -signal.SIGTERM
        return self.returncode

    def signal(self, pid, number):
        if pid != self.pid:
            raise AssertionError('Controlled signal targeted the wrong stub')
        self.signals.append(number)
        self.shielded_observations.append(anyio.current_effective_deadline() == float('inf'))
        if number == signal.SIGTERM:
            self.terminating = True
        elif number == signal.SIGKILL:
            self.returncode = -signal.SIGKILL
        else:
            raise AssertionError('Unexpected controlled signal')

    def wait(self):
        self.wait_calls += 1
        self.shielded_observations.append(anyio.current_effective_deadline() == float('inf'))
        if self.returncode is None:
            raise AssertionError('Controlled wait ran before simulated termination')
        return self.returncode


class SharedLifecycleTests(unittest.TestCase):
    def assert_source_and_sentinel(self, store, gid, output, source_before):
        self.assertEqual(read_at(store.fd, gid), source_before)
        self.assertEqual(sha(source_before), store.graphs[gid]['source_sha256'])
        self.assertEqual((output / 'unrelated.txt').read_text(), 'keep')
        self.assertFalse(store.artifacts)

    def test_legacy_cancellation_propagates_after_shielded_reap(self):
        async def exercise(store, gid, output, operation, hard_kill):
            supervisor = server.Supervisor(store)
            process = ControlledProcess(hard_kill=hard_kill)
            source_before = read_at(store.fd, gid)
            request = lifecycle_request(operation)
            observed = {'propagated': False, 'cleanup_awaits': 0}
            real_sleep = anyio.sleep

            async def checkpoint(delay):
                if process.terminating:
                    observed['cleanup_awaits'] += 1
                    self.assertEqual(anyio.current_effective_deadline(), float('inf'))
                await real_sleep(delay)

            with anyio.CancelScope() as scope:
                def launch(command, **kwargs):
                    self.assertEqual(command[command.index('fibernet_mcp.worker') + 1], operation)
                    index = command.index(operation)
                    observed['output'] = Path(command[index + 2])
                    observed['output'].write_bytes(b'controlled partial output')
                    if request is not None:
                        observed['request'] = Path(command[index + 4])
                        self.assertEqual(observed['request'].read_bytes(), canonical(request))
                    scope.cancel()
                    return process

                with patch.object(server.subprocess, 'Popen', side_effect=launch) as started, \
                     patch.object(server.os, 'killpg', side_effect=process.signal) as signalled, \
                     patch.object(server.anyio, 'sleep', side_effect=checkpoint):
                    try:
                        await supervisor.run(operation, gid, request)
                    except anyio.get_cancelled_exc_class():
                        observed['propagated'] = True
                        # Cleanup must finish before cancellation escapes run().
                        self.assertEqual(process.wait_calls, 1)
                        self.assertIsNone(supervisor.active)
                        self.assertFalse(observed['output'].exists())
                        if request is not None:
                            self.assertFalse(observed['request'].exists())
                        raise
                    self.fail('Supervisor swallowed cancellation')
                # The exception leaves the mocks before the outer scope catches it.
            self.assertTrue(scope.cancelled_caught)
            self.assertTrue(observed['propagated'])
            self.assertEqual(started.call_count, 1)
            expected_signals = [signal.SIGTERM, signal.SIGKILL] if hard_kill else [signal.SIGTERM]
            self.assertEqual(process.signals, expected_signals)
            self.assertEqual(signalled.call_count, len(expected_signals))
            self.assertGreaterEqual(observed['cleanup_awaits'], 1)
            self.assertTrue(all(process.shielded_observations))
            self.assertEqual(process.wait_calls, 1)
            self.assertIsNone(supervisor.active)
            self.assertEqual(len(supervisor.audit), 1)
            audit = supervisor.audit[0]
            self.assertEqual(audit['pid'], process.pid)
            self.assertEqual(audit['reason'], 'cancelled')
            self.assertTrue(audit['reaped'])
            self.assertEqual(audit['returncode'], -expected_signals[-1])
            self.assertEqual({path.name for path in Path(store.path).iterdir()}, {gid})
            self.assert_source_and_sentinel(store, gid, output, source_before)

        for operation in LEGACY_OPERATIONS:
            for hard_kill in (False, True):
                with self.subTest(operation=operation, controlled_hard_kill=hard_kill), \
                     managed_store() as (store, gid, output):
                    anyio.run(exercise, store, gid, output, operation, hard_kill)

    def test_legacy_output_collision_refuses_before_launch(self):
        for operation in LEGACY_OPERATIONS:
            with self.subTest(operation=operation), managed_store() as (store, gid, output):
                supervisor = server.Supervisor(store)
                source_before = read_at(store.fd, gid)
                collision = Path(store.path) / ('worker-' + 'c' * 32)
                collision.write_bytes(b'preexisting output sentinel')
                before = {path.name for path in Path(store.path).iterdir()}
                try:
                    with patch.object(server.uuid, 'uuid4', return_value=SimpleNamespace(hex='c' * 32)), \
                         patch.object(server.subprocess, 'Popen') as started, \
                         patch.object(server.os, 'killpg') as signalled:
                        with self.assertRaises(ContractError):
                            anyio.run(supervisor.run, operation, gid, lifecycle_request(operation))
                    started.assert_not_called()
                    signalled.assert_not_called()
                    self.assertEqual(collision.read_bytes(), b'preexisting output sentinel')
                    self.assertEqual({path.name for path in Path(store.path).iterdir()}, before)
                    self.assertEqual(supervisor.audit, [])
                    self.assertIsNone(supervisor.active)
                    self.assert_source_and_sentinel(store, gid, output, source_before)
                finally:
                    # Remove only this test's sentinel after preservation is checked.
                    collision.unlink()

    def test_legacy_linear_request_collision_preserves_nonowned_file(self):
        with managed_store() as (store, gid, output):
            supervisor = server.Supervisor(store)
            source_before = read_at(store.fd, gid)
            collision = Path(store.path) / ('request-' + 'b' * 32)
            collision.write_bytes(b'preexisting request sentinel')
            before = {path.name for path in Path(store.path).iterdir()}
            identifiers = [SimpleNamespace(hex='a' * 32), SimpleNamespace(hex='b' * 32)]
            try:
                with patch.object(server.uuid, 'uuid4', side_effect=identifiers), \
                     patch.object(server.subprocess, 'Popen') as started, \
                     patch.object(server.os, 'killpg') as signalled:
                    with self.assertRaises(FileExistsError):
                        anyio.run(supervisor.run, 'linear_transport', gid, lifecycle_request('linear_transport'))
                started.assert_not_called()
                signalled.assert_not_called()
                self.assertEqual(collision.read_bytes(), b'preexisting request sentinel')
                self.assertEqual({path.name for path in Path(store.path).iterdir()}, before)
                self.assertEqual(supervisor.audit, [])
                self.assertIsNone(supervisor.active)
                self.assert_source_and_sentinel(store, gid, output, source_before)
            finally:
                collision.unlink()

    def test_legacy_failed_launch_removes_only_owned_request(self):
        for operation in LEGACY_OPERATIONS:
            with self.subTest(operation=operation), managed_store() as (store, gid, output):
                supervisor = server.Supervisor(store)
                source_before = read_at(store.fd, gid)
                request = lifecycle_request(operation)
                created = []
                def fail_launch(command, **kwargs):
                    index = command.index(operation)
                    self.assertFalse(Path(command[index + 2]).exists())
                    if request is not None:
                        path = Path(command[index + 4])
                        self.assertEqual(path.read_bytes(), canonical(request))
                        created.append(path)
                    raise OSError('controlled launch failure')
                with patch.object(server.subprocess, 'Popen', side_effect=fail_launch) as started, \
                     patch.object(server.os, 'killpg') as signalled:
                    with self.assertRaisesRegex(OSError, 'controlled launch failure'):
                        anyio.run(supervisor.run, operation, gid, request)
                self.assertEqual(started.call_count, 1)
                signalled.assert_not_called()
                self.assertEqual(len(created), 1 if request is not None else 0)
                self.assertTrue(all(not path.exists() for path in created))
                self.assertEqual({path.name for path in Path(store.path).iterdir()}, {gid})
                self.assertEqual(supervisor.audit, [])
                self.assertIsNone(supervisor.active)
                self.assert_source_and_sentinel(store, gid, output, source_before)

    def test_legacy_duplicate_artifact_retains_descriptor_identity(self):
        with managed_store() as (store, gid, output):
            payload = {'controlled_lifecycle_payload': [1, 2, 3]}
            first = store.artifact(gid, payload, PROVENANCE, kind='topology_metrics')
            again = store.artifact(gid, copy.deepcopy(payload), PROVENANCE, kind='topology_metrics')
            self.assertIs(again, first)
            self.assertIs(again, store.info(first['artifact_id']))
            self.assertEqual(len(store.artifacts), 1)

    def test_preparation_duplicate_reply_remains_detached_from_store(self):
        async def exercise(store, gid):
            value = bundle()
            value['source_sha256'] = store.graphs[gid]['source_sha256']
            first = await server.publish_preparation(store, gid, value, PROVENANCE, lambda reply: None)
            aid = first['artifact']['artifact_id']
            stored = store.info(aid)
            stored_before = canonical(stored)
            source_before = read_at(store.fd, gid)
            artifact_before = read_at(store.fd, aid)
            total_before = store.total
            commit = store.commit_artifact
            def existing_commit(stage):
                self.assertTrue(stage.existing)
                result = commit(stage)
                self.assertIs(result, store.info(aid))
                return result
            with patch.object(store, 'commit_artifact', side_effect=existing_commit) as committed:
                duplicate = await server.publish_preparation(store, gid, value, PROVENANCE, lambda reply: None)
            self.assertEqual(committed.call_count, 1)
            self.assertEqual(duplicate, first)
            self.assertIsNot(duplicate['artifact'], stored)
            self.assertIsNot(duplicate['artifact']['producer'], stored['producer'])
            self.assertIsNot(duplicate['artifact'], first['artifact'])
            duplicate['artifact']['sha256'] = 'f' * 64
            duplicate['artifact']['producer']['adapter_version'] = 'caller mutation'
            self.assertEqual(canonical(store.info(aid)), stored_before)
            self.assertEqual(canonical(first['artifact']), stored_before)
            self.assertEqual(read_at(store.fd, aid), artifact_before)
            self.assertEqual(read_at(store.fd, gid), source_before)
            self.assertEqual(store.total, total_before)
            self.assertEqual(len(store.artifacts), 1)
        with managed_store() as (store, gid, output):
            anyio.run(exercise, store, gid)
            self.assertEqual((output / 'unrelated.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main(verbosity=2)
