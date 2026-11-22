"""Provider-free tests of the standalone isolated BCC example."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/taichi_isolated_bcc.py"
spec = importlib.util.spec_from_file_location("bcc_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class IsolatedBCCExampleTests(unittest.TestCase):
    def runtime_args(self, parent):
        return ["--worker-python", str(parent / "env" / "python"),
                "--worker-site", str(parent / "env" / "site"),
                "--native-sha256", "a" * 64]

    def test_preflight_checks_actual_generated_request_without_starting(self):
        from fibernet.sim import isolated_taichi

        output = io.StringIO()
        with patch.object(isolated_taichi, "start_isolated_taichi") as start:
            with redirect_stdout(output):
                example.main(["--preflight"])
            start.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual((report["nodes"], report["rods"], len(report["fixed_node_ids"])),
                         (546, 2542, 42))
        self.assertNotIn(report["loaded_node_id"], report["fixed_node_ids"])
        self.assertGreater(report["request_bytes"], 0)
        self.assertLess(report["request_bytes"], 8 * 2**20)
        self.assertIs(report["worker_started"], False)

    def test_default_and_bad_runtime_never_start(self):
        from fibernet.sim import isolated_taichi

        with tempfile.TemporaryDirectory() as temporary:
            full = self.runtime_args(Path(temporary).resolve())
            with patch.object(example, "prepare", return_value=(object(), {}, {})) as prepare:
                with patch.object(isolated_taichi, "start_isolated_taichi") as start:
                    with redirect_stdout(io.StringIO()):
                        example.main([])
                    prepare.assert_called_once_with()
                    prepare.reset_mock()
                    for args in (full[:2], full + ["--preflight"],
                                 [*full[:1], "relative/python", *full[2:]], [*full[:-1], "bad"]):
                        with self.subTest(args=args), redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as exc:
                                example.main(args)
                            self.assertEqual(exc.exception.code, 2)
                    prepare.assert_not_called()
                    start.assert_not_called()

    def check_task_cleanup(self, error_at):
        from fibernet.sim import isolated_taichi

        class MockTask:
            def __init__(self, workdir):
                self.workdir = workdir
                self.closed_with_files = False

            def __enter__(self):
                return self

            def result(self, timeout):
                self.timeout = timeout
                if error_at == "result":
                    raise RuntimeError("mock result error")
                if error_at == "interrupt":
                    raise KeyboardInterrupt
                return object()

            def __exit__(self, *exc):
                self.closed_with_files = self.workdir.is_dir()
                if error_at == "close":
                    raise RuntimeError("mock close error")

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            taskdir = parent / "worker-output"
            taskdir.mkdir()
            evidence = taskdir / "worker.log"
            evidence.write_text("worker evidence", encoding="utf-8")
            unrelated = parent / "unrelated"
            unrelated.write_text("must stay", encoding="utf-8")
            task = MockTask(taskdir)
            with patch.object(example, "prepare", return_value=(object(), {}, {})):
                with patch.object(example, "summarize", return_value={"mode": "completed"}):
                    with patch.object(isolated_taichi, "start_isolated_taichi", return_value=task) as start:
                        with redirect_stdout(io.StringIO()) as stdout:
                            if error_at == "interrupt":
                                with self.assertRaises(KeyboardInterrupt):
                                    example.main(self.runtime_args(parent))
                            elif error_at:
                                with self.assertRaisesRegex(RuntimeError, f"mock {error_at} error"):
                                    example.main(self.runtime_args(parent))
                            else:
                                example.main(self.runtime_args(parent))
            self.assertTrue(task.closed_with_files)
            self.assertEqual(task.timeout, 45)
            self.assertTrue(unrelated.is_file())
            self.assertEqual(start.call_args.kwargs["wall_seconds"], 35)
            if error_at:
                self.assertEqual(evidence.read_text(encoding="utf-8"), "worker evidence")
                self.assertEqual(stdout.getvalue(), "")
            else:
                self.assertFalse(taskdir.exists())
                self.assertEqual(json.loads(stdout.getvalue()), {"mode": "completed"})

    def test_result_error_preserves_worker_evidence(self):
        self.check_task_cleanup("result")

    def test_close_error_preserves_worker_evidence(self):
        self.check_task_cleanup("close")

    def test_interrupted_result_preserves_worker_evidence(self):
        self.check_task_cleanup("interrupt")

    def test_success_removes_only_owned_directory_after_close(self):
        self.check_task_cleanup(None)
