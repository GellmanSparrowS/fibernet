"""Inert planning only; no physics or native execution evidence."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples import cg_resource_preflight as example
from fibernet.integrations import ResourceLimits, RuntimeSpec, prepare_cg

MIB = 1024**2
RUNTIME = RuntimeSpec(mode='cli', python_executable='/absent/python',
    lammps_executable='/absent/lmp', expected_version=20250722,
    expected_banner='LAMMPS (22 Jul 2025)')


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.mocks = []
        for target in ('fibernet.integrations.RuntimeSpec.checked',
                       'subprocess.Popen',
                       'fibernet.integrations.lammps_backend.LAMMPSTask',
                       'examples.cg_crosslinked_network.simulate',
                       'examples.cg_bent_branched_network.simulate',
                       'pathlib.Path.mkdir'):
            self.mocks.append(self.stack.enter_context(patch(target,
                side_effect=AssertionError('forbidden side effect: '+target))))

    def report(self, steps, limits, export=16*MIB):
        _, graph, request = example.build_case(steps=steps,
            timestep_s=5e-8 if steps == 1000 else 1e-7)
        prepared = prepare_cg(graph, request)
        with ExitStack() as guard:
            for target in ('builtins.open', 'io.open', 'os.open', 'pathlib.Path.open',
                           'pathlib.Path.stat', 'pathlib.Path.resolve'):
                guard.enter_context(patch(target,
                    side_effect=AssertionError('preflight filesystem access: '+target)))
            return example.preflight(prepared, RUNTIME,
                limits=limits, export_budget_bytes=export)

    def assert_inert(self, report):
        self.assertIs(report['execution_admission'], False)
        self.assertEqual(report['runtime_readiness'], 'not_checked')
        self.assertEqual(report['whole_workflow_resource_acceptance'], 'not_established')
        self.assertIs(report['forecast']['execution_admission'], False)
        self.assertIsNone(report['forecast']['actual_cpu_seconds'])
        self.assertIsNone(report['forecast']['actual_peak_rss_bytes'])
        for mock in self.mocks:
            mock.assert_not_called()

    def test_fine_default_native_refusal(self):
        report = self.report(1000, ResourceLimits())
        self.assertEqual(report['status'], 'refused')
        self.assertIn('native_aggregate', report['failed_comparisons'])
        self.assert_inert(report)

    def test_both_grids_refuse_16_mib_export(self):
        limits = ResourceLimits(cpu_seconds=10, wall_seconds=20.,
            output_bytes=16*MIB, file_size_bytes=16*MIB)
        for steps in (500, 1000):
            with self.subTest(steps=steps):
                report = self.report(steps, limits)
                self.assertEqual(report['status'], 'refused')
                self.assertIn('detailed_export', report['failed_comparisons'])
                self.assertEqual(report['comparisons']['detailed_export']['budget_bytes'], 16*MIB)
                self.assert_inert(report)

    def test_fit_is_not_admission_and_comparisons_are_exact(self):
        report = self.report(0, ResourceLimits())
        self.assertEqual(report['status'], 'storage_fit_only')
        self.assertEqual(report['failed_comparisons'], [])
        for name, comparison in report['comparisons'].items():
            self.assertEqual(comparison['fits'], comparison['forecast_bytes'] <= comparison['budget_bytes'])
            self.assertEqual(comparison['fits'], report['forecast']['budget_comparison'][name])
        self.assert_inert(report)

    def test_export_boundary_and_memory_refusal(self):
        initial = self.report(0, ResourceLimits())
        required = initial['comparisons']['detailed_export']['forecast_bytes']
        self.assertTrue(self.report(0, ResourceLimits(), required)['comparisons']['detailed_export']['fits'])
        self.assertFalse(self.report(0, ResourceLimits(), required-1)['comparisons']['detailed_export']['fits'])
        small_memory = replace(ResourceLimits(), memory_bytes=64*MIB)
        report = self.report(1000, small_memory)
        self.assertIn('modeled_memory_envelope', report['failed_comparisons'])
        self.assert_inert(report)

    def test_explicit_budgets_and_existing_caps(self):
        for export in (None, True, 0, -1):
            with self.subTest(export=export), self.assertRaises(ValueError):
                self.report(0, ResourceLimits(), export)
        with self.assertRaises(ValueError):
            self.report(0, replace(ResourceLimits(), output_bytes=17*MIB))
        with self.assertRaises(ValueError):
            self.report(2000, ResourceLimits())
        _, graph, request = example.build_case(steps=0)
        with self.assertRaises(TypeError):
            example.preflight(prepare_cg(graph, request), RUNTIME, export_budget_bytes=MIB)

    def test_cli_fit_and_refusal_without_output(self):
        # Temporary config files are test inputs, never generated solver output.
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder)/'runtime.json'
            limits = Path(folder)/'limits.json'
            runtime.write_text(json.dumps(asdict(RUNTIME)))
            limits.write_text(json.dumps(asdict(ResourceLimits())))
            for steps, code in ((0, 0), (1000, 1)):
                output = io.StringIO()
                with redirect_stdout(output):
                    actual = example.main(['--runtime-json', str(runtime),
                        '--limits-json', str(limits), '--export-budget-bytes', str(16*MIB),
                        '--steps', str(steps)])
                self.assertEqual(actual, code)
                self.assert_inert(json.loads(output.getvalue()))
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ['limits.json','runtime.json'])
            for malformed in (['ab'], {'ab': 1}, [[1, 'b']]):
                record = asdict(RUNTIME)
                record['environment'] = malformed
                runtime.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    _, graph, request = example.build_case(steps=0)
                    example.preflight(prepare_cg(graph, request), example.load_runtime(runtime),
                        limits=ResourceLimits(), export_budget_bytes=16*MIB)
            limits.write_text('{}')
            with self.assertRaises(ValueError):
                example.load_limits(limits)
            limits.write_text('{"cpu_seconds": 5, "cpu_seconds": 6}')
            with self.assertRaisesRegex(ValueError, 'duplicate configuration key'):
                example.load_limits(limits)
            runtime.write_bytes(b' '*65537)
            with self.assertRaises(ValueError):
                example.load_runtime(runtime)
            limits.write_bytes(b' '*65537)
            with self.assertRaises(ValueError):
                example.load_limits(limits)


if __name__ == '__main__':
    unittest.main()
