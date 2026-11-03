"""Batch input preflight, sized-iterable compatibility and report restrictions.

Generation and simulation use controlled stubs; these tests do not exercise a
physical solver.
"""
import csv
import json
import tempfile
import unittest
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fibernet.easy as easy


class SizedOnce:
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("input iterated twice")
        return iter(self.values)


class DictChild(dict):
    pass


class DuckConfig:
    def get(self, key, default=None):
        return default


class NonCallableGet:
    get = 1


class BatchInputPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "results.csv"
        self.report = self.root / "report.json"
        self.generated = []
        self.simulated = []

        def generate(**kwargs):
            self.generated.append(kwargs)
            if kwargs['unit'] == 'badgen':
                raise ValueError('bad generation')
            return SimpleNamespace(num_nodes=3, num_edges=2, unit=kwargs['unit'])

        def simulate(graph, **kwargs):
            self.simulated.append(graph)
            if graph.unit == 'badsim':
                raise RuntimeError('bad simulation')
            return SimpleNamespace(max_force=1., max_displacement=2., energy=3.)

        for name, replacement in [('pattern_2d', generate), ('simulate', simulate)]:
            patcher = patch.object(easy, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_batch(self, configs, report=False):
        kwargs = {'report_path': self.report} if report else {}
        return easy.batch_simulate(configs, str(self.output), **kwargs)

    def assert_no_work(self):
        self.assertEqual(self.generated, [])
        self.assertEqual(self.simulated, [])

    def test_first_and_later_bad_entries_preflight_both_routes(self):
        for reporting in (False, True):
            for configs, index in (([None, {}], 0), ([{}, None], 1),
                                   ([{}, NonCallableGet()], 1)):
                with self.subTest(reporting=reporting, index=index, configs=configs):
                    self.output.write_bytes(b'old csv')
                    self.report.write_bytes(b'old report')
                    # Reports retain the existing report-specific error contract.
                    message = 'JSON-safe' if reporting else rf'configs\[{index}\]'
                    with self.assertRaisesRegex(ValueError, message):
                        self.run_batch(configs, report=reporting)
                    self.assert_no_work()
                    self.assertEqual(self.output.read_bytes(), b'old csv')
                    self.assertEqual(self.report.read_bytes(), b'old report')

    def test_unsized_inputs_rejected_before_iteration_or_work(self):
        def unsized():
            raise AssertionError('unsized input must not be consumed')
            yield {}
        for configs in (None, 1, unsized()):
            with self.subTest(configs=configs):
                with self.assertRaisesRegex(ValueError, 'sized iterable'):
                    self.run_batch(configs)
                self.assert_no_work()
                self.assertFalse(self.output.exists())

    def test_sized_noniterable_rejected_before_work(self):
        class SizedOnly:
            def __len__(self):
                return 1
        with self.assertRaisesRegex(ValueError, 'sized iterable'):
            self.run_batch(SizedOnly())
        self.assert_no_work()

    def test_sized_one_shot_snapshotted_once(self):
        configs = SizedOnce([{}, {}])
        self.assertEqual(self.run_batch(configs), str(self.output))
        self.assertEqual(configs.iterations, 1)
        self.assertEqual(len(self.generated), 2)
        self.assertEqual(len(self.simulated), 2)

    def test_tuple_mapping_subclass_and_duck_compatibility(self):
        self.run_batch((DictChild(), UserDict(), DuckConfig()))
        self.assertEqual(len(self.generated), 3)
        self.assertEqual(len(self.simulated), 3)
        with self.output.open(newline='') as stream:
            self.assertEqual([row['id'] for row in csv.DictReader(stream)], ['0', '1', '2'])

    def test_default_non_json_values_still_allowed(self):
        self.run_batch([{'unused': object()}])
        self.assertEqual(len(self.simulated), 1)
        self.assertFalse(self.report.exists())

    def test_report_keeps_original_plain_dict_and_json_contract(self):
        for configs in ([DictChild()], [UserDict()], [DuckConfig()],
                        [{'unused': object()}], SizedOnce([{}])):
            with self.subTest(configs=configs):
                with self.assertRaisesRegex(ValueError, 'JSON-safe'):
                    self.run_batch(configs, report=True)
                self.assert_no_work()
                if isinstance(configs, SizedOnce):
                    self.assertEqual(configs.iterations, 0)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())

    def test_empty_preserves_existing_behavior(self):
        self.output.write_bytes(b'old csv')
        for reporting in (False, True):
            with self.subTest(reporting=reporting):
                with self.assertRaisesRegex(RuntimeError, 'All batch'):
                    self.run_batch([], report=reporting)
                self.assert_no_work()
                self.assertEqual(self.output.read_bytes(), b'old csv')
        report = json.loads(self.report.read_text())
        self.assertEqual(report['cases'], [])
        self.assertEqual(report['status'], 'all_failed')

    def test_mixed_case_errors_remain_case_errors(self):
        self.run_batch([{'unit': 'badgen'}, {}, {'unit': 'badsim'}], report=True)
        self.assertEqual(len(self.generated), 3)
        self.assertEqual(len(self.simulated), 2)
        report = json.loads(self.report.read_text())
        self.assertEqual(report['success_indices'], [1])
        self.assertEqual(report['failure_count'], 2)
        self.assertEqual([case.get('stage') for case in report['cases']],
                         ['generation', None, 'simulation'])
        with self.output.open(newline='') as stream:
            self.assertEqual([row['id'] for row in csv.DictReader(stream)], ['1'])


if __name__ == '__main__':
    unittest.main()
