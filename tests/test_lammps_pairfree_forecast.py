"""Provider-free storage and input-identity regressions for pair-free CG.

Run with ``python -m unittest discover -s tests -p 'test_lammps*py'`` or
pytest. No provider runtime, private checkout, or native execution is needed.
"""
import ast
import builtins
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import (
    PreparedRun, ResourceLimits, RuntimeSpec, forecast_cg_fiber_only, prepare_cg,
)
from fibernet.integrations import _harmonic_profile as harmonic
from fibernet.integrations import _cg_network_profile as network
from fibernet.integrations import lammps_backend, lammps_planning
from fibernet.integrations._lammps_forecast import payload_counts

from _lammps_forecast_test_support import (
    build_branched_case, build_crosslinked_case, capped_case, inert_runtime,
)


FIXTURES = Path(__file__).with_name('fixtures')
LIMITS = ResourceLimits(5, 10., 512 * 1024**2, 16 * 1024**2, 16 * 1024**2)
EXPORT = 64 * 1024**2


class TestPairFreeForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='pairfree-forecast-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.runtime = inert_runtime(cls.directory.name)
        graph, request = build_branched_case(steps=200, timestep_s=5e-8)
        cls.small = prepare_cg(graph, request)
        graph, request = capped_case()
        cls.capped = prepare_cg(graph, request)

    def plan(self, prepared, **kwargs):
        return forecast_cg_fiber_only(
            prepared, self.runtime, LIMITS, export_budget_bytes=EXPORT, **kwargs
        )

    def test_existing_harmonic_forecasts_exact(self):
        golden = json.loads((FIXTURES / 'lammps_forecast_legacy_values.json').read_text())
        for case in golden['harmonic']:
            with self.subTest(args=case['args']):
                self.assertEqual(
                    harmonic.resource_forecast(*case['args']), case['expected']
                )

    def test_existing_network_forecasts_exact(self):
        golden = json.loads((FIXTURES / 'lammps_forecast_legacy_values.json').read_text())
        for case in golden['network']:
            with self.subTest(args=case['args']):
                self.assertEqual(network.forecast(*case['args']), case['expected'])

    def test_run_zero_zero_edges(self):
        graph, request = capped_case(n=2, edge_count=0, angle_count=0, steps=0)
        result = self.plan(prepare_cg(graph, request))
        self.assertEqual(
            tuple(result[key] for key in ('nodes', 'edges', 'angles', 'native_frames')),
            (2, 0, 0, 1),
        )
        self.assertEqual(result['thermo_frames'], 1)
        self.assertTrue(result['fits_declared_budgets'])
        self.assertFalse(result['execution_admission'])

    def test_existing_cases_and_exact_setup_bytes(self):
        for steps, timestep in ((0, 1e-7), (100, 1e-7), (200, 5e-8)):
            with self.subTest(case='branched', steps=steps):
                graph, request = build_branched_case(steps=steps, timestep_s=timestep)
                prepared = prepare_cg(graph, request)
                result = self.plan(prepared)
                self.assertEqual(
                    (result['native_frames'], result['output_frames']),
                    (steps + 1, steps + 1),
                )
                runtime = {key: getattr(self.runtime, key)
                           for key in RuntimeSpec.__dataclass_fields__}
                limits = {key: getattr(LIMITS, key)
                          for key in ResourceLimits.__dataclass_fields__}
                manifest = json.dumps(
                    dict(prepared.manifest(), runtime=runtime, limits=limits),
                    sort_keys=True, separators=(',', ':'), allow_nan=False,
                )
                exact_bytes = sum(len(value.encode('utf-8')) for value in (
                    prepared.graph_json, prepared.data_text, prepared.input_text, manifest
                ))
                self.assertEqual(result['prepared_artifact_bytes'], exact_bytes)
                self.assertEqual(result['pair_exclusion_commands'], 28)
        for steps, timestep in ((0, 1e-7), (500, 1e-7), (1000, 5e-8)):
            with self.subTest(case='crosslinked', steps=steps):
                _, graph, request = build_crosslinked_case(steps=steps, timestep_s=timestep)
                result = self.plan(prepare_cg(graph, request))
                self.assertEqual(
                    tuple(result[key] for key in ('nodes', 'edges', 'angles')), (48, 104, 64)
                )
                self.assertEqual(result['pair_exclusion_commands'], 1176)
                self.assertEqual(
                    result['native_type_pair_cutoff_payload_lower_bound_bytes'], 38416
                )

    def test_retained_prepared_and_native_input_archive_hashes(self):
        # Retain exact archive integrity independently of runtime regeneration.
        # math.dist rounding changed in Python 3.10; the derived initial bending
        # energy and completion marker need not match across Python versions.
        fixture = json.loads((FIXTURES / 'lammps_pairfree_prepared_inputs.json').read_text())
        for name, case in fixture['cases'].items():
            with self.subTest(case=name):
                prepared = PreparedRun(
                    fixture['graph_json'], case['contract_json'], fixture['data_text'],
                    case['input_text'], tuple(fixture['node_ids']), tuple(fixture['edge_ids']),
                    case['timestep_s'], case['steps'], 1,
                )
                self.assertEqual(prepared.identity, case['prepared_sha256'])
                for attr, key in (('graph_json', 'graph_sha256'),
                                  ('data_text', 'model_data_sha256'),
                                  ('input_text', 'input_sha256')):
                    self.assertEqual(
                        hashlib.sha256(getattr(prepared, attr).encode('utf-8')).hexdigest(),
                        case[key],
                    )

    def test_current_runtime_prepared_inputs_remain_exact_and_unchanged(self):
        from fibernet.integrations.lammps_backend import validate_prepared

        for steps, timestep in ((100, 1e-7), (200, 5e-8), (0, 1e-7)):
            with self.subTest(steps=steps, timestep=timestep):
                graph, request = build_branched_case(steps=steps, timestep_s=timestep)
                prepared = prepare_cg(graph, request)
                repeated = prepare_cg(graph, request)
                before = {name: getattr(prepared, name)
                          for name in PreparedRun.__dataclass_fields__}
                for field, value in before.items():
                    self.assertEqual(value, getattr(repeated, field), msg=field)
                self.assertIs(validate_prepared(prepared), prepared)
                result = self.plan(prepared)
                self.assertEqual(before, {name: getattr(prepared, name) for name in before})
                self.assertEqual(result['source_identity']['prepared_sha256'], prepared.identity)
                self.assertEqual(result['source_identity']['graph_sha256'],
                                 hashlib.sha256(prepared.graph_json.encode('utf-8')).hexdigest())
                # Valid-looking changes still fail exact compiler revalidation.
                contract = json.loads(prepared.contract_json)
                contract['initial_bending_energy_J'] += 1e-18
                for altered in (
                    replace(prepared, input_text=prepared.input_text + '# altered\n'),
                    replace(prepared, contract_json=json.dumps(
                        contract, sort_keys=True, separators=(',', ':'))),
                ):
                    with self.assertRaises(ValueError):
                        self.plan(altered)

    def test_worst_caps_1001_frames_without_arrays(self):
        import numpy as np
        with patch.object(np, 'zeros', side_effect=AssertionError('trajectory allocation forbidden')), \
             patch.object(np, 'empty', side_effect=AssertionError('trajectory allocation forbidden')):
            result = self.plan(self.capped)
        self.assertEqual(
            tuple(result[key] for key in ('nodes', 'edges', 'angles', 'native_frames')),
            (64, 128, 128, 1001),
        )
        self.assertEqual(result['trajectory_bytes'], 16336352)
        self.assertEqual(result['pair_exclusion_commands'], 2080)
        self.assertEqual(result['native_type_pair_cutoff_payload_lower_bound_bytes'], 67600)
        self.assertGreater(result['native_aggregate_forecast_bytes'], 16 * 1024**2)
        self.assertFalse(result['budget_comparison']['native_aggregate'])
        self.assertFalse(result['fits_declared_budgets'])
        self.assertFalse(result['execution_admission'])

    def test_budget_edges_and_export_is_explicit(self):
        result = self.plan(self.small)
        exact = result['detailed_export_forecast_bytes']
        for cap, expected in ((exact - 1, False), (exact, True), (exact + 1, True)):
            with self.subTest(export_budget=cap):
                actual = forecast_cg_fiber_only(
                    self.small, self.runtime, LIMITS, export_budget_bytes=cap
                )
                self.assertEqual(actual['budget_comparison']['detailed_export'], expected)
        trajectory = result['trajectory_bytes']
        for cap, expected in ((trajectory - 1, False), (trajectory, True), (trajectory + 1, True)):
            with self.subTest(file_budget=cap):
                actual = forecast_cg_fiber_only(
                    self.small, self.runtime, replace(LIMITS, file_size_bytes=cap),
                    export_budget_bytes=EXPORT,
                )
                self.assertEqual(actual['budget_comparison']['trajectory_file'], expected)
        for bad in (True, 0, -1, 1.5, 2**63):
            with self.subTest(invalid_export_budget=bad), self.assertRaises(ValueError):
                forecast_cg_fiber_only(self.small, self.runtime, LIMITS, export_budget_bytes=bad)
        with self.assertRaises(TypeError):
            forecast_cg_fiber_only(self.small, self.runtime, LIMITS)

    def test_valid_low_output_budgets_return_nonfitting_forecast(self):
        graph, request = build_branched_case(steps=0)
        prepared = prepare_cg(graph, request)
        for budget in (1024, 4096, 8192):
            with self.subTest(output_bytes=budget):
                limits = replace(LIMITS, output_bytes=budget)
                self.assertIs(limits.checked(), limits)
                result = forecast_cg_fiber_only(
                    prepared, self.runtime, limits, export_budget_bytes=EXPORT
                )
                self.assertFalse(result['budget_comparison']['native_aggregate'])
                self.assertFalse(result['fits_declared_budgets'])
                self.assertFalse(result['execution_admission'])
                self.assertEqual(result['limits']['output_bytes'], budget)
        # The manifest contains the selected budget, so stabilize its decimal
        # representation before checking the exact aggregate byte boundary.
        budget = LIMITS.output_bytes
        for _ in range(8):
            result = forecast_cg_fiber_only(
                prepared, self.runtime, replace(LIMITS, output_bytes=budget),
                export_budget_bytes=EXPORT,
            )
            required = result['native_aggregate_forecast_bytes']
            if required == budget:
                break
            budget = required
        else:
            self.fail('aggregate byte-count fixed point did not stabilize')
        for cap, expected in ((budget - 1, False), (budget, True), (budget + 1, True)):
            with self.subTest(aggregate_budget=cap):
                result = forecast_cg_fiber_only(
                    prepared, self.runtime, replace(LIMITS, output_bytes=cap),
                    export_budget_bytes=EXPORT,
                )
                self.assertEqual(result['budget_comparison']['native_aggregate'], expected)
        for invalid in (1023, 16 * 1024**2 + 1):
            with self.subTest(invalid_output_budget=invalid), self.assertRaises(ValueError):
                forecast_cg_fiber_only(
                    prepared, self.runtime, replace(LIMITS, output_bytes=invalid),
                    export_budget_bytes=EXPORT,
                )

    def test_structural_serialization_limits_stay_bounded(self):
        result = self.plan(self.small)
        self.assertEqual(result['structural_planning_limits'], {
            'runtime_configuration_bytes': 1024**2, 'counted_text_bytes': 4 * 1024**2,
        })
        runtime = replace(self.runtime, expected_banner='x' * (1024**2 + 1))
        for cap in (1024, 16 * 1024**2):
            with self.subTest(output_budget=cap), self.assertRaisesRegex(ValueError, 'budget'):
                forecast_cg_fiber_only(
                    self.small, runtime, replace(LIMITS, output_bytes=cap),
                    export_budget_bytes=EXPORT,
                )

    def test_configuration_one_mib_boundary(self):
        limit = 1024**2
        runtime = replace(self.runtime, expected_banner='')
        record = {name: getattr(runtime, name) for name in RuntimeSpec.__dataclass_fields__}
        available = limit - harmonic.json_bytes(record, limit)
        exact = replace(runtime, expected_banner='x' * available)
        self.assertEqual(
            harmonic.json_bytes(lammps_planning._runtime_record(exact, limit), limit), limit
        )
        with self.assertRaisesRegex(ValueError, 'budget'):
            lammps_planning._runtime_record(
                replace(exact, expected_banner=exact.expected_banner + 'x'), limit
            )

    def test_counted_text_four_mib_boundary_independent_of_output_budget(self):
        limit = 4 * 1024**2
        self.assertEqual(harmonic.string_bytes(('x' * limit,), limit), limit)
        with self.assertRaisesRegex(ValueError, 'budget'):
            harmonic.string_bytes(('x' * limit, 'x'), limit)
        # Inject only a malformed, oversize metadata producer to exercise the
        # planner's guard. A caller's tiny or maximum output budget must not
        # replace the independent structural 4 MiB ceiling.
        oversized_metadata = {'text': 'x' * limit}
        for cap in (1024, 16 * 1024**2):
            with self.subTest(output_budget=cap), \
                 patch.object(harmonic, 'export_metadata', return_value=oversized_metadata), \
                 self.assertRaisesRegex(ValueError, 'budget'):
                forecast_cg_fiber_only(
                    self.small, self.runtime, replace(LIMITS, output_bytes=cap),
                    export_budget_bytes=EXPORT,
                )

    def test_oversized_runtime_prewalk_precedes_duplicate_set(self):
        # Repeated short pairs exceed the structural cap before the duplicate
        # key set is built. The failure must be the byte budget, not duplicates.
        runtime = replace(self.runtime, file_sha256=(('same', 'hash'),) * 200000)
        with self.assertRaisesRegex(ValueError, 'budget'):
            forecast_cg_fiber_only(self.small, runtime, LIMITS, export_budget_bytes=EXPORT)

    def test_record_false_full_transient_accounting_and_unknowns(self):
        unrecorded = self.plan(self.small, record_state=False)
        recorded = self.plan(self.small, record_state=True)
        self.assertEqual(unrecorded['transient_state_scope'],
                         'raw parsed history remains retained; record_state=False avoids full converted-state materialization; memory estimates remain conservative')
        self.assertEqual(recorded['transient_state_scope'], unrecorded['transient_state_scope'])
        self.assertFalse(unrecorded['record_state'])
        self.assertTrue(recorded['record_state'])
        self.assertEqual(unrecorded['planned_memory_bytes'], recorded['planned_memory_bytes'])
        self.assertEqual(unrecorded['parsed_state_memory_bytes'], 201 * (9 * 7 + 16) * 64)
        self.assertEqual(unrecorded['runtime_readiness'], 'not_checked')
        for name in ('actual_cpu_seconds', 'actual_peak_rss_bytes', 'predicted_cpu_seconds'):
            self.assertIsNone(unrecorded[name])
        self.assertIn('not enforced', unrecorded['resource_semantics'])

    def test_inert_no_provider_paths_processes_threads_or_writes(self):
        original_import = builtins.__import__
        blocked = {'lammps', 'taichi', 'torch', 'torch_geometric', 'exudyn', 'pybullet', 'mujoco'}

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in blocked:
                raise AssertionError('provider import forbidden: ' + name)
            return original_import(name, *args, **kwargs)

        # All patches are scoped to this call, never applied during collection.
        # Runtime placeholders are assembled in setUpClass, not resolved/opened.
        targets = [
            (RuntimeSpec, 'checked'), (lammps_backend.LAMMPSTask, '__init__'),
            (subprocess, 'Popen'), (threading.Thread, 'start'),
            (builtins, 'open'), (io, 'open'), (os, 'open'),
            (Path, 'open'), (Path, 'resolve'), (Path, 'mkdir'),
            (Path, 'stat'), (Path, 'is_file'), (Path, 'is_dir'),
        ]
        with ExitStack() as stack:
            stack.enter_context(patch.object(builtins, '__import__', side_effect=guarded_import))
            for owner, name in targets:
                stack.enter_context(patch.object(
                    owner, name, side_effect=AssertionError('side effect forbidden: ' + name)
                ))
            result = self.plan(self.small)
        self.assertEqual(result['runtime_readiness'], 'not_checked')
        self.assertFalse(result['execution_admission'])

    def test_determinism_detachment_and_shuffled_ids(self):
        graph, request = build_branched_case()
        shuffled = graph.to_dict()
        shuffled['nodes'].reverse()
        shuffled['edges'].reverse()
        prepared = prepare_cg(graph, request)
        other = prepare_cg(StructureGraph.from_dict(shuffled), request)
        first, second = self.plan(prepared), self.plan(other)
        for key in ('nodes', 'edges', 'angles', 'native_frames', 'trajectory_bytes', 'planned_memory_bytes'):
            self.assertEqual(first[key], second[key])
        original_identity = prepared.identity
        first['current_caps']['max_nodes'] = 999
        first['limits']['memory_bytes'] = 0
        first['source_identity']['prepared_sha256'] = 'changed'
        first['budget_comparison']['native_aggregate'] = 'changed'
        fresh = self.plan(prepared)
        self.assertEqual(fresh['current_caps']['max_nodes'], 64)
        self.assertEqual(fresh['limits']['memory_bytes'], LIMITS.memory_bytes)
        self.assertEqual(fresh['source_identity']['prepared_sha256'], original_identity)
        self.assertIs(type(fresh['budget_comparison']['native_aggregate']), bool)
        self.assertEqual(LIMITS.memory_bytes, 512 * 1024**2)

    def test_malformed_and_unsupported_contracts(self):
        for change in (
            {'profile': 'cg_network_sampled_v1'}, {'steps': 1001}, {'sample_every': 2},
            {'node_ids': self.small.node_ids + (999,)}, {'contract_json': '{}'},
        ):
            with self.subTest(prepared_change=change), self.assertRaises((ValueError, KeyError, TypeError)):
                self.plan(replace(self.small, **change))
        for bad in (1, None, 'true'):
            with self.subTest(record_state=bad), self.assertRaises(ValueError):
                self.plan(self.small, record_state=bad)
        for change in (
            {'expected_version': True}, {'mode': 'python'},
            {'file_sha256': (('x', object()),)}, {'environment': (('A', '1'), ('A', '2'))},
        ):
            with self.subTest(runtime_change=change), self.assertRaises(ValueError):
                forecast_cg_fiber_only(
                    self.small, replace(self.runtime, **change), LIMITS, export_budget_bytes=EXPORT
                )
        with self.assertRaises(ValueError):
            self.plan(replace(self.small, graph_json='x' * (1024**2 + 1)))

    def test_preparation_caps_not_relaxed(self):
        for nodes, edges, angles, steps in (
            (65, 1, 0, 0), (64, 129, 0, 0), (64, 128, 129, 0), (2, 0, 0, 1001),
        ):
            graph, request = capped_case(nodes, edges, angles, steps)
            with self.subTest(nodes=nodes, edges=edges, angles=angles, steps=steps):
                self.assertEqual(
                    (len(graph.nodes), len(graph.edges), len(request.angles), request.run.steps),
                    (nodes, edges, angles, steps),
                )
                with self.assertRaises(ValueError):
                    prepare_cg(graph, request)
        graph, request = build_branched_case()
        for change in (
            {'profile': 'cg_network_sampled_v1'},
            {'pair_encoding': 'static-groups-zero-cutoff/1'},
            {'run': replace(request.run, sample_every=2)},
        ):
            with self.subTest(request_change=change), self.assertRaises(ValueError):
                prepare_cg(graph, replace(request, **change))

    def test_shared_helper_refuses_bad_counts(self):
        common = dict(
            row_bytes=239, prepared_bytes=0, setup_bytes=0, metadata_reserve_bytes=0,
            raw_reserve_bytes=0, export_reserve_bytes=0,
        )
        for bad in (True, -1, 1.5):
            with self.subTest(node_count=bad), self.assertRaises(ValueError):
                payload_counts(bad, 0, 1, 1, 1, **common)
        with self.assertRaises(ValueError):
            payload_counts(2, 0, 0, 1, 1, **common)

    def test_changed_sources_python39_grammar(self):
        for name in (
            '_lammps_forecast.py', 'lammps_planning.py', '_harmonic_profile.py',
            '_cg_network_profile.py', '__init__.py',
        ):
            with self.subTest(source=name):
                ast.parse(
                    Path(lammps_planning.__file__).with_name(name).read_text(encoding='utf-8'),
                    feature_version=(3, 9),
                )


if __name__ == '__main__':
    unittest.main()
