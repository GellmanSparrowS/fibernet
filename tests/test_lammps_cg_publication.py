"""Portable publication contracts; no native execution or historical rerun."""
import ast
import hashlib
import math
import tempfile
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from examples.cg_bent_branched_network import build_case
from fibernet.integrations import prepare_cg
from fibernet.integrations.lammps_backend import PreparedRun, validate_prepared

ROOT = Path(__file__).resolve().parents[1]


class CGPublicationTests(unittest.TestCase):
    def retained_cases(self):
        record = json.loads((ROOT/'tests/fixtures/lammps_cg_fiber_only_acceptance.json').read_text())
        frozen = json.loads((ROOT/'tests/fixtures/lammps_cg_fiber_only_prepared.json').read_text())
        for name, case in record['cases'].items():
            fields = frozen['cases'][name]
            yield name, case, PreparedRun(
                frozen['graph_json'], fields['contract_json'], frozen['data_text'],
                fields['input_text'], tuple(record['node_ids']), tuple(record['edge_ids']),
                case['timestep_s'], case['steps'], 1)

    def test_retained_native_input_identities(self):
        # These are the exact inputs used by the retained native run, not a
        # promise that different Python math implementations emit the same bytes.
        record = json.loads((ROOT/'tests/fixtures/lammps_cg_fiber_only_acceptance.json').read_text())
        for name, case, prepared in self.retained_cases():
            with self.subTest(case=name):
                self.assertEqual(prepared.identity, case['prepared_sha256'])
                for field, key in [('graph_json', 'graph_sha256'), ('data_text', 'model_data_sha256'), ('input_text', 'input_sha256')]:
                    self.assertEqual(hashlib.sha256(getattr(prepared, field).encode()).hexdigest(), case[key])
        self.assertEqual(hashlib.sha256((ROOT/'fibernet/integrations/_lammps_cg.py').read_bytes()).hexdigest(), record['accepted_compiler_sha256'])

    def assert_portable_inputs(self, prepared, retained):
        self.assertEqual(prepared.graph_json, retained.graph_json)
        self.assertEqual(prepared.data_text, retained.data_text)
        self.assertEqual(json.dumps(prepared.node_ids), json.dumps(retained.node_ids))
        self.assertEqual(json.dumps(prepared.edge_ids), json.dumps(retained.edge_ids))
        actual = json.loads(prepared.contract_json)
        expected = json.loads(retained.contract_json)
        # CPython 3.10 improved the shared math.dist/math.hypot norm algorithm:
        # https://docs.python.org/3/library/math.html#math.hypot
        # In this fixed case only bond 97's derived rest length and the derived
        # initial bending energy differ, each by one ULP on Python 3.9. All
        # other coefficients, stable IDs, orientations, units and flags are exact.
        self.assertEqual(actual['bonds'][2][0], 97)
        self.assertEqual(expected['bonds'][2][0], 97)
        for value, reference in [
            (actual['bonds'][2][1][1], expected['bonds'][2][1][1]),
            (actual['initial_bending_energy_J'], expected['initial_bending_energy_J']),
        ]:
            self.assertIs(type(value), float)
            self.assertTrue(math.isfinite(value))
            self.assertLessEqual(abs(value-reference), math.ulp(reference))
        actual['bonds'][2][1][1] = expected['bonds'][2][1][1]
        actual['initial_bending_energy_J'] = expected['initial_bending_energy_J']
        self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))
        lines = prepared.input_text.splitlines(keepends=True)
        reference_lines = retained.input_text.splitlines(keepends=True)
        self.assertEqual(len(lines), len(reference_lines))
        # Permit the same single derived coefficient in native text, and verify
        # its exact agreement with this runtime's contract before normalization.
        contract = json.loads(prepared.contract_json)
        coefficient = f"bond_coeff 3 {contract['bonds'][2][1][0]/2:.17g} {contract['bonds'][2][1][1]:.17g}\n"
        index = next(i for i, line in enumerate(reference_lines) if line.startswith('bond_coeff 3 '))
        self.assertEqual(lines[index], coefficient)
        lines[index] = reference_lines[index]
        self.assertEqual(lines[-1], f'print "{prepared.completion_marker}"\n')
        lines[-1] = reference_lines[-1]
        self.assertEqual(lines, reference_lines)

    def test_portable_example_and_compiler_roundtrip(self):
        with patch('subprocess.Popen', side_effect=AssertionError('native execution forbidden')):
            for name, case, retained in self.retained_cases():
                with self.subTest(case=name):
                    graph, request = build_case(steps=case['steps'], timestep_s=case['timestep_s'])
                    prepared = prepare_cg(graph, request)
                    self.assert_portable_inputs(prepared, retained)
                    self.assertEqual(validate_prepared(prepared), prepared)
                    self.assertEqual(prepare_cg(graph, request), prepared)

    def test_portable_check_rejects_other_drift(self):
        _, _, retained = next(self.retained_cases())
        for field in ('length_to_m', 'initial_bending_energy_J'):
            with self.subTest(field=field):
                contract = json.loads(retained.contract_json)
                # Even two ULPs are refused for the derived energy; all changes
                # to the unit scale are refused without any numeric tolerance.
                contract[field] = math.nextafter(math.nextafter(contract[field], math.inf), math.inf)
                with self.assertRaises(AssertionError):
                    self.assert_portable_inputs(replace(retained, contract_json=json.dumps(contract)), retained)
        contract = json.loads(retained.contract_json)
        contract['sample_every'] = True
        with self.assertRaises(AssertionError):
            self.assert_portable_inputs(replace(retained, contract_json=json.dumps(contract)), retained)
        contract = json.loads(retained.contract_json)
        length = contract['bonds'][2][1][1]
        contract['bonds'][2][1][1] = math.nextafter(math.nextafter(length, math.inf), math.inf)
        with self.assertRaises(AssertionError):
            self.assert_portable_inputs(replace(retained, contract_json=json.dumps(contract)), retained)
        contract = json.loads(retained.contract_json)
        contract['angles'][0][1][0], contract['angles'][0][1][2] = contract['angles'][0][1][2], contract['angles'][0][1][0]
        with self.assertRaises(AssertionError):
            self.assert_portable_inputs(replace(retained, contract_json=json.dumps(contract)), retained)
        with self.assertRaises(AssertionError):
            self.assert_portable_inputs(replace(retained, node_ids=tuple(reversed(retained.node_ids))), retained)
        with self.assertRaises(AssertionError):
            self.assert_portable_inputs(replace(retained, input_text=retained.input_text.replace('units si', 'units metal')), retained)

    def test_python39_syntax(self):
        for relative in ['fibernet/integrations/_lammps_cg.py', 'fibernet/integrations/_harmonic_profile.py',
                         'fibernet/integrations/_cg_network_profile.py', 'fibernet/integrations/lammps_backend.py',
                         'fibernet/sim/_lammps_result.py', 'examples/cg_bent_branched_network.py']:
            ast.parse((ROOT/relative).read_text(), filename=relative, feature_version=(3, 9))

    def test_tiny_harmonic_profile_save_load_contract(self):
        from test_lammps_easy_dispatch import graph, options, RUNTIME
        from fibernet.integrations import lammps_backend as b
        from fibernet.sim import _lammps_result as adapter
        from fibernet.sim.accelerated import SimResult
        from fibernet.integrations import _harmonic_profile as hp
        arguments = options()
        arguments['run'] = replace(arguments['run'], steps=32)
        prepared = b.prepare_harmonic_beads(graph(), profile=hp.NAME, **arguments)
        self.assertEqual(b.validate_prepared(prepared), prepared)
        contract = json.loads(prepared.contract_json)
        points = ((0., 0., 0.), (.0011, 0., 0.))
        zeros = ((0., 0., 0.),) * 2
        trajectory = tuple(b.TrajectoryFrame(i, i*prepared.timestep_s, prepared.node_ids,
                                             points, zeros, zeros) for i in range(33))
        thermos = tuple(b.ThermoFrame(i, 1e-8, 0., 1e-8, (0.,)*6) for i in range(33))
        raw = b.LAMMPSRunResult(prepared.identity, trajectory, thermos, '')
        result = adapter.to_sim_result(prepared, raw, RUNTIME, 0., True)
        with tempfile.TemporaryDirectory() as directory:
            for detailed in (False, True):
                path = Path(directory)/'result.json'
                result.save(path, detailed=detailed)
                with self.assertRaisesRegex(ValueError, 'matching load profile'):
                    SimResult.load(path)
                loaded = SimResult.load(path, profile=hp.NAME)
                self.assertEqual(loaded.to_dict(detailed), result.to_dict(detailed))
            with self.assertRaises(ValueError):
                SimResult.load(path, profile='unknown')
        with self.assertRaises(ValueError):
            b.prepare_harmonic_beads(graph(), profile=hp.NAME,
                                    **dict(arguments, run=replace(arguments['run'], steps=31)))
