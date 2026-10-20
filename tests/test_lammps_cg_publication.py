"""Portable publication contracts; no native execution or historical rerun."""
import ast
import hashlib
import tempfile
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from examples.cg_bent_branched_network import build_case
from fibernet.integrations import prepare_cg

ROOT = Path(__file__).resolve().parents[1]


class CGPublicationTests(unittest.TestCase):
    def test_retained_native_input_identities(self):
        record = json.loads((ROOT/'tests/fixtures/lammps_cg_fiber_only_acceptance.json').read_text())
        with patch('subprocess.Popen', side_effect=AssertionError('native execution forbidden')):
            for name, case in record['cases'].items():
                with self.subTest(case=name):
                    graph, request = build_case(steps=case['steps'], timestep_s=case['timestep_s'])
                    prepared = prepare_cg(graph, request)
                    self.assertEqual(prepared.identity, case['prepared_sha256'])
                    for field, key in [('graph_json', 'graph_sha256'), ('data_text', 'model_data_sha256'), ('input_text', 'input_sha256')]:
                        self.assertEqual(hashlib.sha256(getattr(prepared, field).encode()).hexdigest(), case[key])
                    self.assertEqual(list(prepared.node_ids), record['node_ids'])
                    self.assertEqual(list(prepared.edge_ids), record['edge_ids'])
        self.assertEqual(hashlib.sha256((ROOT/'fibernet/integrations/_lammps_cg.py').read_bytes()).hexdigest(), record['accepted_compiler_sha256'])

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
