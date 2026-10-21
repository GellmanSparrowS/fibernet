"""Inert 48-bead example contracts; no native execution or trajectory integration."""
import ast
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from examples import cg_crosslinked_network as example
from fibernet.integrations import RuntimeSpec, prepare_cg
from fibernet.integrations.lammps_backend import validate_prepared

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = RuntimeSpec('cli_serial_static', '/fixture/python', '/fixture/lammps',
                      expected_version=20250722, expected_banner='LAMMPS (22 Jul 2025)',
                      file_sha256=(('/fixture/python', 'a'*64), ('/fixture/lammps', 'b'*64)))


class CrosslinkedExampleTests(unittest.TestCase):
    def test_import_and_build_are_inert(self):
        with patch('subprocess.Popen', side_effect=AssertionError('native forbidden')):
            importlib.reload(example)
            reference, initial, request = example.build_case()
        self.assertEqual(request.run.steps, 500)
        self.assertEqual(request.run.timestep_s, 1e-7)
        self.assertEqual((len(initial.nodes), len(initial.edges), len(request.angles)), (48, 104, 64))
        self.assertEqual(tuple(initial.nodes), tuple(101+13*i for i in range(48)))
        self.assertEqual(tuple(initial.edges), tuple(1001+17*i for i in range(104)))
        self.assertEqual(tuple(request.angles), tuple(5003+19*i for i in range(64)))
        self.assertEqual(request.fixed_node_ids, tuple(101+13*i for i in range(12)))
        self.assertEqual(set(request.masses_kg.values()), {1e-15})
        adjacency = {n: set() for n in initial.nodes}
        for edge in initial.edges.values():
            adjacency[edge.node_i].add(edge.node_j)
            adjacency[edge.node_j].add(edge.node_i)
        reached, pending = set(), [101]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node)
                pending.extend(adjacency[node]-reached)
        self.assertEqual(reached, set(initial.nodes))
        self.assertEqual(len(initial.edges)-len(initial.nodes)+1, 57)
        self.assertEqual(reference.to_dict()['edges'], initial.to_dict()['edges'])

    def test_reference_and_loaded_rest_geometry_are_separate(self):
        reference, initial, request = example.build_case()
        changed_lengths = 0
        for edge_id, edge in initial.edges.items():
            rest = request.bonds[edge_id]
            self.assertEqual(rest.equilibrium_length_m, 1e-6)
            self.assertEqual(rest.stiffness_N_per_m, 2e-6)
            self.assertAlmostEqual(math.dist(reference.nodes[edge.node_i].position,
                                            reference.nodes[edge.node_j].position), 1e-6, places=20)
            loaded_length = math.dist(initial.nodes[edge.node_i].position, initial.nodes[edge.node_j].position)
            changed_lengths += abs(loaded_length-rest.equilibrium_length_m) > 1e-12
        self.assertGreater(changed_lengths, 0)
        for n in request.fixed_node_ids:
            self.assertEqual(list(reference.nodes[n].position), list(initial.nodes[n].position))
        for angle in request.angles.values():
            self.assertEqual(angle.K_J, 3e-18)
            self.assertEqual({initial.edges[angle.edge_ij].node_i, initial.edges[angle.edge_ij].node_j},
                             {angle.node_i, angle.node_j})
            self.assertEqual({initial.edges[angle.edge_jk].node_i, initial.edges[angle.edge_jk].node_j},
                             {angle.node_j, angle.node_k})

    def test_prepare_is_pair_free_straight_cosine_with_unchanged_graph(self):
        with patch('subprocess.Popen', side_effect=AssertionError('native forbidden')):
            reference, initial, request = example.build_case()
            before = initial.to_dict()
            prepared = prepare_cg(initial, request)
            self.assertEqual(validate_prepared(prepared), prepared)
            reference_prepared = prepare_cg(reference, request)
        self.assertEqual(initial.to_dict(), before)
        contract = json.loads(prepared.contract_json)
        self.assertEqual(contract['angle_model'], 'cosine-straight/1')
        self.assertGreater(contract['initial_bending_energy_J'], 0)
        self.assertAlmostEqual(json.loads(reference_prepared.contract_json)['initial_bending_energy_J'], 0., places=28)
        self.assertIsNone(contract['projectile_node_id'])
        self.assertEqual(contract['projectile_pairs'], [])
        self.assertEqual(contract['pair_model'], 'none')
        self.assertEqual(contract['required_packages'], ['MOLECULE'])
        self.assertIn('angle_style cosine', prepared.input_text)
        self.assertNotIn('morse', prepared.input_text)
        self.assertEqual(prepared.node_ids, tuple(sorted(initial.nodes)))
        self.assertEqual(prepared.edge_ids, tuple(sorted(initial.edges)))

    def test_run_uses_public_route_limits_and_both_exports(self):
        result = MagicMock()
        result.metadata = {'model_contract': {'projectile_node_id': None}}
        result.to_dict.side_effect = lambda detailed: {'detailed': detailed}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            with patch.object(example, 'simulate', return_value=result) as simulate, \
                 patch.object(example.SimResult, 'load', return_value=result) as load, \
                 patch('subprocess.Popen', side_effect=AssertionError('native forbidden')):
                self.assertIs(example.run_example(RUNTIME, output, steps=1000, timestep_s=5e-8), result)
            self.assertTrue((output/'reference_graph.json').is_file())
            self.assertTrue((output/'graph.json').is_file())
            args, kwargs = simulate.call_args
            self.assertEqual(len(args[0].nodes), 48)
            self.assertEqual(kwargs['backend'], 'lammps')
            self.assertEqual(kwargs['mode'], 'dynamics')
            self.assertIs(kwargs['runtime'], RUNTIME)
            self.assertEqual(kwargs['workdir'], output/'native')
            self.assertTrue(kwargs['record_state'])
            self.assertEqual(kwargs['cg_request'].run.steps, 1000)
            self.assertEqual(kwargs['cg_request'].run.timestep_s, 5e-8)
            limits = kwargs['limits']
            self.assertEqual((limits.cpu_seconds, limits.wall_seconds), (10, 20.))
            self.assertEqual((limits.output_bytes, limits.file_size_bytes), (16*1024**2,)*2)
            self.assertEqual(result.save.call_count, 2)
            result.save.assert_any_call(output/'result_detailed.json', detailed=True)
            result.save.assert_any_call(output/'result_summary.json', detailed=False)
            load.assert_any_call(output/'result_detailed.json')
            load.assert_any_call(output/'result_summary.json')
            with patch.object(example, 'simulate') as forbidden:
                with self.assertRaises(FileExistsError):
                    example.run_example(RUNTIME, output)
                forbidden.assert_not_called()
        with self.assertRaises(TypeError):
            example.run_example(None, 'unused')

    def test_export_mismatch_is_rejected(self):
        result, loaded = MagicMock(), MagicMock()
        result.to_dict.return_value = {'value': 1}
        loaded.to_dict.return_value = {'value': 2}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(example, 'simulate', return_value=result), \
             patch.object(example.SimResult, 'load', return_value=loaded):
            with self.assertRaisesRegex(ValueError, 'round-trip'):
                example.run_example(RUNTIME, Path(directory)/'new')

    def test_fixture_records_scientific_not_resource_acceptance(self):
        record = json.loads((ROOT/'tests/fixtures/lammps_cg_crosslinked_observations.json').read_text())
        self.assertEqual(record['scientific_status'], 'PASS_FROZEN_SCENE_GATES')
        self.assertEqual(record['overall_resource_acceptance'], 'FAIL')
        self.assertIs(record['native_rerun_for_publication'], False)
        self.assertEqual(hashlib.sha256((ROOT/'fibernet/integrations/_lammps_cg.py').read_bytes()).hexdigest(),
                         record['accepted_compiler_sha256'])
        self.assertEqual(record['parameters']['angle_rest_radians'], 'pi')
        self.assertEqual(record['parameters']['cycle_rank'], 57)
        for name, steps, dt in [('static', 0, 1e-7), ('coarse', 500, 1e-7), ('fine', 1000, 5e-8)]:
            case = record['cases'][name]
            self.assertEqual((case['steps'], case['timestep_s'], case['frames']), (steps, dt, steps+1))
            self.assertEqual(len(case['prepared_sha256']), 64)
            self.assertTrue(case['raw_public_equal'])
            self.assertTrue(all(case['public_reload_equal'].values()))
            self.assertTrue(all(math.isfinite(v) for v in case['public_result'].values()))
        self.assertLess(record['matched']['max_position_over_L'], record['scientific_tolerances']['matched_position_over_L_max'])
        self.assertLess(record['matched']['max_velocity_over_V'], record['scientific_tolerances']['matched_velocity_over_V_max'])

    def test_cli_requires_runtime_and_forwards_fine_grid(self):
        result = MagicMock()
        result.n_nodes, result.n_edges = 48, 104
        result.energy, result.max_force = 0., 0.
        result.metadata = {'final_max_net_nodal_force_N': 0.}
        with patch.object(example, 'load_runtime', return_value=RUNTIME) as load, \
             patch.object(example, 'run_example', return_value=result) as run, \
             patch('sys.stdout', new_callable=io.StringIO) as output:
            example.main(['--runtime-json', 'runtime.json', '--output', 'output',
                          '--steps', '1000', '--timestep-s', '5e-8'])
        load.assert_called_once_with(Path('runtime.json'))
        run.assert_called_once_with(RUNTIME, Path('output'), steps=1000, timestep_s=5e-8)
        self.assertEqual(json.loads(output.getvalue())['nodes'], 48)
        with patch.object(example, 'run_example') as run, patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                example.main(['--output', 'output'])
            run.assert_not_called()

    def test_python39_syntax_and_shared_runtime_loader(self):
        ast.parse((ROOT/'examples/cg_crosslinked_network.py').read_text(), feature_version=(3, 9))
        from examples.cg_bent_branched_network import load_runtime
        self.assertIs(example.load_runtime, load_runtime)

    def test_documented_python_path_uses_bounded_example(self):
        document = (ROOT/'docs/integrations/LAMMPS_CG_CROSSLINKED_NETWORK.md').read_text()
        snippet = document.split('```python\n', 1)[1].split('```', 1)[0]
        sentinel = object()
        with patch('subprocess.Popen', side_effect=AssertionError('native forbidden')), \
             patch.object(example, 'run_example', return_value=sentinel) as run:
            namespace = {'runtime': RUNTIME}
            exec(compile(snippet, '<documented Python example>', 'exec'), namespace)
        run.assert_called_once_with(RUNTIME, 'NEW_OUTPUT')
        self.assertIs(namespace['result'], sentinel)
