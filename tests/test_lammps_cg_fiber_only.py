"""Inert contract/public-API mocks only; no native launch or physics evidence."""
import copy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.cg_bent_branched_network import build_case, load_runtime, main, run_example
from fibernet import simulate
from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import CGRequest, HarmonicBond, MorsePair, prepare_cg
from fibernet.integrations import lammps_backend as b
from fibernet.integrations._lammps_cg import GROUP_ENCODING, regenerate
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult
from test_lammps_cg import fixture, RUNTIME
from test_lammps_cg_network_profile import network


def synthetic_raw(prepared):
    """Arbitrary internally consistent records to exercise mapping, not dynamics."""
    contract = json.loads(prepared.contract_json)
    points = tuple(tuple(x*contract['length_to_m'] for x in n['position'])
                   for n in json.loads(prepared.graph_json)['nodes'])
    zeros = ((0., 0., 0.),)*len(points)
    force = ((1e-12, 2e-12, 3e-12),)*len(points)
    pe = contract['initial_bending_energy_J']
    return b.LAMMPSRunResult(prepared.identity,
        tuple(b.TrajectoryFrame(i, i*prepared.timestep_s, prepared.node_ids,
                               points, zeros, force) for i in range(prepared.steps+1)),
        tuple(b.ThermoFrame(i, pe, 0., pe, (0.,)*6)
              for i in range(prepared.steps+1)), '')


def synthetic_texts(prepared):
    raw = synthetic_raw(prepared)
    contract = json.loads(prepared.contract_json)
    dump = ['ITEM: UNITS', 'si']
    for frame in raw.trajectory:
        dump += ['ITEM: TIME', str(frame.time_s), 'ITEM: TIMESTEP', str(frame.step),
                 'ITEM: NUMBER OF ATOMS', str(len(prepared.node_ids)),
                 'ITEM: BOX BOUNDS ff ff ff']
        dump += [' '.join(map(str, row)) for row in contract['box_m']]
        dump += ['ITEM: ATOMS id type x y z vx vy vz fx fy fz']
        dump += [f'{i} {i} '+' '.join(map(str, (*p, *v, *f)))
                 for i, (p, v, f) in enumerate(zip(frame.positions_m,
                     frame.velocities_m_s, frame.forces_N), 1)]
    log = ['Step Atoms PotEng KinEng TotEng Pxx Pyy Pzz Pxy Pxz Pyz']
    log += [f'{f.step} {len(prepared.node_ids)} {f.potential_J} 0 {f.total_J} 0 0 0 0 0 0'
            for f in raw.thermo]
    log += ['Loop time', prepared.completion_marker]
    return '\n'.join(dump)+'\n', '\n'.join(log)+'\n'


class SyntheticTask:
    def __init__(self, prepared, *args):
        self.result = synthetic_raw(prepared)
        self.error = None
    def start(self): pass
    def wait(self): return b.TaskState.SUCCEEDED
    def cancel(self): pass


class FiberOnlyTests(unittest.TestCase):
    def test_explicit_null_preserves_positional_field_order(self):
        self.assertEqual([f.name for f in fields(CGRequest)], [
            'length_unit', 'maximum_bond_length_m', 'masses_kg', 'bonds', 'angles',
            'projectile_node_id', 'projectile_pairs', 'run', 'fixed_node_ids',
            'velocities_m_s', 'pair_encoding', 'profile', 'thermo_every', 'neighbor_policy'])
        graph, q = build_case(steps=0)
        positional = CGRequest(*(getattr(q, f.name) for f in fields(q)))
        self.assertEqual(prepare_cg(graph, positional), prepare_cg(graph, q))

    def test_graph_ids_atoms_and_truthful_styles_regenerate(self):
        graph, q = build_case(steps=0)
        before = graph.to_dict()
        p = prepare_cg(graph, q)
        c = json.loads(p.contract_json)
        self.assertEqual(graph.to_dict(), before)
        self.assertEqual(p, b.validate_prepared(p))
        self.assertEqual(p, regenerate(p, c))
        self.assertEqual(p.node_ids, (101, 205, 309, 413, 517, 621, 725))
        self.assertEqual(p.edge_ids, (71, 83, 97, 109, 127, 139))
        self.assertEqual(p.manifest()['angle_to_stable_id'],
                         {'1': 47, '2': 59, '3': 61, '4': 73, '5': 89})
        self.assertEqual(len(json.loads(p.graph_json)['nodes']), 7)
        self.assertIn('\n7 atoms\n6 bonds\n5 angles\n', p.data_text)
        self.assertIsNone(c['projectile_node_id'])
        self.assertEqual(c['projectile_pairs'], [])
        self.assertEqual(c['pair_model'], 'none')
        self.assertEqual(c['pair_policy'], 'no nonbonded interactions; all type pairs excluded')
        self.assertEqual(c['required_packages'], ['MOLECULE'])
        self.assertEqual(c['required_styles']['pair'], ['zero'])
        self.assertEqual(c['energy'], 'sum of harmonic bonds and cosine angles; J')
        self.assertNotIn('morse', p.input_text+c['energy'])
        self.assertNotIn('EXTRA-PAIR', p.contract_json)
        self.assertNotIn('pair_encoding', c)
        self.assertNotIn('profile', c)
        self.assertEqual(sum(line.startswith('neigh_modify exclude type ')
                             for line in p.input_text.splitlines()), 28)
        self.assertGreater(c['initial_bending_energy_J'], 0.)

    def test_null_request_combinations_and_integer_isolation(self):
        graph, q = build_case(steps=0)
        for change in ({'projectile_pairs': {101: MorsePair(1., 2., 1., 3.)}},
                       {'pair_encoding': GROUP_ENCODING}, {'profile': 'cg_network_sampled_v1'},
                       {'profile': 'cg_network_mixed_sampled_v1', 'pair_encoding': GROUP_ENCODING},
                       {'projectile_node_id': True}, {'projectile_node_id': 101},
                       {'projectile_pairs': None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_cg(graph, replace(q, **change))
            # Construction checks cannot stand in for the compiler boundary.
            forged = copy.copy(q)
            for name, value in change.items(): object.__setattr__(forged, name, value)
            with self.subTest(compiler_change=change), self.assertRaises(ValueError):
                prepare_cg(graph, forged)
        legacy_graph, legacy_q = fixture()
        # No graph node is silently inserted, deleted or reinterpreted.
        p = prepare_cg(legacy_graph, replace(legacy_q, projectile_node_id=None, projectile_pairs={}))
        self.assertEqual(len(p.node_ids), 4)

    def test_small_pilot_bounds_remain(self):
        for n in (1, 2, 64, 65):
            graph = StructureGraph(dimension=3)
            for i in range(n): graph.add_node((i, 0., 0.))
            q = CGRequest('m', 2., {i: 1. for i in graph.nodes}, {}, {}, None, {},
                          b.RunSpec(.001, 1000, 1, ((-100., 100.),)*3))
            if n in (2, 64): b.validate_prepared(prepare_cg(graph, q))
            else:
                with self.assertRaises(ValueError): prepare_cg(graph, q)
        graph, q = build_case(steps=0)
        for change in ({'run': replace(q.run, steps=1001)},
                       {'run': replace(q.run, steps=2, sample_every=2)},
                       {'thermo_every': 1},
                       {'angles': {i: q.angles[47] for i in range(129)}}):
            with self.assertRaises(ValueError): prepare_cg(graph, replace(q, **change))
        dense = StructureGraph(dimension=3)
        for i in range(18): dense.add_node((i, 0., 0.))
        for i in range(18):
            for j in range(i+1, 18):
                if len(dense.edges) < 129: dense.add_edge(i, j, n_internal=0)
        too_many = CGRequest('m', 20., {n: 1. for n in dense.nodes},
            {e: HarmonicBond(1., 1.) for e in dense.edges}, {}, None, {},
            b.RunSpec(.001, 0, 1, ((-100., 100.),)*3))
        with self.assertRaises(ValueError): prepare_cg(dense, too_many)

    def test_prepared_and_result_forged_null_combinations(self):
        graph, q = build_case(steps=0)
        p = prepare_cg(graph, q)
        result = adapter.to_sim_result(p, synthetic_raw(p), RUNTIME, 0., True)
        for key, value in [('projectile_pairs', [[101, [1., 2., 1., 3.]]]),
                           ('pair_encoding', {'name': GROUP_ENCODING}),
                           ('pair_model', 'morse/smooth/linear'),
                           ('required_packages', ['MOLECULE', 'EXTRA-PAIR']),
                           ('energy', 'sum of harmonic bonds, cosine angles and shifted-force Morse pairs; J')]:
            c = json.loads(p.contract_json); c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.validate_prepared(replace(p, contract_json=b._json(c)))
            rr = copy.deepcopy(result); rr.metadata['model_contract'] = c
            rr.metadata['model_contract_sha256'] = b._digest(b._json(c))
            with self.subTest(result_key=key), self.assertRaises(ValueError): rr.to_dict(True)
        _, _, legacy_network = network(steps=0, every=1, thermo=1)
        c = json.loads(legacy_network.contract_json)
        c.update(projectile_node_id=None, projectile_pairs=[])
        with self.assertRaises(ValueError): regenerate(legacy_network, c)
        with self.assertRaises(ValueError):
            b.validate_prepared(replace(legacy_network, contract_json=b._json(c)))
        metadata = copy.deepcopy(result.metadata)
        metadata['model_contract'] = c
        metadata['model_contract_sha256'] = b._digest(b._json(c))
        from fibernet.integrations._lammps_cg import guard_result
        with self.assertRaises(ValueError): guard_result(metadata, 7, 6, None)

    def test_parser_projection_and_public_roundtrips(self):
        graph, q = build_case(steps=2)
        p = prepare_cg(graph, q)
        raw = b.parse_outputs(p, *synthetic_texts(p))
        result = adapter.to_sim_result(p, raw, RUNTIME, 0., True)
        self.assertEqual(result.metadata['node_ids'], list(p.node_ids))
        self.assertEqual(result.dynamics_state['steps'], [0, 1, 2])
        self.assertEqual(result.dynamics_state['node_ids'], list(p.node_ids))
        self.assertEqual(result.dynamics_state['final']['nodal_forces'], [[1e-12, 2e-12, 3e-12]]*7)
        self.assertLess(result.max_force, 1e-25)
        self.assertGreater(result.metadata['final_max_net_nodal_force_N'], 1e-12)
        self.assertEqual(result.energy, json.loads(p.contract_json)['initial_bending_energy_J'])
        for record in (True, False):
            with patch.object(b, 'LAMMPSTask', SyntheticTask):
                public = simulate(graph, backend='lammps', mode='dynamics', cg_request=q,
                    runtime=RUNTIME, workdir='unused', limits=b.ResourceLimits(), record_state=record)
            with tempfile.TemporaryDirectory() as tmp:
                for detailed in (True, False):
                    path = Path(tmp)/'result.json'
                    public.save(path, detailed=detailed)
                    loaded = SimResult.load(path)
                    self.assertEqual(loaded.to_dict(detailed), public.to_dict(detailed))
                    self.assertIsNone(loaded.metadata['model_contract']['projectile_node_id'])
                    forged = json.loads(path.read_text())
                    forged['model_contract']['projectile_pairs'] = [[101, [1., 2., 1., 3.]]]
                    forged['model_contract_sha256'] = b._digest(b._json(forged['model_contract']))
                    path.write_text(json.dumps(forged))
                    with self.assertRaises(ValueError): SimResult.load(path)

    def test_public_example_cli_with_mock_task_only(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)/'runtime.json'
            runtime.write_text(json.dumps(asdict(RUNTIME)))
            self.assertEqual(load_runtime(runtime), RUNTIME)
            output = Path(tmp)/'example'
            with patch.object(b, 'LAMMPSTask', SyntheticTask), patch('builtins.print'):
                main(['--runtime-json', str(runtime), '--output', str(output), '--steps', '2'])
            self.assertEqual(StructureGraph.load_json(output/'graph.json').num_nodes, 7)
            for name in ('result_detailed.json', 'result_summary.json'):
                self.assertIsNone(SimResult.load(output/name).metadata['model_contract']['projectile_node_id'])
            with self.assertRaises(FileExistsError): run_example(RUNTIME, output)


if __name__ == '__main__':
    unittest.main()
