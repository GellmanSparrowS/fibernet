"""Analytical/source and mocked role contracts; never execute native LAMMPS."""
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fibernet import simulate
from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import MorsePair, prepare_cg, lammps_backend as b
from fibernet.integrations._lammps_cg import GROUP_ENCODING, morse_energy
from fibernet.integrations import _cg_network_profile as net
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult
from test_lammps_cg import fixture, RUNTIME
from test_lammps_cg_group_encoding import groups_and_pairs
from test_lammps_cg_network_profile import texts as mocked_output_texts


ATTRACTIVE = 'attractive_site'
REPULSIVE = 'repulsive_site'
MIXED_PROFILE = 'cg_network_mixed_sampled_v1'


def repulsive_reference(r, D, a, r0):
    """Independent analytic energy and outward radial force, for r > 0."""
    if r >= r0:
        return 0., 0.
    z = math.expm1(-a*(r-r0))
    return D*z*z, 2*a*D*z*(z+1)


def mixed_case(encoding=None, profile=None):
    g, q = fixture(False)
    q = replace(q, projectile_pairs={
        0: MorsePair(1., 2., 1., 3.),
        1: MorsePair(1.5, 2., 2.5, 2.5, role=REPULSIVE),
    }, pair_encoding=encoding)
    if profile:
        q = replace(q, profile=profile, thermo_every=2,
                    run=replace(q.run, steps=8, sample_every=4))
    return g, q, prepare_cg(g, q)


def mocked_raw(p):
    c = json.loads(p.contract_json)
    points = tuple(tuple(n['position']) for n in json.loads(p.graph_json)['nodes'])
    zeros = ((0., 0., 0.),)*len(p.node_ids)
    return b.LAMMPSRunResult(p.identity,
        tuple(b.TrajectoryFrame(i, i*p.timestep_s, p.node_ids, points, zeros, zeros)
              for i in range(0, p.steps+1, p.sample_every)),
        tuple(b.ThermoFrame(i, 0., 0., 0., (0.,)*6)
              for i in range(0, p.steps+1, c.get('thermo_every', 1))), '')


def forged_contracts(original):
    """Type/shape/coverage controls, without trusting a digest as validation."""
    mutations = [
        ('missing role', lambda c: c['projectile_pair_roles'].pop()),
        ('duplicate role ID', lambda c: c['projectile_pair_roles'].append(c['projectile_pair_roles'][0])),
        ('unknown role ID', lambda c: c['projectile_pair_roles'][1].__setitem__(0, 999)),
        ('projectile role ID', lambda c: c['projectile_pair_roles'][1].__setitem__(0, 3)),
        ('unselected role ID', lambda c: c['projectile_pair_roles'][1].__setitem__(0, 2)),
        ('boolean role ID', lambda c: c['projectile_pair_roles'][0].__setitem__(0, False)),
        ('float role ID', lambda c: c['projectile_pair_roles'][0].__setitem__(0, 0.)),
        ('string role ID', lambda c: c['projectile_pair_roles'][0].__setitem__(0, '0')),
        ('boolean role', lambda c: c['projectile_pair_roles'][1].__setitem__(1, True)),
        ('unknown role', lambda c: c['projectile_pair_roles'][1].__setitem__(1, 'radial')),
        ('role/coefficients mismatch', lambda c: c['projectile_pair_roles'][0].__setitem__(1, REPULSIVE)),
        ('noncanonical ordering', lambda c: c['projectile_pair_roles'].reverse()),
        ('both pair tables reordered', lambda c: (c['projectile_pair_roles'].reverse(), c['projectile_pairs'].reverse())),
        ('overlong role row', lambda c: c['projectile_pair_roles'][0].append('extra')),
        ('mapping instead of rows', lambda c: c.__setitem__('projectile_pair_roles', dict(c['projectile_pair_roles']))),
        ('missing role rows', lambda c: c.pop('projectile_pair_roles')),
        ('missing declaration', lambda c: c.pop('pair_roles')),
        ('empty declaration', lambda c: c.__setitem__('pair_roles', {})),
        ('fifth coefficient', lambda c: c['projectile_pairs'][1][1].append(REPULSIVE)),
    ]
    for label, mutate in mutations:
        c = copy.deepcopy(original)
        mutate(c)
        yield label, c


class MixedRoleTests(unittest.TestCase):
    def test_default_attractive_bytes_match_frozen_legacy_cases(self):
        baseline = json.loads((Path(__file__).parent/'fixtures/lammps_cg_default_encoding_v1.json').read_text())
        for name, edges, steps, pairs in [
            ('base', True, 0, None), ('no_pairs', True, 2, {}),
            ('all_pairs', False, 0, {n: MorsePair(1., 2., 1., 3.) for n in range(3)})]:
            g, q = fixture(edges, steps)
            if pairs is not None:
                q = replace(q, projectile_pairs=pairs)
            p = prepare_cg(g, q)
            actual = {'identity': p.identity, **{k: hashlib.sha256(getattr(p, k).encode()).hexdigest()
                      for k in ('graph_json', 'contract_json', 'data_text', 'input_text')}}
            self.assertEqual(actual, baseline['cases'][name])
            self.assertNotIn('projectile_pair_roles', json.loads(p.contract_json))
            self.assertNotIn('pair_roles', json.loads(p.contract_json))

    def test_explicit_default_role_preserves_both_encodings_and_old_profile(self):
        g, q = fixture(False)
        self.assertEqual(MorsePair(1, 2, 1, 3).role, ATTRACTIVE)
        for encoding, profile in ((None, None), (GROUP_ENCODING, None), (GROUP_ENCODING, net.NAME)):
            request = replace(q, pair_encoding=encoding, profile=profile,
                              thermo_every=1 if profile else None)
            explicit = replace(request, projectile_pairs={
                n: replace(v, role=ATTRACTIVE) for n, v in request.projectile_pairs.items()})
            self.assertEqual(prepare_cg(g, request), prepare_cg(g, explicit))
            c = json.loads(prepare_cg(g, explicit).contract_json)
            self.assertNotIn('pair_roles', c)
            self.assertNotIn('projectile_pair_roles', c)

    def test_repulsive_formula_finite_core_cutoff_and_force_sign(self):
        D, a, r0 = 1.5, 2., 2.5
        c = (D, a, r0, r0)
        for r in (0., .1, 1., 2., r0-1e-4, r0-1e-8, r0, r0+1e-8, 4.):
            expected, force = repulsive_reference(r, D, a, r0)
            actual = morse_energy(r, c)
            self.assertTrue(math.isfinite(actual))
            # The production helper preserves native subtraction arithmetic.
            # Near rc this loses relative accuracy; compare in absolute units
            # without silently treating it as a numerically improved solver.
            self.assertGreaterEqual(actual, -2e-14)
            self.assertTrue(math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-14), (r, actual, expected))
            self.assertGreaterEqual(force, 0.)
            if 0 < r < r0:
                self.assertGreater(expected, 0.)
                self.assertGreater(force, 0.)
        self.assertEqual(morse_energy(r0, c), 0.)
        self.assertEqual(repulsive_reference(r0, D, a, r0)[1], 0.)
        for r in (.1, 1., 2.):
            h = 1e-6
            numerical_force = -(morse_energy(r+h, c)-morse_energy(r-h, c))/(2*h)
            self.assertAlmostEqual(numerical_force/repulsive_reference(r, D, a, r0)[1], 1., places=8)
        # A finite limiting energy is not a hard-core/nonpenetration guarantee.
        self.assertGreater(morse_energy(0., c), morse_energy(.1, c))

    def test_roles_compile_four_coefficients_in_both_encodings(self):
        for encoding in (None, GROUP_ENCODING):
            g, q, p = mixed_case(encoding)
            c = json.loads(p.contract_json)
            self.assertEqual(c['projectile_pair_roles'], [[0, ATTRACTIVE], [1, REPULSIVE]])
            self.assertIn('pair_roles', c)
            self.assertEqual(c['projectile_pairs'], [[0, [1., 2., 1., 3.]], [1, [1.5, 2., 2.5, 2.5]]])
            suffix = '' if encoding else 'morse/smooth/linear '
            self.assertIn('pair_coeff 2 4 '+suffix+'1.5 2 2.5 2.5\n', p.input_text)
            self.assertEqual(p, prepare_cg(g, q))
            self.assertEqual(p, b.validate_prepared(p))
            self.assertEqual(p, prepare_cg(g, replace(q, projectile_pairs=dict(reversed(list(q.projectile_pairs.items()))))))
            if encoding:
                _, selected, excluded = groups_and_pairs(p)
                self.assertEqual(selected, {(1, 4), (2, 4)})
                self.assertFalse(excluded(1, 4)); self.assertFalse(excluded(2, 4))
                self.assertTrue(excluded(3, 4)); self.assertTrue(excluded(1, 2))

    def test_sparse_role_ids_and_projectile_not_last(self):
        g, q, _ = mixed_case(GROUP_ENCODING)
        mapping = {0: 10, 1: 9999, 2: 800, 3: 40}
        record = g.to_dict()
        for n in record['nodes']:
            n['id'] = mapping[n['id']]
        record['next_node_id'] = 10000
        q = replace(q, masses_kg={mapping[n]: v for n, v in q.masses_kg.items()}, projectile_node_id=40,
                    projectile_pairs={mapping[n]: v for n, v in q.projectile_pairs.items()})
        p = prepare_cg(StructureGraph.from_dict(record), q)
        self.assertEqual(p.node_ids, (10, 40, 800, 9999))
        self.assertEqual(json.loads(p.contract_json)['projectile_pair_roles'], [[10, ATTRACTIVE], [9999, REPULSIVE]])
        self.assertIn('pair_coeff 2 4 1.5 2 2.5 2.5\n', p.input_text)
        self.assertEqual(p, b.validate_prepared(p))

    def test_repulsive_only_roles_and_native_squared_cutoff_guards(self):
        for encoding, profile in ((None, None), (GROUP_ENCODING, None), (GROUP_ENCODING, MIXED_PROFILE)):
            g, q, _ = mixed_case(encoding, profile)
            p = prepare_cg(g, replace(q, projectile_pairs={1: q.projectile_pairs[1]}))
            self.assertEqual(json.loads(p.contract_json)['projectile_pair_roles'], [[1, REPULSIVE]])
            self.assertEqual(p, b.validate_prepared(p))
        g, q = fixture(False)
        for alpha, radius in ((1e200, 1e-200), (1e-200, 1e200)):
            request = replace(q, projectile_pairs={0: MorsePair(1., alpha, radius, radius, REPULSIVE)})
            for encoding in (None, GROUP_ENCODING):
                with self.assertRaises((ValueError, b.LAMMPSContractError)):
                    prepare_cg(g, replace(request, pair_encoding=encoding))

    def test_request_role_types_unknown_names_and_exact_cutoff_refused(self):
        g, q, _ = mixed_case()
        pair = q.projectile_pairs[1]
        bad_roles = (None, True, False, 1, [], {}, 'repulsive', 'radial', 'REPULSIVE_SITE', 'repulsive_site ')
        for role in bad_roles:
            with self.subTest(role=role), self.assertRaises((ValueError, b.LAMMPSContractError)):
                prepare_cg(g, replace(q, projectile_pairs={1: replace(pair, role=role)}))
        for cutoff in (2., 3., math.nextafter(2.5, 0.), math.nextafter(2.5, math.inf)):
            with self.subTest(cutoff=cutoff), self.assertRaises((ValueError, b.LAMMPSContractError)):
                prepare_cg(g, replace(q, projectile_pairs={1: replace(pair, cutoff_m=cutoff)}))
        with self.assertRaises((ValueError, b.LAMMPSContractError)):
            prepare_cg(g, replace(q, projectile_pairs={1: replace(pair, role=ATTRACTIVE)}))
        for key in (True, 1., '1', -1, 3, 9999):
            with self.subTest(key=key), self.assertRaises((ValueError, b.LAMMPSContractError)):
                prepare_cg(g, replace(q, projectile_pairs={key: pair}))
        for change in ({'D_J': True}, {'alpha_per_m': float('inf')}, {'r0_m': 0., 'cutoff_m': 0.},
                       {'alpha_per_m': 1e308}, {'alpha_per_m': 1e-30}):
            with self.subTest(change=change), self.assertRaises((ValueError, b.LAMMPSContractError)):
                prepare_cg(g, replace(q, projectile_pairs={1: replace(pair, **change)}))

    def test_selected_repulsive_coincidence_refused_initial_and_retained(self):
        for encoding in (None, GROUP_ENCODING):
            g, q, p = mixed_case(encoding)
            record = g.to_dict()
            record['nodes'][1]['position'] = record['nodes'][3]['position']
            with self.assertRaises((ValueError, b.LAMMPSContractError)):
                prepare_cg(StructureGraph.from_dict(record), q)
            raw = mocked_raw(p)
            positions = list(raw.trajectory[0].positions_m)
            positions[1] = positions[3]
            frame = replace(raw.trajectory[0], positions_m=tuple(positions))
            with self.assertRaises((ValueError, b.LAMMPSContractError)):
                adapter.to_sim_result(p, replace(raw, trajectory=(frame,)), RUNTIME, 0, True)

    def test_fixed_repulsive_anchor_keeps_native_and_edge_force_roles_distinct(self):
        g, q, _ = mixed_case(GROUP_ENCODING)
        p = prepare_cg(g, replace(q, fixed_node_ids=(1,)))
        self.assertIn('group held id 2\n', p.input_text)
        self.assertIn('group mobile subtract all held\n', p.input_text)
        self.assertIn('fix integrate mobile nve\n', p.input_text)
        result = adapter.to_sim_result(p, mocked_raw(p), RUNTIME, 0, True)
        c = result.metadata['model_contract']
        self.assertEqual(c['fixed_node_ids'], [1])
        self.assertEqual(dict(c['projectile_pair_roles'])[1], REPULSIVE)
        self.assertEqual(result.metadata['force_roles']['dynamics_state.nodal_forces'], 'native_net_nodal_after_fixes')
        self.assertEqual(result.metadata['force_roles']['max_force'], 'maximum_absolute_harmonic_bond_axial')
        self.assertEqual(result.edge_forces.tolist(), [])
        self.assertEqual(result.max_force, 0.)

    def test_strict_regeneration_refuses_role_forgery(self):
        for encoding in (None, GROUP_ENCODING):
            _, _, p = mixed_case(encoding)
            for label, c in forged_contracts(json.loads(p.contract_json)):
                with self.subTest(encoding=encoding, mutation=label), self.assertRaises((ValueError, b.LAMMPSContractError)):
                    b.validate_prepared(replace(p, contract_json=b._json(c)))
            with self.assertRaises((ValueError, b.LAMMPSContractError)):
                b.validate_prepared(replace(p, input_text=p.input_text.replace('1.5 2 2.5 2.5', '1.5 2 2.5 3.5')))

    def test_pilot_save_load_and_rehashed_role_forgeries_refused(self):
        _, _, p = mixed_case(GROUP_ENCODING)
        result = adapter.to_sim_result(p, mocked_raw(p), RUNTIME, 0, True)
        with tempfile.TemporaryDirectory() as d:
            for detailed in (False, True):
                path = Path(d)/'mixed.json'
                result.save(path, detailed=detailed)
                self.assertEqual(SimResult.load(path).to_dict(detailed), result.to_dict(detailed))
        for label, c in forged_contracts(result.metadata['model_contract']):
            rr = copy.deepcopy(result)
            rr.metadata['model_contract'] = c
            rr.metadata['model_contract_sha256'] = b._digest(b._json(c))
            for detailed in (False, True):
                with self.subTest(mutation=label, detailed=detailed), self.assertRaises(ValueError):
                    rr.to_dict(detailed)

    def test_network_mixed_profile_is_explicit_with_unchanged_numeric_bounds(self):
        g, q, p = mixed_case(GROUP_ENCODING, MIXED_PROFILE)
        c = json.loads(p.contract_json)
        self.assertEqual(p.profile, MIXED_PROFILE)
        self.assertEqual(c['profile']['name'], MIXED_PROFILE)
        for key, value in net.PROFILE.items():
            if type(value) is int:
                self.assertEqual(c['profile'][key], value)
        self.assertNotEqual(c['profile']['physical_scope'], net.PROFILE['physical_scope'])
        self.assertEqual(c['observation_scope'], net.SCOPE)
        self.assertEqual((c['integration_steps'], c['output_frames'], c['thermo_frames']), (8, 3, 5))
        self.assertEqual(p, b.validate_prepared(p))
        with self.assertRaises((ValueError, b.LAMMPSContractError)):
            prepare_cg(g, replace(q, profile=net.NAME))
        with self.assertRaises((ValueError, b.LAMMPSContractError)):
            prepare_cg(g, replace(q, pair_encoding=None))
        with self.assertRaises((ValueError, b.LAMMPSContractError)):
            prepare_cg(g, replace(q, projectile_pairs={0: q.projectile_pairs[0]}))
        for field, value in (('steps', 1600001), ('sample_every', 3)):
            with self.assertRaises((ValueError, b.LAMMPSContractError)):
                prepare_cg(g, replace(q, run=replace(q.run, **{field: value})))

    def test_network_save_load_requires_matching_explicit_profile(self):
        _, _, p = mixed_case(GROUP_ENCODING, MIXED_PROFILE)
        result = adapter.to_sim_result(p, mocked_raw(p), RUNTIME, 0, True)
        self.assertEqual(result.dynamics_state['profile']['name'], MIXED_PROFILE)
        self.assertEqual(result.dynamics_state['steps'], [0, 4, 8])
        with tempfile.TemporaryDirectory() as d:
            for detailed in (False, True):
                path = Path(d)/'mixed_network.json'
                result.save(path, detailed=detailed)
                self.assertEqual(SimResult.load(path, profile=MIXED_PROFILE).to_dict(detailed), result.to_dict(detailed))
                for profile in (None, net.NAME):
                    with self.assertRaises(ValueError):
                        SimResult.load(path, profile=profile)

    def test_mixed_network_parser_checks_both_observation_schedules(self):
        _, _, p = mixed_case(GROUP_ENCODING, MIXED_PROFILE)
        dump, log = mocked_output_texts(p)
        result = b.parse_outputs(p, dump, log, max_output_bytes=net.PROFILE['raw_bytes'])
        self.assertEqual([f.step for f in result.trajectory], [0, 4, 8])
        self.assertEqual([f.step for f in result.thermo], [0, 2, 4, 6, 8])
        for trajectory, thermo in (
            (dump.replace('ITEM: TIMESTEP\n4\n', 'ITEM: TIMESTEP\n5\n'), log),
            (dump, log.replace('2 4 0 0 0', '3 4 0 0 0', 1)),
            (dump, log.replace('6 4 0 0 0 0 0 0 0 0 0\n', '', 1)),
        ):
            with self.assertRaises((ValueError, b.LAMMPSContractError)):
                b.parse_outputs(p, trajectory, thermo)
        with self.assertRaises((ValueError, b.LAMMPSContractError)):
            b.parse_outputs(p, dump, log, max_output_bytes=1024)

    def test_mixed_network_task_forecast_and_limits_without_launch(self):
        _, _, p = mixed_case(GROUP_ENCODING, MIXED_PROFILE)
        limits = b.ResourceLimits(60, 120, 2*1024**3, 64*1024**2, 64*1024**2)
        self.assertEqual(limits.checked(profile=MIXED_PROFILE), limits)
        with tempfile.TemporaryDirectory() as d, patch.object(b.RuntimeSpec, 'checked', lambda s: s):
            workdir = Path(d)/'never_started'
            task = b.LAMMPSTask(p, RUNTIME, workdir, limits)
            self.assertEqual(task.state, b.TaskState.CREATED)
            self.assertFalse(workdir.exists())
            self.assertEqual(task.output_forecast['output_frames'], 3)
            self.assertEqual(task.output_forecast['thermo_frames'], 5)
            self.assertEqual(task.output_forecast['integration_steps'], 8)
            self.assertEqual(task.output_forecast['observation_scope'], net.SCOPE)
            for change in ({'output_bytes': 1024}, {'file_size_bytes': 1024}, {'memory_bytes': 64*1024**2}):
                with self.subTest(change=change), self.assertRaises((ValueError, b.LAMMPSContractError)):
                    b.LAMMPSTask(p, RUNTIME, workdir, replace(limits, **change))
            self.assertFalse(workdir.exists())
        for change in ({'cpu_seconds': 61}, {'wall_seconds': 121}, {'memory_bytes': 2*1024**3+1},
                       {'output_bytes': 64*1024**2+1}, {'file_size_bytes': 64*1024**2+1}):
            with self.subTest(change=change), self.assertRaises((ValueError, b.LAMMPSContractError)):
                replace(limits, **change).checked(profile=MIXED_PROFILE)

    def test_unified_api_is_mocked_and_preserves_roles(self):
        g, q, _ = mixed_case(GROUP_ENCODING, MIXED_PROFILE)
        class Task:
            def __init__(self, prepared, *args):
                self.result = mocked_raw(prepared)
                self.error = None
            def start(self): pass
            def wait(self): return b.TaskState.SUCCEEDED
            def cancel(self): pass
        with patch.object(b, 'LAMMPSTask', Task):
            result = simulate(g, backend='lammps', mode='dynamics', cg_request=q,
                              runtime=RUNTIME, workdir='unused', limits=b.ResourceLimits(), record_state=True)
        self.assertEqual(result.metadata['model_contract']['projectile_pair_roles'], [[0, ATTRACTIVE], [1, REPULSIVE]])
        self.assertEqual(result.dynamics_state['profile']['name'], MIXED_PROFILE)


if __name__ == '__main__':
    unittest.main()
