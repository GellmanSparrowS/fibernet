"""Provider-free saved mapping checks; synthetic states, never native evidence."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import lammps_backend as b
from fibernet.integrations import _harmonic_profile as hp
from fibernet.integrations import _cg_network_profile as network
from fibernet.integrations._lammps_cg import CGRequest, prepare_cg
from fibernet.integrations._lammps_forecast import payload_counts
from fibernet.integrations.lammps_planning import forecast_cg_fiber_only
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult


RUNTIME = b.RuntimeSpec('cli_serial_static', '/inert/python', '/inert/lammps')


def prepared_case(kind='harmonic', unit='mm'):
    scale = {'m': 1., 'mm': .001}[unit]
    # Real authoritative graph API, sparse stable IDs and reversed insertion.
    graph = StructureGraph.from_dict({'dimension': 3, 'nodes': [
        {'id': 101, 'position': [.002/scale, 0., 0.]},
        {'id': 9, 'position': [.001/scale, 0., 0.]}], 'edges': [
        {'id': 77, 'node_i': 9, 'node_j': 101, 'radius': .01,
         'internal_points': None}]})
    run = b.RunSpec(1e-6, 2, 1, ((-.01, .01),)*3)
    masses = {101: 1e-9, 9: 1e-9}
    bonds = {77: b.HarmonicBond(2., .001)}
    if kind == 'cg':
        return prepare_cg(graph, CGRequest(unit, .01, masses, bonds, {},
            None, {}, run, fixed_node_ids=(9,)))
    options = {'profile': 'harmonic_sampled_v1', 'output_stride': 2} if kind == 'profile' else {}
    return b.prepare_harmonic_beads(graph, length_unit=unit,
        maximum_bond_length_m=.01, masses_kg=masses, bonds=bonds,
        run=run, fixed_node_ids=(9,), **options)


def typed_raw(prepared, *, close_initial=False):
    c = json.loads(prepared.contract_json)
    by_id = {n['id']: n['position'] for n in json.loads(prepared.graph_json)['nodes']}
    original = [[float(x)*c['length_to_m'] for x in by_id[nid]] for nid in prepared.node_ids]
    frames = []
    for step in range(prepared.steps+1):
        points = copy.deepcopy(original)
        if close_initial:
            # Static native samples P=F, but P differs from original A within
            # the existing converter allowance. Therefore saved u is nonzero.
            for point in points:
                point[0] = float(np.nextafter(point[0], np.inf))
        elif step:
            points[1][0] += 1e-5*step
        frames.append(b.TrajectoryFrame(step, step*prepared.timestep_s,
            prepared.node_ids, tuple(map(tuple, points)), ((0., 0., 0.),)*2,
            ((0., 0., 0.),)*2))
    thermos = tuple(b.ThermoFrame(step, 0., 0., 0., (0.,)*6)
                    for step in range(prepared.steps+1))
    return b.LAMMPSRunResult(prepared.identity, tuple(frames), thermos, '')


def result_case(kind='harmonic', unit='mm', *, close_initial=False, record=True):
    p = prepared_case(kind, unit)
    return adapter.to_sim_result(p, typed_raw(p, close_initial=close_initial),
                                 RUNTIME, 0., record)


class DisplacementReferenceTests(unittest.TestCase):
    def test_roundtrip_sampled_summary_no_state_and_aliases(self):
        for kind in ('harmonic', 'profile', 'cg'):
            for unit in ('m', 'mm'):
                with self.subTest(kind=kind, unit=unit):
                    p = prepared_case(kind, unit)
                    raw = typed_raw(p)
                    before = copy.deepcopy((p, raw))
                    result = adapter.to_sim_result(p, raw, RUNTIME, .125, True)
                    self.assertEqual((p, raw), before)
                    ref = result.dynamics_state['displacement_reference']
                    self.assertEqual(ref['node_ids'], [9, 101])
                    self.assertEqual(ref['positions'], [[.001, 0., 0.], [.002, 0., 0.]])
                    self.assertEqual(result.lammps_displacement_relationship()['relationship'], 'consistent')
                    profile = 'harmonic_sampled_v1' if kind == 'profile' else None
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory)/'result.json'
                        for detailed in (True, False):
                            payload = result.to_dict(detailed)
                            result.save(path, detailed=detailed)
                            loaded = SimResult.load(path, profile=profile)
                            self.assertEqual(loaded.to_dict(detailed), payload)
                            projected = payload['dynamics_state']['displacement_reference']
                            if detailed:
                                self.assertEqual(projected, ref)
                                projected['positions'][0][0] += 1.
                                projected['node_ids'][0] = 123
                                projected['source_identity']['graph_sha256'] = 'f'*64
                                self.assertEqual(ref['positions'][0][0], .001)
                                self.assertEqual(ref['node_ids'], [9, 101])
                                self.assertEqual(result.lammps_displacement_relationship()['relationship'], 'consistent')
                                self.assertFalse(np.shares_memory(result.displacements,
                                                                  result.deformed_positions))
                            else:
                                self.assertEqual(loaded.lammps_displacement_relationship(),
                                    {'availability': 'unavailable', 'reason': 'omitted_from_summary'})
                                self.assertEqual(set(projected), set(adapter.REFERENCE_DECLARATIONS) |
                                                 {'availability', 'reason', 'source_identity'})
                                self.assertFalse(any(isinstance(v, (list, np.ndarray)) for v in projected.values()))
                    unrecorded = result_case(kind, unit, record=False)
                    self.assertIsNone(unrecorded.dynamics_state)
                    self.assertEqual(unrecorded.lammps_displacement_relationship(),
                                     {'availability': 'unavailable', 'reason': 'not_recorded'})

    def test_sign_flip_rejected_at_detailed_summary_load_and_atomic_save(self):
        for unit in ('m', 'mm'):
            with self.subTest(unit=unit):
                good = result_case(unit=unit)
                payload = good.to_dict(True)
                bad = copy.deepcopy(good)
                self.assertNotEqual(bad.displacements[1, 0], 0.)
                bad.displacements[1, 0] *= -1
                payload['displacements'][1][0] *= -1
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/'result.json'
                    for detailed in (True, False):
                        with self.assertRaisesRegex(ValueError, 'displacement/reference mismatch'):
                            bad.to_dict(detailed)
                        path.write_text('unchanged destination')
                        with self.assertRaises(ValueError):
                            bad.save(path, detailed=detailed)
                        self.assertEqual(path.read_text(), 'unchanged destination')
                    path.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(ValueError, 'displacement/reference mismatch'):
                        SimResult.load(path)
                with self.assertRaises(ValueError):
                    bad.lammps_displacement_relationship()

    def test_legacy_absence_stays_absent_and_unavailable(self):
        for detailed in (True, False):
            result = result_case()
            del result.dynamics_state['displacement_reference']
            payload = result.to_dict(detailed)
            self.assertNotIn('displacement_reference', payload['dynamics_state'])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'legacy.json'
                path.write_text(json.dumps(payload))
                loaded = SimResult.load(path)
                self.assertEqual(loaded.to_dict(detailed), payload)
                self.assertEqual(loaded.lammps_displacement_relationship(),
                                 {'availability': 'unavailable', 'reason': 'not_recorded'})
                self.assertNotIn('displacement_reference', loaded.dynamics_state)

    def test_reference_errors_never_downgrade_to_legacy(self):
        cases = {
            'schema': lambda r: r.update(schema='unknown/2'),
            'units': lambda r: r.update(units='mm'),
            'basis': lambda r: r.update(basis='sample0'),
            'arithmetic': lambda r: r.update(arithmetic='divide_then_subtract'),
            'availability': lambda r: r.update(availability='unavailable'),
            'shape': lambda r: r.update(positions=[[0., 0.]]),
            'nan': lambda r: r['positions'][0].__setitem__(0, float('nan')),
            'inf': lambda r: r['positions'][0].__setitem__(0, float('inf')),
            'boolean': lambda r: r['positions'][0].__setitem__(0, True),
            'numpy_boolean': lambda r: r['positions'][0].__setitem__(0, np.bool_(False)),
            'complex': lambda r: r['positions'][0].__setitem__(0, 1j),
            'string': lambda r: r['positions'][0].__setitem__(0, '.001'),
            'reversed_ids': lambda r: r.update(node_ids=[101, 9]),
            'duplicate_ids': lambda r: r.update(node_ids=[9, 9]),
            'boolean_id': lambda r: r.update(node_ids=[True, 101]),
            'float_id': lambda r: r.update(node_ids=[9., 101]),
            'negative_id': lambda r: r.update(node_ids=[-9, 101]),
            'oversized_id': lambda r: r.update(node_ids=[9, 2**31]),
            'different_source': lambda r: r['source_identity'].update(graph_sha256='f'*64),
            'malformed_source': lambda r: r.update(source_identity={'graph_sha256': 'bad'}),
            'missing_positions': lambda r: r.pop('positions'),
            'missing_source': lambda r: r.pop('source_identity'),
            'unknown_field': lambda r: r.update(extra=1),
        }
        good = result_case()
        for name, mutate in cases.items():
            with self.subTest(name=name):
                bad = copy.deepcopy(good)
                mutate(bad.dynamics_state['displacement_reference'])
                for detailed in (True, False):
                    with self.assertRaises((ValueError, TypeError)):
                        bad.to_dict(detailed)
                with self.assertRaises((ValueError, TypeError)):
                    bad.lammps_displacement_relationship()
                if name not in ('complex', 'numpy_boolean'):
                    payload = good.to_dict(True)
                    mutate(payload['dynamics_state']['displacement_reference'])
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory)/'bad-detail.json'
                        path.write_text(json.dumps(payload))
                        with self.assertRaises((ValueError, TypeError)):
                            SimResult.load(path)
        for kind in ('absent', 'boolean_alias', 'float_alias'):
            with self.subTest(metadata_node_ids=kind):
                bad = copy.deepcopy(good)
                if kind == 'absent':
                    del bad.metadata['node_ids']
                else:
                    # Make equality alone accept True == 1 (or 1.0 == 1).
                    # A strict integer row identity is required nonetheless.
                    bad.dynamics_state['node_ids'][0] = 1
                    bad.dynamics_state['displacement_reference']['node_ids'][0] = 1
                    bad.metadata['node_ids'][0] = True if kind == 'boolean_alias' else 1.
                with self.assertRaises(ValueError):
                    bad.to_dict(True)
                with self.assertRaises(ValueError):
                    bad.lammps_displacement_relationship()
        tuple_ids = copy.deepcopy(good)
        tuple_ids.metadata['node_ids'] = tuple(tuple_ids.metadata['node_ids'])
        tuple_ids.dynamics_state['node_ids'] = tuple(tuple_ids.dynamics_state['node_ids'])
        tuple_ids.dynamics_state['displacement_reference']['node_ids'] = tuple(
            tuple_ids.dynamics_state['displacement_reference']['node_ids'])
        self.assertEqual(tuple_ids.lammps_displacement_relationship()['relationship'], 'consistent')
        for missing in ('displacements',):
            bad = copy.deepcopy(good)
            setattr(bad, missing, None)
            with self.assertRaisesRegex(ValueError, 'requires displacements'):
                bad.to_dict(True)
        payload = good.to_dict(False)
        for mutation in ('positions', 'available', 'source'):
            bad = copy.deepcopy(payload)
            ref = bad['dynamics_state']['displacement_reference']
            if mutation == 'positions':
                ref['positions'] = [[1., 2., 3.]]
            elif mutation == 'available':
                ref['availability'] = 'available'
            else:
                ref['source_identity']['prepared_sha256'] = 'not-a-digest'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'bad-summary.json'
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    SimResult.load(path)

    def test_tolerance_close_sample_zero_is_not_the_reference(self):
        for unit in ('m', 'mm'):
            with self.subTest(unit=unit):
                result = result_case(unit=unit, close_initial=True)
                state = result.dynamics_state
                reference = state['displacement_reference']['positions']
                self.assertNotEqual(reference, state['positions'][0])
                self.assertEqual(state['positions'][0], state['final']['positions'])
                scale = result.metadata['original_graph_length_to_m']
                # Independent scalar binary64 oracle, not the production helper.
                expected = [[(float(f)-float(a))/float(scale) for f, a in zip(fr, ar)]
                            for fr, ar in zip(state['final']['positions'], reference)]
                np.testing.assert_array_equal(result.displacements, expected)
                self.assertTrue(np.any(result.displacements != 0))
                result.to_dict(True)
                bad = copy.deepcopy(result)
                bad.dynamics_state['displacement_reference']['positions'] = copy.deepcopy(state['positions'][0])
                with self.assertRaisesRegex(ValueError, 'displacement/reference mismatch'):
                    bad.to_dict(False)

    def test_binary64_signed_zero_cancellation_scaling_and_nonfinite(self):
        final = np.array([[1., 0., -0.], [np.nextafter(1., np.inf), 2., 3.]])
        reference = np.array([[1., -0., 0.], [1., 2., 3.]])
        for scale in (1., .001):
            actual = adapter._mapped_displacements(final, reference, scale)
            expected = [[(float(f)-float(a))/float(scale) for f, a in zip(fr, ar)]
                        for fr, ar in zip(final, reference)]
            self.assertEqual(actual.dtype, np.dtype('float64'))
            np.testing.assert_array_equal(actual, expected)
        result = result_case()
        result.displacements[0, 1] = -0.
        result.to_dict(True)  # signed zeros are numerically equivalent
        largest = np.finfo(np.float64).max
        for final, reference, scale in (([[largest]], [[-largest]], 1.),
                                        ([[largest]], [[0.]], .001)):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                adapter._mapped_displacements(final, reference, scale)

    def test_forecast_reference_delta_budgets_and_metadata_scope(self):
        common = dict(row_bytes=239, prepared_bytes=100, setup_bytes=200,
                      metadata_reserve_bytes=300, raw_reserve_bytes=400,
                      export_reserve_bytes=500)
        old = payload_counts(2, 1, 3, 3, 3, record_reference=False, **common)
        new = payload_counts(2, 1, 3, 3, 3, **common)
        delta = 4*2*64+2048
        for key in old:
            expected = delta if key == 'detailed_export_forecast_bytes' else (
                4*delta if key in ('planned_memory_bytes', 'selected_copy_memory_bytes') else 0)
            self.assertEqual(new[key]-old[key], expected, key)
        p = prepared_case('cg')
        # Valid inert configuration only; no runtime lookup or admission.
        planning_runtime = replace(RUNTIME, expected_version=20250722,
                                   expected_banner='LAMMPS (22 Jul 2025)')
        options = dict(export_budget_bytes=8*1024**2)
        no = forecast_cg_fiber_only(p, planning_runtime, record_state=False, **options)
        yes = forecast_cg_fiber_only(p, planning_runtime, record_state=True, **options)
        self.assertEqual(no['metadata_base_bytes'], yes['metadata_base_bytes'])
        self.assertEqual(yes['detailed_export_forecast_bytes']-no['detailed_export_forecast_bytes'], delta)
        self.assertEqual(yes['planned_memory_bytes']-no['planned_memory_bytes'], 4*delta)
        cap = no['detailed_export_forecast_bytes']
        self.assertTrue(forecast_cg_fiber_only(p, planning_runtime, record_state=False,
                            export_budget_bytes=cap)['budget_comparison']['detailed_export'])
        self.assertFalse(forecast_cg_fiber_only(p, planning_runtime, record_state=True,
                            export_budget_bytes=cap)['budget_comparison']['detailed_export'])
        # Profile accounting inherits the exact shared extra without native work.
        harmonic = hp.resource_forecast(2, 1, 2, 2, 100, 200)
        harmonic_slots = 12*2*2+16*2+2+16*2+64
        self.assertEqual(harmonic['detailed_export_forecast_bytes'],
                         harmonic_slots*64 + hp.SAMPLED_PROFILE['prepared_bytes'] + 1024**2 + delta)
        net = network.forecast(2, 1, 2, 1, 1, 100, 200)
        net_slots = 12*2*3+16*2+2+16*3+64
        self.assertEqual(net['detailed_export_forecast_bytes'],
                         net_slots*64 + network.PROFILE['prepared_bytes'] + 1024**2 + delta)
        descriptor = result_case().dynamics_state['displacement_reference']
        self.assertLessEqual(hp.json_bytes(descriptor, 1024**2, indent=2), delta)
        # Encoding-only checks at real detailed-export nesting. Max-width stable
        # IDs, finite binary64 tokens and fixed 64-hex source identities are
        # bounded by the declared reference contract, not arbitrary metadata.
        for n in (2, 1024):
            worst = dict(adapter.REFERENCE_DECLARATIONS, availability='available',
                source_identity={'prepared_sha256': 'f'*64, 'graph_sha256': 'f'*64},
                node_ids=[2**31-1-i for i in range(n)],
                positions=[[-np.finfo(np.float64).max, -1.2345678901234567e-300,
                            1.2345678901234567e+300] for _ in range(n)])
            base = result_case().to_dict(True)
            del base['dynamics_state']['displacement_reference']
            encoded_before = hp.json_bytes(base, 1024**2, indent=2)
            base['dynamics_state']['displacement_reference'] = worst
            extra_encoded = hp.json_bytes(base, 1024**2, indent=2)-encoded_before
            self.assertLessEqual(extra_encoded, 4*n*64+2048)
        with self.assertRaises(ValueError):
            payload_counts(2, 1, 3, 3, 3, record_reference=1, **common)


if __name__ == '__main__':
    unittest.main()
