"""Provider-free tests: synthetic trajectories are not native evidence."""
import math
import copy
from dataclasses import replace, FrozenInstanceError
import unittest

import numpy as np
from fibernet.core.structure_graph import StructureGraph

from fibernet.integrations.lammps_backend import (
    HarmonicBond, RunSpec, LAMMPSContractError, edge_length_history,
    parse_outputs, prepare_harmonic_beads,
)
from test_lammps_backend_contract import graph, fixture, prepared
from test_lammps_cg import fixture as cg_fixture
from test_lammps_cg_network_profile import texts as cg_texts
from fibernet.integrations._lammps_cg import prepare_cg


class EdgeLengthHistoryTests(unittest.TestCase):
    def test_stable_edge_ids_and_changed_positions_in_si(self):
        g = graph()
        c = g.add_node([1, 2, 0])
        other = g.add_edge(40, c, n_internal=0)
        p = prepare_harmonic_beads(g, length_unit='mm', maximum_bond_length_m=.01,
            masses_kg={10:1e-9,40:2e-9,c:3e-9},
            bonds={77:HarmonicBond(2,.001),other:HarmonicBond(3,.002)},
            run=RunSpec(1e-6,2,1,((-0.01,0.01),)*3))
        r = parse_outputs(p, *fixture(p))
        last = replace(r.trajectory[-1], positions_m=((0.,0.,0.),(.002,0.,0.),(.001,.002,0.)))
        r = replace(r, trajectory=(*r.trajectory[:-1], last))
        h = edge_length_history(p, r, 77)
        self.assertEqual((h.edge_id,h.node_i,h.node_j), (77,10,40))
        self.assertEqual(h.steps,(0,1,2))
        self.assertEqual(h.times_s,(0.,1e-6,2e-6))
        self.assertEqual(h.lengths_m,(.001,.001,.002))
        other_history = edge_length_history(p,r,other)
        self.assertAlmostEqual(other_history.lengths_m[-1], math.hypot(.001,.002))

    def test_zero_step(self):
        p = prepared(run=RunSpec(1e-6,0,1,((-0.01,0.01),)*3))
        h = edge_length_history(p, parse_outputs(p,*fixture(p)),77)
        self.assertEqual(h.steps,(0,))
        self.assertEqual(h.lengths_m,(.001,))

    def test_sampled_cg_uses_trajectory_stride(self):
        g,q = cg_fixture()
        q = replace(q, run=RunSpec(.001,4,2,q.run.box_m),
                    profile='cg_network_sampled_v1', thermo_every=1,
                    pair_encoding='static-groups-zero-cutoff/1')
        p = prepare_cg(g,q)
        h = edge_length_history(p,parse_outputs(p,*cg_texts(p)),0)
        self.assertEqual(h.steps,(0,2,4))
        self.assertEqual(h.times_s,(0.,.002,.004))
        self.assertEqual(h.lengths_m,(1.,1.,1.))

    def test_mismatch_and_bad_geometry(self):
        p=prepared(); r=parse_outputs(p,*fixture(p)); first=r.trajectory[0]
        cases = [replace(r,identity='other'), replace(r,trajectory=None),
            replace(r,trajectory=r.trajectory[:-1]),
            replace(r,trajectory=(replace(first,node_ids=(40,10)),*r.trajectory[1:])),
            replace(r,trajectory=(replace(first,step=True),*r.trajectory[1:])),
            replace(r,trajectory=(replace(first,time_s=1.),*r.trajectory[1:])),
            replace(r,trajectory=(replace(first,positions_m=((0.,0.,0.),)),*r.trajectory[1:])),
            replace(r,trajectory=(replace(first,positions_m=((0.,0.,0.),(0.,1.))),*r.trajectory[1:])),
            replace(r,trajectory=(replace(first,positions_m=((0.,0.,0.),(float('nan'),0.,0.))),*r.trajectory[1:])),
            replace(r,trajectory=(replace(first,positions_m=((-1e308,0.,0.),(1e308,0.,0.))),*r.trajectory[1:])),
            replace(r,trajectory=(replace(first,positions_m=((0.,0.,0.),(True,0.,0.))),*r.trajectory[1:]))]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(LAMMPSContractError):
                edge_length_history(p,bad,77)
        for eid in (True,78,-1):
            with self.subTest(eid=eid), self.assertRaises(LAMMPSContractError):
                edge_length_history(p,r,eid)
        with self.assertRaises(LAMMPSContractError):
            edge_length_history(replace(p,input_text=p.input_text+'\n'),r,77)

class EdgeLengthHistoryNumericalTests(unittest.TestCase):
    def setUp(self):
        self.prepared = prepared()
        self.result = parse_outputs(self.prepared, *fixture(self.prepared))

    def changed(self, **fields):
        final = replace(self.result.trajectory[-1], **fields)
        return replace(self.result, trajectory=(*self.result.trajectory[:-1], final))

    def test_all_supported_units_use_meter_lengths(self):
        # Independent SI values for a 3-4-12 chord of length 13 graph units.
        expected = (('m', 13.), ('mm', .013), ('um', 13e-6), ('µm', 13e-6),
                    ('nm', 13e-9), ('cm', .13), ('km', 13000.), ('in', .3302),
                    ('ft', 3.9624), ('Å', 13e-10), ('angstrom', 13e-10))
        for unit, meters in expected:
            with self.subTest(unit=unit):
                data = graph().to_dict()
                data['nodes'][1]['position'] = [3., 4., 12.]
                g = StructureGraph.from_dict(data)
                p = prepare_harmonic_beads(
                    g, length_unit=unit, maximum_bond_length_m=20000.,
                    masses_kg={10: 1., 40: 2.}, bonds={77: HarmonicBond(2., meters)},
                    run=RunSpec(1e-6, 2, 1, ((-1e5, 1e5),)*3))
                # The parser fixture deliberately reverses raw atom rows.
                r = parse_outputs(p, *fixture(p))
                h = edge_length_history(p, r, 77)
                self.assertEqual((h.edge_id, h.node_i, h.node_j), (77, 10, 40))
                self.assertEqual(h.times_s, (0., 1e-6, 2e-6))
                for value in h.lengths_m:
                    self.assertTrue(math.isclose(value, meters, rel_tol=2e-15, abs_tol=0.))

    def test_arbitrary_ids_reversed_orientation_and_record_order(self):
        g = graph()
        c = g.add_node([1., 2., 2.])
        g.add_edge(40, c, n_internal=0)
        data = g.to_dict()
        data['edges'][0].update(id=900, node_i=40, node_j=10)
        data['edges'][1]['id'] = 6
        data['next_edge_id'] = 901
        data['nodes'].reverse()
        data['edges'].reverse()
        g = StructureGraph.from_dict(data)
        p = prepare_harmonic_beads(
            g, length_unit='mm', maximum_bond_length_m=.01,
            masses_kg={10: 1., 40: 2., c: 3.},
            bonds={900: HarmonicBond(1., .001), 6: HarmonicBond(2., math.sqrt(8)*.001)},
            run=RunSpec(1e-6, 2, 1, ((-.1, .1),)*3))
        r = parse_outputs(p, *fixture(p))
        first = edge_length_history(p, r, 900)
        second = edge_length_history(p, r, 6)
        self.assertEqual((first.node_i, first.node_j), (40, 10))
        self.assertEqual(first.lengths_m, (.001,)*3)
        self.assertEqual((second.node_i, second.node_j), (40, c))
        for value in second.lengths_m:
            self.assertTrue(math.isclose(value, math.sqrt(8)*.001, rel_tol=2e-15))
        for bond_index in (1, 2):
            with self.assertRaisesRegex(LAMMPSContractError, 'unknown graph edge ID'):
                edge_length_history(p, r, bond_index)

    def test_prepared_parsed_nonaxis_extreme_scales(self):
        for scale in (1e-200, 1e-150, 1., 1e150, 1e200):
            with self.subTest(scale=scale):
                g = StructureGraph(dimension=3, tolerance=scale*1e-5)
                a = g.add_node([0., 0., 0.], merge=False)
                b = g.add_node([3*scale, 4*scale, 12*scale], merge=False)
                e = g.add_edge(a, b, n_internal=0)
                p = prepare_harmonic_beads(
                    g, length_unit='m', maximum_bond_length_m=30*scale,
                    masses_kg={a: 1., b: 2.}, bonds={e: HarmonicBond(2., 13*scale)},
                    run=RunSpec(1e-6, 2, 1, ((-50*scale, 50*scale),)*3))
                h = edge_length_history(p, parse_outputs(p, *fixture(p)), e)
                for value in h.lengths_m:
                    self.assertTrue(math.isclose(value, 13*scale, rel_tol=2e-15, abs_tol=0.))

    def test_immutable_detached_output_and_unchanged_inputs(self):
        positions = [[0., 0., 0.], [3., 4., 12.]]
        r = self.changed(positions_m=positions)
        before = copy.deepcopy((self.prepared, r))
        h = edge_length_history(self.prepared, r, 77)
        self.assertEqual((self.prepared, r), before)
        self.assertTrue(all(type(getattr(h, key)) is tuple
                            for key in ('steps', 'times_s', 'lengths_m')))
        positions[1][:] = [0., 0., 0.]
        self.assertEqual(h.lengths_m[-1], 13.)
        with self.assertRaises(FrozenInstanceError):
            h.edge_id = 1
        with self.assertRaises(TypeError):
            h.lengths_m[0] = 2.

    def test_integer_ids_and_schedule_types(self):
        self.assertEqual(edge_length_history(self.prepared, self.result, np.int64(77)).edge_id, 77)
        for edge_id in (False, 77., '77', None, 2**31):
            with self.subTest(edge_id=edge_id), self.assertRaises(LAMMPSContractError):
                edge_length_history(self.prepared, self.result, edge_id)
        changes = (dict(step=2.), dict(time_s=True), dict(time_s=float('nan')),
                   dict(time_s=2e-6*(1+2e-12)), dict(node_ids=(10, 40.)),
                   dict(node_ids=(10, 10)), dict(node_ids=None))
        for fields in changes:
            with self.subTest(fields=fields), self.assertRaises(LAMMPSContractError):
                edge_length_history(self.prepared, self.changed(**fields), 77)
        for frames in ((), self.result.trajectory[::-1], (*self.result.trajectory, self.result.trajectory[-1])):
            with self.subTest(frames=frames), self.assertRaises(LAMMPSContractError):
                edge_length_history(self.prepared, replace(self.result, trajectory=frames), 77)

    def test_malformed_geometry_is_not_silently_coerced(self):
        bad_positions = (None, ((0., 0., 0.), None), ((0., 0., 0.), (1., 2., '3')),
                         ((0., 0., 0.), (1., 2., float('inf'))),
                         ((0., 0., 0.), (10**400, 0., 0.)))
        for positions in bad_positions:
            with self.subTest(positions=positions), self.assertRaises(LAMMPSContractError):
                edge_length_history(self.prepared, self.changed(positions_m=positions), 77)

    def test_extractor_does_not_revalidate_native_physics(self):
        # Typed caller edits below deliberately bypass the parser. The extractor
        # reports geometry only; it does not authenticate these forged samples.
        r = replace(self.result, thermo=(), trajectory=tuple(
            replace(f, velocities_m_s=None, forces_N=None) for f in self.result.trajectory))
        self.assertEqual(edge_length_history(self.prepared, r, 77).lengths_m, (.001,)*3)
        for points, expected in ((((1., 2., 3.), (1., 2., 3.)), 0.),
                                 (((5., -9., 7.), (8., -5., 19.)), 13.)):
            h = edge_length_history(self.prepared, self.changed(positions_m=points), 77)
            self.assertEqual(h.lengths_m[-1], expected)

if __name__ == '__main__': unittest.main()
