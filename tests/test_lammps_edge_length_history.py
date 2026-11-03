"""Provider-free tests: synthetic trajectories are not native evidence."""
import math
from dataclasses import replace
import unittest

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

if __name__ == '__main__': unittest.main()
