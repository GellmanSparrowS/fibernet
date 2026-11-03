"""Synthetic protocol fixtures; no native execution or scientific trajectories."""
from dataclasses import replace
import math
import unittest
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations.lammps_backend import HarmonicBond, RunSpec, prepare_harmonic_beads, parse_outputs
from fibernet.sim._lammps_result import to_sim_result
from test_lammps_backend_contract import fixture
from test_lammps_cg import RUNTIME


def sample(length, *, rest=None, stiffness=2., steps=0):
    g=StructureGraph(dimension=3,tolerance=length*1e-5)
    a=g.add_node([0,0,0]); b=g.add_node([length,0,0],merge=False)
    e=g.add_edge(a,b,n_internal=0)
    p=prepare_harmonic_beads(g,length_unit='m',maximum_bond_length_m=3*length,
        masses_kg={a:1.,b:1.},bonds={e:HarmonicBond(stiffness,length if rest is None else rest)},
        run=RunSpec(.001,steps,1,((-4*length,4*length),)*3))
    return p,parse_outputs(p,*fixture(p))


class StableLAMMPSNormTests(unittest.TestCase):
    def test_balanced_bond_across_extreme_scales(self):
        for length in (1.,1e-170,1e160):
            p,r=sample(length)
            for record in (False,True):
                with self.subTest(length=length,record=record):
                    o=to_sim_result(p,r,RUNTIME,0.,record)
                    self.assertEqual(o.edge_stretches.tolist(),[1.])
                    self.assertEqual(o.edge_forces.tolist(),[0.])
                    self.assertEqual(o.max_displacement,0.)
    def test_tiny_representable_displacement_norm(self):
        p,r=sample(1e-160,steps=1)
        frame=r.trajectory[1]; actual=1e-160+1e-170
        frame=replace(frame,positions_m=((0.,0.,0.),(actual,0.,0.)))
        r=replace(r,trajectory=(r.trajectory[0],frame))
        o=to_sim_result(p,r,RUNTIME,0.,False)
        expected=actual-1e-160
        self.assertGreater(expected,0.)
        self.assertEqual(o.max_displacement,expected)
        self.assertEqual(o.displacements[1,0],expected)
    def test_nonfinite_derived_force_or_stretch_rejected(self):
        for kwargs in (dict(length=2.,rest=.1,stiffness=1e308),
                       dict(length=1e200,rest=1e-200)):
            p,r=sample(**kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError,'nonfinite LAMMPS derived result'):
                to_sim_result(p,r,RUNTIME,0.,False)
    def test_nonzero_force_underflow_rejected(self):
        p,r=sample(2e-150,rest=1e-150,stiffness=1e-200)
        with self.assertRaisesRegex(ValueError,'underflow'):
            to_sim_result(p,r,RUNTIME,0.,False)
    def test_nonzero_stretch_underflow_rejected(self):
        g=StructureGraph(dimension=3,tolerance=1e-180)
        a=g.add_node([0,0,0]);b=g.add_node([1e-170,0,0],merge=False)
        e=g.add_edge(a,b,n_internal=0)
        p=prepare_harmonic_beads(g,length_unit='m',maximum_bond_length_m=2e160,
            masses_kg={a:1.,b:1.},bonds={e:HarmonicBond(2.,1e160)},
            run=RunSpec(.001,0,1,((-3e160,3e160),)*3))
        with self.assertRaisesRegex(ValueError,'underflow'):
            to_sim_result(p,parse_outputs(p,*fixture(p)),RUNTIME,0.,False)
    def test_stretch_mean_and_spread_avoid_square_overflow_underflow(self):
        for lengths,rest,box,tolerance in (([1e-70,2e-70],1e100,2e100,1e-80),
                                           ([1e100,1e100],1e-208,3e100,1e95)):
            g=StructureGraph(dimension=3,tolerance=tolerance)
            nodes=[g.add_node([x,0,0],merge=False) for x in (0.,lengths[0],sum(lengths))]
            edges=[g.add_edge(nodes[i],nodes[i+1],n_internal=0) for i in (0,1)]
            p=prepare_harmonic_beads(g,length_unit='m',maximum_bond_length_m=box/2,
                masses_kg={n:1. for n in nodes},bonds={e:HarmonicBond(2.,rest) for e in edges},
                run=RunSpec(.001,0,1,((-box,box),)*3))
            o=to_sim_result(p,parse_outputs(p,*fixture(p)),RUNTIME,0.,False)
            if rest == 1e100:
                self.assertTrue(math.isclose(o.mean_stretch,1.5e-170,rel_tol=1e-14))
                self.assertTrue(math.isclose(o.std_stretch,.5e-170,rel_tol=1e-14))
            else:
                self.assertTrue(math.isclose(o.mean_stretch,1e308,rel_tol=1e-14))
                self.assertEqual(o.std_stretch,0.)
    def test_nonfinite_graph_unit_history_is_not_returned(self):
        g=StructureGraph(dimension=3,tolerance=1e300)
        a=g.add_node([0,0,0]);b=g.add_node([1e308,0,0],merge=False)
        e=g.add_edge(a,b,n_internal=0)
        p=prepare_harmonic_beads(g,length_unit='nm',maximum_bond_length_m=1.5e300,
            masses_kg={a:1.,b:1.},bonds={e:HarmonicBond(1.,1e299)},
            run=RunSpec(.001,2,1,((-2e300,2e300),)*3))
        r=parse_outputs(p,*fixture(p))
        middle=replace(r.trajectory[1],positions_m=((0.,0.,0.),(1e300,0.,0.)))
        r=replace(r,trajectory=(r.trajectory[0],middle,r.trajectory[2]))
        self.assertEqual(to_sim_result(p,r,RUNTIME,0.,False).positions_trajectory, [])
        with self.assertRaisesRegex(ValueError,'nonfinite LAMMPS derived trajectory'):
            to_sim_result(p,r,RUNTIME,0.,True)

if __name__ == '__main__': unittest.main()
