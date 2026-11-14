"""Synthetic protocol fixtures; no native execution or scientific trajectories."""
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
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

    def test_near_equal_stretches_keep_representable_spread(self):
        for length, rest in ((10., 1.), (1e100, 1e-208), (1e-170, 1.)):
            g = StructureGraph(dimension=3, tolerance=length*1e-5)
            nodes = []
            edges = []
            for y, x in enumerate((math.nextafter(length, 0.), length)):
                a = g.add_node([0., y*length, 0.], merge=False)
                b = g.add_node([x, y*length, 0.], merge=False)
                nodes.extend((a, b))
                edges.append(g.add_edge(a, b, n_internal=0))
            p = prepare_harmonic_beads(
                g, length_unit='m', maximum_bond_length_m=2*max(length, rest),
                masses_kg={n: 1. for n in nodes},
                bonds={e: HarmonicBond(2., rest) for e in edges},
                run=RunSpec(.001, 0, 1, ((-4*max(length, rest), 4*max(length, rest)),)*3))
            for record in (False, True):
                with self.subTest(length=length, rest=rest, record=record):
                    o = to_sim_result(p, parse_outputs(p, *fixture(p)), RUNTIME, 0., record)
                    expected = (max(o.edge_stretches)-min(o.edge_stretches))/2
                    self.assertGreater(expected, 0.)
                    self.assertEqual(o.std_stretch, expected)

    def test_nonaxial_norms_across_scales(self):
        for scale in (1e-170, 1., 1e160):
            g = StructureGraph(dimension=3, tolerance=scale*1e-5)
            a = g.add_node([0., 0., 0.])
            b = g.add_node([3*scale, 4*scale, 0.], merge=False)
            e = g.add_edge(a, b, n_internal=0)
            p = prepare_harmonic_beads(
                g, length_unit='m', maximum_bond_length_m=20*scale,
                masses_kg={a: 1., b: 1.}, bonds={e: HarmonicBond(2., 5*scale)},
                run=RunSpec(.001, 1, 1, ((-30*scale, 30*scale),)*3))
            r = parse_outputs(p, *fixture(p))
            points = tuple(tuple(x+d*scale for x, d in zip(point, (1., 2., 2.)))
                           for point in r.trajectory[-1].positions_m)
            r = replace(r, trajectory=(r.trajectory[0], replace(r.trajectory[1], positions_m=points)))
            for record in (False, True):
                with self.subTest(scale=scale, record=record):
                    o = to_sim_result(p, r, RUNTIME, 0., record)
                    self.assertTrue(math.isclose(o.max_displacement, 3*scale, rel_tol=1e-14))
                    self.assertTrue(math.isclose(o.edge_stretches[0], 1., rel_tol=1e-14))

    def test_subnormal_force_respects_active_runtime(self):
        for length in (1e-150, 3e-150):
            p, r = sample(length, rest=2e-150, stiffness=1e-160)
            expected = 1e-160*(length-2e-150)
            for record in (False, True):
                with self.subTest(length=length, record=record):
                    if expected == 0.:
                        # A previously initialized Taichi runtime enables FTZ.
                        # Projection must reject that loss, never return zero force.
                        with self.assertRaisesRegex(ValueError, 'LAMMPS derived force or stretch underflow'):
                            to_sim_result(p, r, RUNTIME, 0., record)
                    else:
                        o = to_sim_result(p, r, RUNTIME, 0., record)
                        self.assertEqual(o.edge_forces[0], expected)
                        self.assertEqual(o.max_force, abs(expected))

    def test_representable_subnormal_force_in_clean_process(self):
        # Windows runs its full suite in one process. Test gradual underflow in
        # a fresh interpreter, without initializing any optional runtime.
        code = '''
import sys
sys.path.insert(0, 'tests')
from test_lammps_stable_norm import sample, to_sim_result, RUNTIME
for length in (1e-150, 3e-150):
    prepared, raw = sample(length, rest=2e-150, stiffness=1e-160)
    expected = 1e-160*(length-2e-150)
    assert expected != 0., 'fresh interpreter did not preserve subnormal force'
    for record in (False, True):
        result = to_sim_result(prepared, raw, RUNTIME, 0., record)
        assert result.edge_forces[0] == expected
        assert result.max_force == abs(expected)
'''
        completed = subprocess.run(
            [sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)

    def test_collapsed_final_bond_remains_rejected(self):
        p, r = sample(1., steps=1)
        final = replace(r.trajectory[1], positions_m=((0., 0., 0.), (0., 0., 0.)))
        r = replace(r, trajectory=(r.trajectory[0], final))
        for record in (False, True):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, 'native bond left declared reach'):
                to_sim_result(p, r, RUNTIME, 0., record)

    def test_graph_units_and_history_remain_consistent(self):
        for unit, scale in (('mm', 1e-3), ('nm', 1e-9)):
            g = StructureGraph(dimension=3)
            a = g.add_node([0., 0., 0.])
            b = g.add_node([3., 4., 0.], merge=False)
            e = g.add_edge(a, b, n_internal=0)
            p = prepare_harmonic_beads(
                g, length_unit=unit, maximum_bond_length_m=20*scale,
                masses_kg={a: 1., b: 1.}, bonds={e: HarmonicBond(2., 5*scale)},
                run=RunSpec(.001, 1, 1, ((-30*scale, 30*scale),)*3))
            r = parse_outputs(p, *fixture(p))
            final = replace(r.trajectory[1], positions_m=((scale, 2*scale, 2*scale),
                                                        (4*scale, 6*scale, 2*scale)))
            r = replace(r, trajectory=(r.trajectory[0], final))
            for record in (False, True):
                with self.subTest(unit=unit, record=record):
                    o = to_sim_result(p, r, RUNTIME, 0., record)
                    np.testing.assert_allclose(o.displacements, [[1., 2., 2.]]*2, rtol=1e-14, atol=0.)
                    np.testing.assert_allclose(o.deformed_positions, [[1., 2., 2.], [4., 6., 2.]], rtol=1e-14, atol=0.)
                    self.assertTrue(math.isclose(o.max_displacement, 3., rel_tol=1e-14))
                    if record:
                        np.testing.assert_array_equal(o.positions_trajectory[-1], o.deformed_positions)
                    else:
                        self.assertEqual(o.positions_trajectory, [])

if __name__ == '__main__': unittest.main()
