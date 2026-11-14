"""Beam radii are physical inputs, not signed or truth-valued graph labels."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


class FEMRadiusAdmissionTests(unittest.TestCase):
    def test_invalid_scalar_sections_reject(self):
        for radius in (0, -0.0, -.05, np.nan, np.inf, -np.inf,
                       True, np.bool_(False), '.05', [.05], np.array(.05),
                       .05 + 0j, np.timedelta64(1, 'D'), 10 ** 1000):
            with self.subTest(radius=repr(radius)):
                with self.assertRaisesRegex(ValueError, 'radius'):
                    BeamFrameFEM.section_properties(radius)

    def test_representable_positive_sections_keep_original_radius(self):
        for radius in (1, .05, np.int32(1), np.int64(1),
                       np.float32(.05), np.float64(.05)):
            with self.subTest(radius=repr(radius)):
                section = BeamFrameFEM.section_properties(radius)
                self.assertIs(section['r'], radius)
                r = float(radius)
                self.assertEqual(section['A'], np.pi * r ** 2)
                self.assertEqual(section['I'], np.pi * r ** 4 / 4)
                self.assertEqual(section['J'], np.pi * r ** 4 / 2)

    def test_unrepresentable_section_properties_reject(self):
        for radius in (1e-100, 1e100, np.float64(1e100)):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, 'section properties'):
                    BeamFrameFEM.section_properties(radius)

    def test_numpy_integer_section_avoids_integer_power_overflow(self):
        r = np.int64(100_000)
        actual = BeamFrameFEM.section_properties(r)
        self.assertEqual(actual['I'], np.pi * float(r) ** 4 / 4)

    def test_representable_large_section_avoids_intermediate_overflow(self):
        section = BeamFrameFEM.section_properties(1e77)
        self.assertEqual(section['I'], (np.pi / 4) * 1e77 ** 4)
        self.assertEqual(section['J'], (np.pi / 2) * 1e77 ** 4)
        self.assertTrue(all(np.isfinite(section[key]) for key in ('A', 'I', 'J')))

    def test_generated_invalid_radius_rejects_before_assembly_without_mutation(self):
        for dimension in (2, 3):
            for radius in (0., -.05, True):
                with self.subTest(dimension=dimension, radius=radius):
                    if dimension == 2:
                        graph = fn.pattern_2d(unit='square', box=(1, 1),
                                              grid=(1, 1), radius=radius)
                    else:
                        graph = fn.pattern_3d(unit='cubic', box=(1, 1, 1),
                                              grid=(1, 1, 1), radius=radius)
                    before = copy.deepcopy(graph.to_dict())
                    with patch.object(BeamFrameFEM, 'solve_2d') as solve2:
                        with patch.object(BeamFrameFEM, 'solve_3d') as solve3:
                            with self.assertRaisesRegex(ValueError, 'radius'):
                                fn.simulate(graph, backend='fem', stretch_ratio=1.01,
                                            nonlinear=False, E=1000.)
                    solve2.assert_not_called()
                    solve3.assert_not_called()
                    self.assertEqual(graph.to_dict(), before)

    def test_mixed_graph_radii_reject_before_numpy_can_coerce_bool(self):
        graph = fn.pattern_2d(unit='square', box=(1, 1), grid=(1, 1), radius=.05)
        list(graph.edges.values())[-1].radius = True
        with self.assertRaisesRegex(ValueError, 'radius'):
            BeamFrameFEM.graph_to_fem_input(graph)

    def test_direct_2d_3d_solvers_reject_before_sparse_solve(self):
        edges = np.array([[0], [1]])
        for dim in (2, 3):
            positions = np.array([[0., 0., 0.], [1., 0., 0.]])[:, :dim]
            for radius in (0., -.05, True):
                with self.subTest(dim=dim, radius=radius):
                    with patch('fibernet.ml.beam_frame_fem.spsolve') as solve:
                        with self.assertRaisesRegex(ValueError, 'radius'):
                            getattr(BeamFrameFEM(), 'solve_%dd' % dim)(
                                edges, positions, np.array([radius]), fixed_nodes=[0])
                    solve.assert_not_called()

    def test_generated_graph_material_roundtrip_and_explicit_solver_precedence(self):
        for dim in (2, 3):
            with self.subTest(dim=dim):
                material = fn.Material(youngs_modulus=1234.)
                factory = fn.pattern_2d if dim == 2 else fn.pattern_3d
                graph = factory(unit='square' if dim == 2 else 'cubic',
                                box=(1,) * dim, grid=(1,) * dim,
                                radius=.05, material=material)
                before = copy.deepcopy(graph.to_dict())
                restored = fn.StructureGraph.from_dict(before)
                for edge in restored.edges.values():
                    self.assertEqual(edge.radius, .05)
                    self.assertEqual(edge.material.youngs_modulus, 1234.)
                    edge.material.youngs_modulus = 9876.
                options = dict(backend='fem', stretch_ratio=1.01,
                               nonlinear=False, E=1e9, nu=.3)
                a = fn.simulate(graph, **options)
                b = fn.simulate(restored, **options)
                np.testing.assert_array_equal(a.edge_forces, b.edge_forces)
                np.testing.assert_array_equal(a.displacements, b.displacements)
                self.assertTrue(np.isfinite(a.edge_forces).all())
                self.assertEqual(graph.to_dict(), before)
                with tempfile.TemporaryDirectory() as folder:
                    target = Path(folder) / 'result.json'
                    a.save(target, detailed=True)
                    saved = fn.SimResult.load(target)
                np.testing.assert_array_equal(a.edge_forces, saved.edge_forces)

    def test_explicit_metre_millimetre_units_preserve_axial_force(self):
        # N/m^2 with metres versus N/mm^2 with millimetres. No auto-conversion.
        forces = []
        for length_scale in (1., 1000.):
            graph = fn.StructureGraph(dimension=2)
            a = graph.add_node([0., 0.])
            b = graph.add_node([length_scale, 0.])
            graph.add_edge(a, b, radius=.05 * length_scale)
            result = fn.simulate(graph, backend='fem', stretch_ratio=1.01,
                                 nonlinear=False, E=1000. / length_scale ** 2)
            forces.append(result.edge_forces[0])
        np.testing.assert_allclose(forces, [1000. * np.pi * .05 ** 2 * .01] * 2)


if __name__ == '__main__':
    unittest.main()
