"""Material admission at the shared FEM constructor, without optional runtimes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


class FEMMaterialAdmissionTests(unittest.TestCase):
    def test_invalid_modulus_rejected(self):
        for value in (0, -1, np.nan, np.inf, -np.inf, True, np.bool_(False),
                      '1000', [1000], np.array(1000.), 1 + 0j,
                      np.timedelta64(1, 'D'), 10 ** 1000):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, 'E'):
                BeamFrameFEM(E=value)

    def test_invalid_poisson_ratio_rejected(self):
        for value in (-1, .5, -2, 1, np.nan, np.inf, -np.inf,
                      True, np.bool_(False), '.3', [.3], np.array(.3),
                      .3 + 0j, np.timedelta64(1, 'D'), 10 ** 1000):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, 'nu'):
                BeamFrameFEM(nu=value)

    def test_invalid_easy_parameters_reject_before_graph_conversion(self):
        with patch.object(BeamFrameFEM, 'graph_to_fem_input') as convert:
            for key, value in (('E', np.nan), ('E', -1), ('nu', np.nan), ('nu', .5)):
                for dimension in (2, 3):
                    with self.subTest(key=key, dim=dimension):
                        graph = fn.StructureGraph(dimension=dimension)
                        with self.assertRaisesRegex(ValueError, key):
                            fn.simulate(graph, backend='fem', stretch_ratio=1.01,
                                        nonlinear=False, **{key: value})
            convert.assert_not_called()

    def test_derived_shear_modulus_must_remain_finite_positive(self):
        with self.assertRaisesRegex(ValueError, 'shear modulus'):
            BeamFrameFEM(E=1e308, nu=np.nextafter(-1., 0.))
        with self.assertRaisesRegex(ValueError, 'shear modulus'):
            BeamFrameFEM(E=np.nextafter(0., 1.), nu=.3)

    def test_valid_python_numpy_scalars_and_auxetic_materials(self):
        for modulus in (1000, 1000., np.int64(1000), np.float32(1000), np.float64(1000)):
            for ratio in (-.75, 0, .3, np.float32(.3), np.float64(.49)):
                with self.subTest(E=repr(modulus), nu=repr(ratio)):
                    fem = BeamFrameFEM(E=modulus, nu=ratio)
                    self.assertIs(fem.E, modulus)
                    self.assertIs(fem.nu, ratio)
                    self.assertEqual(fem.G, modulus / (2 * (1 + ratio)))

    def test_finite_inputs_that_overflow_stiffness_reject_before_solve(self):
        fem = BeamFrameFEM(E=1e308, nu=.3)  # Admitted inputs, failed computation.
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                positions = np.array([[0., 0., 0.], [1., 0., 0.]])[:, :dimension]
                with patch('fibernet.ml.beam_frame_fem.spsolve') as solve:
                    with np.errstate(over='ignore', invalid='ignore'):
                        with self.assertRaisesRegex(ValueError, 'nonfinite .*local beam stiffness'):
                            getattr(fem, 'solve_%dd' % dimension)(
                                np.array([[0], [1]]), positions, np.array([1.]),
                                fixed_nodes=[0])
                    solve.assert_not_called()

    def test_finite_element_entries_that_overflow_sparse_sum_reject(self):
        # Axial stiffness is finite per member but overflows when summed.
        fem = BeamFrameFEM(E=1e307, nu=.3)
        edges = np.tile(np.array([[0], [1]]), (1, 200))
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                positions = np.array([[0., 0., 0.], [1., 0., 0.]])[:, :dimension]
                with patch('fibernet.ml.beam_frame_fem.spsolve') as solve:
                    with np.errstate(over='ignore', invalid='ignore'):
                        with self.assertRaisesRegex(ValueError, 'nonfinite assembled'):
                            getattr(fem, 'solve_%dd' % dimension)(
                                edges, positions, np.full(200, .2), fixed_nodes=[0])
                    solve.assert_not_called()

    def test_overflowed_geometric_length_rejects_before_solve(self):
        fem = BeamFrameFEM(E=1000.)
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                positions = np.array([[0., 0., 0.], [1e308, 0., 0.]])[:, :dimension]
                with patch('fibernet.ml.beam_frame_fem.spsolve') as solve:
                    with np.errstate(over='ignore', invalid='ignore'):
                        with self.assertRaisesRegex(ValueError, 'nonfinite .*beam length'):
                            getattr(fem, 'solve_%dd' % dimension)(
                                np.array([[0], [1]]), positions, np.array([.05]),
                                fixed_nodes=[0])
                    solve.assert_not_called()

    def test_valid_linear_axial_response_and_save_reload_in_both_dimensions(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                graph = fn.StructureGraph(dimension=dimension)
                left = graph.add_node([0., 0., 0.])
                right = graph.add_node([1., 0., 0.])
                graph.add_edge(left, right, radius=.05)
                result = fn.simulate(graph, backend='fem', stretch_ratio=1.01,
                                     nonlinear=False, E=1000., nu=-.25)
                expected = 1000. * np.pi * .05 ** 2 * .01
                self.assertAlmostEqual(result.edge_forces[0], expected)
                self.assertTrue(np.isfinite(result.displacements).all())
                with tempfile.TemporaryDirectory() as folder:
                    target = Path(folder) / 'result.json'
                    result.save(target, detailed=True)
                    loaded = fn.SimResult.load(target)
                np.testing.assert_array_equal(loaded.displacements, result.displacements)
                np.testing.assert_array_equal(loaded.edge_forces, result.edge_forces)


if __name__ == '__main__':
    unittest.main()
