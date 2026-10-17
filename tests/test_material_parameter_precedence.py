"""Stored edge Material does not silently override existing solver parameters."""
import importlib.util

import numpy as np
import pytest

import fibernet as fn
from fibernet.sim.reduced_beam import ReducedBeamConfig, ReducedBeamSolver


def bar(modulus):
    graph = fn.StructureGraph(dimension=2)
    graph.add_node([0, 0]); graph.add_node([1, 0])
    graph.add_edge(0, 1, radius=.05,
                   material=fn.Material(youngs_modulus=modulus))
    return graph


def test_easy_fem_uses_explicit_global_E_not_stored_edge_material():
    a = fn.simulate(bar(1234.), backend='fem', strain=1.01, E=1000., nonlinear=False)
    b = fn.simulate(bar(9876.), backend='fem', strain=1.01, E=1000., nonlinear=False)
    np.testing.assert_array_equal(a.edge_forces, b.edge_forces)
    c = fn.simulate(bar(1234.), backend='fem', strain=1.01, E=2000., nonlinear=False)
    np.testing.assert_allclose(c.edge_forces, 2*a.edge_forces)


def test_reduced_model_parameters_do_not_infer_edge_modulus():
    cfg = ReducedBeamConfig(target_stretch=1.01, num_steps=4, save_interval=1,
                            stiffness=100., use_bending=False, use_contact=False)
    a = ReducedBeamSolver(bar(1234.), cfg).run()
    b = ReducedBeamSolver(bar(9876.), cfg).run()
    np.testing.assert_array_equal(a.frames_xy, b.frames_xy)
    np.testing.assert_array_equal(a.force_curve, b.force_curve)


@pytest.mark.skipif(importlib.util.find_spec('taichi') is None,
                    reason='Real Taichi CPU runtime is optional')
def test_spring_global_coefficient_is_independent_of_stored_modulus():
    from fibernet.sim.accelerated import TaichiEngine
    engine = TaichiEngine()
    kwargs = dict(fixed_nodes=[0], displacement_schedule={1: [(1, [.01, 0, 0])]},
                  spring_k=100., stiffness=100., num_steps=2, save_interval=1)
    a = engine.dynamics(bar(1234.), **kwargs)
    b = engine.dynamics(bar(9876.), **kwargs)
    np.testing.assert_array_equal(a.edge_forces, b.edge_forces)
    np.testing.assert_array_equal(a.displacements, b.displacements)


def test_fracture_checks_homogeneous_material_match_instead_of_overriding():
    pytest.importorskip('psutil')
    from fibernet.sim.fracture import FractureConfig, FractureSolver
    graph = bar(1234.)
    config = FractureConfig(young_modulus_Pa=1000., poisson_ratio=.3,
                           normal_strength_Pa=1e6, shear_strength_Pa=1e6,
                           axis=0, maximum_displacement_m=.001)
    with pytest.raises(ValueError, match='homogeneous isotropic'):
        FractureSolver(graph, config, fixed_nodes=[0], loaded_nodes=[1], length_unit='m')


def test_spring_force_like_coefficient_and_energy_length_scaling():
    forces, energies = [], []
    for length in (1., 2.):
        graph = bar(1234.)
        graph.set_node_position(1, [length, 0])
        reference = graph.node_positions()
        deformed = reference.copy(); deformed[1, 0] *= 1.1
        result = fn.SimResult(deformed_positions=deformed,
                              displacements=deformed-reference)
        result.compute_detailed(graph, stiffness=100.)
        forces.append(result.edge_forces[0]); energies.append(result.energy)
    np.testing.assert_allclose(forces, [10., 10.])  # F=k*strain, independent of L0.
    np.testing.assert_allclose(energies, [.5, 1.])  # U=.5*k*strain**2*L0.
