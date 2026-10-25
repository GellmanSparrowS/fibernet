"""Analytical reduced-spring checks for state-aligned forces and options."""

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver


def _graph(points, edges, dimension=2):
    graph = StructureGraph(dimension=dimension)
    for position in points:
        graph.add_node(position)
    for first, second in edges:
        graph.add_edge(first, second)
    return graph


@pytest.mark.parametrize("num_steps", [1, 2, 10])
@pytest.mark.parametrize("n_increments", [0, 1, 2])
def test_gripped_spring_force_and_work_match_recorded_extension(num_steps, n_increments):
    """Both integrators must recover F=k*extension/L0 and W=k*extension²/(2L0)."""
    graph = _graph([[0., 0.], [1., 0.]], [(0, 1)])
    result = ReducedBeamSolver(graph, ReducedBeamConfig(
        target_stretch=1.1, stiffness=100., use_bending=False,
        use_contact=False, num_steps=num_steps, save_interval=1,
        ramp_fraction=1., n_increments=n_increments,
    )).run()

    extension = result.edge_strain[:, 0].astype(float)
    np.testing.assert_allclose(result.force_curve, 100. * extension, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.energies['work'], result.energies['axial'], rtol=1e-12)
    assert result.force_curve[-1] == pytest.approx(10.)
    assert result.energies['work'][-1] == pytest.approx(.5)
    assert result.metadata['force_sampling'] == 'recorded_configuration'
    assert result.metadata['work_quadrature'] == 'trapezoidal_grip_reaction'


@pytest.mark.parametrize("n_increments", [0, 2])
def test_free_node_reaction_matches_same_saved_configuration(n_increments):
    """Check a moving free node, beyond the fully gripped analytical case."""
    graph = _graph([[0., 0.], [1., .2], [2., 0.]], [(0, 1), (1, 2)])
    solver = ReducedBeamSolver(graph, ReducedBeamConfig(
        target_stretch=1.1, stiffness=100., use_bending=True,
        use_contact=False, num_steps=20, save_interval=1,
        ramp_fraction=1., n_increments=n_increments, dt=.001,
    ))
    result = solver.run()

    for frame, reported in zip(result.frames_xy, result.force_curve):
        forces, _ = solver._forces(frame.astype(float), np.empty((0, 2), dtype=int), 1)
        expected = -forces[solver.right, 0].sum()
        # Saved coordinates use float32; compare at their storage precision.
        assert reported == pytest.approx(expected, rel=3e-5, abs=3e-5)


@pytest.mark.parametrize("use_contact", [False, True])
def test_bending_can_be_disabled_for_a_degree_two_chain(use_contact):
    graph = _graph([[0., 0.], [1., 0.], [2., 0.]], [(0, 1), (1, 2)])
    result = ReducedBeamSolver(graph, ReducedBeamConfig(
        use_bending=False, use_contact=use_contact,
        num_steps=4, save_interval=1,
    )).run()
    assert result.n_frames == 5
    assert np.isfinite(result.force_curve).all()
    np.testing.assert_array_equal(result.energies['bend'], 0.)


def test_every_step_contact_rebuild_detects_new_contacts():
    graph = _graph([[0., 0.], [1., 0.], [0., 3.], [1., 3.]], [(0, 1), (2, 3)])
    solver = ReducedBeamSolver(graph, ReducedBeamConfig(
        use_bending=False, use_contact=True, r_contact=1.,
        contact_rebuild=1, k_contact=20.,
    ))
    candidates = solver._contact_candidates(solver.pos0, fresh=True)
    assert not len(candidates)

    for step, separation in [(1, .5), (2, 3.), (3, .25)]:
        positions = solver.pos0.copy()
        positions[2:, 1] = separation
        forces, candidates = solver._forces(positions, candidates, step)
        expected_vertical = 20. * max(1. - separation, 0.)
        np.testing.assert_allclose(forces[:, 1],
                                   [-expected_vertical, -expected_vertical,
                                    expected_vertical, expected_vertical])
        np.testing.assert_allclose(forces.sum(axis=0), 0.)


@pytest.mark.parametrize('dimension', [2, 3])
def test_nonplanar_positions_are_rejected_before_contact_search(dimension, monkeypatch):
    graph = _graph([[0., 0., 0.], [1., 0., 1.]], [(0, 1)], dimension=dimension)

    def unexpected(*args, **kwargs):
        pytest.fail('nonplanar input reached contact search')

    monkeypatch.setattr(ReducedBeamSolver, '_contact_candidates', unexpected)
    with pytest.raises(ValueError, match='one XY plane only'):
        ReducedBeamSolver(graph)


def test_constant_z_plane_has_the_same_planar_response():
    graph = _graph([[0., 0., 0.], [1., 0., 0.]], [(0, 1)])
    config = ReducedBeamConfig(num_steps=2, save_interval=1, use_contact=False)
    reference = ReducedBeamSolver(graph, config).run()
    graph = _graph([[0., 0., 17.], [1., 0., 17.]], [(0, 1)], dimension=3)
    translated = ReducedBeamSolver(graph, config).run()
    np.testing.assert_array_equal(reference.frames_xy, translated.frames_xy)
    np.testing.assert_array_equal(reference.force_curve, translated.force_curve)


@pytest.mark.parametrize('point', [[.5, 0., 0.], [.5, .5, 0.]])
def test_edge_owned_samples_require_explicit_graph_subdivision(point, monkeypatch):
    graph = _graph([[0., 0.], [1., 0.]], [(0, 1)])
    graph.edges[0].internal_points = np.array([point])

    def unexpected(*args, **kwargs):
        pytest.fail('edge-owned internal points reached contact search')

    monkeypatch.setattr(ReducedBeamSolver, '_contact_candidates', unexpected)
    with pytest.raises(ValueError, match='actual graph nodes'):
        ReducedBeamSolver(graph)


def test_empty_internal_point_array_remains_supported():
    graph = _graph([[0., 0.], [1., 0.]], [(0, 1)])
    graph.edges[0].internal_points = np.empty((0, 3))
    result = ReducedBeamSolver(graph, ReducedBeamConfig(
        num_steps=1, save_interval=1, use_contact=False,
    )).run()
    assert result.n_frames == 2
