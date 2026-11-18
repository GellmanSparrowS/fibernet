"""Physical end moments for the sparse planar Euler-Bernoulli solver."""

import math

import numpy as np
import pytest

from fibernet.ml.beam_frame_fem_sparse import SparseBeamFrameFEM


YOUNG = 2.1e9
TIP_TRANSVERSE = 1.75


def _solve(points, radii, tip_force, reverse, young=YOUNG):
    n_nodes = len(points)
    edges = np.array([np.arange(n_nodes - 1), np.arange(1, n_nodes)])
    if reverse:
        edges = edges[::-1].copy()
    forces = np.zeros_like(points)
    forces[-1] = tip_force
    solver = SparseBeamFrameFEM(E=young)
    u, axial, moments, retained = solver.solve_2d(edges, points, radii, forces, [0], damping=0.)
    np.testing.assert_array_equal(retained, np.arange(n_nodes - 1))
    K, _ = solver.build_sparse_stiffness_2d(edges, points, radii)
    applied = np.column_stack((forces, np.zeros(n_nodes)))
    reactions = (K @ u.ravel() - applied.ravel()).reshape(n_nodes, 3)
    return u, axial, moments, reactions


@pytest.mark.parametrize(("length", "angle"), [(2., 0.), (.7, .47)])
@pytest.mark.parametrize("reverse", [False, True])
def test_sparse_cantilever_recovers_section_moment_in_force_length_units(length, angle, reverse):
    tangent = np.array([math.cos(angle), math.sin(angle)])
    transverse = np.array([-tangent[1], tangent[0]])
    radius, axial_force = .04, .8
    inertia = math.pi * radius**4 / 4
    area = math.pi * radius**2
    load = axial_force*tangent + TIP_TRANSVERSE*transverse
    u, axial, moments, reactions = _solve(np.stack((np.zeros(2), length*tangent)),
                                          np.array([radius]), load, reverse)

    expected_u = (axial_force * length / (YOUNG * area) * tangent
                  + TIP_TRANSVERSE * length**3 / (3 * YOUNG * inertia) * transverse)
    expected_rotation = TIP_TRANSVERSE * length**2 / (2 * YOUNG * inertia)
    np.testing.assert_allclose(u[-1], np.r_[expected_u, expected_rotation], atol=1e-14)
    np.testing.assert_allclose(axial, [axial_force/area], atol=1e-8)
    np.testing.assert_allclose(reactions[0], np.r_[-load, -length*TIP_TRANSVERSE], atol=1e-10)
    expected_moment = [[0., length*TIP_TRANSVERSE]] if reverse else [[-length*TIP_TRANSVERSE, 0.]]
    np.testing.assert_allclose(moments, expected_moment, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("reverse", [False, True])
def test_two_unequal_members_have_the_same_section_moment_at_the_joint(reverse):
    points = np.array([[0., 0.], [2., 0.], [5., 0.]])
    _, _, moments, _ = _solve(points, np.array([.041, .037]), [0., TIP_TRANSVERSE], reverse)
    expected = TIP_TRANSVERSE * np.array([[-5., -3.], [-3., 0.]])
    if reverse:
        expected = -expected[:, ::-1]
    np.testing.assert_allclose(moments, expected, rtol=1e-10, atol=1e-10)


def test_mixed_frame_section_moments_follow_tip_force_levers():
    points = np.array([[0., 0.], [1.1, 0.], [1.1, .71]])
    load = np.array([.8, TIP_TRANSVERSE])
    _, _, moments, _ = _solve(points, np.array([.041, .037]), load, reverse=False)
    arms = points[-1] - points
    # A cut balances the load on the outboard side of each member.
    section_moment = arms[:, 1]*load[0] - arms[:, 0]*load[1]
    expected = np.column_stack((section_moment[:-1], section_moment[1:]))
    np.testing.assert_allclose(moments, expected, rtol=1e-10, atol=1e-10)


def test_section_moments_scale_with_consistent_length_units():
    points = np.array([[0., 0.], [2., 0.]])
    radii = np.array([.04])
    force = np.array([.8, TIP_TRANSVERSE])
    u_m, sigma_m2, moment_nm, _ = _solve(points, radii, force, reverse=False)
    u_mm, sigma_mm2, moment_nmm, _ = _solve(
        points * 1000, radii * 1000, force, reverse=False, young=YOUNG / 1000**2
    )
    np.testing.assert_allclose(moment_nmm, moment_nm * 1000, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(u_mm, u_m * [1000, 1000, 1], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(sigma_mm2, sigma_m2 / 1000**2, rtol=1e-10)
