"""Analytical linear Euler-Bernoulli recovery invariants (zero damping).

These tests do not validate finite rotations or nonlinear equilibrium.
"""
import numpy as np
import pytest

from fibernet.ml.beam_frame_fem import BeamFrameFEM

E, R, L, P = 2000., .2, 2., .01
I = np.pi * R**4 / 4
EDGES = np.array([[0], [1]])
RADII = np.array([R])


def test_2d_infinitesimal_rigid_rotation_has_zero_stress():
    fem = BeamFrameFEM(E=E)
    pos = np.array([[0., 0.], [L, 0.]])
    angle = .001
    u = np.array([[0., 0., angle], [0., angle * L, angle]])
    k, selected = fem.build_stiffness_2d(EDGES, pos, RADII)
    np.testing.assert_allclose(k @ u.ravel(), 0., atol=1e-14)
    assert abs(.5 * u.ravel() @ k @ u.ravel()) < 1e-14
    for field in fem._compute_element_stress_2d(pos, u, EDGES, selected, RADII):
        np.testing.assert_allclose(field, 0., atol=1e-12)


def test_2d_tip_force_cantilever_analytical_response():
    fem = BeamFrameFEM(E=E)
    out = fem.solve_2d(EDGES, np.array([[0., 0.], [L, 0.]]), RADII,
                       forces=np.array([[0., 0.], [0., P]]),
                       fixed_nodes=[0], damping=0)
    np.testing.assert_allclose(out['u'][1],
                               [0., P*L**3/(3*E*I), P*L**2/(2*E*I)], atol=1e-14)
    np.testing.assert_allclose(out['reactions'][0], [0., -P, -P*L], atol=1e-14)
    flat = out['u'].ravel()
    np.testing.assert_allclose(.5 * flat @ out['K'] @ flat, P**2*L**3/(6*E*I))
    np.testing.assert_allclose(out['moments'][0], [-P*L, 0.], atol=1e-14)
    np.testing.assert_allclose(out['edge_forces'][0], [0., P, -P*L/2], atol=1e-14)
    np.testing.assert_allclose(out['sigma_bending'], P*L*R/I)


def test_2d_recovery_agrees_with_local_stiffness_end_forces():
    fem = BeamFrameFEM(E=E)
    pos = np.array([[0., 0.], [L, 0.]])
    u = np.array([[.001, .002, -.003], [-.004, .005, .006]])
    k, selected = fem.build_stiffness_2d(EDGES, pos, RADII)
    q = k @ u.ravel()
    _, _, _, moments, forces = fem._compute_element_stress_2d(pos, u, EDGES, selected, RADII)
    np.testing.assert_allclose(moments[0], [q[2], -q[5]], atol=1e-14)
    np.testing.assert_allclose(forces[0, :2], [-q[0], q[1]], atol=1e-14)


@pytest.mark.parametrize('tip', [[0., P, 0.], [0., 0., P],
                               [0., P, P], [0., .6*P, -.8*P]])
@pytest.mark.parametrize('rotated', [False, True])
def test_3d_circular_cantilever_analytical_response(tip, rotated):
    fem = BeamFrameFEM(E=E)
    pos = np.array([[0., 0., 0.], [L, 0., 0.]])
    tip = np.array(tip)
    # Proper rotation moves the beam across the solver's local-basis branch.
    q = np.array([[.36, -.48, .8], [.8, .6, 0.], [-.48, .64, .6]]) if rotated else np.eye(3)
    pos, tip = pos @ q.T, q @ tip
    out = fem.solve_3d(EDGES, pos, RADII, forces=np.array([np.zeros(3), tip]),
                      fixed_nodes=[0], damping=0)
    expected_translation = tip * L**3/(3*E*I)
    expected_rotation = np.cross(pos[1]/L, tip) * L**2/(2*E*I)
    np.testing.assert_allclose(out['u'][1, :3], expected_translation, atol=1e-14)
    np.testing.assert_allclose(out['u'][1, 3:], expected_rotation, atol=1e-14)
    np.testing.assert_allclose(out['reactions'][0, :3], -tip, atol=1e-14)
    np.testing.assert_allclose(out['reactions'][0, 3:], -np.cross(pos[1], tip), atol=1e-14)
    flat = out['u'].ravel()
    np.testing.assert_allclose(.5*flat @ out['K'] @ flat,
                               np.dot(tip, tip)*L**3/(6*E*I))
    np.testing.assert_allclose(out['sigma_bending'], np.linalg.norm(tip)*L*R/I)


@pytest.mark.parametrize('reverse', [False, True])
def test_3d_combines_moment_components_at_same_endpoint(reverse):
    # x-axis local e2=global z, e3=-global y. Cubics v=a*x^3,
    # w=b*x^2+c*x^3 give endpoint curvatures (0, .004), (.006, 0).
    a, b, c = .006/(6*L), .002, -.004/(6*L)
    v, vp = a*L**3, 3*a*L**2
    w, wp = b*L**2+c*L**3, 2*b*L+3*c*L**2
    u = np.array([[0., 0., 0., 0., 0., 0.], [0., -w, v, 0., -vp, -wp]])
    edges = EDGES[::-1] if reverse else EDGES
    out = BeamFrameFEM(E=E).solve_3d(edges, np.array([[0., 0., 0.], [L, 0., 0.]]),
                                    RADII, prescribed_disp=dict(enumerate(u)), damping=0)
    np.testing.assert_allclose(out['sigma_bending'], E*R*.006, atol=1e-12)


def test_3d_infinitesimal_rigid_motion_has_zero_stress():
    pos = np.array([[.4, -.5, .7], [1.5, .8, -.3]])
    omega = np.array([.001, -.002, .003])
    translation = np.array([.004, -.005, .006])
    u = np.column_stack((translation + np.cross(omega, pos), np.tile(omega, (2, 1))))
    out = BeamFrameFEM(E=E).solve_3d(EDGES, pos, RADII,
                                    prescribed_disp=dict(enumerate(u)), damping=0)
    np.testing.assert_allclose(out['K'] @ u.ravel(), 0., atol=1e-14)
    assert abs(.5*u.ravel() @ out['K'] @ u.ravel()) < 1e-14
    np.testing.assert_allclose(out['sigma_total'], 0., atol=1e-12)
