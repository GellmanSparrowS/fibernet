"""Physical beam energy against analytical laws and independent quadrature.

Synthetic linear Euler--Bernoulli cases, not experimental validation. Numerical
regularization contributes to equilibrium but is not physical strain energy.
"""

import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from test_static_network_workflow import (
    dense_oracle,
    quadrature_matrix,
    synthetic_graph,
)


def energy(raw):
    record = raw["beam_energy"]
    assert record["status"] == "available"
    assert record["excluded_terms"] == ["numerical_regularization"]
    assert np.isfinite(record["strain_energy"])
    return record["strain_energy"]


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("axis", [0, 1])
def test_cantilever_energy_closed_form_and_length_unit_rescaling(dimension, axis):
    """N/m and N/mm descriptions have energy values differing by 1000."""
    modulus, length, radius, force = 2.7e6, 1.3, 0.035, 0.004
    area, inertia = np.pi * radius**2, np.pi * radius**4 / 4
    expected = (
        force**2 * length / (2 * modulus * area)
        if axis == 0
        else force**2 * length**3 / (6 * modulus * inertia)
    )
    values = []
    for scale in [1.0, 1000.0]:
        positions = np.zeros((2, dimension))
        positions[1, 0] = length * scale
        loads = np.zeros_like(positions)
        loads[1, axis] = force
        fem = BeamFrameFEM(E=modulus / scale**2)
        solve = fem.solve_2d if dimension == 2 else fem.solve_3d
        raw = solve(
            np.array([[0], [1]]),
            positions,
            np.array([radius * scale]),
            forces=loads,
            fixed_nodes=[0],
            damping=0.0,
        )
        value = energy(raw)
        np.testing.assert_allclose(value, expected * scale, rtol=2e-12)
        np.testing.assert_allclose(value, 0.5 * force * raw["u"][1, axis], rtol=2e-12)
        assert raw["beam_energy"]["unit_system"] == "not_declared"
        values.append(value)
    np.testing.assert_allclose(values[1] / values[0], 1000, rtol=2e-12)


@pytest.mark.parametrize("rotate", [False, True])
def test_3d_mixed_axial_biaxial_bending_and_torsion_energy(rotate):
    modulus, poisson, length, radius = 3e6, 0.31, 1.4, 0.04
    area, inertia, polar = (
        np.pi * radius**2,
        np.pi * radius**4 / 4,
        np.pi * radius**4 / 2,
    )
    shear = modulus / (2 * (1 + poisson))
    axial, fy, fz, twist = 0.007, 0.003, -0.005, 0.002
    tip = np.array(
        [
            axial * length / (modulus * area),
            fy * length**3 / (3 * modulus * inertia),
            fz * length**3 / (3 * modulus * inertia),
            twist,
            -fz * length**2 / (2 * modulus * inertia),
            fy * length**2 / (2 * modulus * inertia),
        ]
    )
    expected_parts = np.array(
        [
            axial**2 * length / (2 * modulus * area),
            (fy**2 + fz**2) * length**3 / (6 * modulus * inertia),
            shear * polar * twist**2 / (2 * length),
        ]
    )
    rotation = np.eye(3)
    if rotate:
        # Rodrigues rotation about an oblique axis, unlike the solver's frame.
        axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
        cross = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        rotation += np.sin(0.73) * cross + (1 - np.cos(0.73)) * cross @ cross
    positions = np.array([[0, 0, 0], [length / 2, 0, 0], [length, 0, 0]]) @ rotation.T
    tip = np.r_[rotation @ tip[:3], rotation @ tip[3:]]
    raw = BeamFrameFEM(E=modulus, nu=poisson).solve_3d(
        np.array([[0, 1], [1, 2]]),
        positions,
        np.full(2, radius),
        fixed_nodes=[0],
        prescribed_disp={2: tip},
        damping=0.0,
    )
    np.testing.assert_allclose(energy(raw), expected_parts.sum(), rtol=2e-10)
    np.testing.assert_allclose(energy(raw), 0.5 * tip @ raw["reactions"][2], rtol=2e-10)
    assert energy(raw) > expected_parts[0] + expected_parts[1]
    assert raw["beam_energy"]["included_terms"] == ["axial", "bending", "torsion"]


def test_heterogeneous_series_torsion_energy_uses_each_poisson_ratio():
    lengths = np.array([0.7, 1.6])
    radii = np.array([0.017, 0.029])
    laws = [
        fn.Material(youngs_modulus=2e6, poissons_ratio=0.12),
        fn.Material(youngs_modulus=7e6, poissons_ratio=0.43),
    ]
    shear = np.array([m.youngs_modulus / (2 * (1 + m.poissons_ratio)) for m in laws])
    compliance = np.sum(lengths / (shear * np.pi * radii**4 / 2))
    angle = 0.003
    raw = BeamFrameFEM(E=9e8, nu=0.25).solve_3d(
        np.array([[0, 1], [1, 2]]),
        np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [2.3, 0.0, 0.0]]),
        radii,
        fixed_nodes=[0],
        prescribed_disp={2: (0, 0, 0, angle, 0, 0)},
        damping=0.0,
        edge_materials=laws,
    )
    np.testing.assert_allclose(energy(raw), angle**2 / (2 * compliance), rtol=2e-13)
    np.testing.assert_allclose(raw["sigma_axial"], 0.0, atol=1e-14)
    np.testing.assert_allclose(raw["sigma_bending"], 0.0, atol=1e-14)
    assert energy(raw) > 0


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("motion", ["zero", "translation", "infinitesimal_rotation"])
def test_rigid_motion_has_zero_physical_energy(dimension, motion):
    positions = np.array([[0.0, 0.0, 0.0], [0.8, 0.3, 0.2], [1.4, -0.2, 0.5]])[
        :, :dimension
    ]
    if dimension == 2:
        omega = 0.002 if motion == "infinitesimal_rotation" else 0.0
        displacement = omega * positions[:, [1, 0]] * [-1, 1]
    else:
        omega = (
            np.array([0.001, -0.002, 0.003])
            if motion == "infinitesimal_rotation"
            else np.zeros(3)
        )
        displacement = np.cross(omega, positions)
    if motion == "translation":
        displacement += np.arange(1, dimension + 1) * 0.002
    prescribed = {i: tuple(d) for i, d in enumerate(displacement)}
    if dimension == 3:
        prescribed = {i: tuple(d) + tuple(omega) for i, d in enumerate(displacement)}
    fem = BeamFrameFEM(E=2e6)
    solve = fem.solve_2d if dimension == 2 else fem.solve_3d
    raw = solve(
        np.array([[0, 1], [1, 2]]),
        positions,
        np.array([0.02, 0.03]),
        prescribed_disp=prescribed,
        damping=0.0,
    )
    # A rigid infinitesimal motion has no axial strain, no curvature and no
    # relative twist. Check those kinematics independently of assembled K.
    np.testing.assert_allclose(raw["u"][:, :dimension], displacement, atol=1e-16)
    expected_rotation = (
        np.full((3, 1), omega) if dimension == 2 else np.tile(omega, (3, 1))
    )
    np.testing.assert_allclose(raw["u"][:, dimension:], expected_rotation, atol=1e-14)
    for i, j in [(0, 1), (1, 2)]:
        tangent = positions[j] - positions[i]
        tangent /= np.linalg.norm(tangent)
        slope = (displacement[j] - displacement[i]) / np.linalg.norm(
            positions[j] - positions[i]
        )
        np.testing.assert_allclose(slope @ tangent, 0.0, atol=1e-17)
        rigid_slope = (
            omega * np.array([-tangent[1], tangent[0]])
            if dimension == 2
            else np.cross(omega, tangent)
        )
        np.testing.assert_allclose(slope, rigid_slope, atol=1e-17)
    # The public snapshot evaluates the matrix quadratic without clipping.
    # Valid rigid motions can therefore be unavailable for tiny negative
    # roundoff; this is a documented diagnostic limitation, not finite energy.
    u = raw["u"].ravel()
    quadratic = float(0.5 * u @ (raw["K"] @ u))
    np.testing.assert_allclose(quadratic, 0.0, atol=2e-16)
    record = raw["beam_energy"]
    if quadratic < 0:
        assert record["status"] == "unavailable"
        assert record["strain_energy"] is None
        assert record["reason"] == "negative_quadratic_form"
    else:
        np.testing.assert_allclose(energy(raw), quadratic, rtol=1e-13, atol=1e-30)


@pytest.mark.parametrize("grid", [2, 4])
def test_heterogeneous_network_energy_quadrature_and_regularized_work(grid):
    graph = synthetic_graph(grid)
    fem = BeamFrameFEM(E=9e8)
    prepared = fem.graph_to_fem_input(graph)
    pos, edges, radii = (
        prepared["node_pos"],
        prepared["edge_index"].T,
        prepared["radii"],
    )
    oracle = np.zeros((3 * len(pos), 3 * len(pos)))
    for edge, radius, law in zip(
        edges, radii, [e.material for e in graph.edges.values()]
    ):
        part, _ = quadrature_matrix(pos, [edge], [radius], law.youngs_modulus)
        oracle += part
    damping = 1e-6
    delta = prepared["x_range"] * (1.001 - 1)
    u, free, fixed = dense_oracle(
        oracle,
        prepared["boundaries"]["left"],
        prepared["boundaries"]["right"],
        delta,
        damping,
    )
    raw = fem.stretch_test(
        graph,
        target_stretch=1.001,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
    )
    result = fem.to_sim_result(raw, graph)
    expected = 0.5 * u @ oracle @ u
    np.testing.assert_allclose(energy(raw), expected, rtol=2e-8, atol=1e-13)
    np.testing.assert_allclose(raw["u"].ravel(), u, rtol=2e-8, atol=2e-11)
    residual = oracle @ u
    work = 0.5 * u[fixed] @ residual[fixed]
    penalty = 0.5 * damping * (u[free] @ u[free])
    np.testing.assert_allclose(energy(raw), work - penalty, rtol=2e-8, atol=1e-13)
    assert penalty > 0
    assert raw["beam_energy"]["unit_system"] == "SI"
    assert result.metadata["beam_energy"] == raw["beam_energy"]
    assert result.metadata["energy_kind"] == "axial_stress_proxy"
    assert energy(raw) > 1.1 * result.energy


@pytest.mark.parametrize("dimension", [2, 3])
def test_numerical_regularization_is_excluded_from_force_loaded_energy(dimension):
    modulus, radius, length, force, damping = 2e6, 0.03, 1.2, 0.004, 0.1
    positions = np.zeros((2, dimension))
    positions[1, 0] = length
    loads = np.zeros_like(positions)
    loads[1, 0] = force
    fem = BeamFrameFEM(E=modulus)
    solve = fem.solve_2d if dimension == 2 else fem.solve_3d
    raw = solve(
        np.array([[0], [1]]),
        positions,
        np.array([radius]),
        fixed_nodes=[0],
        forces=loads,
        damping=damping,
    )
    stiffness = modulus * np.pi * radius**2 / length
    extension = force / (stiffness + damping)
    expected = 0.5 * stiffness * extension**2
    np.testing.assert_allclose(energy(raw), expected, rtol=1e-13)
    np.testing.assert_allclose(
        energy(raw), 0.5 * force * extension - 0.5 * damping * extension**2, rtol=1e-13
    )
    assert abs(energy(raw) - 0.5 * force * extension) > 1e-6 * expected
