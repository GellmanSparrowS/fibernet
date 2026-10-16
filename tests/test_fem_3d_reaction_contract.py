"""Linear reactions use physical stiffness; regularization still affects u.

These small analytical fixtures do not certify nonlinear or arbitrary-network
equilibrium. In particular, physical free-row residuals need not vanish.
"""
import json

import numpy as np
import pytest

from fibernet import StructureGraph, simulate
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim.accelerated import SimResult


E, LENGTH, EXTENSION = 1000., 1., .01
RTOL = ATOL = 1e-12
DAMPING = [pytest.param(None, id="default"), 0., 1e-3, .25]


def bar_inputs(dim, parallel):
    # The second, unequal member reverses orientation without changing tension.
    edges = np.array([[0, 1], [1, 0]]) if parallel else np.array([[0], [1]])
    radii = np.array([.05, .1] if parallel else [.05])
    positions = np.array([[0., 0., 0.], [LENGTH, 0., 0.]])[:, :dim]
    return edges, positions, radii


def damping_kwargs(damping):
    return {} if damping is None else {"damping": damping}


def damping_value(damping):
    return 1e-6 if damping is None else damping


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("damping", DAMPING)
def test_prescribed_axial_reactions_exclude_regularization(dim, parallel, damping):
    edges, positions, radii = bar_inputs(dim, parallel)
    forces = np.zeros((2, dim))
    forces[:, 0] = [.02, .03]  # Loads on constrained DOFs must be subtracted.
    raw = getattr(BeamFrameFEM(E=E), f"solve_{dim}d")(
        edges, positions, radii, forces=forces, fixed_nodes=[0],
        prescribed_disp={1: (EXTENSION,) + (0.,) * (dim - 1)},
        **damping_kwargs(damping))
    member_forces = E * np.pi * radii**2 * EXTENSION / LENGTH
    expected = np.zeros_like(raw["reactions"])
    expected[:, 0] = [-member_forces.sum() - .02, member_forces.sum() - .03]
    np.testing.assert_allclose(raw["reactions"], expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(raw["u"][:, 0], [0., EXTENSION], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(raw["edge_forces"][:, 0], member_forces,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(raw["sigma_axial"], E * EXTENSION / LENGTH,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(raw["edge_list"], np.arange(len(radii)))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("damping", DAMPING)
def test_free_axial_row_retains_physical_regularization_residual(dim, parallel, damping):
    edges, positions, radii = bar_inputs(dim, parallel)
    forces = np.zeros((2, dim))
    forces[:, 0] = [.02, .03]
    root_displacement = .004
    raw = getattr(BeamFrameFEM(E=E), f"solve_{dim}d")(
        edges, positions, radii, forces=forces, fixed_nodes=[0],
        prescribed_disp={0: (root_displacement,) + (0.,) * (dim - 1)},
        **damping_kwargs(damping))
    stiffness = E * np.pi * np.sum(radii**2) / LENGTH
    regularization = damping_value(damping)
    tip_displacement = (.03 + stiffness * root_displacement) / (stiffness + regularization)
    tension = stiffness * (tip_displacement - root_displacement)
    expected = np.zeros_like(raw["reactions"])
    expected[:, 0] = [-tension - .02, -regularization * tip_displacement]
    np.testing.assert_allclose(raw["u"][:, 0], [root_displacement, tip_displacement],
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(raw["reactions"], expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(raw["reactions"][1], -regularization * raw["u"][1],
                               rtol=RTOL, atol=ATOL)
    if regularization:
        assert abs(raw["reactions"][1, 0]) > ATOL


@pytest.mark.parametrize("damping", DAMPING)
def test_mixed_3d_translations_and_rotations_use_physical_stiffness(damping):
    edges, positions, radii = bar_inputs(3, True)
    forces = np.array([[.02, -.01, .015], [.03, .005, -.007]])
    root = np.array([.004, -.002, .003, .005, -.004, .006])
    raw = BeamFrameFEM(E=E).solve_3d(
        edges, positions, radii, forces=forces,
        prescribed_disp={0: root, 1: (EXTENSION,)}, **damping_kwargs(damping))
    external = np.zeros((2, 6))
    external[:, :3] = forces
    physical = (raw["K"] @ raw["u"].ravel()).reshape(2, 6) - external
    np.testing.assert_array_equal(raw["u"][0], root)
    np.testing.assert_allclose(raw["reactions"], physical, rtol=RTOL, atol=ATOL)
    # The other five tip DOFs are free, including all three rotations.
    np.testing.assert_allclose(raw["reactions"][1, 1:],
                               -damping_value(damping) * raw["u"][1, 1:],
                               rtol=RTOL, atol=ATOL)
    assert np.all(np.abs(raw["u"][1, 1:]) > ATOL)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("damping", DAMPING)
def test_planar_2d_and_3d_recovery_agree(parallel, damping):
    results = []
    for dim in (2, 3):
        edges, positions, radii = bar_inputs(dim, parallel)
        forces = np.zeros((2, dim))
        forces[:, :2] = [[.02, -.01], [.03, .005]]
        results.append(getattr(BeamFrameFEM(E=E), f"solve_{dim}d")(
            edges, positions, radii, forces=forces, fixed_nodes=[0],
            prescribed_disp={0: (.004, -.002) + (0.,) * (dim - 2)},
            **damping_kwargs(damping)))
    planar, spatial = results
    for field in ("u", "reactions"):
        np.testing.assert_allclose(spatial[field][:, [0, 1, 5]], planar[field],
                                   rtol=RTOL, atol=ATOL)
    for field in ("sigma_axial", "sigma_bending", "sigma_total"):
        np.testing.assert_allclose(spatial[field], planar[field], rtol=RTOL, atol=ATOL)


def sparse_reversed_graph(parallel):
    graph = StructureGraph(dimension=3)
    graph.add_node([0, 0, 0])
    graph.add_node([LENGTH, 0, 0])
    graph.add_edge(0, 1, radius=.05)
    if parallel:
        graph.add_edge(1, 0, radius=.1, allow_parallel=True)
    data = graph.to_dict()
    node_ids = {0: 7, 1: 42}
    for node in data["nodes"]:
        node["id"] = node_ids[node["id"]]
    for edge in data["edges"]:
        edge["id"] = 11 + 8 * edge["id"]
        edge["node_i"] = node_ids[edge["node_i"]]
        edge["node_j"] = node_ids[edge["node_j"]]
    data["nodes"].reverse()
    data.pop("next_node_id")
    data.pop("next_edge_id")
    return StructureGraph.from_dict(data)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("detailed", [False, True])
def test_reactions_survive_json_and_sparse_node_alignment(parallel, detailed, tmp_path):
    from fibernet.viz.render import _align_result

    graph = sparse_reversed_graph(parallel)
    fem = BeamFrameFEM(E=E)
    inputs = fem.graph_to_fem_input(graph, dim=3)
    raw = fem.solve_3d(
        inputs["edge_index"], inputs["node_pos"], inputs["radii"],
        fixed_nodes=[1], prescribed_disp={0: (EXTENSION, 0., 0.)},
        forces=np.array([[.03, 0., 0.], [.02, 0., 0.]]))
    result = fem.to_sim_result(raw, graph)
    path = tmp_path / "reactions.json"
    result.save(path, detailed=detailed)
    saved = json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
    restored = SimResult.load(path)
    total = E * np.pi * np.sum(inputs["radii"]**2) * EXTENSION / LENGTH
    expected = np.zeros((2, 6))
    expected[:, 0] = [total - .03, -total - .02]  # Raw row order: IDs 42, 7.
    for reactions in (raw["reactions"], saved["reactions"], restored.metadata["reactions"]):
        np.testing.assert_allclose(reactions, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(restored.metadata["reactions"], raw["reactions"])
    assert restored.metadata["node_ids"] == [42, 7]
    assert restored.metadata["edge_ids"] == ([11, 19] if parallel else [11])
    assert restored.max_force == result.max_force == max(abs(raw["edge_forces"][:, 0]))
    if detailed:
        aligned = _align_result(graph, restored, require_edges=True)
        assert aligned.metadata["node_ids"] == [7, 42]
        np.testing.assert_allclose(aligned.metadata["reactions"], expected[::-1],
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(restored.metadata["reactions"], raw["reactions"])


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("route", ["quick", "stretch"])
def test_default_public_3d_routes_recover_physical_grip_forces(parallel, route):
    graph = sparse_reversed_graph(parallel)
    if route == "quick":
        result = simulate(graph, backend="fem", strain=1.01, E=E, nonlinear=False)
    else:
        fem = BeamFrameFEM(E=E)
        result = fem.to_sim_result(fem.stretch_test(graph, target_stretch=1.01, dim=3), graph)
    radii = np.array([.05, .1] if parallel else [.05])
    expected = E * np.pi * radii**2 * EXTENSION / LENGTH
    np.testing.assert_allclose(result.edge_forces, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(result.metadata["reactions"])[:, 0],
                               [expected.sum(), -expected.sum()], rtol=RTOL, atol=ATOL)
