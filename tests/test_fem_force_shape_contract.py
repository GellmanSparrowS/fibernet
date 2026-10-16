"""Linear beam loads contain translations only, in exact input-node row order."""
import numpy as np
import pytest
from scipy import sparse

from fibernet import StructureGraph
import fibernet.ml.beam_frame_fem as beam_module
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def bar_inputs(dim):
    return (np.array([[0], [1]]),
            np.array([[0., 0., 0.], [1., 0., 0.]])[:, :dim],
            np.array([.05]))


def solve(dim, forces):
    return getattr(BeamFrameFEM(E=1000.), f"solve_{dim}d")(
        *bar_inputs(dim), forces=forces, fixed_nodes=[0])


def assert_same_result(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if sparse.issparse(actual[name]):
            assert actual[name].format == expected[name].format
            np.testing.assert_array_equal(actual[name].toarray(), expected[name].toarray())
        else:
            np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", [
    "moment_columns", "one_extra_column", "missing_column", "no_columns",
    "broadcastable_single_row", "extra_row", "no_rows", "vector", "scalar", "rank_three",
])
def test_invalid_shapes_rejected_before_assembly_or_solve(dim, kind, monkeypatch):
    shapes = {
        "moment_columns": (2, 3 if dim == 2 else 6),
        "one_extra_column": (2, dim + 1),
        "missing_column": (2, dim - 1),
        "no_columns": (2, 0),
        "broadcastable_single_row": (1, dim),
        "extra_row": (3, dim),
        "no_rows": (0, dim),
        "vector": (2 * dim,),
        "scalar": (),
        "rank_three": (2, dim, 1),
    }
    shape = shapes[kind]
    forces = np.zeros(shape)
    fem = BeamFrameFEM(E=1000.)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid forces reached stiffness assembly or the linear solver")

    # 2D has a separate builder; 3D assembles inline via section_properties/coo.
    monkeypatch.setattr(fem, "build_stiffness_2d", forbidden)
    monkeypatch.setattr(fem, "section_properties", forbidden)
    monkeypatch.setattr(beam_module.sparse, "coo_matrix", forbidden)
    monkeypatch.setattr(beam_module, "spsolve", forbidden)
    with pytest.raises(ValueError) as exc:
        getattr(fem, f"solve_{dim}d")(*bar_inputs(dim), forces=forces, fixed_nodes=[0])
    message = str(exc.value)
    assert "forces" in message and "shape" in message
    assert str((2, dim)) in message and str(shape) in message
    assert "translational" in message and "moment" in message


@pytest.mark.parametrize("dim", [2, 3])
def test_native_moment_columns_are_not_silently_ignored(dim):
    forces = np.zeros((2, 3 if dim == 2 else 6))
    forces[1, -1] = .01
    with pytest.raises(ValueError, match="moment"):
        solve(dim, forces)


@pytest.mark.parametrize("dim", [2, 3])
def test_none_matches_explicit_zero_translational_loads(dim):
    result = solve(dim, None)
    assert_same_result(result, solve(dim, np.zeros((2, dim))))
    np.testing.assert_array_equal(result["u"], 0.)
    np.testing.assert_array_equal(result["reactions"], 0.)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_correct_translational_arrays_retain_supported_numeric_dtypes(dim, dtype):
    forces = np.zeros((2, dim), dtype=dtype)
    forces[1] = np.arange(1, dim + 1)
    before = forces.copy()
    result = solve(dim, forces)
    assert_same_result(result, solve(dim, forces.astype(np.float64)))
    np.testing.assert_array_equal(forces, before)
    assert np.any(result["u"] != 0.)
    assert np.isfinite(result["u"]).all()
    assert np.isfinite(result["reactions"]).all()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("readonly", [False, True])
def test_noncontiguous_and_readonly_load_arrays_remain_supported(dim, readonly):
    forces = np.zeros((2, 2 * dim))[:, ::2]
    forces[1] = np.arange(1, dim + 1)
    forces.setflags(write=not readonly)
    assert not forces.flags.c_contiguous
    assert_same_result(solve(dim, forces), solve(dim, forces.copy()))


@pytest.mark.parametrize("dim", [2, 3])
def test_translational_load_rows_follow_sparse_graph_input_order(dim):
    graph = StructureGraph(dimension=dim)
    graph.add_node([0., 0., 0.])
    graph.add_node([1., 0., 0.])
    graph.add_edge(0, 1, radius=.05)
    data = graph.to_dict()
    ids = {0: 7, 1: 42}
    for node in data["nodes"]:
        node["id"] = ids[node["id"]]
    for edge in data["edges"]:
        edge["id"] = 11
        edge["node_i"], edge["node_j"] = ids[edge["node_i"]], ids[edge["node_j"]]
    data["nodes"].reverse()
    data.pop("next_node_id")
    data.pop("next_edge_id")
    graph = StructureGraph.from_dict(data)
    fem = BeamFrameFEM(E=1000.)
    inputs = fem.graph_to_fem_input(graph, dim=dim)
    forces = np.zeros((2, dim))
    forces[0, 0] = .125  # First raw row is tip ID 42; fixed row is root ID 7.
    raw = getattr(fem, f"solve_{dim}d")(
        inputs["edge_index"], inputs["node_pos"], inputs["radii"],
        forces=forces, fixed_nodes=[1])
    result = fem.to_sim_result(raw, graph)
    assert result.metadata["node_ids"] == [42, 7]
    assert result.metadata["edge_ids"] == [11]
    np.testing.assert_array_equal(result.metadata["reactions"], raw["reactions"])
    np.testing.assert_allclose(raw["u"][0, 0], .125 / (1000 * np.pi * .05**2 + 1e-6))
    assert raw["u"][1, 0] == 0.


@pytest.mark.parametrize("dim", [2, 3])
def test_supported_cpu_torch_inputs_keep_existing_conversion(dim):
    torch = pytest.importorskip("torch")
    inputs = bar_inputs(dim)
    forces = np.zeros((2, dim))
    forces[1] = np.arange(1, dim + 1)
    result = getattr(BeamFrameFEM(E=1000.), f"solve_{dim}d")(
        *(torch.from_numpy(value) for value in inputs),
        forces=torch.from_numpy(forces), fixed_nodes=[0])
    assert_same_result(result, solve(dim, forces))


@pytest.mark.parametrize("dim", [2, 3])
def test_invalid_torch_force_shape_is_checked_after_conversion(dim, monkeypatch):
    torch = pytest.importorskip("torch")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid tensor forces reached the linear solver")

    monkeypatch.setattr(beam_module, "spsolve", forbidden)
    with pytest.raises(ValueError, match="translational"):
        solve(dim, torch.zeros((2, 3 if dim == 2 else 6), dtype=torch.float64))


@pytest.mark.parametrize("shape", [(2, 3), (1, 2)])
def test_nonlinear_forwarding_inherits_linear_shape_rejection(shape, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Forwarded invalid forces reached stiffness assembly")

    monkeypatch.setattr(BeamFrameFEM, "build_stiffness_2d", forbidden)
    with pytest.raises(ValueError, match="translational"):
        BeamFrameFEM(E=1000.).solve_2d_nonlinear(
            *bar_inputs(2), forces=np.zeros(shape), fixed_nodes=[0],
            prescribed_disp={1: (.01, 0.)}, n_steps=1, max_iter=1)
