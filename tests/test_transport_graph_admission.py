"""Graph identity and coordinate admission shared by both transport owners."""

import copy
import json

import numpy as np
import pytest

from fibernet import StructureGraph
from fibernet.sim.linear_transport import solve_linear_transport
from fibernet.sim.transient_thermal import solve_transient_thermal


def graph():
    return StructureGraph.from_dict(
        {
            "dimension": 2,
            "nodes": [
                {"id": 30, "position": [1.0, 0.0, 0.0]},
                {"id": 10, "position": [0.0, 0.0, 0.0]},
                {"id": 90, "position": [2.0, 0.0, 0.0]},
            ],
            "edges": [
                {"id": 7, "node_i": 10, "node_j": 30},
                {"id": 2, "node_i": 30, "node_j": 10},
                {"id": 81, "node_i": 30, "node_j": 90},
            ],
        }
    )


def electrical(g):
    return solve_linear_transport(
        g,
        {7: 2.0, 2: 1.0, 81: 0.0},
        {10: 1.0, 30: 0.0, 90: 7.0},
        length_unit="m",
    )


def thermal(g):
    return solve_transient_thermal(
        g,
        {10: 1.0, 30: 1.0, 90: 1.0},
        {7: 2.0, 2: 1.0, 81: 0.0},
        {10: 300.0, 30: 310.0, 90: 305.0},
        [0.0, 0.1],
        length_unit="m",
    )


@pytest.fixture(params=[electrical, thermal], ids=["electrical", "thermal"])
def solve(request):
    return request.param


def invalid_positions():
    return [
        np.ma.array([9.0, 0.0, 0.0], mask=[True, False, False]),
        np.ma.array([0.0, 0.0, 0.0], mask=False),
        [np.ma.array(9.0, mask=True), 0.0, 0.0],
        [np.ma.array(0.0, mask=False), 0.0, 0.0],
        np.array([np.ma.array(9.0, mask=True), 0.0, 0.0], dtype=object),
        np.array([1j, 0.0, 0.0]),
        [1j, 0.0, 0.0],
        np.array([False, False, False]),
        [False, 0.0, 0.0],
        (np.bool_(False), 0.0, 0.0),
        np.array([0.0, 0.0, 0.0], dtype=object),
        ["0", 0.0, 0.0],
        np.array(["0", "0", "0"]),
        [None, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
        [np.inf, 0.0, 0.0],
        [0.0, 0.0],
        [[0.0, 0.0, 0.0]],
        [],
        [10**1000, 0, 0],
    ]


@pytest.mark.parametrize("position", invalid_positions())
def test_invalid_geometry_rejected_before_sparse_assembly(solve, position, monkeypatch):
    import fibernet.sim.linear_transport as el
    import fibernet.sim.transient_thermal as th

    def must_not_assemble(*args, **kwargs):
        pytest.fail("invalid graph reached sparse assembly")

    monkeypatch.setattr(el, "coo_matrix", must_not_assemble)
    monkeypatch.setattr(th, "coo_matrix", must_not_assemble)
    g = graph()
    g.nodes[10].position = position
    with pytest.raises(ValueError, match="position"):
        solve(g)
    assert g.nodes[10].position is position


@pytest.mark.parametrize(
    "value", [99, -1, 2**63, True, np.bool_(False), 10.0, "10", None]
)
@pytest.mark.parametrize("kind", ["node", "edge", "isolated_node"])
def test_record_id_must_be_valid_and_match_key(solve, value, kind):
    g = graph()
    record, field = (
        (g.edges[7], "edge_id")
        if kind == "edge"
        else (g.nodes[90 if kind == "isolated_node" else 10], "node_id")
    )
    setattr(record, field, value)
    with pytest.raises(ValueError, match="record ID"):
        solve(g)
    assert getattr(record, field) is value


@pytest.mark.parametrize("sign", [-1, 1])
def test_extended_precision_conversion_cannot_erase_nonzero(solve, sign):
    if np.finfo(np.longdouble).tiny >= np.finfo(np.float64).tiny:
        pytest.skip("longdouble has no wider exponent range on this platform")
    for value in [np.longdouble("1e-400"), np.longdouble("1e400")]:
        g = graph()
        g.nodes[10].position = np.array([sign * value, 0, 0], dtype=np.longdouble)
        with pytest.raises(ValueError, match="representable"):
            solve(g)


@pytest.mark.parametrize(
    "position",
    [
        [0, 0, 0],
        (0.0, 0.0, 0.0),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.uint64),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.longdouble),
    ],
)
def test_supported_coordinates_preserve_fields_and_fingerprint(solve, position):
    expected = solve(graph()).to_dict()
    g = graph()
    g.nodes[10].position = position
    # NumPy integer record IDs remain legal and agree with ordinary keys.
    g.nodes[10].node_id = np.int64(10)
    g.edges[7].edge_id = np.uint64(7)
    assert solve(g).to_dict() == expected
    assert g.nodes[10].position is position


def test_stable_order_parallel_inactive_and_detached_results(solve):
    g = graph()
    before = copy.deepcopy(g.to_dict())
    result = solve(g)
    assert result.node_ids.tolist() == [10, 30, 90]
    assert result.edge_ids.tolist() == [7, 2, 81]
    assert result.endpoints_by_id.tolist() == [[10, 30], [30, 10], [30, 90]]
    assert g.to_dict() == before
    saved = result.to_dict()
    g.nodes[10].position[:] = 99
    assert result.to_dict() == saved
    json.dumps(saved, allow_nan=False)


def test_max_int64_ids_and_singleton_transport_remain_supported(solve):
    nid = np.iinfo(np.int64).max
    g = StructureGraph.from_dict(
        {
            "dimension": 3,
            "nodes": [{"id": int(nid), "position": [0, 0, 0]}],
            "edges": [],
        }
    )
    g.nodes[nid].node_id = np.int64(nid)
    if solve is electrical:
        result = solve_linear_transport(g, {}, {nid: 3.0}, length_unit="m")
        assert result.voltage_V.tolist() == [3.0]
    else:
        result = solve_transient_thermal(
            g,
            {nid: 1.0},
            {},
            {nid: 300.0},
            [0, 1],
            source_W={nid: 2.0},
            length_unit="m",
        )
        assert result.temperature_K.tolist() == [[300.0], [302.0]]
    assert result.node_ids.tolist() == [nid]
    assert result.endpoints_by_id.shape == (0, 2)


def test_thermal_saved_temperature_binding_retains_exact_graph_fields():
    from fibernet.sim.saved_temperature import prepare_saved_temperature

    g = graph()
    result = thermal(g)
    saved = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    snapshot = prepare_saved_temperature(g, saved, frame_index=1, length_unit="m")
    assert snapshot.mechanical_kwargs(g)["temperature_K"] == dict(
        zip(result.node_ids, result.temperature_K[1])
    )


def test_geometry_is_provenance_not_a_conductance_law():
    g = graph()
    original = electrical(g)
    # Coincident nodes, curved edges and material/radius changes do not infer g.
    for node in g.nodes.values():
        node.position[:] = 0
    g.edges[7].internal_points = np.array([[0.0, 2.0, 0.0]])
    g.edges[7].radius = 100.0
    moved = electrical(g)
    np.testing.assert_array_equal(moved.current_A, original.current_A)
    assert moved.provenance["input_sha256"] != original.provenance["input_sha256"]
