"""Reject nested masked geometry before building detached learning inputs."""

import copy

import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.graph_data import graph_batch
from fibernet.ml.graph_training_contract import bind_graph_inputs, project_graph_inputs
from fibernet.sim.linear_transport import solve_linear_transport


def _batch(graph, provenance=False):
    return graph_batch(
        [graph],
        graph_ids=["synthetic-cell"],
        length_unit="m",
        coordinate_frame="cell_xyz",
        edge_features=["radius", "chord_length", "polyline_length"],
        include_provenance=provenance,
    )


def _nested(masked, kind):
    row = np.ma.array([123.0, 0.5, 0.5], mask=[masked, False, False])
    scalars = [np.ma.array(123.0, mask=masked), 0.5, 0.5]
    if kind == "node-list":
        return scalars
    if kind == "node-tuple":
        return tuple(scalars)
    if kind == "node-object":
        out = np.empty(3, dtype=object)
        out[:] = scalars
        return out
    if kind == "internal-list-row":
        return [row]
    if kind == "internal-tuple-row":
        return (row,)
    if kind == "internal-list-scalars":
        return [scalars]
    if kind == "internal-tuple-scalars":
        return (tuple(scalars),)
    out = np.empty(1, dtype=object)
    out[0] = row
    return out


@pytest.mark.parametrize("provenance", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "node-list",
        "node-tuple",
        "node-object",
        "internal-list-row",
        "internal-tuple-row",
        "internal-list-scalars",
        "internal-tuple-scalars",
        "internal-object-row",
    ],
)
def test_nested_masked_geometry_is_not_a_valid_snapshot(kind, masked, provenance):
    graph = fn.pattern_3d(unit="cubic", box=(2, 2, 2), grid=(1, 1, 1))
    value = _nested(masked, kind)
    target = next(
        iter(graph.nodes.values() if kind.startswith("node") else graph.edges.values())
    )
    field = "position" if kind.startswith("node") else "internal_points"
    setattr(target, field, value)
    with pytest.raises(ValueError, match="MaskedArray"):
        _batch(graph, provenance)
    assert getattr(target, field) is value


@pytest.mark.parametrize("provenance", [False, True])
def test_masked_empty_internal_array_still_rejects(provenance):
    graph = fn.pattern_3d(unit="cubic", box=(2, 2, 2), grid=(1, 1, 1))
    next(iter(graph.edges.values())).internal_points = np.ma.array(
        np.empty((0, 3)), mask=False
    )
    with pytest.raises(ValueError, match="MaskedArray"):
        _batch(graph, provenance)


@pytest.mark.parametrize("container", [list, tuple, np.asarray])
@pytest.mark.parametrize("provenance", [False, True])
def test_generated_transport_and_bound_learning_rows_agree(container, provenance):
    # Small real generation/transport path, not a trained-model quality claim.
    graph = fn.pattern_3d(unit="cubic", box=(2, 3, 4), grid=(1, 1, 1))
    remap = {nid: 10 + 7 * i for i, nid in enumerate(graph.nodes)}
    nodes = {}
    for nid, node in reversed(list(graph.nodes.items())):
        node.node_id = remap[nid]
        node.position = container(node.position.tolist())
        nodes[node.node_id] = node
    graph._nodes = nodes
    edges = {}
    for i, edge in enumerate(reversed(list(graph.edges.values()))):
        edge.node_i, edge.node_j = remap[edge.node_i], remap[edge.node_j]
        edge.edge_id = 100 + 11 * i
        edges[edge.edge_id] = edge
    graph._edges = edges
    curved = next(iter(graph.edges.values()))
    midpoint = (
        np.asarray(graph.nodes[curved.node_i].position)
        + np.asarray(graph.nodes[curved.node_j].position)
    ) / 2
    midpoint += [0.25, 0.5, 0.75]
    curved.internal_points = container([container(midpoint.tolist())])
    parallel = copy.deepcopy(curved)
    parallel.edge_id = 999
    parallel.node_i, parallel.node_j = parallel.node_j, parallel.node_i
    graph.edges[999] = parallel

    view = _batch(graph, provenance)
    conductance = {eid: 0.25 + i / 20 for i, eid in enumerate(graph.edges)}
    ordered = sorted(graph.nodes)
    boundary = {ordered[0]: 0.0, ordered[-1]: 1.0}
    solved = solve_linear_transport(graph, conductance, boundary, length_unit="m")
    np.testing.assert_array_equal(solved.node_ids, view.node_ids)
    np.testing.assert_array_equal(solved.edge_ids, view.edge_ids)
    np.testing.assert_array_equal(
        solved.endpoints_by_id, view.node_ids[view.edge_index.T]
    )
    # Independent dense resistor-network oracle using the snapshot's row map.
    incidence = np.zeros((len(view.edge_ids), len(view.node_ids)))
    for row, (left, right) in enumerate(view.edge_index.T):
        incidence[row, left], incidence[row, right] = 1.0, -1.0
    weights = np.array([conductance[int(eid)] for eid in view.edge_ids])
    laplacian = incidence.T @ (weights[:, None] * incidence)
    voltage = np.zeros(len(view.node_ids))
    voltage[-1] = 1.0
    free = np.arange(1, len(voltage) - 1)
    fixed = np.array([0, len(voltage) - 1])
    voltage[free] = np.linalg.solve(
        laplacian[np.ix_(free, free)],
        -laplacian[np.ix_(free, fixed)] @ voltage[fixed],
    )
    np.testing.assert_allclose(solved.voltage_V, voltage, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        solved.current_A, weights * (incidence @ voltage), rtol=1e-12, atol=1e-14
    )
    assert solved.dissipated_power_W == pytest.approx(
        float(np.dot(weights, (incidence @ voltage) ** 2)), rel=1e-12
    )
    spec = {
        "schema": "fibernet.graph_feature_spec.v1",
        "length_unit": "m",
        "coordinate_frame": "cell_xyz",
        "dtype": "float32",
        "node_offset": [0.0] * 4,
        "node_scale": [1.0] * 4,
        "features": [
            {
                "name": "path_length",
                "source": {"snapshot": "polyline_length"},
                "unit": "m",
                "offset": 0.0,
                "scale": 1.0,
                "required": True,
                "parity": "even",
            }
        ],
    }
    conditions = {
        "schema": "fibernet.electrical_conditions.v1",
        "task_rule_id": "synthetic_two_terminal",
        "conditioning": "metadata_only_fixed_rule",
        "boundary_voltages": [
            {"node_id": nid, "value": value, "unit": "V"}
            for nid, value in boundary.items()
        ],
        "nodal_currents": [],
    }
    refs = {name: "synthetic/" + name for name in ("graph", "external", "conditions")}
    source = dict(batch=view, external=None, conditions=conditions, source_refs=refs)
    gid = view.graph_ids[0]
    bound = bind_graph_inputs(
        [view],
        source_groups={gid: "synthetic-cell-family"},
        feature_spec=spec,
        external={gid: None},
        conditions={gid: conditions},
        source_refs={gid: refs},
    )
    projected = project_graph_inputs(bound, sources={gid: source})[gid]
    np.testing.assert_array_equal(projected["edge_index"], view.message_index)
    np.testing.assert_allclose(
        projected["edge_features"][:, 0],
        view.edge_features[view.message_parent_rows, 2],
        rtol=1e-6,
    )
    assert view.edge_ids[-1] == 999
    assert view.message_parent_ids[-2:].tolist() == [999, 999]
    assert view.message_orientation[-2:].tolist() == [1, -1]
