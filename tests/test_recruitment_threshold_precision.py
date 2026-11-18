"""Reported recruitment thresholds must reproduce the float64 decisions."""

import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import fibernet as fn
from fibernet.analysis import analyze_tensile_recruitment, compute_percolation
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def replay(strain, thresholds, edges, left, right, n, hysteresis, grip_connected):
    """Independent NetworkX connectivity oracle using only reported thresholds."""
    previous = np.zeros(len(edges), dtype=bool)
    masks, spanning, node_fractions = [], [], []
    for frame, threshold in zip(strain, thresholds):
        active = (frame > threshold) | (previous & (frame > hysteresis * threshold))
        active &= frame > 0
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges[active])
        retained = np.zeros(len(edges), dtype=bool)
        span = np.zeros(len(edges), dtype=bool)
        span_nodes = set()
        for component in nx.connected_components(graph):
            touches_left = bool(component.intersection(left))
            touches_right = bool(component.intersection(right))
            rows = np.array([a in component and b in component for a, b in edges])
            if not grip_connected or touches_left or touches_right:
                retained |= active & rows
            if touches_left and touches_right:
                span |= active & rows
                span_nodes.update(component)
        masks.append(retained)
        spanning.append(span)
        node_fractions.append(len(span_nodes) / n)
        previous = retained
    return np.asarray(masks), np.asarray(spanning), node_fractions


def check_replay(result, strain, edges, left, right, n, hysteresis, grip_connected):
    # Strict JSON is a real consumer of the public NumPy threshold history.
    thresholds = json.loads(
        json.dumps(result.threshold_curve.tolist(), allow_nan=False)
    )
    expected, spanning, fractions = replay(
        strain, thresholds, edges, left, right, n, hysteresis, grip_connected
    )
    np.testing.assert_array_equal(result.active_edges, expected)
    np.testing.assert_array_equal(result.edge_in_spanning, spanning)
    np.testing.assert_allclose(result.spanning_frac, fractions, rtol=1e-7, atol=0)
    np.testing.assert_allclose(
        result.backbone_frac, spanning.mean(axis=1), rtol=1e-7, atol=0
    )
    first = np.flatnonzero(np.asarray(fractions) > 0)
    assert result.perc_frame == (int(first[0]) if len(first) else -1)


@pytest.mark.parametrize("mode", ["relative", "quantile", "floor"])
@pytest.mark.parametrize("grip_connected", [False, True])
def test_reported_threshold_replays_strict_boundary_and_hysteresis(
    mode, grip_connected
):
    threshold = 0.1
    edges = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
    strain = np.array(
        [
            [1.0, np.nextafter(threshold, np.inf), 0.4, 0.1],
            [1.0, np.nextafter(0.5 * threshold, np.inf), 0.4, 0.1],
            [1.0, 0.5 * threshold, 0.4, 0.1],
        ]
    )
    options = dict(alpha=0.1, min_active=threshold if mode == "floor" else 1e-12)
    if mode == "quantile":
        options.update(quantile=0.0)
        # A disconnected positive edge sets the onset to exactly 0.1.
        strain[:, 3] = threshold
        strain[1:, 1] = [np.nextafter(threshold, np.inf), threshold]
    result = analyze_tensile_recruitment(
        strain,
        edges,
        [0],
        [2],
        6,
        hysteresis=0.5,
        grip_connected=grip_connected,
        **options,
    )
    assert result.threshold_curve.dtype == np.float64
    check_replay(result, strain, edges, [0], [2], 6, 0.5, grip_connected)
    # The first edge-to-edge path is genuinely recruited, not only the oracle.
    assert result.edge_in_spanning[0, :2].all()


@pytest.mark.parametrize("threshold", [1e-50, 1e-12, 0.1, 1e40])
def test_threshold_floor_is_not_narrowed(threshold):
    strain = np.array([[2 * threshold, threshold]])
    edges = np.array([[0, 1], [1, 2]])
    result = analyze_tensile_recruitment(
        strain, edges, [0], [2], 3, alpha=0, min_active=threshold
    )
    assert result.threshold_curve[0] == threshold
    assert np.isfinite(result.threshold_curve).all()
    assert result.threshold_curve[0] > 0
    check_replay(result, strain, edges, [0], [2], 3, 0.6, True)


@pytest.mark.parametrize("grid", [2, 4])
def test_saved_fem_network_replays_threshold_decisions(tmp_path, grid):
    graph = fn.pattern_2d(
        unit="honeycomb", grid=(grid, grid), box=(1, 1), radius=0.02, seed=23
    )
    data = graph.to_dict()
    ids = {node["id"]: 101 + 7 * row for row, node in enumerate(data["nodes"])}
    for node in data["nodes"]:
        node["id"] = ids[node["id"]]
    data["nodes"].reverse()
    for row, edge in enumerate(data["edges"]):
        edge.update(
            id=503 + 11 * row, node_i=ids[edge["node_i"]], node_j=ids[edge["node_j"]]
        )
    data["edges"].reverse()
    data.pop("next_node_id", None)
    data.pop("next_edge_id", None)
    graph = fn.StructureGraph.from_dict(data)
    graph.save_json(tmp_path / "graph.json")
    graph = fn.StructureGraph.load_json(tmp_path / "graph.json")
    result = fn.simulate(
        graph,
        backend="fem",
        stretch_ratio=1.001,
        nonlinear=False,
        E=1e6,
        nu=0.3,
        pct=0.1,
    )
    result.save(tmp_path / "result.json", detailed=True)
    result = fn.SimResult.load(tmp_path / "result.json")
    node_rows = {nid: row for row, nid in enumerate(result.metadata["node_ids"])}
    edges = np.array(
        [
            [node_rows[graph.edges[eid].node_i], node_rows[graph.edges[eid].node_j]]
            for eid in result.metadata["edge_ids"]
        ]
    )
    boundaries = BeamFrameFEM.graph_to_fem_input(graph, dim=2, pct=0.1)["boundaries"]
    left, right = boundaries["left"], boundaries["right"]
    strain = np.asarray(result.metadata["sigma_axial"]) / 1e6
    candidates = strain[
        (strain > 0) & (strain != strain.astype(np.float32).astype(float))
    ]
    assert candidates.size
    selected = candidates[0]
    rounded = float(np.float32(selected))
    # Put the threshold on the opposite side of an actual strain from float32.
    threshold = np.nextafter(selected, -np.inf if rounded > selected else np.inf)
    history = np.stack([np.zeros_like(strain), strain, 0.75 * strain, 0.25 * strain])
    recruitment = analyze_tensile_recruitment(
        history,
        edges,
        left,
        right,
        graph.num_nodes,
        alpha=0,
        min_active=threshold,
        hysteresis=0.5,
        grip_connected=False,
    )
    check_replay(recruitment, history, edges, left, right, graph.num_nodes, 0.5, False)
    # Ensure this network actually exercises a float32 replay error.
    narrowed, _, _ = replay(
        history,
        recruitment.threshold_curve.astype(np.float32),
        edges,
        left,
        right,
        graph.num_nodes,
        0.5,
        False,
    )
    assert np.any(narrowed != recruitment.active_edges)
    run = SimpleNamespace(
        edge_strain=history,
        edges=edges,
        left_nodes=left,
        right_nodes=right,
        frames_xy=np.zeros((len(history), graph.num_nodes, 2)),
    )
    adapted = compute_percolation(
        run, alpha=0, min_active=threshold, hysteresis=0.5, grip_connected=False
    )
    np.testing.assert_array_equal(adapted.threshold_curve, recruitment.threshold_curve)
    np.testing.assert_array_equal(adapted.active_edges, recruitment.active_edges)
