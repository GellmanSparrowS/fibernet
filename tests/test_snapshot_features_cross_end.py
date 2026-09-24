"""The desktop snapshot-feature contract remains numerically unchanged."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from fibernet.analysis import ContactConfig, SnapshotFeatureExtractor
from fibernet.gen import PlanarManufacturingConfig, manufacturable_graph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FiberScope"))
from fslab.features import compute_features as app_compute_features


def _digest(features):
    def ordinary(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    data = {key: ordinary(value) for key, value in features.items()}
    return hashlib.sha256(json.dumps(data, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_snapshot_feature_golden_and_app_alias():
    positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                          [.5, -.2], [.5, 1.2], [-.2, .5], [1.2, .5]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [6, 7]], dtype=int)
    config = ContactConfig(width=.16, resolution=64, max_pixels=200000)
    library = SnapshotFeatureExtractor(include_contact=True,
                                       contact_config=config)
    actual = library.extract(positions, edges)
    expected = app_compute_features(positions, edges, include_contact=True,
                                    contact_config=config)
    original_digest = "34d2f2a09348ee062d2bbb2aaea1a80893eab0a25ffd7037085744f4a9cba685"
    assert len(actual) == 54
    assert _digest(actual) == _digest(expected) == original_digest
    assert actual["pore_count"] == 4
    assert actual["contact_pair_count"] == 5


def test_snapshot_feature_public_input_budgets():
    extractor = SnapshotFeatureExtractor(max_nodes=4, max_edges=5)
    with pytest.raises(ValueError, match="integer"):
        extractor.extract(np.zeros((2, 2)), np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="indices"):
        extractor.extract(np.zeros((2, 2)), np.array([[0, 2]]))
    with pytest.raises(MemoryError, match="budget"):
        extractor.extract(np.zeros((5, 2)), np.empty((0, 2), dtype=int))


def test_public_graph_and_real_example_schema():
    graph = manufacturable_graph(PlanarManufacturingConfig(
        unit="ring", grid_x=2, grid_y=2, n_pts_per_side=1, seed=11))
    features = SnapshotFeatureExtractor().extract_graph(graph)
    assert features["n_node"] == graph.num_nodes
    assert features["n_edge"] == graph.num_edges
    assert features["pore_count"] > 0
    positions = np.asarray(graph.node_positions(), dtype=float)
    positions[0, 2] += 0.1
    graph.set_node_positions({i: position for i, position in enumerate(positions)})
    with pytest.raises(ValueError, match="planar"):
        SnapshotFeatureExtractor().extract_graph(graph)
