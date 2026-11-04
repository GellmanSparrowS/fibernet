"""Snapshot projection refuses nonfinite source coordinates before dropping Z."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from fibernet.core.structure_graph import StructureGraph
from fibernet.analysis.snapshot_features import SnapshotFeatureExtractor


def graph():
    g = StructureGraph(dimension=3)
    ids = [g.add_node(p) for p in ((0., 0., 0.), (3., 0., 0.), (1., 2., 0.))]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        g.add_edge(ids[a], ids[b])
    return g


def record(result):
    return json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in result.items()}, allow_nan=False)


@pytest.mark.parametrize('value', (float('nan'), float('inf'), -float('inf')))
@pytest.mark.parametrize('all_nodes', (False, True))
def test_native_nonfinite_z_refused_without_mutation(value, all_nodes):
    g = graph()
    for node in (list(g.nodes.values()) if all_nodes else [g.nodes[0]]):
        node.position[2] = value
    before = g.node_positions().tobytes()
    with pytest.raises(ValueError, match='finite'):
        SnapshotFeatureExtractor().extract_graph(g)
    assert g.node_positions().tobytes() == before


@pytest.mark.parametrize('value', (float('nan'), float('inf'), -float('inf')))
def test_generic_nonfinite_constant_z_precedes_edge_access(value):
    def forbidden():
        raise AssertionError('invalid positions must fail before edge access')
    g = SimpleNamespace(node_positions=lambda: np.array([[0., 0., value], [1., 1., value]]),
                        edge_array=forbidden)
    with pytest.raises(ValueError, match='finite'):
        SnapshotFeatureExtractor().extract_graph(g)


@pytest.mark.parametrize('z', (0., 3.5, -2.))
def test_constant_finite_z_preserves_all_features(z):
    g = graph(); extractor = SnapshotFeatureExtractor()
    expected = record(extractor.extract_graph(g))
    for node in g.nodes.values():
        node.position[2] = z
    assert record(extractor.extract_graph(g)) == expected


def test_empty_graph_preserves_reference_features():
    extractor = SnapshotFeatureExtractor()
    assert record(extractor.extract_graph(StructureGraph(dimension=3))) == record(
        extractor.extract(np.empty((0, 2)), np.empty((0, 2), dtype=int)))


def test_finite_nonplanar_error_stays_planar_error():
    g = graph(); g.nodes[0].position[2] = .1
    with pytest.raises(ValueError, match='planar'):
        SnapshotFeatureExtractor().extract_graph(g)
