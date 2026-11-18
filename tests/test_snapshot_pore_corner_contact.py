"""Coincident endpoint IDs share one vertex only in the pore embedding."""

import numpy as np
import pytest

from fibernet.analysis import SnapshotFeatureExtractor
from fibernet.analysis.snapshot_features import _planarize


POINTS = np.asarray([(0, -1), (3, 3), (3, 0), (0, 0),
                     (0, 0), (-1, 0), (0, 3), (-1, -1)], float)
EDGES = np.asarray([(4, 2), (2, 1), (1, 6), (6, 4),
                    (3, 5), (5, 7), (7, 0), (0, 3)])


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('angle', [0., 0.37])
def test_point_touching_separate_ids_keep_both_full_areas(seed, angle):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(POINTS))
    inverse = np.argsort(order)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    points = POINTS[order] @ rotation.T + [7., -11.]
    edges = inverse[EDGES][rng.permutation(len(EDGES))]
    before = points.copy(), edges.copy()
    f = SnapshotFeatureExtractor().extract(points, edges)
    np.testing.assert_allclose(np.sort(f['pore_areas']), [1., 9.], atol=1e-12)
    np.testing.assert_array_equal(f['pore_neighbor'], [0., 0.])
    assert f['pore_count'] == 2
    assert f['pore_perim_mean'] == pytest.approx(8.)
    # Pore-only embedding does not rewrite the input IDs or the structure count.
    assert f['n_node'] == 8
    np.testing.assert_array_equal(points, before[0])
    np.testing.assert_array_equal(edges, before[1])


def test_nested_corner_contact_matches_explicit_shared_vertex():
    points = np.asarray([(0, 0), (4, 0), (4, 4), (0, 4),
                         (0, 0), (2, 1), (1, 2)], float)
    edges = np.asarray([(0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 4)])
    shared = np.where(edges == 4, 0, edges)
    extractor = SnapshotFeatureExtractor()
    duplicate_features = extractor.extract(points, edges)
    shared_features = extractor.extract(points, shared)
    for key in duplicate_features:
        if key.startswith('pore_') or key in ('porosity', 'largest_pore_ratio'):
            np.testing.assert_array_equal(duplicate_features[key], shared_features[key])
    np.testing.assert_allclose(np.sort(duplicate_features['pore_areas']), [1.5, 14.5])
    np.testing.assert_array_equal(duplicate_features['pore_neighbor'], [1., 1.])


def test_exact_embedding_canonicalization_does_not_snap_nearby_endpoints():
    points = POINTS.copy()
    points[[0, 3, 5, 7]] -= 1e-8
    interaction = dict(cross_pairs=np.empty((0, 2), dtype=int),
                       cross_t=np.empty((0, 2)), cross_pts=np.empty((0, 2)))
    _, edges = _planarize(points, EDGES, interaction)
    # The two distinct IDs at the almost-touching corners are still separate.
    assert 3 in edges and 4 in edges
    assert not np.any(np.all(edges == [3, 4], axis=1))


def test_duplicate_geometric_edges_deduplicate_after_vertex_mapping():
    points = np.asarray([(0, 0), (1, 0), (1, 1), (0, 1),
                         (0, 1), (1, 1), (1, 0), (0, 0)], float)
    edges = np.asarray([(0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4)])
    f = SnapshotFeatureExtractor().extract(points, edges)
    np.testing.assert_array_equal(f['pore_areas'], [1.])
    np.testing.assert_array_equal(f['pore_neighbor'], [0.])
    assert f['pore_perim_mean'] == 4.
    assert f['n_node'] == 8 and f['n_edge'] == 8
