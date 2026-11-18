"""Exact pore ownership with fewer winding queries for disjoint components."""

import numpy as np
import pytest

import fibernet.analysis.snapshot_features as snapshot


def add_polygon(points, edges, polygon):
    first = len(points)
    points.extend(polygon)
    edges.extend((first + i, first + (i + 1) % len(polygon))
                 for i in range(len(polygon)))


def squares(count):
    points, edges = [], []
    for i in range(count):
        x = 3.0 * i
        add_polygon(points, edges, [(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)])
    return points, edges


def pore_metrics(points, edges, monkeypatch):
    calls = []
    original = snapshot._inside_boundary

    def counted(point, polygon):
        calls.append(point.copy())
        return original(point, polygon)

    monkeypatch.setattr(snapshot, '_inside_boundary', counted)
    interaction = dict(cross_pairs=np.empty((0, 2), dtype=int),
                       cross_t=np.empty((0, 2)), cross_pts=np.empty((0, 2)))
    metrics, arrays = snapshot._pore_features(
        np.asarray(points, float), np.asarray(edges, int), interaction, 1.0)
    return metrics, arrays, calls


def test_disjoint_loops_skip_all_winding_queries(monkeypatch):
    count = 128
    points, edges = squares(count)
    metrics, arrays, calls = pore_metrics(points, edges, monkeypatch)
    assert not calls  # The unfiltered path made count * (count - 1) queries.
    assert metrics['pore_count'] == count
    np.testing.assert_array_equal(arrays['areas'], np.ones(count))
    np.testing.assert_array_equal(arrays['aspects'], np.ones(count))
    np.testing.assert_array_equal(arrays['neighbors'], np.zeros(count))
    assert metrics['pore_perim_mean'] == 4.0


def test_many_inner_loops_keep_only_actual_parent_candidates(monkeypatch):
    count = 40
    points, edges = [], []
    add_polygon(points, edges, [(-1, -1), (3 * count, -1),
                                (3 * count, 2), (-1, 2)])
    inner_points, inner_edges = squares(count)
    offset = len(points)
    points.extend(inner_points)
    edges.extend((a + offset, b + offset) for a, b in inner_edges)
    metrics, arrays, calls = pore_metrics(points, edges, monkeypatch)
    assert len(calls) == count
    assert metrics['pore_count'] == count + 1
    np.testing.assert_array_equal(arrays['areas'],
                                  [3 * (3 * count + 1) - count] + [1] * count)
    np.testing.assert_array_equal(arrays['neighbors'], [count] + [1] * count)
    assert metrics['pore_perim_mean'] == pytest.approx(
        (2 * (3 * count + 4) + 8 * count) / (count + 1))


def test_concave_bbox_is_only_prefilter_not_membership(monkeypatch):
    points, edges = [], []
    add_polygon(points, edges, [(0, 0), (4, 0), (4, 1),
                                (1, 1), (1, 4), (0, 4)])
    add_polygon(points, edges, [(2, 2), (3, 2), (3, 3), (2, 3)])
    metrics, arrays, calls = pore_metrics(points, edges, monkeypatch)
    assert len(calls) == 1  # In parent bbox, outside the actual L-shaped face.
    np.testing.assert_array_equal(arrays['areas'], [7, 1])
    np.testing.assert_array_equal(arrays['neighbors'], [0, 0])
    assert metrics['pore_perim_mean'] == 10.0


def test_inclusive_bbox_keeps_boundary_aligned_query(monkeypatch):
    points, edges = [], []
    add_polygon(points, edges, [(0, 2), (2, 0), (4, 2), (2, 4)])
    # This component lies outside the diamond, but its first query has x equal
    # to the diamond's minimum. The inclusive prefilter must defer to winding.
    add_polygon(points, edges, [(0, 0.75), (-0.5, 0.75),
                                (-0.5, 1.25), (0, 1.25)])
    _, arrays, calls = pore_metrics(points, edges, monkeypatch)
    assert any(point[0] == 0.0 for point in calls)
    np.testing.assert_array_equal(arrays['areas'], [8, 0.25])
    np.testing.assert_array_equal(arrays['neighbors'], [0, 0])
