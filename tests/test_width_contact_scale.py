"""Finite-width raster decisions must not depend on the length unit."""

from fractions import Fraction

import numpy as np
import pytest

from fibernet.analysis import ContactConfig, SnapshotFeatureExtractor

POINTS = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, -0.5], [1.0, 0.5]])
EDGES = np.array([[0, 1], [2, 3]])
DIMENSIONLESS = (
    "contact_resolution",
    "contact_pair_count",
    "contact_edge_ratio",
    "contact_coverage_ratio",
    "contact_patch_count",
    "contact_graph_max",
)


def exact_crossing_pixels(resolution=64):
    """Independent rational full-canvas oracle for the two crossing capsules."""
    radius = Fraction(1, 10)
    step = Fraction(12, 5) / (resolution - 1)
    union = overlap = 0
    for ix in range(resolution):
        x = -Fraction(1, 5) + ix * step
        for iy in range(resolution):
            y = -Fraction(7, 10) + iy * step
            horizontal = (x - min(max(x, 0), 2)) ** 2 + y**2 <= radius**2
            vertical = (x - 1) ** 2 + (
                y - min(max(y, -Fraction(1, 2)), Fraction(1, 2))
            ) ** 2 <= radius**2
            union += horizontal or vertical
            overlap += horizontal and vertical
    return union, overlap, float(step)


@pytest.mark.parametrize("scale", [1e-150, 1e-50, 1e-16, 1e-8, 1.0, 1e8, 1e150])
def test_crossing_contacts_match_exact_pixel_oracle_at_every_unit(scale):
    union, overlap, step = exact_crossing_pixels()
    actual = ContactConfig(width=0.2 * scale, resolution=64).compute(
        POINTS * scale, EDGES
    )
    assert actual["contact_pair_count"] == 1
    assert actual["contact_edge_ratio"] == 1
    assert actual["contact_patch_count"] == 1
    assert actual["contact_graph_max"] == 2
    assert actual["contact_coverage_ratio"] == overlap / union
    assert actual["contact_pixel_size"] / scale == pytest.approx(step)
    assert actual["contact_overlap_area"] / scale / scale == pytest.approx(
        overlap * step**2
    )
    assert actual["contact_width"] == 0.2 * scale


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
@pytest.mark.parametrize("offset", [[0, 0], [16, -32], [-256, 128]])
def test_translation_edge_reversal_and_no_input_mutation(scale, offset):
    pos = (POINTS + offset) * scale
    edges = EDGES[::-1, ::-1].copy()
    before_pos, before_edges = pos.copy(), edges.copy()
    config = ContactConfig(width=0.2 * scale, resolution=128)
    actual = config.compute(pos, edges)
    baseline = ContactConfig(width=0.2, resolution=128).compute(POINTS, EDGES)
    for key in DIMENSIONLESS:
        assert actual[key] == baseline[key]
    assert actual["contact_overlap_area"] / scale / scale == pytest.approx(
        baseline["contact_overlap_area"]
    )
    np.testing.assert_array_equal(pos, before_pos)
    np.testing.assert_array_equal(edges, before_edges)


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_non_crossing_overlapping_capsules_and_shared_nodes(scale):
    # Distinct node identities are retained even at coincident coordinates.
    points = np.array([[0.0, 0], [2, 0], [0, 0.1], [2, 0.1], [3, 0]])
    config = ContactConfig(width=0.2 * scale, resolution=128)
    actual = config.compute(points * scale, EDGES)
    assert actual["contact_pair_count"] == 1
    assert actual["contact_overlap_area"] > 0
    shared = config.compute(points * scale, np.array([[0, 1], [1, 4]]))
    assert shared["contact_pair_count"] == 0
    assert shared["contact_overlap_area"] == 0
    apart = config.compute(points * scale, np.array([[0, 2], [1, 4]]))
    assert apart["contact_pair_count"] == 0


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_coincident_zero_length_segments_are_disks(scale):
    actual = ContactConfig(width=0.2 * scale, resolution=64).compute(
        np.zeros((4, 2)), EDGES
    )
    assert actual["contact_pair_count"] == 1
    assert actual["contact_coverage_ratio"] == 1
    assert actual["contact_graph_max"] == 2
    assert actual["contact_overlap_area"] > 0


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_snapshot_public_route_preserves_contact_unit_contract(scale):
    config = ContactConfig(width=0.2 * scale, resolution=64)
    actual = SnapshotFeatureExtractor(
        include_contact=True, contact_config=config
    ).extract(POINTS * scale, EDGES)
    direct = config.compute(POINTS * scale, EDGES)
    for key, value in direct.items():
        assert actual[key] == value


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_pixel_budget_is_not_changed_by_units(scale):
    with pytest.raises(MemoryError, match="coverage budget"):
        ContactConfig(width=0.2 * scale, resolution=64, max_pixels=1).compute(
            POINTS * scale, EDGES
        )


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_unrepresentable_pixel_area_fails_explicitly(scale):
    with pytest.raises(ValueError, match="area scale"):
        ContactConfig(width=0.2 * scale).compute(POINTS * scale, EDGES)


def test_unrepresentable_coordinate_extent_fails_explicitly():
    with pytest.raises(ValueError, match="coordinate/width scale"):
        ContactConfig(width=1.0).compute(
            np.array([[-1e308, 0], [1e308, 0]]), np.array([[0, 1]])
        )


def test_zero_width_and_empty_edges_preserve_short_circuits():
    for width, edges in [(0, EDGES), (1e200, np.empty((0, 2), dtype=int))]:
        actual = ContactConfig(width=width).compute(POINTS, edges)
        assert actual["contact_width"] == width
        assert actual["contact_overlap_area"] == 0
        assert actual["contact_pixel_size"] == 0


def test_large_integer_coordinates_normalize_without_integer_overflow():
    points = np.array(
        [[-6 * 10**18, 0], [6 * 10**18, 0], [0, -3 * 10**18], [0, 3 * 10**18]],
        dtype=np.int64,
    )
    config = ContactConfig(width=1.2e18, resolution=64)
    actual = config.compute(points, EDGES)
    assert actual == config.compute(points.astype(float), EDGES)
    assert actual["contact_pair_count"] == 1
