"""Scalar representation must not change a graph's cross-section scaling."""

import copy

import numpy as np
import pytest

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.transforms import compose, make_scale, scale
from fibernet.ml.beam_frame_fem import BeamFrameFEM

SCALARS = [
    np.int32(2),
    np.int64(2),
    np.uint64(2),
    np.float16(2),
    np.float32(2),
    np.float64(2),
    np.longdouble(2),
    np.array(2, dtype=np.int64),
    np.array(2, dtype=np.uint64),
    np.array(2.0, dtype=np.float32),
    np.array(2.0),
    np.ma.array(2.0, mask=False),
]


def graph(dimension=3):
    result = StructureGraph.from_dict(
        {
            "version": "3.1",
            "dimension": dimension,
            "box_size": [4, 5, 6 if dimension == 3 else 0],
            "next_node_id": 100,
            "next_edge_id": 200,
            "nodes": [
                {"id": 40, "position": [0, 0, 0]},
                {"id": 7, "position": [2, 1, 1 if dimension == 3 else 0]},
            ],
            "edges": [
                {
                    "id": 99,
                    "node_i": 40,
                    "node_j": 7,
                    "radius": 0.1,
                    "segments": 7,
                    "internal_points": [[0.5, 0.8, 0]],
                },
                {"id": 3, "node_i": 7, "node_j": 40, "radius": 0.2},
            ],
        }
    )
    result.metadata["units"] = {"length": "m"}
    for edge in result.edges.values():
        edge.material = Material(youngs_modulus=1234.0, extra={"sample": [1]})
        edge.metadata["source"] = [edge.edge_id]
    return result


@pytest.mark.parametrize("factor", SCALARS)
@pytest.mark.parametrize("dimension", [2, 3])
def test_numpy_scalar_matches_python_geometry_radii_and_ownership(factor, dimension):
    original = graph(dimension)
    before = copy.deepcopy(original.to_dict())
    actual = scale(original, factor, center=[1, 2, 0])
    expected = scale(original, float(factor), center=[1, 2, 0])
    assert actual.to_dict() == expected.to_dict()
    assert list(actual.nodes) == [40, 7] and list(actual.edges) == [99, 3]
    assert (actual._next_node_id, actual._next_edge_id) == (100, 200)
    assert actual.add_node(actual.nodes[40].position) == 40
    assert actual.add_edge(40, 7) == 99
    actual.nodes[40].position[0] = 1000
    actual.edges[99].internal_points[0, 0] = 1000
    actual.edges[99].material.extra["sample"].append(2)
    actual.edges[99].metadata["source"].append(2)
    actual.metadata["units"]["length"] = "changed"
    assert original.to_dict() == before


@pytest.mark.parametrize("value", [-2.0, 0.0, 0.5])
@pytest.mark.parametrize("wrap", [np.float32, np.array])
def test_signed_zero_fractional_and_empty_scalar_parity(value, wrap):
    for original in (graph(), StructureGraph(box_size=[1, 2])):
        actual = scale(original, wrap(value))
        assert actual.to_dict() == scale(original, value).to_dict()
    if value == 0:
        # A geometry operation may collapse members; it is not FEM admission.
        with pytest.raises(ValueError, match="radius"):
            BeamFrameFEM.section_properties(
                scale(graph(), wrap(value)).edges[99].radius
            )


@pytest.mark.parametrize(
    "factor", [[2], [2, 2], [2, 2, 2], np.array([2.0]), np.array([[2.0, 2.0, 2.0]])]
)
def test_vectors_retain_historical_coordinate_only_policy(factor):
    original = graph()
    actual = scale(original, factor)
    assert [edge.radius for edge in actual.edges.values()] == [0.1, 0.2]
    assert actual.nodes[7].position[0] == 4


@pytest.mark.parametrize("factor", [np.ma.array(2.0, mask=True), np.ma.masked])
def test_masked_scalar_rejects_without_mutation_even_for_empty_graph(factor):
    for original in (graph(), StructureGraph()):
        before = copy.deepcopy(original.to_dict())
        with pytest.raises(ValueError, match="masked scalar"):
            scale(original, factor)
        assert original.to_dict() == before


def test_composed_transform_and_json_roundtrip(tmp_path):
    original = graph()
    actual = compose(original, make_scale(np.float32(2)))
    path = tmp_path / "scaled.json"
    actual.save_json(path)
    restored = StructureGraph.load_json(path)
    assert restored.to_dict() == scale(original, 2.0).to_dict()


@pytest.mark.parametrize("unit", ["D", "ns"])
def test_timedelta_is_not_reclassified_as_a_numeric_scalar(unit):
    # NumPy considers timedelta64 an integer subclass; retain the legacy
    # coordinate-only coercion rather than treating a duration as real scaling.
    original = graph()
    assert (
        scale(original, np.timedelta64(2, unit)).to_dict()
        == scale(original, [2]).to_dict()
    )


@pytest.mark.parametrize("factor", [np.int64(-(2**63)), np.uint64(2**64 - 1)])
def test_numpy_integer_extrema_use_float64_radius_arithmetic(factor):
    original = graph()
    actual = scale(original, factor)
    assert actual.to_dict() == scale(original, float(factor)).to_dict()
    assert actual.edges[99].radius > 0


@pytest.mark.parametrize("radius", [np.float16(0.1), np.float32(0.1)])
@pytest.mark.parametrize("factor", [np.float64(2), np.float64(1.0e5)])
def test_existing_float64_scalar_radius_promotion_is_preserved(radius, factor):
    original = StructureGraph()
    left, right = original.add_node([0, 0]), original.add_node([1, 0])
    original.add_edge(left, right, radius=radius)
    expected = abs(factor) * radius
    actual = scale(original, factor).edges[0].radius
    assert np.isfinite(actual)
    assert actual == expected
    assert type(actual) is type(expected)
    assert original.edges[0].radius is radius


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("direction", [0, 1])
@pytest.mark.parametrize("factor", [np.float32(2), np.int64(2), np.array(2.0)])
def test_scaled_cantilever_fem_matches_analytical_response(
    dimension, direction, factor
):
    original = StructureGraph(dimension=dimension)
    a, b = original.add_node([0, 0, 0]), original.add_node([2, 0, 0])
    original.add_edge(a, b, radius=0.1, material=Material(youngs_modulus=1.0e7))
    results = []
    for source in (original, scale(original, factor)):
        positions, edges, radii = source.to_numpy()
        forces = np.zeros((2, dimension))
        forces[1, direction] = 0.25
        result = getattr(BeamFrameFEM(E=1.0e7), f"solve_{dimension}d")(
            edges.T,
            positions[:, :dimension],
            radii,
            forces=forces,
            fixed_nodes=[0],
            damping=0.0,
        )
        length, radius = source.edge_lengths()[0], radii[0]
        area, inertia = np.pi * radius**2, np.pi * radius**4 / 4
        expected = (
            0.25 * length / (1.0e7 * area)
            if direction == 0
            else 0.25 * length**3 / (3 * 1.0e7 * inertia)
        )
        np.testing.assert_allclose(result["u"][1, direction], expected, rtol=1e-10)
        results.append(result["u"][1, direction])
    np.testing.assert_allclose(results[1], results[0] / 2, rtol=1e-10)
