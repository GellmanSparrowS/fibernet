"""Reject nonfinite merge tolerances before geometry can be collapsed."""

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph


@pytest.mark.parametrize('scalar', [float, np.float16, np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_constructor_rejects_nonfinite_tolerance(scalar, value):
    with pytest.raises(ValueError, match='^tolerance must be finite$'):
        StructureGraph(tolerance=scalar(value))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_from_dict_rejects_nonfinite_tolerance_without_changing_source(value):
    source = StructureGraph()
    source.add_node([0, 0])
    source.add_node([100, 100])
    data = source.to_dict()
    data['tolerance'] = value
    nodes = data['nodes']
    with pytest.raises(ValueError, match='^tolerance must be finite$'):
        StructureGraph.from_dict(data)
    assert data['tolerance'] is value
    assert data['nodes'] is nodes
    assert source.num_nodes == 2


@pytest.mark.parametrize('scalar', [float, np.float16, np.float32, np.float64])
@pytest.mark.parametrize('value', [0.0, -0.0, -0.25, 0.25])
def test_finite_tolerance_identity_is_preserved(scalar, value):
    tolerance = scalar(value)
    graph = StructureGraph(tolerance=tolerance)
    assert graph.tolerance is tolerance


@pytest.mark.parametrize('scalar', [float, np.float32, np.float64])
@pytest.mark.parametrize('value', [0.0, -0.0, -0.25, 0.25])
def test_existing_finite_merge_policy_and_roundtrip_are_preserved(scalar, value):
    graph = StructureGraph(tolerance=scalar(value))
    first = graph.add_node([0, 0])
    duplicate = graph.add_node([0, 0])
    assert (first == duplicate) == (value > 0)
    distant = graph.add_node([100, 100])
    assert distant != first
    restored = StructureGraph.from_dict(graph.to_dict())
    assert restored.to_dict() == graph.to_dict()


def test_default_tolerance_and_integer_tolerance_remain_supported():
    graph = StructureGraph()
    assert graph.tolerance == 1e-6
    assert graph.add_node([0, 0]) == graph.add_node([1e-7, 0])
    graph = StructureGraph(tolerance=1)
    assert graph.add_node([0, 0]) == graph.add_node([0.5, 0])


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.longdouble, object])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_scalar_array_nonfinite_tolerance_is_rejected(dtype, value):
    tolerance = np.array(value, dtype=dtype)
    with pytest.raises(ValueError, match='^tolerance must be finite$'):
        StructureGraph(tolerance=tolerance)

    source = StructureGraph()
    source.add_node([0, 0])
    source.add_node([100, 100])
    data = source.to_dict()
    data['tolerance'] = tolerance
    nodes = data['nodes']
    with pytest.raises(ValueError, match='^tolerance must be finite$'):
        StructureGraph.from_dict(data)
    assert data['tolerance'] is tolerance
    assert data['nodes'] is nodes
    assert source.num_nodes == 2
    assert tolerance.shape == ()
    assert tolerance.dtype == np.dtype(dtype)
    assert (np.isnan(tolerance.item()) if np.isnan(value) else tolerance.item() == value)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.longdouble, object])
@pytest.mark.parametrize('value', [0.0, -0.0, -0.25, 0.25])
def test_scalar_array_finite_tolerance_preserves_identity(dtype, value):
    tolerance = np.array(value, dtype=dtype)
    graph = StructureGraph(tolerance=tolerance)
    assert graph.tolerance is tolerance
    restored = StructureGraph.from_dict(graph.to_dict())
    assert restored.tolerance is tolerance
    assert bool(np.signbit(tolerance.item())) == bool(np.signbit(value))


@pytest.mark.parametrize('dtype,value', [
    (dtype, value)
    for dtype in (np.float16, np.float32, np.float64, np.longdouble, object)
    for value in (0.0, -0.0, -0.25, 0.25)
    # Float16 zero-array hashing depends on NumPy scalar-promotion rules.
    # Constructor/roundtrip admission above is the version-independent contract.
    if not (dtype is np.float16 and value == 0)
])
def test_scalar_array_finite_tolerance_preserves_merge_policy(dtype, value):
    tolerance = np.array(value, dtype=dtype)
    graph = StructureGraph(tolerance=tolerance)
    first = graph.add_node([0, 0])
    duplicate = graph.add_node([0, 0])
    assert (first == duplicate) == (value > 0)
    assert graph.add_node([100, 100]) != first
    restored = StructureGraph.from_dict(graph.to_dict())
    assert restored.tolerance is tolerance
    assert restored.to_dict() == graph.to_dict()
    assert bool(np.signbit(tolerance.item())) == bool(np.signbit(value))


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.bool_])
def test_scalar_array_integral_tolerance_remains_supported(dtype):
    tolerance = np.array(1, dtype=dtype)
    graph = StructureGraph(tolerance=tolerance)
    assert graph.tolerance is tolerance
    assert graph.add_node([0, 0]) == graph.add_node([0.5, 0])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_unmasked_scalar_array_nonfinite_tolerance_is_rejected(value):
    with pytest.raises(ValueError, match='^tolerance must be finite$'):
        StructureGraph(tolerance=np.ma.array(value, mask=False))


def test_scalar_array_validation_does_not_unmask_or_coerce_other_types():
    masked = np.ma.array(np.inf, mask=True)
    assert StructureGraph(tolerance=masked).tolerance is masked

    class Uncoercible:
        def __float__(self):
            raise AssertionError('unexpected coercion')

    tolerance = np.array(Uncoercible(), dtype=object)
    assert StructureGraph(tolerance=tolerance).tolerance is tolerance
