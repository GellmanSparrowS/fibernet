"""Node-index admission only; no simulation, plotting or external provider."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import fibernet.analysis.tensile_recruitment as module


def _inputs():
    return dict(edge_strain=[[0.02]], edges=[[0, 1]],
                left_nodes=[0], right_nodes=[1], n_nodes=2)


def _forbid_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid indices must fail before graph analysis')
    monkeypatch.setattr(module, '_grip_prune', forbidden)
    monkeypatch.setattr(module, '_frame_analysis', forbidden)


@pytest.mark.parametrize('field', ['edges', 'left_nodes', 'right_nodes'])
@pytest.mark.parametrize('bad', [-0.2, 0.9, np.nan, np.inf, -np.inf,
                                  True, '0', 2**63])
def test_invalid_indices_reject_before_graph_analysis(field, bad, monkeypatch):
    _forbid_analysis(monkeypatch)
    inputs = _inputs()
    inputs[field] = [[bad, 1]] if field == 'edges' else [bad]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match=field):
            module.analyze_tensile_recruitment(**inputs)


@pytest.mark.parametrize('edges', [
    np.array([[-0.2, 1.0]], dtype=np.float32),
    np.array([[0.9, 1.0]], dtype=np.float64),
    np.array([[np.nan, 1.0]]),
    np.array([[np.inf, 1.0]]),
    np.array([[-np.inf, 1.0]]),
    np.array([[2**63, 1]], dtype=np.uint64),
    np.array([[2**64 - 1, 1]], dtype=np.uint64),
    np.array([[float(2**63), 1.0]]),
    np.array([[False, True]]),
    np.array([['0', '1']]),
    np.array([[0.9, 1]], dtype=object),
])
def test_typed_and_object_arrays_reject_without_cast_warnings(edges, monkeypatch):
    _forbid_analysis(monkeypatch)
    inputs = _inputs()
    inputs['edges'] = edges
    before = edges.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='edges'):
            module.analyze_tensile_recruitment(**inputs)
    np.testing.assert_array_equal(edges, before)


@pytest.mark.parametrize('kind', [
    'list', 'tuple', 'numpy_scalars', np.int32, np.int64, np.uint64,
    np.float32, np.float64, np.longdouble, object,
])
def test_integer_and_exact_integral_float_inputs_preserve_results(kind):
    def convert(values):
        if kind == 'list':
            return values
        if kind == 'tuple':
            return tuple(tuple(row) if isinstance(row, list) else row for row in values)
        if kind == 'numpy_scalars':
            return [[np.int64(v) for v in row] for row in values] if isinstance(values[0], list) else [np.int64(v) for v in values]
        return np.array(values, dtype=kind)

    edges = convert([[0, 1], [1, 2], [0, 2]])
    left, right = convert([0]), convert([2])
    before = [np.array(value, copy=True) for value in (edges, left, right)]
    strain = [[0.2, 0.3, -0.1], [0.02, 0.3, 0.0]]
    expected = module.analyze_tensile_recruitment(
        strain, [[0, 1], [1, 2], [0, 2]], [0], [2], 3,
        alpha=0.1, hysteresis=0.5)
    actual = module.analyze_tensile_recruitment(
        strain, edges, left, right, 3, alpha=0.1, hysteresis=0.5)
    for field in ('active_edges', 'edge_in_spanning', 'spanning_frac',
                  'backbone_frac', 'edge_depth_norm', 'threshold_curve'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.perc_frame == expected.perc_frame == 0
    assert actual.active_edges.tolist() == [[True, True, False], [True, True, False]]
    for value, original in zip((edges, left, right), before):
        np.testing.assert_array_equal(value, original)


def test_mixed_list_preserves_large_integer_without_float_inference():
    result = module._node_indices([[2**53 + 1, 1.0]], 'edges')
    assert result.dtype == np.int64
    assert int(result[0, 0]) == 2**53 + 1
    assert int(result[0, 1]) == 1


@pytest.mark.parametrize('dtype', [np.int64, np.uint64])
def test_typed_int64_max_is_not_rounded_and_result_is_detached(dtype):
    source = np.array([[0, 2**63 - 1]], dtype=dtype)
    result = module._node_indices(source, 'edges')
    assert result.dtype == np.int64
    assert int(result[0, 1]) == 2**63 - 1
    assert not np.shares_memory(result, source)


def test_object_integer_overflow_rejects_before_int64_storage():
    with pytest.raises(ValueError, match='edges'):
        module._node_indices(np.array([[2**200, 1]], dtype=object), 'edges')


def test_run_adapter_uses_the_same_index_gate(monkeypatch):
    _forbid_analysis(monkeypatch)
    run = SimpleNamespace(edge_strain=np.array([[0.02]]),
                          edges=np.array([[-0.2, 1.0]]),
                          left_nodes=[0], right_nodes=[1],
                          frames_xy=np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match='edges'):
        module.compute_percolation(run)
