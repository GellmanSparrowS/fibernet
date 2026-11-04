"""Ordinary extraction rejects unavailable values; legacy opt-in stays explicit."""
import inspect

import networkx as nx
import numpy as np
import pytest

import fibernet.analysis.graph_features as module
from fibernet.analysis.graph_features import FeatureUnavailableError, GraphFeatureExtractor

METHODS = ('extract', 'extract_vector', 'batch_extract', 'batch_extract_matrix')


def graph():
    result = nx.Graph()
    result.add_node(0, pos=(0., 0.))
    result.add_node(1, pos=(1., 0.))
    result.add_edge(0, 1)
    return result


def invoke(ext, method, obj, **kwargs):
    return getattr(ext, method)([obj] if method.startswith('batch') else obj, **kwargs)


@pytest.mark.parametrize('method', METHODS)
def test_missing_cv2_default_rejects_complete_ordered_features(monkeypatch, method):
    monkeypatch.setattr(module, 'HAS_CV2', False)
    ext = GraphFeatureExtractor(canvas_size=32)
    with pytest.raises(FeatureUnavailableError) as caught:
        invoke(ext, method, graph())
    error = caught.value
    expected = ext.MESH_FEATURES + ext.PORE_FEATURES + ext.CONTACT_MEASUREMENTS
    assert isinstance(error, RuntimeError)
    assert error.unavailable_features == expected
    assert list(error.reasons) == list(expected)
    assert set(error.reasons.values()) == {'dependency_missing.cv2'}
    for name in expected:
        assert name in str(error)
    for hint in ('opencv-python-headless', 'extract_with_status()', 'allow_unavailable=True'):
        assert hint in str(error)
    report = ext.extract_with_status(graph())
    assert len(report['values']) == 94
    assert sum(s['state'] == 'unknown' for s in report['status'].values()) == 31


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('flag', [None, 0, 1, 'false', [], np.bool_(True)])
def test_flag_must_be_explicit_python_bool_even_for_empty_batch(method, flag):
    ext = GraphFeatureExtractor()
    with pytest.raises(TypeError, match='allow_unavailable must be a bool'):
        getattr(ext, method)([] if method.startswith('batch') else object(), allow_unavailable=flag)
    parameter = inspect.signature(getattr(ext, method)).parameters['allow_unavailable']
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


@pytest.mark.parametrize('flag', [False, True])
def test_empty_batches_preserve_shapes_and_do_not_extract(monkeypatch, flag):
    ext = GraphFeatureExtractor()
    def forbidden(*args, **kwargs):
        pytest.fail('empty batch must not extract')
    monkeypatch.setattr(ext, 'extract', forbidden)
    assert ext.batch_extract([], allow_unavailable=flag) == []
    assert ext.batch_extract_matrix(iter(()), allow_unavailable=flag).shape == (0, 94)


@pytest.mark.parametrize('method', METHODS)
def test_forwarding_and_legacy_raw_path(monkeypatch, method):
    ext = GraphFeatureExtractor()
    values = {name: float(i) for i, name in enumerate(ext.FEATURE_COLS)}
    calls = []
    obj = object()
    monkeypatch.setattr(ext, '_extract', lambda item: calls.append(item) or values)
    def forbidden(*args):
        pytest.fail('explicit legacy must not compute status')
    monkeypatch.setattr(ext, 'extract_with_status', forbidden)
    result = invoke(ext, method, obj, allow_unavailable=True)
    assert calls == [obj]
    if method == 'extract':
        assert result is values
    elif method == 'batch_extract':
        assert result == [values]
    else:
        expected = list(values.values())
        np.testing.assert_equal(result, [expected] if method.startswith('batch') else expected)


@pytest.mark.parametrize('method', METHODS)
def test_default_single_status_call_allows_non_unavailable_states(monkeypatch, method):
    ext = GraphFeatureExtractor()
    values = {name: float(i) for i, name in enumerate(ext.FEATURE_COLS)}
    states = ('unknown', 'fallback', 'defined_empty', 'not_applicable', 'computed')
    status = {name: {'state': states[i % len(states)], 'reason': 'test'}
              for i, name in enumerate(ext.FEATURE_COLS)}
    calls = []
    obj = object()
    monkeypatch.setattr(ext, 'extract_with_status', lambda item: calls.append(item) or
                        {'values': values, 'status': status})
    invoke(ext, method, obj)
    assert calls == [obj]


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('obj,kind,message', [
    (object(), TypeError, 'Expected nx.Graph or StructureGraph'),
    (nx.MultiGraph(), ValueError, 'simple undirected graph'),
    (nx.DiGraph(), ValueError, 'simple undirected graph'),
])
def test_input_errors_precede_guard(monkeypatch, method, obj, kind, message):
    monkeypatch.setattr(module, 'HAS_CV2', False)
    with pytest.raises(kind, match=message):
        invoke(GraphFeatureExtractor(), method, obj)


def test_compute_error_and_conversion_order_unchanged(monkeypatch):
    monkeypatch.setattr(module, 'HAS_CV2', False)
    ext = GraphFeatureExtractor(canvas_size=32)
    original = ext._to_graph
    calls = []
    def convert(obj):
        calls.append('convert')
        return original(obj)
    def fail(obj):
        calls.append('compute')
        raise ValueError('compute sentinel')
    monkeypatch.setattr(ext, '_to_graph', convert)
    monkeypatch.setattr(ext, '_basic_size', fail)
    with pytest.raises(ValueError, match='compute sentinel'):
        ext.extract(graph())
    assert calls == ['convert', 'compute']


def test_no_positions_reason_is_not_treated_as_dependency_missing(monkeypatch):
    monkeypatch.setattr(module, 'HAS_CV2', True)
    ext = GraphFeatureExtractor(canvas_size=32)
    # Empty graph returns from rendering before any native cv2 access.
    with pytest.raises(FeatureUnavailableError) as caught:
        ext.extract(nx.Graph())
    assert 'input.no_positions' in caught.value.reasons.values()
    assert 'dependency_missing.cv2' not in caught.value.reasons.values()


@pytest.mark.parametrize('method', ('extract_vector', 'batch_extract', 'batch_extract_matrix'))
def test_default_preserves_old_extract_override_signature(method):
    class ExistingExtractor(GraphFeatureExtractor):
        def extract(self, obj):
            return {name: 1. for name in self.FEATURE_COLS}
    result = invoke(ExistingExtractor(), method, object())
    assert len(result) == (1 if method.startswith('batch') else 94)


def test_matrix_default_preserves_old_vector_override_signature():
    class ExistingExtractor(GraphFeatureExtractor):
        def extract_vector(self, obj):
            return np.arange(94)
    np.testing.assert_equal(ExistingExtractor().batch_extract_matrix([object()]),
                            [np.arange(94)])
