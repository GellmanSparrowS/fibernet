"""Missing-cv2 real-path and MOCK image branch routing, never native-cv2 parity."""
import importlib.util
import inspect
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import fibernet.analysis.graph_features as candidate

# Frozen pre-availability reference from this project's legacy extractor.
BASE = Path(os.environ.get('FIBERNET_BASELINE_FEATURE_FILE',
    str(Path(__file__).with_name('fixtures') / 'graph_features_legacy_availability.py')))
spec = importlib.util.spec_from_file_location('_availability_original_graph_features', BASE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
C = candidate.GraphFeatureExtractor
B = baseline.GraphFeatureExtractor


@pytest.fixture(autouse=True)
def controlled_spectral_starts(monkeypatch, request, tmp_path):
    """Identical starts only for unaudited stochastic spectra; NOT API determinism."""
    counts = {'baseline': {'algebraic': 0, 'eigsh': 0}, 'candidate': {'algebraic': 0, 'eigsh': 0}}
    old_algebraic = nx.algebraic_connectivity
    def algebraic(*args, **kwargs):
        caller = inspect.currentframe().f_back.f_globals.get('__name__')
        owner = 'baseline' if caller == baseline.__name__ else 'candidate'
        counts[owner]['algebraic'] += 1
        kwargs.setdefault('seed', 173)
        return old_algebraic(*args, **kwargs)
    monkeypatch.setattr(nx, 'algebraic_connectivity', algebraic)
    for owner, module in (('baseline', baseline), ('candidate', candidate)):
        original = module.eigsh
        def eigen(matrix, *args, _owner=owner, _original=original, **kwargs):
            counts[_owner]['eigsh'] += 1
            kwargs.setdefault('v0', np.arange(1, matrix.shape[0] + 1, dtype=float))
            return _original(matrix, *args, **kwargs)
        monkeypatch.setattr(module, 'eigsh', eigen)
    yield
    assert counts['baseline'] == counts['candidate'], counts
    with (tmp_path / 'spectral_call_counts.json').open('w') as handle:
        handle.write(json.dumps({'test': request.node.nodeid, 'counts': counts,
            'classification': 'controlled-start parity, not old API determinism'}) + '\n')


def graph(kind='square'):
    g = nx.Graph()
    positions = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    for i, pos in enumerate(positions):
        g.add_node(i, pos=pos)
    edges = {'square': ((0, 1), (1, 2), (2, 3), (3, 0)),
             'path': ((0, 1), (1, 2), (2, 3)),
             'crossing': ((0, 2), (1, 3))}[kind]
    g.add_edges_from(edges)
    return g


def equal_values(a, b):
    assert list(a) == list(b)
    for name in a:
        np.testing.assert_equal(a[name], b[name], err_msg=name)


def pair_modules(monkeypatch, cv=False, scipy=None):
    for mod in (candidate, baseline):
        monkeypatch.setattr(mod, 'HAS_CV2', bool(cv))
        if cv:
            monkeypatch.setattr(mod, 'cv2', cv, raising=False)
        if scipy is not None:
            monkeypatch.setattr(mod, 'HAS_SCIPY', scipy)


@pytest.mark.parametrize('kind', ['square', 'path', 'crossing'])
def test_missing_cv2_real_original_parity(monkeypatch, kind):
    pair_modules(monkeypatch)
    old, new = B(canvas_size=64), C(canvas_size=64)
    g = graph(kind)
    expected = old.extract(g)
    actual = new.extract(g, allow_unavailable=True)
    result = new.extract_with_status(g)
    equal_values(expected, actual)
    equal_values(old.extract(g), result['values'])
    np.testing.assert_equal(old.extract_vector(g), new.extract_vector(g, allow_unavailable=True))
    assert len(result['status']) == len(result['values']) == 94
    assert len(result['audited_names']) == 63
    unknown = [n for n, r in result['status'].items() if r['state'] == 'unknown']
    assert len(unknown) == 31
    for n in C.MESH_FEATURES + C.PORE_FEATURES + C.CONTACT_MEASUREMENTS:
        assert result['status'][n] == {'state': 'unavailable', 'reason': 'dependency_missing.cv2'}
    for n in C.CONTACT_CONFIGURATION:
        assert result['status'][n] == {'state': 'computed', 'reason': 'input_recorded'}
    old_batch, new_batch = old.batch_extract([g]), new.batch_extract([g], allow_unavailable=True)
    assert isinstance(new_batch, list)
    equal_values(old_batch[0], new_batch[0])
    np.testing.assert_equal(old.batch_extract_matrix([g]), new.batch_extract_matrix([g], allow_unavailable=True))
    candidate._checked_feature_contract({k: v for k, v in result.items() if k != 'values'})
    json.dumps(result, allow_nan=False)  # this specific missing-cv2 fixture is finite


@pytest.mark.parametrize('kind', [nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph])
def test_original_exception_parity(monkeypatch, kind):
    pair_modules(monkeypatch)
    g = kind(graph())
    with pytest.raises(ValueError) as old:
        B().extract(g)
    with pytest.raises(ValueError) as new:
        C().extract_with_status(g)
    assert str(old.value) == str(new.value)


class MockCV2:
    """Synthetic deterministic branch inputs, NOT an OpenCV implementation."""
    CC_STAT_AREA = 4
    RETR_EXTERNAL = CHAIN_APPROX_SIMPLE = LINE_AA = 1
    __version__ = 'MOCK-NOT-NATIVE'

    def __init__(self, areas=(8., 12.), shape='normal', centers=None):
        self.areas = tuple(areas)
        self.shape = shape
        self.centers = centers

    def line(self, img, *args):
        img[0, 0] = 255

    def floodFill(self, img, *args):
        return None

    def connectedComponentsWithStats(self, img, *args):
        n = len(self.areas) + 1
        stats = np.zeros((n, 5), dtype=float)
        stats[1:, 4] = self.areas
        centers = np.zeros((n, 2))
        for i in range(1, n):
            centers[i] = (20. + 10 * (i - 1), 20. + 10 * (i - 1))
        if self.centers is not None:
            centers[1:] = self.centers
        labels = np.zeros_like(img, dtype=int)
        for i in range(1, n):
            labels[i, i] = i
        return n, labels, stats, centers

    def findContours(self, img, *args):
        return ([], None) if self.shape == 'no_contour' else ([np.array([[0, 0]])], None)

    def contourArea(self, contour):
        return 0. if self.shape == 'zero_hull' else 4.

    def convexHull(self, contour):
        return contour

    def arcLength(self, contour, closed):
        return 0. if self.shape == 'zero_perimeter' else 8.


@pytest.mark.parametrize('areas,scipy,options,shape', [
    ((), True, {}, 'normal'), ((8.,), True, {}, 'normal'),
    ((8., 12.), True, {}, 'normal'), ((8., 8.), True, {}, 'normal'),
    ((8., 12.), False, {}, 'normal'), ((8., 12.), True, {'area_thresh': 1.}, 'normal'),
    ((8., 12.), True, {'top_k': 0}, 'normal'),
    ((8., 12.), True, {'area_thresh': 0.}, 'no_contour'),
    ((8., 12.), True, {'area_thresh': 0.}, 'zero_hull'),
    ((8., 12.), True, {'area_thresh': 0.}, 'zero_perimeter'),
])
def test_mock_pore_branch_original_value_parity(monkeypatch, areas, scipy, options, shape):
    fake = MockCV2(areas, shape)
    pair_modules(monkeypatch, fake, scipy)
    old, new = B(canvas_size=64, **options), C(canvas_size=64, **options)
    image = np.full((64, 64), 255, np.uint8)
    status = {}
    with np.errstate(all='ignore'):
        expected = old._pore_features(image)
        actual = new._pore_features(image, _status=status)
    equal_values(expected, actual)
    assert set(status) == set(C.PORE_FEATURES)
    if not areas:
        assert status['total_pore_count']['state'] == 'defined_empty'
        assert status['pore_density']['state'] == 'computed'
        assert status['top_convexity_mean']['state'] == 'not_applicable'
    elif len(areas) == 1:
        assert status['pore_area_skew']['reason'] == 'domain.insufficient_pores'
        assert status['pore_area_cv']['reason'] == 'stat.single_pore_default'
    elif not scipy:
        names = ('pore_area_cv', 'pore_area_skew', 'pore_area_kurtosis',
                 'pore_large_area_frac', 'pore_count_large_frac')
        assert all(status[n]['reason'] == 'dependency_missing.scipy' for n in names)
        assert status['pore_area_max_over_mean']['state'] == 'computed'
    if options.get('top_k') == 0:
        assert status['top_area_sum_ratio'] == {'state': 'defined_empty', 'reason': 'selection.disabled_top_k'}
    if options.get('area_thresh') == 1.:
        assert status['largest_pore_ratio']['reason'] == 'domain.no_selected_pores'
    if shape != 'normal':
        name = 'top_circularity_min' if shape == 'zero_perimeter' else 'top_convexity_mean'
        assert status[name]['state'] == 'fallback'


@pytest.mark.parametrize('areas', [(), (8.,), (8., 12.)])
def test_mock_mesh_original_value_parity(monkeypatch, areas):
    pair_modules(monkeypatch, MockCV2(areas))
    image = np.full((64, 64), 255, np.uint8)
    status = {}
    equal_values(B(canvas_size=64)._mesh_holes(image), C(canvas_size=64)._mesh_holes(image, _status=status))
    assert all(v['state'] == ('computed' if areas else 'not_applicable') for v in status.values())


def test_mock_mesh_no_mask_and_missing_render_precedence(monkeypatch):
    pair_modules(monkeypatch, MockCV2())
    ext = C(canvas_size=64)
    status = {}
    ext._mesh_holes(np.zeros((64, 64), np.uint8), _status=status)
    assert status['mesh_cv_area']['reason'] == 'domain.no_pores'
    ext._mesh_holes(None, _status=status, _missing_reason='input.no_positions')
    assert status['mesh_cv_area']['reason'] == 'input.no_positions'
    monkeypatch.setattr(candidate, 'HAS_CV2', False)
    ext._mesh_holes(None, _status=status, _missing_reason='input.no_positions')
    assert status['mesh_cv_area']['reason'] == 'dependency_missing.cv2'


@pytest.mark.parametrize('pixels,positions', [
    ([{(0, 0)}, {(2, 2)}], {0: (0, 0), 2: (2, 2), 1: (0, 2), 3: (2, 0)}),
    ([{(1, 1)}, {(1, 1)}], {0: (0, 0), 2: (2, 2), 1: (0, 2), 3: (2, 0)}),
    ([{(1, 1)}, {(1, 1)}], {0: (1, 1), 2: (1, 1), 1: (1, 1), 3: (1, 1)}),
    ([set(), set()], {0: (0, 0), 2: (2, 2), 1: (0, 2), 3: (2, 0)}),
])
def test_mock_contact_original_value_parity(monkeypatch, pixels, positions):
    pair_modules(monkeypatch, MockCV2())
    g = graph('crossing')
    old, new = B(canvas_size=64), C(canvas_size=64)
    def run(ext, status=None):
        remaining = iter(pixels)
        monkeypatch.setattr(ext, '_get_edge_pixels', lambda *args: next(remaining))
        return ext._contact_features(g, positions, **({'_status': status} if status is not None else {}))
    expected = run(old)
    status = {}
    actual = run(new, status)
    equal_values(expected, actual)
    assert len(status) == 42
    if not any(pixels):
        assert all(status[n]['reason'] == 'render_empty_edge' for n in C.CONTACT_MEASUREMENTS)
    elif not pixels[0] & pixels[1]:
        assert status['contact_overlap_pair_size_mean']['state'] == 'not_applicable'
        assert status['contact_overlap_pair_size_sum']['state'] == 'defined_empty'
        assert status['contact_edge_contact_degree_mean']['state'] == 'computed'
        assert actual['contact_edge_contact_degree_mean'] == 0
    elif len(set(positions.values())) == 1:
        assert status['contact_overlap_length_ratio_centerline']['reason'] == 'domain.zero_centerline_length'


def test_contact_no_edges_not_claimed_rendered(monkeypatch):
    pair_modules(monkeypatch, MockCV2())
    g = nx.Graph()
    g.add_node(0, pos=(0, 0))
    status = {}
    equal_values(B()._contact_features(g, {0: (0, 0)}),
                 C()._contact_features(g, {0: (0, 0)}, _status=status))
    assert all(status[n]['reason'] == 'domain.no_edges' for n in C.CONTACT_MEASUREMENTS)
    assert status['contact_nodes']['state'] == 'computed'


def test_mock_full_nonfinite_result_preserves_old_python_values(monkeypatch):
    pair_modules(monkeypatch, MockCV2((8., 8.)), True)
    old, new = B(canvas_size=64), C(canvas_size=64)
    result = new.extract_with_status(graph())
    equal_values(old.extract(graph()), result['values'])
    assert np.isnan(result['values']['pore_area_skew'])
    assert np.isnan(result['values']['pore_area_kurtosis'])
    assert result['status']['pore_area_skew'] == {'state': 'fallback', 'reason': 'stat.nonfinite'}
    meta = {k: v for k, v in result.items() if k != 'values'}
    json.dumps(meta, allow_nan=False)
    candidate._checked_feature_contract(meta)
    with pytest.raises(ValueError):
        json.dumps(result, allow_nan=False)
    # Per-call status must not leak after the environment changes.
    monkeypatch.setattr(candidate, 'HAS_CV2', False)
    monkeypatch.setattr(baseline, 'HAS_CV2', False)
    fresh = new.extract_with_status(graph())
    equal_values(old.extract(graph()), fresh['values'])
    assert fresh['status']['pore_area_skew']['reason'] == 'dependency_missing.cv2'


def test_unexpected_renderer_failure_propagates(monkeypatch):
    def fail(*args):
        raise RuntimeError('raster sentinel')
    for ext in (B(), C()):
        monkeypatch.setattr(ext, '_render_image', fail)
        with pytest.raises(RuntimeError, match='raster sentinel'):
            ext.extract(graph())
    with pytest.raises(RuntimeError, match='raster sentinel'):
        ext.extract_with_status(graph())


def test_frozen_named_partition_and_no_new_wildcard_imports():
    expected_pore = ("largest_pore_ratio", "top_area_sum_ratio", "top_convexity_min",
        "top_convexity_mean", "top_circularity_min", "big_pore_count", "total_pore_count",
        "total_pore_ratio", "center_pore_ratio", "edge_pore_ratio", "pore_area_cv",
        "pore_area_skew", "pore_area_kurtosis", "pore_area_max_over_mean",
        "pore_large_area_frac", "pore_count_large_frac", "pore_density", "pore_spatial_cv")
    assert C.PORE_FEATURES == expected_pore
    assert C.CONTACT_MEASUREMENTS == tuple(n for n in B.FEATURE_COLS
        if n.startswith('contact_') and n not in C.CONTACT_CONFIGURATION)
    assert len(set(C.AUDITED_NAMES)) == 63
    assert C.FEATURE_COLS == B.FEATURE_COLS
    assert {n for n in vars(candidate) if not n.startswith('_')} == {n for n in vars(baseline) if not n.startswith('_')} | {'FeatureUnavailableError'}
