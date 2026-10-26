"""Regression coverage for the legacy feature boundary.
"""
import builtins
import importlib
import sys

import numpy as np
import pytest


GROUPS = ('morphology', 'topology', 'spectral', 'pore', 'anisotropy')
KEYS = ['node_features', 'edge_index', 'edge_features', 'num_nodes', 'num_edges', 'graph']


def classes():
    from fibernet.core.fiber import Fiber
    from fibernet.core.network import FiberNetwork, Crosslink
    from fibernet.ml.features import FeatureExtractor, GNNFeatureExtractor
    return Fiber, FiberNetwork, Crosslink, FeatureExtractor, GNNFeatureExtractor


def fiber(points=None, radius=0.1):
    Fiber, _, _, _, _ = classes()
    return Fiber(centerline=np.array(points if points is not None else [[0, 0, 0], [0, 1, 0]], dtype=float), radius=radius)


def network(fibers=(), contacts=()):
    _, Network, Crosslink, _, _ = classes()
    net = Network()
    for item in fibers:
        net.add_fiber(item)
    for i, j in contacts:
        net.add_crosslink(Crosslink(i, j, 0.5, 0.5, np.array([0., 0.5, 0.])))
    return net


def basic():
    return classes()[3](**{'include_' + name: False for name in GROUPS})


def gnn(**kwargs):
    return classes()[4](**kwargs)


def pair():
    return network([fiber(), fiber([[2, 0, 0], [2, 1, 0]])], [(0, 1)])


def check_output(result, n, e, node_width=6, edge_width=2):
    assert list(result) == KEYS
    assert result['num_nodes'] == n and result['num_edges'] == e
    assert result['graph'].number_of_nodes() == n
    assert result['graph'].number_of_edges() == e
    assert result['node_features'].shape == (n, node_width)
    assert result['edge_features'].shape == (e, edge_width)
    assert result['edge_index'].shape == (2, e)
    assert result['node_features'].dtype == np.float32
    assert result['edge_features'].dtype == np.float32
    assert result['edge_index'].dtype == np.int64
    assert np.isfinite(result['node_features']).all()
    assert np.isfinite(result['edge_features']).all()


def test_import_isolation(monkeypatch):
    assert 'fibernet.ml.features' not in sys.modules, 'requires cold feature-module import'
    original = builtins.__import__
    forbidden = ('fibernet.analysis._archived', 'fibernet.analysis.morphology',
                 'fibernet.analysis.topology', 'fibernet.analysis.advanced')
    attempts = []
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden):
            attempts.append(name)
            raise AssertionError('deleted/archived analyzer import: ' + name)
        return original(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', guarded)
    module = importlib.import_module('fibernet.ml.features')
    ml = importlib.import_module('fibernet.ml')
    assert ml.FeatureExtractor is module.FeatureExtractor
    assert callable(module.extract_features) and callable(module.GNNFeatureExtractor)
    assert not attempts
    assert not any(name.startswith('fibernet.analysis._archived') for name in sys.modules)


@pytest.mark.parametrize('nonempty', [False, True])
def test_basic_schema(nonempty):
    net = pair() if nonempty else network()
    extractor = basic()
    expected = {'num_fibers': 2 if nonempty else 0, 'num_crosslinks': 1 if nonempty else 0, 'dimension': 3}
    result = extractor.extract(net)
    assert result == expected and list(result) == list(expected)
    assert extractor.get_feature_names() == list(expected)
    values = extractor.extract_to_array(net)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, np.array(list(expected.values())))


@pytest.mark.parametrize('group', list(GROUPS) + ['all'])
@pytest.mark.parametrize('operation', ['extract', 'extract_to_array', 'get_feature_names'])
def test_enabled_groups_are_explicit(group, operation):
    kwargs = {'include_' + name: group == 'all' or name == group for name in GROUPS}
    extractor = classes()[3](**kwargs)
    with pytest.raises(NotImplementedError) as error:
        method = getattr(extractor, operation)
        method() if operation == 'get_feature_names' else method(network())
    message = str(error.value)
    for name in GROUPS if group == 'all' else [group]:
        assert name + ':' in message
    assert 'FiberNetwork' in message and 'include_morphology=False' in message
    assert 'not a drop-in replacement' in message


@pytest.mark.parametrize('as_array', [False, True])
def test_convenience_is_diagnostic(as_array):
    from fibernet.ml.features import extract_features
    with pytest.raises(NotImplementedError, match='No complete legacy vector'):
        extract_features(network(), as_array=as_array)


@pytest.mark.parametrize('kind', ['basic', 'gnn'])
def test_wrong_representation(kind):
    extractor = basic() if kind == 'basic' else gnn()
    method = extractor.extract if kind == 'basic' else extractor.extract_graph
    with pytest.raises(TypeError, match='no implicit StructureGraph conversion'):
        method(object())


@pytest.mark.parametrize('node,edge,nw,ew', [
    (None, None, 6, 2), (['degree'], ['length'], 3, 1),
    (['centrality'], ['angle'], 3, 1), ([], [], 6, 2),
    (['unknown'], ['unknown'], 2, 0),
])
def test_empty_shapes(node, edge, nw, ew):
    check_output(gnn(node_features=node, edge_features=edge).extract_graph(network()), 0, 0, nw, ew)


def test_nonempty_values_and_input_unchanged():
    net = pair()
    before = [f.centerline.copy() for f in net.fibers]
    result = gnn().extract_graph(net)
    check_output(result, 2, 1)
    np.testing.assert_allclose(result['node_features'], [[0, .5, 0, 1, 1, .1], [2, .5, 0, 1, 1, .1]])
    expected_angle = np.arccos(1. / (1. + 1e-10) ** 2)
    np.testing.assert_allclose(result['edge_features'], [[2, expected_angle]], rtol=1e-5)
    np.testing.assert_array_equal(result['edge_index'], [[0], [1]])
    for old, f in zip(before, net.fibers):
        np.testing.assert_array_equal(old, f.centerline)
    assert [f.fiber_id for f in net.fibers] == [0, 1]


def test_membership_order_and_duplicates():
    net = pair()
    base = gnn().extract_graph(net)
    result = gnn(node_features=['degree', 'position', 'degree'], edge_features=['angle', 'length', 'angle']).extract_graph(net)
    check_output(result, 2, 1)
    np.testing.assert_array_equal(result['node_features'], base['node_features'])
    np.testing.assert_array_equal(result['edge_features'], base['edge_features'])
    no_columns = gnn(node_features=['degree'], edge_features=['unknown']).extract_graph(net)
    check_output(no_columns, 2, 1, 3, 0)


def test_colocated_fibers_stay_distinct():
    result = gnn().extract_graph(network([fiber(), fiber()]))
    check_output(result, 2, 0)
    assert list(result['graph'].nodes) == [0, 1]


@pytest.mark.parametrize('shape', ['empty', 'disconnected', 'chain'])
def test_centrality_once(monkeypatch, shape):
    import networkx as nx
    net = network() if shape == 'empty' else network([fiber(), fiber(), fiber()] if shape == 'chain' else [fiber(), fiber()], [(0, 1), (1, 2)] if shape == 'chain' else [])
    original = nx.betweenness_centrality
    calls = []
    def once(graph):
        calls.append(graph)
        return original(graph)
    monkeypatch.setattr(nx, 'betweenness_centrality', once)
    result = gnn(node_features=['centrality']).extract_graph(net)
    check_output(result, len(net.fibers), len(net.crosslinks), 3, 2)
    assert len(calls) == 1
    expected = [0., 1., 0.] if shape == 'chain' else [0.] * len(net.fibers)
    np.testing.assert_array_equal(result['node_features'][:, 0], expected)


def test_centrality_failure_propagates(monkeypatch):
    import networkx as nx
    def fail(graph):
        raise RuntimeError('centrality sentinel')
    monkeypatch.setattr(nx, 'betweenness_centrality', fail)
    with pytest.raises(RuntimeError, match='centrality sentinel'):
        gnn(node_features=['centrality']).extract_graph(pair())


@pytest.mark.parametrize('case', ['negative', 'dangling', 'float', 'bool', 'self', 'duplicate', 'reverse', 'position_shape', 'position_nan'])
def test_contact_preflight(case, monkeypatch):
    import networkx as nx
    net = pair()
    cl = net.crosslinks[0]
    if case == 'negative': cl.fiber_i = -1
    elif case == 'dangling': cl.fiber_j = 2
    elif case == 'float': cl.fiber_i = 0.0
    elif case == 'bool': cl.fiber_i = True
    elif case == 'self': cl.fiber_j = 0
    elif case == 'duplicate': net.crosslinks.append(cl)
    elif case == 'reverse':
        _, _, Crosslink, _, _ = classes()
        net.crosslinks.append(Crosslink(1, 0, .5, .5, np.zeros(3)))
    elif case == 'position_shape': cl.position = np.zeros(2)
    elif case == 'position_nan': cl.position = np.array([np.nan, 0., 0.])
    extractor = gnn()
    def forbidden(*args):
        raise AssertionError('feature construction reached after invalid preflight')
    monkeypatch.setattr(extractor, '_extract_node_features', forbidden)
    monkeypatch.setattr(nx, 'Graph', forbidden)
    with pytest.raises(ValueError):
        extractor.extract_graph(net)


@pytest.mark.parametrize('case', ['shape', 'one_point', 'nan', 'radius_negative', 'radius_inf', 'zero_chord', 'integer_mutation'])
def test_geometry_preflight(case, monkeypatch):
    import networkx as nx
    item = fiber()
    if case == 'shape': item.centerline = np.zeros((2, 2))
    elif case == 'one_point': item.centerline = np.zeros((1, 3))
    elif case == 'nan': item.centerline[0, 0] = np.nan
    elif case == 'radius_negative': item.radius = -1.
    elif case == 'radius_inf': item.radius = np.inf
    elif case == 'zero_chord': item.centerline[-1] = item.centerline[0]
    elif case == 'integer_mutation': item.centerline = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int64)
    net = network([item])
    def forbidden(*args, **kwargs):
        raise AssertionError('graph construction reached after invalid geometry preflight')
    monkeypatch.setattr(nx, 'Graph', forbidden)
    with pytest.raises(ValueError):
        gnn().extract_graph(net)


def test_curve_projection_warning():
    item = fiber([[0, 0, 0], [1, 1, 0], [2, 0, 0]])
    net = network([item, fiber([[0, 2, 0], [2, 2, 0]])], [(0, 1)])
    with pytest.warns(UserWarning, match='not a lossless curved-fiber representation'):
        result = gnn().extract_graph(net)
    check_output(result, 2, 1)
    np.testing.assert_allclose(result['node_features'][0, :3], [1, 0, 0])
    np.testing.assert_allclose(result['node_features'][0, -2], 2 * np.sqrt(2))
    expected_angle = np.arccos((2. / (2. + 1e-10)) ** 2)
    np.testing.assert_allclose(result['edge_features'], [[2., expected_angle]], rtol=1e-5)


def test_weight_is_explicit_constant():
    net = pair()
    net.crosslinks[0].stiffness = 123.
    with pytest.warns(UserWarning, match='constant 1.0, not crosslink stiffness'):
        result = gnn(edge_features=['weight']).extract_graph(net)
    check_output(result, 2, 1, 6, 1)
    np.testing.assert_array_equal(result['edge_features'], [[1.]])


@pytest.mark.parametrize('length', [1e-12, 1.01e-12, 1e-10])
def test_legacy_regularized_angle(length):
    net = network([fiber([[0, 0, 0], [length, 0, 0]]), fiber([[0, 1, 0], [length, 1, 0]])], [(0, 1)])
    if length <= 1e-12:
        with pytest.raises(ValueError, match='no defined endpoint-chord angle'):
            gnn().extract_graph(net)
    else:
        result = gnn().extract_graph(net)
        expected = np.arccos((length / (length + 1e-10)) ** 2)
        np.testing.assert_allclose(result['edge_features'][0, 1], expected, rtol=1e-6)


@pytest.mark.parametrize('case', ['midpoint', 'arc_norm', 'chord_norm', 'difference', 'position_cast', 'arc_cast', 'radius_cast', 'distance_cast', 'radius_underflow', 'distance_underflow', 'arc_underflow', 'cosine_nonfinite'])
def test_extreme_numerical_rejection(case, monkeypatch):
    kwargs = {}
    if case == 'midpoint':
        net = network([fiber([[1e308, 0, 0], [1e308, 1, 0]])])
        kwargs = {'edge_features': ['length']}
    elif case == 'arc_norm':
        net = network([fiber([[0, 0, 0], [1e200, 0, 0]])])
        kwargs = {'node_features': ['degree'], 'edge_features': ['length']}
    elif case == 'chord_norm':
        net = network([fiber([[0, 0, 0], [1e200, 0, 0]]), fiber([[0, 1, 0], [1e200, 1, 0]])], [(0, 1)])
    elif case == 'difference':
        net = network([fiber([[-1e308, 0, 0], [1e308, 0, 0]])])
    elif case == 'position_cast':
        net = network([fiber([[1e40, 0, 0], [1e40, 1, 0]])])
    elif case == 'arc_cast':
        net = network([fiber([[0, 0, 0], [1e39, 0, 0]])])
        kwargs = {'node_features': ['degree'], 'edge_features': ['length']}
    elif case == 'radius_cast':
        net = network([fiber(radius=1e40)])
    elif case == 'distance_cast':
        net = network([fiber([[-2e38, 0, 0], [-2e38, 1, 0]]), fiber([[2e38, 0, 0], [2e38, 1, 0]])], [(0, 1)])
        kwargs = {'node_features': ['degree'], 'edge_features': ['length']}
    elif case == 'radius_underflow':
        net = network([fiber(radius=1e-50)])
    elif case == 'distance_underflow':
        net = network([fiber(), fiber([[1e-200, 0, 0], [1e-200, 1, 0]])], [(0, 1)])
        kwargs = {'node_features': ['degree'], 'edge_features': ['length']}
    elif case == 'arc_underflow':
        net = network([fiber([[0, 0, 0], [1e-200, 0, 0]])])
        kwargs = {'edge_features': ['length']}
    else:
        net = pair()
        monkeypatch.setattr(np, 'dot', lambda *args, **kwargs: np.inf)
    with pytest.raises(ValueError, match='range|finite|underflow'):
        gnn(**kwargs).extract_graph(net)


@pytest.mark.parametrize('radius', [float(np.finfo(np.float32).max), 2.0 ** -126])
def test_float32_portable_normal_boundary(radius):
    result = gnn().extract_graph(network([fiber(radius=radius)]))
    check_output(result, 1, 0)
    assert result['node_features'][0, -1] != 0
    assert result['node_features'][0, -1] == np.float32(radius)


def test_float32_subnormal_radius_requires_explicit_rescaling():
    prior = np.geterr()
    try:
        with pytest.raises(ValueError, match='below the portable float32 normal range; rescale'):
            gnn().extract_graph(network([fiber(radius=1e-40)]))
    finally:
        assert np.geterr() == prior


@pytest.mark.parametrize('fail', [False, True])
def test_numpy_error_state_restored(fail):
    prior = np.geterr()
    requested = dict(over='ignore', invalid='warn', divide='raise', under='ignore')
    np.seterr(**requested)
    try:
        net = network([fiber(radius=1e40)]) if fail else pair()
        if fail:
            with pytest.raises(ValueError):
                gnn().extract_graph(net)
        else:
            gnn().extract_graph(net)
        assert np.geterr() == requested
    finally:
        np.seterr(**prior)


def test_names_never_generate_or_analyze(monkeypatch):
    extractor = basic()
    original = builtins.__import__
    def forbidden_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith('fibernet.gen') or name.startswith('fibernet.analysis'):
            raise AssertionError('names attempted generation/analysis import')
        return original(name, globals, locals, fromlist, level)
    def no_extract(*args):
        raise AssertionError('names attempted extraction')
    def no_generation(*args, **kwargs):
        raise AssertionError('names attempted generator invocation')
    generator = sys.modules.get('fibernet.gen')
    assert generator is not None, 'normal pinned package initialization is required'
    monkeypatch.setattr(generator, 'random_straight_2d', no_generation)
    for name, module in list(sys.modules.items()):
        if name.startswith('fibernet.gen.'):
            for attribute in ('random_straight_2d', 'pattern_2d', 'pattern_3d'):
                if attribute in vars(module):
                    monkeypatch.setattr(module, attribute, no_generation)
    monkeypatch.setattr(builtins, '__import__', forbidden_import)
    monkeypatch.setattr(extractor, 'extract', no_extract)
    assert extractor.get_feature_names() == ['num_fibers', 'num_crosslinks', 'dimension']


def test_networkx_absence_does_not_break_basic(monkeypatch):
    extractor = basic()
    original = builtins.__import__
    def no_networkx(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'networkx' or name.startswith('networkx.'):
            raise ImportError('networkx sentinel')
        return original(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', no_networkx)
    assert extractor.extract(network())['num_fibers'] == 0
    with pytest.raises(ImportError, match='networkx sentinel'):
        gnn().extract_graph(network())
