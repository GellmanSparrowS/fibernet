"""Analytic small-graph spectral oracles and controlled eigensolver failures."""
import numpy as np
import pytest

from fibernet.analysis.spectrum_metrics import analyze_spectrum
from fibernet.core.structure_graph import StructureGraph


def graph(n, edges):
    ids = [10 + 20 * i for i in range(n)]
    return StructureGraph.from_dict({'nodes': [{'id': x, 'position': [i, 0, 0]}
        for i, x in enumerate(ids)], 'edges': [{'id': 100 + i, 'node_i': ids[a],
        'node_j': ids[b]} for i, (a, b) in enumerate(edges)]})


@pytest.mark.parametrize('n,edges,expected,components', [
    (3, [(0, 1)], [0, 0, 2], 2),
    (3, [(0, 1), (1, 2)], [0, 1, 3], 1),
    (4, [(0, 1), (1, 2), (2, 3), (3, 0)], [0, 2, 2, 4], 1),
    (4, [(i, j) for i in range(4) for j in range(i + 1, 4)], [0, 4, 4, 4], 1),
    (4, [(0, 1), (2, 3)], [0, 0, 2, 2], 2),
    (3, [], [0, 0, 0], 3)])
def test_analytic_spectra(n, edges, expected, components):
    m = analyze_spectrum(graph(n, edges))
    assert m.schema == 'fibernet.spectrum/1'
    np.testing.assert_allclose(m.raw_eigenvalues, expected, atol=1e-13)
    assert m.n_components == m.topological_zero_count == components
    assert m.lambda_max == pytest.approx(expected[-1], abs=1e-13)
    assert m.fiedler_value == (0 if components > 1 else pytest.approx(expected[1]))
    assert m.residual_inf_norm <= m.residual_tolerance
    assert m.orthogonality_inf_norm <= m.orthogonality_tolerance
    assert m.residual_tolerance == 64 * np.finfo(float).eps * n * max(1, m.laplacian_inf_norm)
    assert m.orthogonality_tolerance == 64 * np.finfo(float).eps * n


def test_parallel_policy_and_multiplicity_scaling():
    g = graph(2, [(0, 1), (1, 0)])
    np.testing.assert_allclose(analyze_spectrum(g).raw_eigenvalues, [0, 4])
    np.testing.assert_allclose(analyze_spectrum(g, adjacency='unique').raw_eigenvalues, [0, 2])
    a = analyze_spectrum(graph(3, [(0, 1), (1, 2)]))
    b = analyze_spectrum(graph(3, [(0, 1), (1, 2)] * 7))
    np.testing.assert_allclose(b.raw_eigenvalues, np.array(a.raw_eigenvalues) * 7, atol=1e-13)
    assert b.fiedler_value == pytest.approx(a.fiedler_value * 7)
    assert b.lambda_max == pytest.approx(a.lambda_max * 7)


@pytest.mark.parametrize('n,values,largest,solver', [(0, (), None, 'analytic_empty'),
                                                    (1, (0,), 0, 'analytic_singleton')])
def test_undefined_small_graphs(n, values, largest, solver):
    m = analyze_spectrum(graph(n, []))
    assert m.raw_eigenvalues == values and m.lambda_max == largest
    assert m.fiedler_value is None
    assert m.topological_zero_count == n
    assert m.solver == solver


def test_geometry_weights_orientation_and_order_do_not_change_definition():
    g = graph(3, [(0, 1), (1, 2)])
    a = analyze_spectrum(g)
    for node in g.nodes.values():
        node.position *= 1e200
    for edge in g.edges.values():
        edge.node_i, edge.node_j = edge.node_j, edge.node_i
        edge.radius = 999
        edge.metadata['weight'] = 1e100
    g._nodes = dict(reversed(list(g.nodes.items())))
    b = analyze_spectrum(g)
    assert a == b
    assert b.node_ids == (10, 30, 50) and b.edge_ids == (100, 101)


@pytest.mark.parametrize('kind', ['values_shape', 'vectors_shape', 'values_nan', 'vectors_inf',
                                  'values_complex', 'negative', 'residual', 'orthogonality'])
def test_bad_eigensolver_output_is_explicit_failure(kind, monkeypatch):
    original = np.linalg.eigh
    def bad(matrix):
        w, v = original(matrix)
        if kind == 'values_shape': w = w[:, None]
        elif kind == 'vectors_shape': v = v[:, :1]
        elif kind == 'values_nan': w[0] = np.nan
        elif kind == 'vectors_inf': v[0, 0] = np.inf
        elif kind == 'values_complex': w = w.astype(complex) + 1j
        elif kind == 'negative': w[0] = -1
        elif kind == 'residual': w[-1] += 1
        elif kind == 'orthogonality': v *= 2
        return w, v
    monkeypatch.setattr(np.linalg, 'eigh', bad)
    with pytest.raises(ArithmeticError):
        analyze_spectrum(graph(3, [(0, 1), (1, 2)]))


def test_solver_failure_not_caught_as_zero(monkeypatch):
    def fail(matrix):
        raise np.linalg.LinAlgError('controlled failure')
    monkeypatch.setattr(np.linalg, 'eigh', fail)
    with pytest.raises(np.linalg.LinAlgError, match='controlled'):
        analyze_spectrum(graph(2, [(0, 1)]))


def test_small_roundoff_kept_raw_and_sort_preserves_eigenpairs(monkeypatch):
    original = np.linalg.eigh
    def perturbed(matrix):
        w, v = original(matrix)
        w[0] = -1e-15
        return w[::-1], v[:, ::-1]
    monkeypatch.setattr(np.linalg, 'eigh', perturbed)
    m = analyze_spectrum(graph(3, [(0, 1)]))
    assert m.raw_eigenvalues[0] == -1e-15
    assert m.topological_zero_count == 2
    assert m.fiedler_value == 0


@pytest.mark.parametrize('name', ['max_nodes', 'max_edges', 'max_matrix_bytes'])
@pytest.mark.parametrize('bad', [True, 1.0, -1, 0])
def test_strict_budgets(name, bad):
    with pytest.raises(ValueError, match=name):
        analyze_spectrum(graph(0, []), **{name: bad})


@pytest.mark.parametrize('kwargs', [{'max_nodes': 1}, {'max_edges': 1}, {'max_matrix_bytes': 575}])
def test_budget_precedes_dense_allocation(kwargs, monkeypatch):
    def fail(*args, **kw):
        raise AssertionError('must fail before matrix allocation')
    monkeypatch.setattr(np, 'zeros', fail)
    with pytest.raises(MemoryError):
        analyze_spectrum(graph(3, [(0, 1), (1, 2)]), **kwargs)


def test_hard_dense_node_cap():
    with pytest.raises(MemoryError, match='512'):
        analyze_spectrum(graph(513, []), max_nodes=1000, max_matrix_bytes=10**9)


def test_full_supported_small_matrix_path_oracle():
    n = 512
    m = analyze_spectrum(graph(n, [(i, i + 1) for i in range(n - 1)]),
                         max_nodes=512, max_matrix_bytes=8 * 8 * 512 * 512)
    expected = 2 - 2 * np.cos(np.pi * np.arange(n) / n)
    np.testing.assert_allclose(m.raw_eigenvalues, expected, atol=1e-13)


@pytest.mark.parametrize('bad', [True, 10.0, -1])
@pytest.mark.parametrize('kind', ['node_key', 'node_record', 'edge_key', 'edge_record', 'endpoint'])
def test_strict_identity(kind, bad):
    g = graph(2, [(0, 1)])
    if kind == 'node_key': g.nodes[bad] = g.nodes.pop(10)
    elif kind == 'node_record': g.nodes[10].node_id = bad
    elif kind == 'edge_key': g.edges[bad] = g.edges.pop(100)
    elif kind == 'edge_record': g.edges[100].edge_id = bad
    else: g.edges[100].node_i = bad
    with pytest.raises(ValueError):
        analyze_spectrum(g)


def test_policy_type_loops_and_numpy_ids():
    with pytest.raises(TypeError): analyze_spectrum(None)
    with pytest.raises(ValueError): analyze_spectrum(graph(0, []), adjacency='weighted')
    g = graph(2, [(0, 1)])
    g.nodes[10].node_id = np.int64(10)
    g.edges[100].edge_id = np.uint64(100)
    g.edges[100].node_i = np.int64(10)
    m = analyze_spectrum(g, max_nodes=np.int64(2))
    assert all(type(i) is int for i in m.node_ids + m.edge_ids)
    g.edges[100].node_i = 30
    with pytest.raises(ValueError, match='self-loops'):
        analyze_spectrum(g)


@pytest.mark.parametrize('target', ['values', 'vectors'])
@pytest.mark.parametrize('mask', [False, True])
def test_masked_solver_output_rejected_before_array_conversion(target, mask, monkeypatch):
    original = np.linalg.eigh
    def masked(matrix):
        values, vectors = original(matrix)
        if target == 'values':
            values = np.ma.array(values, mask=mask)
        else:
            vectors = np.ma.array(vectors, mask=mask)
        return values, vectors
    monkeypatch.setattr(np.linalg, 'eigh', masked)
    with pytest.raises(ArithmeticError, match='masked'):
        analyze_spectrum(graph(2, [(0, 1)]))
