"""Bounded small-dense, unweighted combinatorial Laplacian spectra."""
from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Tuple

import numpy as np
from fibernet.core.structure_graph import StructureGraph


@dataclass(frozen=True)
class LaplacianSpectrumV1:
    node_ids: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    adjacency: str
    n_components: int
    topological_zero_count: int
    raw_eigenvalues: Tuple[float, ...]
    lambda_max: Optional[float]
    fiedler_value: Optional[float]
    solver: str
    laplacian_inf_norm: float
    residual_inf_norm: float
    residual_tolerance: float
    orthogonality_inf_norm: float
    orthogonality_tolerance: float
    schema: str = field(default='fibernet.spectrum/1', init=False)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def analyze_spectrum(graph, *, adjacency='physical_multiplicity', max_nodes=256,
                     max_edges=20000, max_matrix_bytes=4194304) -> LaplacianSpectrumV1:
    """Analyze L=D-A with explicit edge multiplicity and raw numerical values.

    Hard limits are 512 nodes and 20000 physical edges. Planned dense workspace
    is estimated as eight N-by-N float64 arrays, not an OS memory guarantee.
    No coordinate, radius, material or metadata weights enter this spectrum.
    Invalid eigensolver results fail explicitly; small roundoff is not clipped.
    """
    if not isinstance(graph, StructureGraph):
        raise TypeError('analyze_spectrum requires StructureGraph')
    if adjacency not in ('physical_multiplicity', 'unique'):
        raise ValueError('adjacency must be physical_multiplicity or unique')
    for name, value in [('max_nodes', max_nodes), ('max_edges', max_edges),
                        ('max_matrix_bytes', max_matrix_bytes)]:
        if not _integer(value) or value == 0:
            raise ValueError(f'{name} must be a positive integer, not bool')
    n, e = len(graph.nodes), len(graph.edges)
    if n > min(max_nodes, 512):
        raise MemoryError('graph exceeds max_nodes or hard 512-node dense limit')
    if e > min(max_edges, 20000):
        raise MemoryError('graph exceeds max_edges or hard 20000-edge limit')
    if 8 * n * n * np.dtype(np.float64).itemsize > max_matrix_bytes:
        raise MemoryError('planned dense arrays exceed max_matrix_bytes')
    for nid, node in graph.nodes.items():
        if not _integer(nid) or not _integer(node.node_id) or nid != node.node_id:
            raise ValueError('node IDs must be matching nonnegative integers')
    node_ids = tuple(sorted(int(nid) for nid in graph.nodes))
    rows = {nid: row for row, nid in enumerate(node_ids)}
    edge_ids, pairs = [], []
    for eid, edge in graph.edges.items():
        a, b = edge.node_i, edge.node_j
        if not _integer(eid) or not _integer(edge.edge_id) or eid != edge.edge_id:
            raise ValueError('edge IDs must be matching nonnegative integers')
        if not _integer(a) or not _integer(b) or a not in rows or b not in rows:
            raise ValueError('edge endpoints must be existing nonnegative integer IDs')
        if a == b:
            raise ValueError('spectrum v1 does not support self-loops')
        edge_ids.append(int(eid))
        pairs.append((min(int(a), int(b)), max(int(a), int(b))))
    edge_ids = tuple(edge_ids)
    if n < 2:
        return LaplacianSpectrumV1(node_ids, edge_ids, adjacency, n, n,
            () if n == 0 else (0.0,), None if n == 0 else 0.0, None,
            'analytic_empty' if n == 0 else 'analytic_singleton', 0., 0., 0., 0., 0.)
    import networkx as nx
    simple = nx.Graph()
    simple.add_nodes_from(node_ids)
    simple.add_edges_from(pairs)
    components = nx.number_connected_components(simple)
    laplacian = np.zeros((n, n), dtype=np.float64)
    selected_pairs = pairs if adjacency == 'physical_multiplicity' else sorted(set(pairs))
    for a, b in selected_pairs:
        i, j = rows[a], rows[b]
        laplacian[i, i] += 1
        laplacian[j, j] += 1
        laplacian[i, j] -= 1
        laplacian[j, i] -= 1
    norm = float(np.linalg.norm(laplacian, ord=np.inf))
    epsilon = np.finfo(np.float64).eps
    residual_tolerance = float(64 * epsilon * n * max(1., norm))
    orthogonality_tolerance = float(64 * epsilon * n)
    # Solver failures intentionally propagate, including numpy.linalg.LinAlgError.
    values, vectors = np.linalg.eigh(laplacian)
    if np.ma.isMaskedArray(values) or np.ma.isMaskedArray(vectors):
        raise ArithmeticError('eigensolver returned masked values or vectors')
    values, vectors = np.asarray(values), np.asarray(vectors)
    if (values.shape != (n,) or vectors.shape != (n, n)
            or values.dtype.kind not in 'iuf' or vectors.dtype.kind not in 'iuf'
            or not np.isfinite(values).all() or not np.isfinite(vectors).all()):
        raise ArithmeticError('eigensolver returned invalid shape, dtype or nonfinite values')
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    if values[0] < -residual_tolerance:
        raise ArithmeticError('eigensolver returned a materially negative Laplacian eigenvalue')
    with np.errstate(over='ignore', invalid='ignore'):
        residual = float(np.linalg.norm(laplacian @ vectors - vectors * values, ord=np.inf))
        orthogonality = float(np.linalg.norm(vectors.T @ vectors - np.eye(n), ord=np.inf))
    if not np.isfinite(residual) or residual > residual_tolerance:
        raise ArithmeticError('eigensolver residual exceeds acceptance tolerance')
    if not np.isfinite(orthogonality) or orthogonality > orthogonality_tolerance:
        raise ArithmeticError('eigensolver orthogonality exceeds acceptance tolerance')
    if components == 1 and values[1] <= 0:
        raise ArithmeticError('connected graph Fiedler value is not numerically positive')
    return LaplacianSpectrumV1(node_ids, edge_ids, adjacency, components, components,
        tuple(float(value) for value in values), float(values[-1]),
        0.0 if components > 1 else float(values[1]), 'numpy.linalg.eigh', norm,
        residual, residual_tolerance, orthogonality, orthogonality_tolerance)
