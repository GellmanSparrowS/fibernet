"""Shared existing planar frame geometry and strict constrained sparse solve."""
import math
import warnings
import numpy as np
from scipy.sparse.linalg import spsolve, MatrixRankWarning

def _finite(value, shape, name):
    obj = np.asarray(value, dtype=object)
    if any(isinstance(item, (bool, np.bool_)) or
           not isinstance(item, (int, float, np.integer, np.floating)) for item in obj.ravel()):
        raise ValueError(f'{name} requires real numbers, not bools or strings')
    try:
        x = np.array(value, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if x.size == 0 and math.prod(shape) == 0:
        x = x.reshape(shape)  # JSON [] loses trailing dimensions of empty branches.
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f'{name} requires finite shape {shape}')
    return x
def _geometry(graph, length_unit):
    scales = {'m': 1.0, 'mm': 1e-3, 'um': 1e-6}
    if length_unit not in scales:
        raise ValueError('length_unit must explicitly be m, mm or um')
    if graph.dimension != 2:
        raise ValueError('viscoelastic frame history supports dimension=2 only')
    node_ids, edge_ids = sorted(graph.nodes), sorted(graph.edges)
    if len(node_ids) < 2 or not edge_ids:
        raise ValueError('a frame requires at least two nodes and an edge')
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i < 0
           for i in node_ids + edge_ids):
        raise ValueError('stable IDs must be nonnegative integers')
    node_ids, edge_ids = list(map(int, node_ids)), list(map(int, edge_ids))
    rows = {nid: i for i, nid in enumerate(node_ids)}
    xyz = _finite([graph.nodes[n].position for n in node_ids], (len(node_ids), 3), 'positions')
    if np.any(xyz[:, 2] != xyz[0, 2]):
        raise ValueError('frame geometry must lie in one XY plane')
    pos = xyz[:, :2] * scales[length_unit]
    edges, radii = [], []
    for eid in edge_ids:
        edge = graph.edges[eid]
        if edge.node_i not in rows or edge.node_j not in rows or edge.node_i == edge.node_j:
            raise ValueError('invalid edge endpoints')
        a, b = rows[edge.node_i], rows[edge.node_j]
        rad = float(edge.radius) * scales[length_unit]
        if isinstance(edge.radius, (bool, np.bool_)) or not math.isfinite(rad) or rad <= 0:
            raise ValueError('radius must be positive and finite')
        d = xyz[b] - xyz[a]
        length = float(np.linalg.norm(pos[b] - pos[a]))
        if not math.isfinite(length) or length < 1e-12:
            raise ValueError('SI member length must be at least 1e-12 m')
        if edge.internal_points is not None and np.size(edge.internal_points):
            points = np.array(edge.internal_points, dtype=float, copy=True)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError('invalid internal points')
            xi = (points - xyz[a]) @ d / np.dot(d, d)
            deviation = points - xyz[a] - xi[:, None] * d
            if (np.any(xi < 0) or np.any(xi > 1) or np.any(np.diff(xi) <= 0)
                    or np.any(np.linalg.norm(deviation, axis=1) > 1e-10 * np.linalg.norm(d))):
                raise ValueError('curved members require explicit straight-edge subdivision')
        edges.append((a, b)); radii.append(rad)
    edges, radii = np.array(edges, dtype=np.int64), np.array(radii)
    dofs = np.array([[3*a, 3*a+1, 3*a+2, 3*b, 3*b+1, 3*b+2] for a, b in edges])
    B, metric = [], []
    for (a, b), r in zip(edges, radii):
        dx, dy = pos[b] - pos[a]
        L = math.hypot(dx, dy); c, s = dx/L, dy/L
        T = np.array([[c,s,0,0,0,0],[-s,c,0,0,0,0],[0,0,1,0,0,0],
                      [0,0,0,c,s,0],[0,0,0,-s,c,0],[0,0,0,0,0,1]])
        local = [[-1/L, 0, 0, 1/L, 0, 0]]
        for xi in ((1-1/math.sqrt(3))/2, (1+1/math.sqrt(3))/2):
            local.append([0, (-6+12*xi)/L**2, (-4+6*xi)/L,
                          0, (6-12*xi)/L**2, (-2+6*xi)/L])
        B.append(np.array(local) @ T)
        A, I = math.pi*r*r, math.pi*r**4/4
        metric.append((A*L, I*L/2, I*L/2))
    B, metric = np.array(B), np.array(metric)
    if not np.isfinite(B).all() or not np.isfinite(metric).all() or np.any(metric <= 0):
        raise ValueError('section geometry underflow/overflow')
    return node_ids, edge_ids, pos, edges, radii, dofs, B, metric


def _histories(times, node_ids, loads, constraints):
    n = len(node_ids); nt = len(times); rows = {nid: i for i, nid in enumerate(node_ids)}
    force = np.zeros((nt, n, 3)); prescribed = np.zeros_like(force)
    mask = np.zeros((n, 3), dtype=bool)
    for mapping, kind in ((loads, 'loads'), (constraints, 'constraints')):
        if mapping is None:
            continue
        if not isinstance(mapping, dict):
            raise ValueError(f'{kind} must map stable node IDs to (times,3) samples')
        for nid, history in mapping.items():
            if isinstance(nid, (bool, np.bool_)) or not isinstance(nid, (int, np.integer)) or nid not in rows:
                raise ValueError(f'{kind} references an invalid stable node ID')
            i = rows[nid]
            if kind == 'loads':
                force[:, i] = _finite(history, (nt, 3), 'load history')
            else:
                h = np.asarray(history, dtype=object)
                if h.shape != (nt, 3):
                    raise ValueError('constraint history requires (times,3) samples')
                active = np.array([[x is not None for x in row] for row in h])
                if np.any(active != active[0]):
                    raise ValueError('constraint masks must be constant in time')
                mask[i] = active[0]
                for component in range(3):
                    if active[0, component]:
                        prescribed[:, i, component] = _finite(h[:, component], (nt,), 'prescribed history')
    return force.reshape(nt, -1), prescribed.reshape(nt, -1), mask.ravel()


def _check_supports(pos, edges, mask):
    """Positive welded beam elements have only component rigid null modes."""
    parent = list(range(len(pos))); degree = np.zeros(len(pos), dtype=int)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in edges:
        parent[root(a)] = root(b); degree[a] += 1; degree[b] += 1
    if np.any(degree == 0):
        raise ValueError('isolated nodes are unsupported in a beam history')
    components = {}
    for i in range(len(pos)):
        components.setdefault(root(i), []).append(i)
    for nodes in components.values():
        xy = pos[nodes] - pos[nodes].mean(axis=0)
        scale = max(float(np.max(np.abs(xy))), 1e-12)
        restrictions = []
        for i, (x, y) in zip(nodes, xy/scale):
            modes = ((1,0,-y),(0,1,x),(0,0,1))
            restrictions.extend(modes[k] for k in range(3) if mask[3*i+k])
        if len(restrictions) < 3 or np.linalg.matrix_rank(np.array(restrictions), tol=1e-10) < 3:
            raise ValueError('each connected component requires three independent rigid-mode restraints')


def _solve_constrained_sparse(K, rhs, prescribed, mask, *, solve=None):
    """Existing unregularized history solve, with injectable legacy solve seam."""
    if not np.isfinite(K.data).all() or not np.isfinite(rhs).all():
        raise FloatingPointError('nonfinite history system')
    free = np.flatnonzero(~mask)
    fixed = np.flatnonzero(mask)
    candidate = prescribed.copy()
    if free.size:
        right = rhs[free] - K[free][:, fixed] @ candidate[fixed]
        if not np.isfinite(right).all():
            raise FloatingPointError('nonfinite reduced system')
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            candidate[free] = (spsolve if solve is None else solve)(K[free][:, free], right)
    if not np.isfinite(candidate).all():
        raise FloatingPointError('nonfinite history solution')
    return candidate
