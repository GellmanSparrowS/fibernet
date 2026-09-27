"""Pure quad mapping and bounded surface coarsening helpers."""
import numpy as np


def bilinear(quad, uv):
    uv = np.asarray(uv, float)
    u, v = uv[:, 0:1], uv[:, 1:2]
    return ((1-u)*(1-v)*quad[0] + u*(1-v)*quad[1] + u*v*quad[2] + (1-u)*v*quad[3])


def corner_quads(vertices, faces):
    vertices = np.asarray(vertices, float).tolist()
    mids, result = {}, []
    for face in faces:
        face = list(face)
        center = len(vertices)
        vertices.append(np.mean([vertices[i] for i in face], axis=0).tolist())
        edge_mid = []
        for a, b in zip(face, face[1:] + face[:1]):
            key = tuple(sorted((a, b)))
            if key not in mids:
                mids[key] = len(vertices)
                vertices.append(((np.asarray(vertices[a]) + vertices[b])*.5).tolist())
            edge_mid.append(mids[key])
        result.extend([[node, edge_mid[i], center, edge_mid[i-1]] for i, node in enumerate(face)])
    return np.asarray(vertices), result


def coarsen_quads(vertices, faces, budget):
    faces = np.asarray(faces, int)
    triangles = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
    v, f = reduce_triangles(vertices, triangles, max(4, int(budget)//3))
    if len(f) * 3 > budget:
        # Tiny disconnected accessories can prevent further edge collapse.
        # Retain components covering at least 95% of the source surface area.
        parent = np.arange(len(v))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a, b, c in f:
            parent[root(b)] = root(a)
            parent[root(c)] = root(a)
        groups = np.array([root(int(face[0])) for face in f])
        areas = np.linalg.norm(np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]]), axis=1)
        totals = np.bincount(groups, weights=areas, minlength=len(v))
        order = np.argsort(totals)[::-1]
        count = np.searchsorted(np.cumsum(totals[order]), .95*totals.sum())+1
        f = f[np.isin(groups, order[:count])]
        v, f = reduce_triangles(v, f, max(4, int(budget)//3))
    v, f = corner_quads(v, f)
    if len(f) > budget:
        raise MemoryError('mesh cannot fit the mapping budget; reduce spectrum points')
    return v, f


def reduce_triangles(vertices, faces, target=500):
    """Bounded edge collapse; weld fine seams if topology prevents reduction.

    Clustering uses at most 1/16 of the longest bounding-box dimension.
    This is a display mesh approximation, never a change to the source file.
    """
    import fast_simplification
    v, f = np.asarray(vertices, float), np.asarray(faces, np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('invalid triangle vertices')
    center, scale = v.mean(0), max(float(np.ptp(v, axis=0).max()), 1e-12)
    v = (v-center)/scale
    for resolution in (None, 256, 128, 64, 32, 16):
        if resolution:
            _, inverse = np.unique(np.round(v*resolution).astype(np.int64), axis=0, return_inverse=True)
            counts = np.bincount(inverse)
            v = np.column_stack([np.bincount(inverse, weights=v[:, k])/counts for k in range(3)])
            f = inverse[f]
            f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
            _, ids = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
            f = f[np.sort(ids)]
        if not len(f):
            raise ValueError('mesh collapsed during reduction')
        if len(f) > target:
            v, f = fast_simplification.simplify(v, f, target_count=int(target), agg=10)
        if len(f) <= target:
            break
    return v*scale+center, f


