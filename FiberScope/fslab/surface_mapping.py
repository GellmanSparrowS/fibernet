"""Actual cell topology mapped to bilinear patches with explicit seam stitches."""
import numpy as np
from fibernet.gen.surface_mapping import (MappingConfig,
                                           map_cells as _shared_map_cells)


def bilinear(quad, uv):
    uv = np.asarray(uv, float)
    u, v = uv[:, 0:1], uv[:, 1:2]
    return ((1-u)*(1-v)*quad[0] + u*(1-v)*quad[1] + u*v*quad[2] + (1-u)*v*quad[3])


def map_cells(vertices, faces, spectrum, unit='square', config=None):
    """Map APP unit cells through the shared library geometry core."""
    from .structure import StructureFactory
    graph = StructureFactory(
        unit=unit, grid_x=1, grid_y=1, n_pts_per_side=len(spectrum),
        line_displacements=np.asarray(spectrum).tolist()).build()
    return _shared_map_cells(vertices, faces, spectrum, unit=unit,
                             config=config, base_graph=graph)


def front_basis(vertices):
    """Deterministic principal front plane; viewing only, never edits assets."""
    vertices = np.asarray(vertices, float)
    centered = vertices - vertices.mean(0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    normal = axes[:, 0]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    up = np.array([0., 1., 0.])
    up -= normal * np.dot(up, normal)
    if np.linalg.norm(up) < .1:
        up = np.array([0., 0., 1.])
        up -= normal * np.dot(up, normal)
    up /= np.linalg.norm(up)
    right = np.cross(up, normal)
    return np.vstack([right, up, normal])


def subdivide_quads(vertices, faces, levels=1, max_faces=10000):
    if not 0 <= int(levels) <= 3:
        raise ValueError('quad subdivision levels must be 0..3')
    if len(faces)*4**int(levels)>max_faces:
        raise MemoryError('subdivision exceeds 10000 faces; reduce input target or subdivision levels')
    for _ in range(int(levels)):
        vertices,faces=corner_quads(vertices,faces)
    return np.asarray(vertices,float),faces


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


def load_obj(path, return_info=False, target_faces=None):
    from fibernet.gen.obj_import import load_obj as shared_load_obj
    return shared_load_obj(path, return_info=return_info,
                           target_faces=target_faces)
