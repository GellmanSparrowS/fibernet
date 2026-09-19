"""Actual cell topology mapped to bilinear patches with explicit seam stitches."""
from dataclasses import dataclass
import numpy as np
from .structure import StructureFactory


@dataclass
class MappingConfig:
    overscale: float = 1.06
    max_points: int = 150000
    max_segments: int = 200000
    stitch: bool = True


def bilinear(quad, uv):
    uv = np.asarray(uv, float)
    u, v = uv[:, 0:1], uv[:, 1:2]
    return ((1-u)*(1-v)*quad[0] + u*(1-v)*quad[1] + u*v*quad[2] + (1-u)*v*quad[3])


def map_cells(vertices, faces, spectrum, unit='square', config=None):
    """Map one actual unit per patch; output contains no supporting mesh edges.

    Each shared patch edge gets one common midpoint anchor. Short stitches
    join it to the nearest point on each adjacent cell fiber. This is an
    explicit manufactured connection, not a claim that visual overlap bonds.
    """
    config = config or MappingConfig()
    if not 1. <= config.overscale <= 1.3:
        raise ValueError('mapping overscale must be 1.00..1.30')
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('invalid surface vertices')
    if faces.ndim != 2 or faces.shape[1] != 4 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('surface requires valid quad faces')
    graph = StructureFactory(unit=unit, grid_x=1, grid_y=1, n_pts_per_side=len(spectrum),
                             line_displacements=np.asarray(spectrum).tolist()).build()
    pos = np.asarray(graph.node_positions(), float)[:, :2]
    edges = np.asarray(graph.edge_array(), int)[:, :2]
    span = np.ptp(pos, axis=0)
    if len(edges) == 0 or np.any(span < 1e-9):
        raise ValueError('unit must have fibers and nonzero width/height')
    uv = ((pos - pos.min(axis=0)) / span - .5) * config.overscale + .5
    # Reserve for up to four seam split points and anchors per patch.
    if len(faces) * (len(pos) + 8) > config.max_points or len(faces) * (len(edges) + 16) > config.max_segments:
        allowed = min(config.max_points // (len(pos)+8), config.max_segments // (len(edges)+16))
        if allowed < 12:
            raise MemoryError('surface mapping budget exceeded; reduce mesh faces or spectrum points')
        source_count = len(faces)
        vertices, faces = coarsen_quads(vertices, faces, allowed)
        output = map_cells(vertices, faces, spectrum, unit, config)
        config.source_faces = source_count
        return output
    mapped = np.concatenate([bilinear(vertices[face], uv) for face in faces])
    points = list(mapped)
    output = []
    splits = {}
    seams = {}
    for fid, face in enumerate(faces):
        for side in range(4):
            a, b = int(face[side]), int(face[(side+1) % 4])
            seams.setdefault(tuple(sorted((a, b))), []).append((fid, side))
    boundary_uv = np.array([[.5, 0.], [1., .5], [.5, 1.], [0., .5]])
    start, vector = uv[edges[:, 0]], uv[edges[:, 1]] - uv[edges[:, 0]]
    norm = np.maximum((vector * vector).sum(axis=1), 1e-15)
    stitch_count = 0
    if config.stitch:
        for seam, adjacent in seams.items():
            if len(adjacent) < 2:
                continue
            anchor = len(points)
            points.append(vertices[list(seam)].mean(axis=0))
            for fid, side in adjacent:
                target = boundary_uv[side]
                t = np.clip(((target-start) * vector).sum(axis=1) / norm, 0., 1.)
                foot = start + t[:, None] * vector
                edge_id = int(np.argmin(((foot-target) ** 2).sum(axis=1)))
                value = float(t[edge_id])
                a, b = edges[edge_id] + fid * len(pos)
                if value < 1e-9:
                    node = int(a)
                elif value > 1-1e-9:
                    node = int(b)
                else:
                    node = len(points)
                    # Linear interpolation on the rendered fiber, not the curved patch.
                    points.append((1-value)*mapped[a] + value*mapped[b])
                    splits.setdefault((fid, edge_id), []).append((value, node))
                output.append((node, anchor))
                stitch_count += 1
    for fid in range(len(faces)):
        for eid, (a, b) in enumerate(edges):
            chain = [int(a)+fid*len(pos)]
            chain.extend(node for _, node in sorted(splits.get((fid, eid), [])))
            chain.append(int(b)+fid*len(pos))
            output.extend(zip(chain[:-1], chain[1:]))
    # Weld numerical coincidences, including seam feet repeated by two sides.
    points = np.asarray(points)
    scale = max(float(np.ptp(vertices, axis=0).max()), 1.)
    _, index, inverse = np.unique(np.round(points / (scale*1e-9)).astype(np.int64),
                                  axis=0, return_index=True, return_inverse=True)
    segments = inverse[np.asarray(output, int)]
    segments = np.unique(np.sort(segments, axis=1), axis=0)
    segments = segments[segments[:, 0] != segments[:, 1]]
    if len(index) > config.max_points or len(segments) > config.max_segments:
        raise MemoryError('surface mapping exceeds output budget')
    config.source_faces = config.mapped_faces = len(faces)
    return points[index], segments


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
    """Bounded OBJ parser; polygons become conforming corner quads as needed."""
    from pathlib import Path
    if Path(path).stat().st_size > 64 * 1024 * 1024:
        raise MemoryError('OBJ exceeds the 64 MB import limit')
    vertices, faces = [], []
    with open(path, encoding='utf-8', errors='strict') as stream:
        for line in stream:
            parts = line.split('#', 1)[0].split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                raw = [int(x.split('/')[0]) for x in parts[1:]]
                ids = [x-1 if x > 0 else len(vertices)+x for x in raw]
                if 0 in raw or not 3 <= len(ids) <= 256 or len(set(ids)) != len(ids) or any(x < 0 or x >= len(vertices) for x in ids):
                    raise ValueError('invalid OBJ face indices')
                faces.append(ids)
            if len(vertices) > (500000 if target_faces else 100000) or len(faces) > (1000000 if target_faces else 10000):
                raise MemoryError('OBJ import budget exceeded')
    if not faces or not vertices:
        raise ValueError('OBJ has no valid faces')
    info = dict(source_vertices=len(vertices), source_faces=len(faces),
                triangles=sum(len(f) == 3 for f in faces), reduced=False)
    if target_faces is not None:
        if not 12 <= int(target_faces) <= 10000:
            raise ValueError('target quad budget must be 12..10000')
        if len(faces) > target_faces or (any(len(f)!=4 for f in faces) and sum(len(f) for f in faces)>target_faces):
            from .obj_polygons import triangulate
            triangles = triangulate(vertices, faces)
            v, f = reduce_triangles(vertices, triangles, target=max(4, int(target_faces)//3))
            vertices, faces = v.tolist(), f.tolist()
            info['reduced'] = True
    if any(len(face) != 4 for face in faces):
        vertices, faces = corner_quads(vertices, faces)
    if target_faces is not None and len(faces) > int(target_faces):
        vertices, faces = coarsen_quads(vertices, faces, int(target_faces))
        info['reduced'] = True
    array = np.asarray(vertices, float)
    if array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError('OBJ has non-finite or malformed vertices')
    info.update(quad_faces=len(faces), converted=bool(info['triangles'] or info['source_faces'] != len(faces)))
    return (array, faces, info) if return_info else (array, faces)
