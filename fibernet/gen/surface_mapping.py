"""Actual cell topology mapped to bilinear patches with explicit seam stitches."""
from dataclasses import dataclass
import numpy as np
from .manufacturing import PlanarManufacturingConfig, manufacturable_graph
from .surface_geometry import coarsen_quads


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


def map_cells(vertices, faces, spectrum, unit='square', config=None, base_graph=None):
    """Map one actual unit per patch; output contains no supporting mesh edges.

    Each shared patch edge gets one common midpoint anchor. Short stitches
    join it to the nearest point on each adjacent cell fiber. This is an
    explicit manufactured connection, not a claim that visual overlap bonds.
    """
    config = config or MappingConfig()
    if not 1. <= config.overscale <= 1.3:
        raise ValueError('mapping overscale must be 1.00..1.30')
    vertices = np.asarray(vertices, float)
    raw_faces = np.asarray(faces)
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.size == 0:
        spectrum = spectrum.reshape(0, 2)
    if (spectrum.ndim != 2 or spectrum.shape[1] != 2 or
            not np.isfinite(spectrum).all()):
        raise ValueError('spectrum must contain finite [dx, dy] pairs')
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or
            len(vertices) < 4 or not np.isfinite(vertices).all()):
        raise ValueError('invalid surface vertices')
    if (raw_faces.ndim != 2 or raw_faces.shape[1] != 4 or
            not len(raw_faces) or
            not np.issubdtype(raw_faces.dtype, np.integer)):
        raise ValueError('surface requires four integer indices per face')
    faces = np.asarray(raw_faces, dtype=np.int64)
    if (faces.min() < 0 or faces.max() >= len(vertices) or
            np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)):
        raise ValueError('surface requires valid quad faces')
    graph = base_graph
    if graph is None:
        spec = PlanarManufacturingConfig(
            unit=unit, grid_x=1, grid_y=1, n_pts_per_side=len(spectrum),
            line_displacements=np.asarray(spectrum).tolist())
        graph = manufacturable_graph(spec)
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
        output = map_cells(vertices, faces, spectrum, unit, config, base_graph=graph)
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

