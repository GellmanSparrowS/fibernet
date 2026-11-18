"""Bounded projected ear clipping for ordinary OBJ polygon faces."""
import numpy as np


_FLOAT_EPS = np.finfo(float).eps


def _cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _cross_roundoff(a, b):
    """Bound determinant roundoff without treating short edges as collinear."""
    return 8 * _FLOAT_EPS * (np.abs(a[..., 0] * b[..., 1]) + np.abs(a[..., 1] * b[..., 0]))


def _clip_ears(xy, face, sign, tolerance=None):
    # The exact import path preserves the boundary; legacy reduction may drop collinear corners.
    keep_boundary = tolerance is None
    output = []
    active = list(range(len(face)))
    while len(active) > 3:
        for j, b in enumerate(active):
            a, c = active[j - 1], active[(j + 1) % len(active)]
            before, after = xy[b] - xy[a], xy[c] - xy[b]
            turn = sign * _cross_2d(before, after)
            turn_error = _cross_roundoff(before, after) if keep_boundary else tolerance
            if abs(turn) <= turn_error:
                if keep_boundary:
                    continue
                active.pop(j)
                break
            if turn < 0:
                continue
            others = [k for k in active if k not in (a, b, c)]
            points = xy[others]
            inside = np.ones(len(points), bool)
            for u, v in ((a, b), (b, c), (c, a)):
                edge, offsets = xy[v] - xy[u], points - xy[u]
                edge_error = _cross_roundoff(edge, offsets) if keep_boundary else tolerance
                inside &= sign * _cross_2d(edge, offsets) >= -edge_error
            if inside.any():
                continue
            output.append([face[a], face[b], face[c]])
            active.pop(j)
            break
        else:
            raise ValueError('OBJ polygon is self-intersecting or cannot be triangulated')
    if len(active) == 3:
        output.append([face[k] for k in active])
    return output


def triangulate(vertices, faces):
    vertices = np.asarray(vertices, float)
    output = []
    for face in faces:
        if len(face) == 3:
            output.append(face)
            continue
        points = vertices[face]
        normal = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
        xy = np.delete(points, int(np.argmax(np.abs(normal))), axis=1)
        area = sum(_cross_2d(a, b) for a, b in zip(xy, np.roll(xy, -1, axis=0)))
        if abs(area) < 1e-20:
            continue
        sign = 1 if area > 0 else -1
        tolerance = max(float(np.ptp(xy, axis=0).max()) ** 2 * 1e-12, 1e-24)
        output.extend(_clip_ears(xy, face, sign, tolerance))
    if not output:
        raise ValueError('OBJ has no non-degenerate triangles')
    return np.asarray(output, np.int64)


def _split_planar_concave_faces(vertices, faces):
    """Triangulate planar concave faces without dropping shared boundary vertices."""
    output = []
    tolerance = 1e-12
    for face in faces:
        if len(face) == 3:
            output.append(face)
            continue
        source = vertices[face]
        points = source - source[0]
        scale = float(np.abs(points).max())
        if scale == 0:
            output.append(face)
            continue
        # A power-of-two normalization avoids rounding away distinctions on short edges.
        _, exponent = np.frexp(scale)
        points = np.ldexp(points, -exponent)
        normal = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
        axis = int(np.argmax(np.abs(normal)))
        xy = points[:, [(axis + 1) % 3, (axis + 2) % 3]]
        area = normal[axis]
        area_error = _cross_roundoff(xy, np.roll(xy, -1, axis=0)).sum()
        # Project source-coordinate resolution onto the normal for the planarity test.
        resolution = _FLOAT_EPS * np.abs(source).max(axis=0)
        normal_roundoff = np.ldexp(resolution @ np.abs(normal), -exponent)
        plane_error = tolerance * np.linalg.norm(normal) + 4 * normal_roundoff
        if abs(area) <= area_error or np.abs(points @ normal).max() > plane_error:
            # A warped face has no unique planar interior. Keep its prior mapping.
            output.append(face)
            continue
        edges = np.roll(xy, -1, axis=0) - xy
        next_edges = np.roll(edges, -1, axis=0)
        sign = 1 if area > 0 else -1
        turns = _cross_2d(edges, next_edges)
        if (sign * turns < -_cross_roundoff(edges, next_edges)).any():
            output.extend(_clip_ears(xy, face, sign))
        else:
            output.append(face)
    return output
