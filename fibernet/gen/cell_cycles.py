"""Reference-space closed fiber walks; geometric crossings never weld fibers."""
import numpy as np


def closed_cell_edges(points, edges):
    """Trace bounded faces of a rotation system, then close uncovered branches.

    Face boundaries have independent directed identities. Thus an internal
    shared side occurs twice with opposite directions. Open branches receive
    an explicit return fiber, before tiling or deformation takes place.
    """
    points = np.asarray(points, float)
    source = sorted({tuple(sorted(map(int, e))) for e in edges if e[0] != e[1]})
    adjacency = [[] for _ in points]
    for a, b in source:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for a, neighbors in enumerate(adjacency):
        neighbors.sort(key=lambda b: (np.arctan2(points[b, 1]-points[a, 1],
                                               points[b, 0]-points[a, 0]), b))
    visited, result, covered = set(), [], set()
    for a, b in source:
        for initial in ((a, b), (b, a)):
            if initial in visited:
                continue
            walk, current = [], initial
            while current not in visited:
                visited.add(current)
                walk.append(current)
                u, v = current
                around = adjacency[v]
                current = (v, around[(around.index(u)-1) % len(around)])
            area = sum(points[u, 0]*points[v, 1]-points[v, 0]*points[u, 1]
                       for u, v in walk)
            if current == initial and area > 1e-10:
                result.extend(walk)
                covered.update(tuple(sorted(e)) for e in walk)
    for a, b in source:
        if (a, b) not in covered:
            result.extend(((a, b), (b, a)))
    return np.asarray(result, dtype=np.int64).reshape(-1, 2)
