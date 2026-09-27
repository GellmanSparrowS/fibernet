"""Bounded OBJ import for mapped manufacturing surfaces."""
from pathlib import Path
import numpy as np
from .surface_geometry import corner_quads, coarsen_quads, reduce_triangles

def load_obj(path, return_info=False, target_faces=None):
    """Read a bounded OBJ surface and return a conforming quad mesh."""
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
                if (0 in raw or not 3 <= len(ids) <= 256 or
                        len(set(ids)) != len(ids) or
                        any(x < 0 or x >= len(vertices) for x in ids)):
                    raise ValueError('invalid OBJ face indices')
                faces.append(ids)
            if (len(vertices) > (500000 if target_faces else 100000) or
                    len(faces) > (1000000 if target_faces else 10000)):
                raise MemoryError('OBJ import budget exceeded')
    if not faces or not vertices:
        raise ValueError('OBJ has no valid faces')
    info = dict(source_vertices=len(vertices), source_faces=len(faces),
                triangles=sum(len(f) == 3 for f in faces), reduced=False)
    if target_faces is not None:
        if not 12 <= int(target_faces) <= 10000:
            raise ValueError('target quad budget must be 12..10000')
        if (len(faces) > target_faces or
                (any(len(f) != 4 for f in faces) and
                 sum(len(f) for f in faces) > target_faces)):
            from .obj_polygons import triangulate
            triangles = triangulate(vertices, faces)
            v, f = reduce_triangles(vertices, triangles,
                                    target=max(4, int(target_faces)//3))
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
    info.update(quad_faces=len(faces),
                converted=bool(info['triangles'] or
                               info['source_faces'] != len(faces)))
    return (array, faces, info) if return_info else (array, faces)

