"""Bounded projected ear clipping for ordinary OBJ polygon faces."""
import numpy as np


def triangulate(vertices, faces):
    vertices = np.asarray(vertices,float)
    output = []
    for face in faces:
        if len(face)==3:
            output.append(face)
            continue
        points = vertices[face]
        normal = np.cross(points, np.roll(points,-1,axis=0)).sum(axis=0)
        xy = np.delete(points,int(np.argmax(np.abs(normal))),axis=1)
        cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
        area = sum(cross(a,b) for a,b in zip(xy,np.roll(xy,-1,axis=0)))
        if abs(area)<1e-20:
            continue
        sign = 1 if area>0 else -1
        active = list(range(len(face)))
        tolerance = max(float(np.ptp(xy,axis=0).max())**2*1e-12,1e-24)
        while len(active)>3:
            found = False
            for j,b in enumerate(active):
                a,c = active[j-1],active[(j+1)%len(active)]
                turn = sign*cross(xy[b]-xy[a],xy[c]-xy[b])
                if abs(turn)<=tolerance:
                    active.pop(j)
                    found = True
                    break
                if turn<0:
                    continue
                others = [k for k in active if k not in (a,b,c)]
                p = xy[others]
                inside = np.ones(len(p),bool)
                for u,v in ((a,b),(b,c),(c,a)):
                    delta = xy[v]-xy[u]
                    inside &= sign*(delta[0]*(p[:,1]-xy[u,1])-delta[1]*(p[:,0]-xy[u,0]))>=-tolerance
                if inside.any():
                    continue
                output.append([face[a],face[b],face[c]])
                active.pop(j)
                found = True
                break
            if not found:
                raise ValueError('OBJ polygon is self-intersecting or cannot be triangulated')
        if len(active)==3:
            output.append([face[k] for k in active])
    if not output:
        raise ValueError('OBJ has no non-degenerate triangles')
    return np.asarray(output,np.int64)
