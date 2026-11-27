"""Shared triangle connectivity for cap inspection and streamline assignment."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def edge_connected_triangles(c):
    """Return labels for triangles connected through complete shared index edges."""
    faces=len(c)
    parent={}
    a,b=[],[]
    for j,tri in enumerate(c):
        for i,k in ((0,1),(0,2),(1,2)):
            e=tuple(sorted((int(tri[i]),int(tri[k]))))
            old=parent.setdefault(e,j)
            if old!=j:
                a.append(old);b.append(j)
    mat=coo_matrix((np.ones(len(a),dtype=np.uint8),(a,b)),shape=(faces,faces)).tocsr()
    n,labels=connected_components(mat,directed=False)
    return n,labels
