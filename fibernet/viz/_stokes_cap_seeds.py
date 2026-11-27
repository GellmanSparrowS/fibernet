"""Bounded streamline seeding over planar cap components of an affine tetra mesh."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator


def _groups(faces, canonical_order, budget):
    incidence, links = {}, []
    for i, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (a, c), (b, c)):
            prior = incidence.setdefault((min(u, v), max(u, v)), i)
            if prior != i:
                links.append((i, prior))
    i, j = np.asarray(links).T if links else ([], [])
    graph = csr_matrix((np.ones(len(i)), (i, j)), shape=(len(faces), len(faces)))
    num, labels = connected_components(graph, directed=False)
    if num > budget:
        raise ValueError("More cap components than seeds; reduce the selected cap complexity")
    ordered_labels = labels[canonical_order]
    first = np.unique(ordered_labels, return_index=True)[1]
    return [canonical_order[ordered_labels == ordered_labels[k]] for k in sorted(first)]


def _allocation(areas, budget):
    residual = (budget - len(areas)) * (areas / areas.sum())
    floor = np.floor(residual).astype(int)
    count = 1 + floor
    rank = np.argsort(-(residual - floor), kind="stable")
    count[rank[:budget - count.sum()]] += 1
    return count


def cap_component_seeds(linear, surface, xyz, tetra4, reverse, *, budget=49):
    """Return stable seed coordinates and the zero-based cap component of each."""
    if not 1 <= budget <= 64:
        raise ValueError("The cap seed budget must be 1 through 64")
    xlo, xhi = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    plane = xhi if reverse else xlo
    pts = vtk_to_numpy(surface.GetPoints().GetData())
    polys = surface.GetPolys()
    all_faces = vtk_to_numpy(polys.GetConnectivityArray()).reshape(-1, 3)
    on_plane = np.all(pts[all_faces, 0] == plane, axis=1)
    faces = all_faces[on_plane]
    if not len(faces):
        raise ValueError("The selected extreme x plane contains no planar boundary triangles")
    owners = vtk_to_numpy(surface.GetCellData().GetArray("fibernet_surface_owner"))[on_plane]
    tets = xyz[tetra4[owners]]
    off = tets[:, :, 0] != plane
    if np.any(off.sum(axis=1) != 1):
        raise ValueError("The planar cap lacks an adjacent affine tetrahedron")
    apex = tets[off]
    triangle = pts[faces].astype(float, copy=False)
    vertex_order = np.lexsort((triangle[:,:,2], triangle[:,:,1]), axis=1)
    triangle = np.take_along_axis(triangle, vertex_order[:,:,None], axis=1)
    uv = triangle[:,1:,1:] - triangle[:,:1,1:]
    areas = .5 * abs(uv[:,0,0]*uv[:,1,1] - uv[:,0,1]*uv[:,1,0])
    if np.any(areas <= 0) or not np.isfinite(areas).all():
        raise ValueError("The selected planar cap contains degenerate triangles")
    cent = triangle.mean(axis=1)
    yz = triangle[:,:,1:].reshape(len(triangle), 6)
    rank = np.lexsort((*yz[:,::-1].T, cent[:,2], cent[:,1]))
    groups = _groups(faces, rank, budget)
    totals = np.array([areas[g].sum() for g in groups])
    counts = _allocation(totals, budget)
    loc = vtkStaticCellLocator()
    loc.SetDataSet(linear)
    loc.BuildLocator()
    points, labels = [], []
    for i, (g, n) in enumerate(zip(groups, counts)):
        cumulative = np.cumsum(areas[g])
        pick = np.searchsorted(cumulative, ((np.arange(n) + .5) / n) * cumulative[-1])
        for j, f in enumerate(g[pick]):
            # Quasi-uniform seeds strictly inside the selected triangle. Move
            # toward the adjacent tetrahedral apex, not blindly along x.
            u = .18 + .62 * ((.5 + j*.6180339887498949) % 1)
            v = .18 + .62 * ((.5 + j*.4142135623730950) % 1)
            q = np.sqrt(u)
            sample = triangle[f].T @ [1-q, q*(1-v), q*v] if j else cent[f].copy()
            fraction = min(.08, (.003*(xhi-xlo)) / abs(apex[f, 0] - plane))
            seed = sample + fraction*(apex[f] - sample)
            if loc.FindCell(seed) != int(owners[f]):
                raise ValueError("Could not verify an interior seed in its cap tetrahedron")
            points.append(seed)
            labels.append(i)
    return np.asarray(points), np.asarray(labels), totals
