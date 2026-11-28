# SPDX-License-Identifier: MIT
"""Reusable harmonic wall-motion transfer for a conforming Tet4 pore."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from ..io.tet4_meshio import Tet4Bundle, _integers, _points
from . import tet4
from .tet4_pore import _MIB, _matched_volume_range, _validated_matched_mesh

__all__ = ['PoreWallMotion', 'prepare_conforming_pore_wall']


def _faces(cells):
    if len(np.unique(np.sort(cells, axis=1), axis=0)) != len(cells):
        raise ValueError('pore wall transfer rejects duplicate tetrahedra')
    cols = tuple(combinations(range(4), 3))
    faces = np.sort(cells[:, cols].reshape(-1, 3), axis=1)
    unique, counts = np.unique(faces, return_counts=True, axis=0)
    if np.any(counts > 2):
        raise ValueError('solid or pore has a nonmanifold face')
    boundary = unique[counts == 1]
    if not len(boundary):
        raise ValueError('pore wall transfer requires a nonempty mesh boundary')
    return boundary


def _mesh(bundle):
    return tet4._validate(bundle.points, bundle.tetrahedra, bundle.node_ids,
                          bundle.cell_ids, bundle.material_ids, bundle.materials,
                          bundle.length_unit)


class PoreWallMotion:
    """Prepared wall correspondence and harmonic extension, created by
    :func:`prepare_conforming_pore_wall`.

    Input displacement is provided in meters, with an explicit stable solid
    node ID for each row. The returned pore bundle retains its own original
    unit, IDs, connectivity and labels. The outer box stays fixed.
    """

    def __init__(self, reference, solid_ids, solid_rows, pore_rows, free, system, coupling, factor):
        self._reference = reference
        self._solid_ids = solid_ids
        self._solid_rows = solid_rows
        self._pore_rows = pore_rows
        self._free = free
        self._system = system
        self._coupling = coupling
        self._factor = factor

    def _offsets(self, displacement, node_ids, scale):
        count = len(self._solid_ids)
        ids = _integers(node_ids, (count,), 'solid displacement node_ids', unique=True).astype(np.uint64)
        order = np.argsort(ids)
        if not np.array_equal(np.sort(self._solid_ids), ids[order]):
            raise ValueError('solid displacement node IDs must match the prepared wall')
        rows = order[np.searchsorted(ids[order], self._solid_ids)]
        if tet4._row_count(displacement, 'solid displacement_m') != count:
            raise ValueError('solid displacement_m must have one three-component row per solid node')
        u = _points(displacement)
        if u.shape != (count, 3):
            raise ValueError('solid displacement_m must have shape (N, 3)')
        wall = u[rows[self._solid_rows]]
        ref = self._reference
        d = np.zeros_like(ref.points)
        with np.errstate(over='ignore', invalid='ignore'):
            d[self._pore_rows] = wall * scale
        if not np.isfinite(d).all():
            raise ValueError('solid wall motion is not finite in the pore length unit')
        if self._factor is not None:
            rhs = -(self._coupling @ d[self._pore_rows])
            if not np.isfinite(rhs).all():
                raise ValueError('harmonic pore right-hand side must be finite')
            solved = self._factor.solve(rhs)
            residual = np.linalg.norm(self._system @ solved - rhs, axis=0)
            if (not np.isfinite(solved).all() or not np.isfinite(residual).all()
                    or np.any(residual > 1e-10 * np.linalg.norm(rhs, axis=0))):
                raise ValueError('harmonic pore interior motion did not close')
            d[self._free] = solved
        return d

    def deform(self, displacement_m, *, node_ids):
        """Return a detached pore mesh after applying one displacement field.

        The prepared correspondence and sparse factorization are reused.
        This transfers wall motion to a fixed fluid geometry, not fluid
        tractions or a moving-mesh fluid time integration.
        """
        ref = self._reference
        d = self._offsets(displacement_m, node_ids, 1. / tet4._LENGTH_TO_M[ref.length_unit])
        with np.errstate(over='ignore', invalid='ignore'):
            points = ref.points + d
        if not np.isfinite(points).all():
            raise ValueError('deformed pore points must be finite')
        moved = replace(ref, points=points, tetrahedra=ref.tetrahedra.copy(),
                        node_ids=ref.node_ids.copy(), cell_ids=ref.cell_ids.copy(),
                        material_ids=ref.material_ids.copy(), materials=deepcopy(ref.materials))
        _matched_volume_range(ref, moved)
        return moved


def prepare_conforming_pore_wall(solid, pore):
    """Prepare one conforming solid/pore pair for repeated wall movements.

    Both Tet4 bundles must use the same length unit. The enclosed solid and
    the pore must share their entire inner triangulated surface exactly. All
    remaining pore boundary vertices must be on the fixed axis-aligned box.
    Distinct solid/pore node IDs are supported. Nonconforming boundaries and
    solids meeting the outer box are outside this transfer model. Matching
    boundary faces is not a certificate against overlapping cells.
    """
    if not isinstance(solid, Tet4Bundle) or not isinstance(pore, Tet4Bundle):
        raise TypeError('solid and pore must be Tet4Bundle objects')
    if solid.length_unit != pore.length_unit:
        raise ValueError('solid and pore must use the same length unit')
    if not (4 <= tet4._row_count(solid.points, 'solid points') <= 1800
            and 1 <= tet4._row_count(solid.tetrahedra, 'solid tetrahedra') <= 8000):
        raise ValueError('solid point/cell admission limit for pore wall transfer')
    s = _mesh(solid)
    p = _validated_matched_mesh(pore, 512*_MIB)
    x, cells = p.points, p.tetrahedra
    lookup = {tuple(xyz): row for row, xyz in enumerate(x)}
    if len(lookup) != len(x) or len(np.unique(s.points, axis=0)) != len(s.points):
        raise ValueError('pore wall transfer requires distinct coordinates within each mesh')
    if len(np.unique(cells)) != len(x):
        raise ValueError('pore wall transfer requires every pore point to belong to a cell')
    solid_faces = _faces(s.tetrahedra)
    solid_rows = np.unique(solid_faces)
    try:
        pore_rows = np.asarray([lookup[tuple(s.points[row])] for row in solid_rows])
    except KeyError as exc:
        raise ValueError('solid boundary vertex has no matching pore point') from exc
    solid_to_pore = np.full(len(s.points), -1, dtype=int)
    solid_to_pore[solid_rows] = pore_rows
    boundary = _faces(cells)
    outer = np.any((x == x.min(axis=0)) | (x == x.max(axis=0)), axis=1)
    wetted = set(map(tuple, np.sort(solid_to_pore[solid_faces], axis=1)))
    inner_faces = boundary[~np.any(outer[boundary], axis=1)]
    if wetted != set(map(tuple, inner_faces)):
        raise ValueError('solid and pore must share the complete inner triangulated interface')
    fixed_rows = np.unique(boundary)
    if np.any(outer[pore_rows]) or not np.array_equal(np.sort(np.r_[pore_rows, np.flatnonzero(outer)]), fixed_rows):
        raise ValueError('solid wall and fixed outer box must partition the pore boundary')
    boundary_on_box = boundary[np.any(outer[boundary], axis=1)]
    if (not len(boundary_on_box)
            or np.any(~np.any(np.all(x[boundary_on_box] == x.min(axis=0), axis=1)
                             | np.all(x[boundary_on_box] == x.max(axis=0), axis=1), axis=1))):
        raise ValueError('remaining pore boundary faces must lie on the fixed axis-aligned box')
    fixed = np.zeros(len(x), dtype=bool)
    fixed[fixed_rows] = True
    free = np.flatnonzero(~fixed)
    if not len(free):
        return PoreWallMotion(p, s.node_ids.astype(np.uint64), solid_rows, pore_rows, free, None, None, None)
    cols = tuple(combinations(range(4), 2))
    edges = np.unique(np.sort(cells[:, cols].reshape(-1, 2), axis=1), axis=0)
    a, b = edges.T
    lengths = np.linalg.norm(x[a] - x[b], axis=1)
    if np.any(lengths == 0) or not np.isfinite(lengths).all():
        raise ValueError('harmonic pore edge lengths must be finite and positive')
    weights = 1. / lengths
    if not np.isfinite(weights).all():
        raise ValueError('harmonic pore edge weights must be finite')
    adjacency = coo_matrix((np.tile(weights, 2), (np.r_[a, b], np.r_[b, a])),
                           shape=(len(x), len(x))).tocsr()
    laplace = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    system = laplace[free][:, free].tocsr()
    try:
        factor = splu(system.tocsc())
    except RuntimeError as exc:
        raise ValueError('pore interior must be connected to the prepared boundary') from exc
    coupling = laplace[free][:, pore_rows].tocsr()
    return PoreWallMotion(p, s.node_ids.astype(np.uint64), solid_rows, pore_rows, free, system, coupling, factor)
