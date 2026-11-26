# SPDX-License-Identifier: MIT
"""Small one-way solid-to-pore calculation on a bonded three-path scaffold.

Run from the source checkout with Python 3.10+ and the pore-flow extra:
    python -m examples.tet4_solid_to_pore_flow
"""
import json
from dataclasses import replace
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_pore import solve_matched_pore_geometry
from .tet4_periodic_pore import _scaffold_voxels, _voxel_bundle, bundle


def _boundary_faces(t):
    face_columns = tuple(combinations(range(4), 3))
    faces = np.sort(t[:, face_columns].reshape(-1, 3), axis=1)
    unique, count = np.unique(faces, axis=0, return_counts=True)
    if np.max(count) > 2:
        raise ValueError("scaffold or pore has a nonmanifold face")
    return unique[count == 1]


def _extend_interface(solid, fluid, displacements_mm):
    x, t = fluid.points, fluid.tetrahedra
    rows = {tuple(row): k for k, row in enumerate(x)}
    solid_faces = _boundary_faces(solid.tetrahedra)
    on_solid = np.unique(solid_faces)
    try:
        s = np.array([rows[tuple(solid.points[row])] for row in on_solid])
    except KeyError as exc:
        raise ValueError("solid boundary vertex has no matching pore point") from exc
    solid_to_fluid = np.full(len(solid.points), -1, dtype=int)
    solid_to_fluid[on_solid] = s
    wetted = {tuple(f) for f in np.sort(solid_to_fluid[solid_faces], axis=1)}
    boundary_faces = _boundary_faces(t)
    boundary = np.unique(boundary_faces)
    outer = np.any((x == x.min(axis=0)) | (x == x.max(axis=0)), axis=1)
    inner_faces = boundary_faces[~np.any(outer[boundary_faces], axis=1)]
    if wetted != {tuple(f) for f in inner_faces}:
        raise ValueError("solid and fluid must share the complete inner triangulated interface")
    if np.any(outer[s]) or not np.array_equal(np.sort(np.r_[s, np.flatnonzero(outer)]), boundary):
        raise ValueError("the solid wall and fixed duct exterior must partition the boundary")
    offset = np.zeros_like(x)
    offset[s] = displacements_mm[on_solid]
    fixed = np.zeros(len(x), dtype=bool)
    fixed[boundary] = True
    if np.all(fixed):
        return offset
    edge_columns = tuple(combinations(range(4), 2))
    edge = np.unique(np.sort(t[:, edge_columns].reshape(-1, 2), axis=1), axis=0)
    a, b = edge.T
    weights = 1. / np.linalg.norm(x[a] - x[b], axis=1)
    adjacency = coo_matrix((np.tile(weights, 2), (np.r_[a, b], np.r_[b, a])),
                           shape=(len(x), len(x))).tocsr()
    laplace = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    system = laplace[~fixed][:, ~fixed]
    rhs = -(laplace @ offset)[~fixed]
    offset[~fixed] = spsolve(system.tocsc(), rhs)
    if (not np.isfinite(offset).all()
            or np.linalg.norm(system @ offset[~fixed] - rhs) > 1e-10 * np.linalg.norm(rhs)):
        raise ValueError("pore interior extension did not close")
    return offset


def demo():
    # Scope: linear elasticity drives a stationary periodic fluid problem.
    # The fluid does not feed forces back into the solid.
    solid = _voxel_bundle(_scaffold_voxels(), elastic_modulus=2e9)
    fluid = bundle()
    p = solid.points[:, 0]
    left, right = p == p.min(), p == p.max()
    fixed = {int(k): ("x", "y", "z") for k in solid.node_ids[left]}
    fixed.update({int(k): ("y", "z") for k in solid.node_ids[right]})
    stroke_m = 2e-6
    imposed = {int(k): {"x": stroke_m} for k in solid.node_ids[right]}
    motion = solve_static_tet4(solid, forces_by_id={}, fixed_dofs_by_id=fixed,
                               prescribed_displacements_m_by_id=imposed)
    if not np.array_equal(motion.node_ids, solid.node_ids):
        raise ValueError("solid result node IDs must match the source bundle")
    force_l = float(motion.reaction_N[left, 0].sum())
    force_r = float(motion.reaction_N[right, 0].sum())
    if not force_r > 0 or abs(force_l + force_r) > 1e-8 * force_r:
        raise ValueError("solid grip reaction is unbalanced")
    offsets_mm = _extend_interface(solid, fluid, motion.displacement_m * 1e3)
    out = solve_matched_pore_geometry(fluid, replace(fluid, points=fluid.points + offsets_mm),
                                     viscosity_Pa_s=.001, pressure_gradient_Pa_per_m=500.)
    return dict(solid_tetrahedra=len(solid.cell_ids), pore_tetrahedra=len(fluid.cell_ids),
                grip_stroke_m=stroke_m, grip_reaction_N=force_r,
                scope="small-strain solid to fixed-geometry stationary Stokes",
                baseline_flow_m3_s=out.baseline.axial_flow_m3_s,
                loaded_flow_m3_s=out.deformed.axial_flow_m3_s,
                relative_bulk_flow_change=out.relative_bulk_flow_change,
                relative_end_flow_change=list(out.relative_cap_flow_change),
                minimum_cell_volume_ratio=out.min_cell_volume_ratio,
                maximum_cell_volume_ratio=out.max_cell_volume_ratio)


if __name__ == "__main__":
    print(json.dumps(demo(), allow_nan=False))
