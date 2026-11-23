#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare two electrical materials on the same welded-cylinder solid.

Run beside tet4_static_welded_junction.py with FiberNet >= 4.20.0:
    python examples/tet4_dc_convection_welded_junction.py

Three 3 mm cylinders of radius 0.5 mm share one conforming Tet4 mesh.
This uses the existing voxel approximation of their union, not CAD geometry
or a mesh convergence study. Cell centroids in the central region identify
the second material. This is steady one-way coupling: the solved temperature
does not change the electrical conductivity.
"""
from collections import Counter
from itertools import combinations
from math import fsum

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_convection
from fibernet.sim.tet4_dc import solve_dc_tet4
from tet4_static_welded_junction import boundary, make_geometry


VOLTAGE_V, AMBIENT_K, CAP_K = 0.3, 299.95, 300.0
CONVECTION_W_M2_K, THERMAL_W_M_K = 100.0, 0.25
BODY, CORE = 7, 17


def run():
    grid, points_mm, tets, core, _, digest = make_geometry(intervals=12)
    left, right, exterior_count = boundary(grid, tets, right_x_index=12)
    counts = Counter(face for tet in tets for face in combinations(sorted(tet), 3))
    exterior = {face for face, count in counts.items() if count == 1}
    sides = np.array(sorted(exterior - set(left) - set(right)))
    if len(exterior) != exterior_count or not (np.any(core) and np.any(~core)):
        raise RuntimeError("expected a shared solid with both material regions")
    nids = np.arange(len(points_mm), dtype=np.uint64)*19 + np.uint64(2**53+1)
    cids = np.arange(len(tets), dtype=np.uint64)*23 + np.uint64(2**62+11)
    labels = np.where(core, CORE, BODY)
    materials = dict.fromkeys((BODY, CORE), {"young_modulus_pa": 1e9, "poisson_ratio": .30})
    mesh = Tet4Bundle(points_mm, tets, nids, cids, labels, materials, "mm")
    left_nodes, right_nodes = np.unique(left), np.unique(right)
    cap_nodes = np.r_[left_nodes, right_nodes]
    electrodes = {**dict.fromkeys(map(int, nids[left_nodes]), 0.0),
                  **dict.fromkeys(map(int, nids[right_nodes]), VOLTAGE_V)}
    prescribed = dict.fromkeys(electrodes, CAP_K)
    points_m = points_mm * 1e-3
    edges = points_m[tets[:, 1:]] - points_m[tets[:, :1]]
    vols = np.abs(np.linalg.det(edges)) / 6
    tri = points_m[sides]
    areas = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    results = {}
    for case, core_S_m in (("homogeneous", 2.0), ("lower_conductivity_junction", .5)):
        dc = solve_dc_tet4(mesh, conductivity_S_per_m_by_material={BODY: 2., CORE: core_S_m},
                           electrode_voltage_V_by_id=electrodes)
        hot = solve_steady_conduction_tet4_with_convection(
            mesh, conductivity_W_per_m_K_by_material=dict.fromkeys(materials, THERMAL_W_M_K),
            prescribed_K_by_id=prescribed,
            source_W_per_m3_by_cell=dict(zip(map(int, dc.cell_ids), dc.joule_density_W_per_m3)),
            faces_by_node_id=nids[sides], convection_W_per_m2_K=CONVECTION_W_M2_K,
            ambient_K=AMBIENT_K)
        np.testing.assert_array_equal(dc.node_ids, nids)
        np.testing.assert_array_equal(hot.node_ids, nids)
        np.testing.assert_array_equal(dc.cell_ids, cids)
        np.testing.assert_array_equal(hot.cell_ids, cids)
        # Recompute physical powers from mesh coordinates and solved fields.
        electrical = dc.voltage_V[tets]
        grad = np.linalg.solve(edges, (electrical[:, 1:] - electrical[:, :1])[..., None])[..., 0]
        joule = np.where(core, core_S_m, 2.) * np.sum(grad**2, axis=1) * vols
        conv = CONVECTION_W_M2_K * fsum(areas * np.mean(hot.temperature_K[sides] - AMBIENT_K, axis=1))
        terminals = VOLTAGE_V * fsum(dc.electrode_current_into_solid_A[right_nodes])
        caps = -fsum(hot.heat_into_solid_W[cap_nodes])
        np.testing.assert_allclose(joule, hot.integrated_source_W_by_cell, rtol=1e-9, atol=1e-18)
        np.testing.assert_allclose(fsum(joule), terminals, rtol=1e-10, atol=1e-18)
        np.testing.assert_allclose(conv, -hot.total_convective_heat_into_solid_W, rtol=1e-9, atol=1e-17)
        np.testing.assert_allclose(fsum(joule), conv + caps, rtol=1e-9, atol=1e-17)
        results[case] = (fsum(joule), conv, caps, min(hot.temperature_K), fsum(joule) - conv - caps)
    return (len(points_mm), len(tets), int(np.sum(core)), len(sides), digest), results


if __name__ == "__main__":
    geometry, runs = run()
    nodes, tets, core_count, side_count, digest = geometry
    print(f"{nodes} nodes | {tets} tets | {core_count} junction tets | {side_count} convection faces")
    print(f"Geometry and material-region digest: {digest}")
    print("Positive powers leave through the named boundary; negative caps out means heat enters.")
    for case, (joule, conv, caps, coolest, residual) in runs.items():
        print(f"{case}: Joule {joule*1e6:.4f} µW; convection out {conv*1e6:.4f} µW; "
              f"caps out {caps*1e6:.4f} µW; minimum {coolest:.5f} K; heat residual {residual:.2e} W")
