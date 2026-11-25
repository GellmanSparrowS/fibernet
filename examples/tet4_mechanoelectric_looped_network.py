# SPDX-License-Identifier: MIT
"""Predict how small axial strain changes DC resistance in a fused 3D lattice.

Run: python -m examples.tet4_mechanoelectric_looped_network
Three axial paths at different transverse positions share four crosslinks.
They are idealized square-section bars, meshed as one bonded solid. Electrical
conductivity is fixed: changes come from deformation, not piezoresistivity.
"""
from dataclasses import dataclass, replace
from itertools import permutations, product
from math import fsum

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_dc import solve_dc_tet4


VOLTAGE_V = 1.0
STRAIN = 0.001
BASE_ID, STIFF_ID = 7, 19
MODES = ("uniform", "uniform_stiff", "stiff_half")


@dataclass(frozen=True)
class Result:
    load_reaction_N: float
    rest_resistance_ohm: float
    resistance_slope_per_strain: float


def _lattice():
    """Generate a quarter-mm voxel scaffold with shared nodes at intersections."""
    boxes = set()

    def fill(x, y, z):
        boxes.update(product(x, y, z))

    for y, z in ((1, 1), (5, 1), (1, 5)):
        fill(range(12), range(y, y + 2), range(z, z + 2))
    for x in (2, 8):
        fill(range(x, x + 2), range(1, 7), range(1, 3))
        fill(range(x, x + 2), range(1, 3), range(1, 7))

    shifts = list(product(range(2), repeat=3))
    vertices = sorted({tuple(i + j for i, j in zip(box, step))
                       for box in boxes for step in shifts})
    rows = {v: i for i, v in enumerate(vertices)}
    axes = np.eye(3, dtype=int)
    patterns = [np.vstack((np.zeros(3, dtype=int),
                           np.cumsum(axes[list(order)], axis=0)))
                for order in permutations(range(3))]
    connectivity = np.asarray([[rows[tuple(box + v)] for v in pattern]
                                for box in sorted(boxes) for pattern in patterns])
    return np.asarray(vertices, dtype=float) * .25, connectivity


def _bundle(points_mm, tetrahedra, mode):
    modulus = 2e9 if mode == "uniform_stiff" else 1e9
    materials = {BASE_ID: dict(young_modulus_pa=modulus, poisson_ratio=.3)}
    labels = np.full(len(tetrahedra), BASE_ID, dtype=np.uint64)
    if mode == "stiff_half":
        # The plane x = 1.5 mm is already a shared mesh interface.
        on_stiff_side = np.all(points_mm[tetrahedra, 0] >= 1.5, axis=1)
        labels[on_stiff_side] = STIFF_ID
        materials[STIFF_ID] = dict(young_modulus_pa=2e9, poisson_ratio=.3)
    return Tet4Bundle(points_mm, tetrahedra,
                      np.arange(len(points_mm), dtype=np.uint64) * 7 + 100,
                      np.arange(len(tetrahedra), dtype=np.uint64) * 11 + 200,
                      labels, materials, "mm")


def _scan(bundle, strain):
    points_mm = bundle.points
    connectivity = bundle.tetrahedra
    x = points_mm[:, 0]
    left, right = x == x.min(), x == x.max()
    left_ids, right_ids = [list(map(int, bundle.node_ids[mask])) for mask in (left, right)]
    fixed = {**dict.fromkeys(left_ids, ("x", "y", "z")),
             **dict.fromkeys(right_ids, ("y", "z"))}
    electrodes = {**dict.fromkeys(left_ids, 0.), **dict.fromkeys(right_ids, VOLTAGE_V)}
    conductivity = dict.fromkeys(bundle.materials, 1.)
    reaction = 0.
    deformed_mm = points_mm
    if strain:
        grip_motion_m = strain * np.ptp(x) * 1e-3
        imposed = {nid: {"x": grip_motion_m} for nid in right_ids}
        mechanics = solve_static_tet4(bundle, forces_by_id={}, fixed_dofs_by_id=fixed,
                                     prescribed_displacements_m_by_id=imposed)
        reaction = fsum(mechanics.reaction_N[right, 0])
        deformed_mm = points_mm + 1e3 * mechanics.displacement_m
        old = points_mm[connectivity]
        new = deformed_mm[connectivity]
        same_orientation = (np.linalg.det(old[:, 1:] - old[:, :1]) *
                            np.linalg.det(new[:, 1:] - new[:, :1]) > 0.)
        if not np.all(same_orientation):
            raise RuntimeError("Deformation inverted or collapsed a tetrahedron")
    current = solve_dc_tet4(replace(bundle, points=deformed_mm),
                           conductivity_S_per_m_by_material=conductivity,
                           electrode_voltage_V_by_id=electrodes)
    into_high = fsum(current.electrode_current_into_solid_A[right])
    if into_high <= 0.:
        raise RuntimeError("Expected positive current at the high electrode")
    return VOLTAGE_V / into_high, reaction


def run():
    """Compare mechanical stiffness at fixed structure and conductivity."""
    points, tetrahedra = _lattice()
    results = {}
    for mode in MODES:
        mesh = _bundle(points, tetrahedra, mode)
        rest, _ = _scan(mesh, 0.)
        compress, _ = _scan(mesh, -STRAIN)
        stretch, reaction = _scan(mesh, STRAIN)
        results[mode] = Result(reaction, rest, (stretch - compress) / (2. * STRAIN * rest))
    return results


if __name__ == "__main__":
    for name, sample in run().items():
        print(f"{name}: force {sample.load_reaction_N:.6g} N at {STRAIN:.2%} strain; "
              f"R0 {sample.rest_resistance_ohm:.6g} ohm; "
              f"(dR/R0)/dstrain {sample.resistance_slope_per_strain:.6g}")
