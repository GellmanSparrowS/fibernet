# SPDX-License-Identifier: MIT
"""Predict how small axial strain changes DC resistance in a fused 3D lattice.

Run: python -m examples.tet4_mechanoelectric_looped_network [--matched-force | --geometry-breakdown]
Three axial paths at different transverse positions share four crosslinks.
They are idealized square-section bars, meshed as one bonded solid. Electrical
conductivity is fixed: changes come from deformation, not piezoresistivity.
"""
import argparse
from dataclasses import dataclass, replace
from itertools import permutations, product
from math import fsum, isclose

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_dc import solve_dc_tet4


VOLTAGE_V = 1.0
STRAIN = 0.001
MATCHED_FORCE_N = 0.5
BASE_ID, STIFF_ID = 7, 19
MODES = ("uniform", "uniform_stiff", "stiff_half")


@dataclass(frozen=True)
class Result:
    load_reaction_N: float
    rest_resistance_ohm: float
    resistance_slope_per_strain: float


@dataclass(frozen=True)
class MatchedForceResult:
    axial_strain: float
    compressive_reaction_N: float
    tensile_reaction_N: float
    rest_resistance_ohm: float
    resistance_slope_per_N: float


@dataclass(frozen=True)
class GeometryBreakdown:
    stretch_along_current_per_N: float
    transverse_contraction_per_N: float

    @property
    def resistance_slope_per_N(self):
        return self.stretch_along_current_per_N + self.transverse_contraction_per_N


def _lattice(*, refinement=1):
    """Generate a quarter-mm voxel scaffold with shared nodes at intersections."""
    if type(refinement) is not int or not 1 <= refinement <= 4:
        raise ValueError("refinement must be an integer between 1 and 4")
    boxes = set()

    def fill(x, y, z):
        boxes.update(product(x, y, z))

    for y, z in ((1, 1), (5, 1), (1, 5)):
        fill(range(12), range(y, y + 2), range(z, z + 2))
    for x in (2, 8):
        fill(range(x, x + 2), range(1, 7), range(1, 3))
        fill(range(x, x + 2), range(1, 3), range(1, 7))

    if refinement > 1:
        boxes = {tuple(refinement*i+d for i,d in zip(box,shift))
                 for box in boxes for shift in product(range(refinement), repeat=3)}
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
    return np.asarray(vertices, dtype=float) * (.25/refinement), connectivity


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


def _fixture(bundle):
    x = bundle.points[:, 0]
    left, right = x == x.min(), x == x.max()
    left_ids, right_ids = [list(map(int, bundle.node_ids[mask])) for mask in (left, right)]
    fixed = {**dict.fromkeys(left_ids, ("x", "y", "z")),
             **dict.fromkeys(right_ids, ("y", "z"))}
    electrodes = {**dict.fromkeys(left_ids, 0.), **dict.fromkeys(right_ids, VOLTAGE_V)}
    return right, right_ids, fixed, electrodes, np.ptp(x) * 1e-3


def _scan(bundle, strain):
    points_mm = bundle.points
    connectivity = bundle.tetrahedra
    right, right_ids, fixed, electrodes, length_m = _fixture(bundle)
    conductivity = dict.fromkeys(bundle.materials, 1.)
    reaction = 0.
    deformed_mm = points_mm
    if strain:
        grip_motion_m = strain * length_m
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


def run_matched_force():
    """Compare equal total cap reactions under unchanged rigid-grip kinematics.

    In linear elasticity, first calibrate the axial stiffness using a
    prescribed cap motion, then re-solve both signed motions with that same
    grip. This is not uniform traction on an axially free cap.
    """
    points, tetrahedra = _lattice()
    results = {}
    for mode in MODES:
        mesh = _bundle(points, tetrahedra, mode)
        rest, _ = _scan(mesh, 0.)
        _, reference_force = _scan(mesh, STRAIN)
        if not reference_force >= MATCHED_FORCE_N:
            raise RuntimeError("Matched force lies outside the calibrated strain range")
        strain = STRAIN * (MATCHED_FORCE_N / reference_force)
        compress, force_minus = _scan(mesh, -strain)
        stretch, force_plus = _scan(mesh, strain)
        if not (isclose(-force_minus, MATCHED_FORCE_N, rel_tol=1e-9)
                and isclose(force_plus, MATCHED_FORCE_N, rel_tol=1e-9)):
            raise RuntimeError("Computed cap reaction did not match the target")
        slope = (stretch - compress) / ((force_plus - force_minus) * rest)
        results[mode] = MatchedForceResult(strain, force_minus, force_plus, rest, slope)
    return results


def run_geometry_breakdown():
    """Split the small-force response using the undeformed DC field.

    Unit directions refer to each cell's local current, not the loading
    axis. The reference-voltage energy weights sum to total Joule power.
    With fixed conductivity and electrode voltages, first-order resistance
    responds to stretch along the current minus the transverse area strain.
    """
    points, tetrahedra = _lattice()
    results = {}
    for mode in MODES:
        b = _bundle(points, tetrahedra, mode)
        right, right_ids, held, electrodes, length_m = _fixture(b)
        mechanical = solve_static_tet4(
            b, forces_by_id={}, fixed_dofs_by_id=held,
            prescribed_displacements_m_by_id={n: {"x": STRAIN * length_m} for n in right_ids},
            return_cell_fields=True,
        )
        electric = solve_dc_tet4(b, electrode_voltage_V_by_id=electrodes,
                                conductivity_S_per_m_by_material=dict.fromkeys(b.materials, 1.))
        cells = mechanical.cell_fields
        if not np.array_equal(cells.cell_ids, electric.cell_ids):
            raise RuntimeError("Mechanical and electrical cell IDs differ")
        force = fsum(mechanical.reaction_N[right, 0])
        power = electric.joule_power_W_by_cell
        total = fsum(power)
        if force <= 0. or total <= 0.:
            raise RuntimeError("Expected positive tensile reaction and reference power")

        J = electric.current_density_A_per_m2
        den = np.einsum("ij,ij->i", J, J)
        along = np.einsum("ei,eij,ej->e", J, cells.strain, J)
        along = np.divide(along, den, out=np.zeros_like(along), where=den > 0.)
        transverse = np.trace(cells.strain, axis1=1, axis2=2) - along
        results[mode] = GeometryBreakdown(
            fsum(power * along) / (total * force),
            -fsum(power * transverse) / (total * force),
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--matched-force", action="store_true",
                        help="compare the materials at the same signed total axial reaction")
    output.add_argument("--geometry-breakdown", action="store_true",
                        help="split the geometric force response along and across local current")
    args = parser.parse_args()
    if args.geometry_breakdown:
        for name, sample in run_geometry_breakdown().items():
            print(f"{name}: current-aligned stretch {sample.stretch_along_current_per_N:.6g} 1/N; "
                  f"transverse contraction {sample.transverse_contraction_per_N:.6g} 1/N; "
                  f"total {sample.resistance_slope_per_N:.6g} 1/N")
        return
    if args.matched_force:
        for name, sample in run_matched_force().items():
            print(f"{name}: forces {sample.compressive_reaction_N:.6g}, "
                  f"{sample.tensile_reaction_N:.6g} N; "
                  f"axial strain amplitude {sample.axial_strain:.4%}; "
                  f"R0 {sample.rest_resistance_ohm:.6g} ohm; "
                  f"(dR/R0)/dF {sample.resistance_slope_per_N:.6g} 1/N")
        return
    for name, sample in run().items():
        print(f"{name}: force {sample.load_reaction_N:.6g} N at {STRAIN:.2%} strain; "
              f"R0 {sample.rest_resistance_ohm:.6g} ohm; "
              f"(dR/R0)/dstrain {sample.resistance_slope_per_strain:.6g}")


if __name__ == "__main__":
    main()
