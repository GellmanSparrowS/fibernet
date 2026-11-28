# SPDX-License-Identifier: MIT
"""Small one-way solid-to-pore calculation on a bonded three-path scaffold.

Run after installing FiberNet with the pore-flow extra on Python 3.10+:
    python -m fibernet.examples.tet4_solid_to_pore_flow
"""
import argparse
import json
from dataclasses import replace

import numpy as np

from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_pore import solve_matched_pore_geometry, solve_matched_pore_geometry_cases
from fibernet.sim.tet4_pore_interface import _faces as _boundary_faces, prepare_conforming_pore_wall
from .tet4_periodic_pore import _scaffold_voxels, _voxel_bundle, bundle


def _extend_interface(solid, fluid, displacements_mm):
    # Compatibility helper for this fixed mm example.
    wall = prepare_conforming_pore_wall(solid, fluid)
    return wall._offsets(displacements_mm, solid.node_ids, 1.)


def _mechanics(solid, stroke_m):
    p = solid.points[:, 0]
    left, right = p == p.min(), p == p.max()
    fixed = {int(k): ("x", "y", "z") for k in solid.node_ids[left]}
    fixed.update({int(k): ("y", "z") for k in solid.node_ids[right]})
    imposed = {int(k): {"x": stroke_m} for k in solid.node_ids[right]}
    motion = solve_static_tet4(solid, forces_by_id={}, fixed_dofs_by_id=fixed,
                               prescribed_displacements_m_by_id=imposed)
    if not np.array_equal(motion.node_ids, solid.node_ids):
        raise ValueError("solid result node IDs must match the source bundle")
    force_l = float(motion.reaction_N[left, 0].sum())
    force_r = float(motion.reaction_N[right, 0].sum())
    if not force_r * stroke_m > 0 or abs(force_l + force_r) > 1e-8 * abs(force_r):
        raise ValueError("solid grip reaction is unbalanced")
    return motion, force_r


def _reshaped_pore(solid, fluid, motion, *, wall=None):
    if wall is None:
        wall = prepare_conforming_pore_wall(solid, fluid)
    return wall.deform(motion.displacement_m, node_ids=motion.node_ids)


def _flow(solid, fluid, motion):
    return solve_matched_pore_geometry(fluid, _reshaped_pore(solid, fluid, motion),
                                       viscosity_Pa_s=.001, pressure_gradient_Pa_per_m=500.)


def demo():
    solid = _voxel_bundle(_scaffold_voxels(), elastic_modulus=2e9)
    fluid = bundle()
    stroke_m = 2e-6
    motion, force_r = _mechanics(solid, stroke_m)
    out = _flow(solid, fluid, motion)
    return dict(solid_tetrahedra=len(solid.cell_ids), pore_tetrahedra=len(fluid.cell_ids),
                grip_stroke_m=stroke_m, grip_reaction_N=force_r,
                scope="small-strain solid to fixed-geometry stationary Stokes",
                baseline_flow_m3_s=out.baseline.axial_flow_m3_s,
                loaded_flow_m3_s=out.deformed.axial_flow_m3_s,
                relative_bulk_flow_change=out.relative_bulk_flow_change,
                relative_end_flow_change=list(out.relative_cap_flow_change),
                minimum_cell_volume_ratio=out.min_cell_volume_ratio,
                maximum_cell_volume_ratio=out.max_cell_volume_ratio)


def compare_materials():
    """Compare the same scaffold and fluid at equal signed grip reaction."""
    soft = _voxel_bundle(_scaffold_voxels(), elastic_modulus=1e9)
    stiff = replace(soft, materials={1: dict(young_modulus_pa=2e9, poisson_ratio=.3)})
    fluid = bundle()
    calibration_m = 2e-6
    force_N = .15
    preparations = []
    wall = prepare_conforming_pore_wall(soft, fluid)
    for solid in (soft, stiff):
        _, calibration_N = _mechanics(solid, calibration_m)
        for sign in (-1, 1):
            target_N = sign * force_N
            stroke_m = calibration_m * target_N / calibration_N
            motion, reaction_N = _mechanics(solid, stroke_m)
            if not np.isclose(reaction_N, target_N, rtol=2e-10, atol=0.):
                raise ValueError("actual grip reaction does not match the target")
            preparations.append((stroke_m, reaction_N, _reshaped_pore(solid, fluid, motion, wall=wall)))
    flow = solve_matched_pore_geometry_cases(
        fluid, [item[2] for item in preparations],
        viscosity_Pa_s=.001, pressure_gradient_Pa_per_m=500.,
    )
    loaded = [
        dict(stroke_m=stroke_m, reaction_N=reaction_N, bulk_flow_m3_s=q.deformed.axial_flow_m3_s,
             relative_bulk_change=q.relative_bulk_flow_change,
             relative_end_change=list(q.relative_cap_flow_change))
        for (stroke_m, reaction_N, _), q in zip(preparations, flow)
    ]
    results = []
    for index, solid in enumerate((soft, stiff)):
        cases = loaded[2*index:2*index+2]
        minus, plus = cases
        slope = (plus["relative_bulk_change"] - minus["relative_bulk_change"]) / (2 * force_N)
        results.append(dict(young_modulus_Pa=solid.materials[1]["young_modulus_pa"],
                            cases=cases, relative_bulk_slope_per_N=slope))
    return dict(target_force_magnitude_N=force_N,
                scope="same scaffold, equal signed grip reaction, one-way stationary Stokes",
                materials=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-force", action="store_true",
                        help="compare two elastic moduli at the same signed grip forces")
    args = parser.parse_args()
    print(json.dumps(compare_materials() if args.matched_force else demo(), allow_nan=False))


if __name__ == "__main__":
    main()
