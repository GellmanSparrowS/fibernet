#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare heat expansion by material on one voxel-welded Tet4 junction."""
import json

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission
from fibernet.sim.tet4_thermoelastic import solve_prescribed_temperature_tet4

if __package__:
    from .tet4_static_welded_junction import boundary, make_geometry
else:
    from tet4_static_welded_junction import boundary, make_geometry


def compare_welded_junction():
    """Use one fixed mesh and the same volume-integrated expansion in both cases."""
    n = 12
    grid, points_mm, cells, core, _, mesh_hash = make_geometry(n)
    left, right, _ = boundary(grid, cells, n)
    grips = [np.unique(a) for a in (left, right)]
    ids = np.arange(len(grid), dtype=np.uint64)
    mids = np.where(core, 2, 1)
    young, nu, reference, uniform_alpha = 1e9, .30, 293., 12e-6
    temp = 313. + 10. / 3. * points_mm[:, 0]
    delta = temp[cells] - reference
    cell_pts_m = 1e-3 * points_mm[cells]
    volume = abs(np.linalg.det(cell_pts_m[:, 1:] - cell_pts_m[:, :1])) / 6
    delta_mean = delta.mean(axis=1)
    delta_variance = np.sum((delta - delta_mean[:, None])**2, axis=1) / 20
    if not np.all(volume > 0.) or not np.any(core) or np.all(core):
        raise ValueError("The comparison needs positive-volume cells and two regions.")
    budget = volume * delta_mean
    center = 2 * uniform_alpha
    outer = (uniform_alpha * budget.sum() - center * budget[core].sum()) / budget[~core].sum()
    mesh = Tet4Bundle(
        points_mm, cells, ids, np.arange(len(cells), dtype=np.uint64), mids,
        {i: {"young_modulus_pa": young, "poisson_ratio": nu} for i in (1, 2)}, "mm",
    )
    holds = {int(i): ("x", "y", "z") for i in np.concatenate(grips)}
    plan = inspect_static_tet4_admission(mesh, fixed_dofs_by_id=holds)
    if not plan.within_factor_budget:
        raise ValueError("The quick junction exceeds the static factor budget.")
    bulk = young / (3 * (1 - 2 * nu))
    result_rows = {}
    for case, alpha in (("uniform", {1: uniform_alpha, 2: uniform_alpha}),
                        ("larger_core_expansion", {1: float(outer), 2: center})):
        local_alpha = np.where(core, alpha[2], alpha[1])
        expanded = float(budget @ local_alpha)
        np.testing.assert_allclose(expanded, uniform_alpha * budget.sum(), rtol=2e-14)
        out = solve_prescribed_temperature_tet4(
            mesh, temperature_K_by_id=dict(zip(map(int, ids), map(float, temp))),
            stress_free_temperature_K=reference, alpha_per_K_by_material=alpha,
            fixed_dofs_by_id=holds, forces_by_id={},
        )
        np.testing.assert_array_equal(out.node_ids, ids)
        np.testing.assert_array_equal(out.material_ids, mids)
        total_force = out.reaction_N.sum(axis=0)
        total_moment = np.cross(points_mm * 1e-3, out.reaction_N).sum(axis=0)
        np.testing.assert_allclose(total_force, 0., rtol=0., atol=2e-10)
        np.testing.assert_allclose(total_moment, 0., rtol=0., atol=2e-13)
        zero_motion_energy = 4.5 * bulk * float(volume @ (
            local_alpha**2 * (delta_mean**2 + delta_variance)
        ))
        if not 0. < out.elastic_energy_J < zero_motion_energy:
            raise ValueError("Stored energy exceeds the admissible zero-motion trial.")
        if out.terminal_mechanical_pairing_J != 0.:
            raise ValueError("The fixed-cap test should have zero mechanical pairing.")
        result_rows[case] = {
            "alpha_per_K": alpha,
            "integrated_linear_thermal_eigenstrain_m3": expanded,
            "zero_motion_trial_energy_J": zero_motion_energy,
            "elastic_energy_J": out.elastic_energy_J,
            "core_energy_share": float(out.elastic_energy_J_by_cell[core].sum() / out.elastic_energy_J),
            "axial_reactions_N": [float(out.reaction_N[g, 0].sum()) for g in grips],
            "max_force_balance_error_N": float(np.max(abs(total_force))),
            "max_moment_balance_error_Nm": float(np.max(abs(total_moment))),
            "max_free_force_residual_N": out.physical_free_residual_inf_N,
        }
    return {
        "geometry": "voxel approximation of three joined circular cylinders",
        "mesh_digest": mesh_hash, "nodes": len(grid), "tetrahedra": len(cells),
        "core_tetrahedra": int(core.sum()),
        "free_dofs": plan.free_dofs,
        "factor_entries": plan.factor_entries,
        "cases": result_rows,
    }


if __name__ == "__main__":
    print(json.dumps(compare_welded_junction(), indent=2, allow_nan=False))
