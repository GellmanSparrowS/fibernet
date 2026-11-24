"""Compare force and displacement control on the same tagged fused junction."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction

if __package__:
    from .tet4_fused_cylinder_thermomechanics import load_named_mesh
    from .tet4_stable_material_retag import mesh_identity, reassign_cells
else:
    from tet4_fused_cylinder_thermomechanics import load_named_mesh
    from tet4_stable_material_retag import mesh_identity, reassign_cells


def compare(bundle, faces, phase):
    """Perturb only junction E on a validated three-cylinder CAD fixture."""
    bulk = bundle.materials[phase["bulk"]]
    junction = bundle.materials[phase["junction"]]
    nominal = 3 * junction["young_modulus_pa"]
    mask = bundle.material_ids == phase["junction"]
    cell_ids = [int(i) for i in bundle.cell_ids[mask]]
    identity = mesh_identity(bundle)
    target = max(bundle.materials) + 1
    left, right = faces["cap_neg_x"], faces["cap_pos_x"]
    fixed = dict.fromkeys(map(int, bundle.node_ids[np.unique(left)]), ("x", "y", "z"))
    xyz = bundle.points[right] * 1e-3
    area = .5 * float(np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0],
                                             xyz[:, 2] - xyz[:, 0]), axis=1).sum())
    load = np.array([1., .27, -.41]) * 1e-3
    traction = nodal_forces_from_traction(bundle, bundle.node_ids[right],
                                        traction_Pa=load / area)

    def at_offset(offset, prescribed=None, fields=False):
        model = reassign_cells(bundle, expected_mesh_identity=identity, cell_ids=cell_ids,
                               material_id=target,
                               material={**junction, "young_modulus_pa": nominal * math.exp(offset)})
        return solve_static_tet4(model, forces_by_id=traction if prescribed is None else {},
                                 fixed_dofs_by_id=fixed, return_cell_fields=fields,
                                 prescribed_displacements_m_by_id=prescribed)

    base = at_offset(0, fields=True)
    lookup = {int(i): row for row, i in enumerate(bundle.node_ids)}
    # Replay all three cap components, including the nonuniform transverse response.
    cap = {int(i): dict(zip("xyz", map(float, base.displacement_m[lookup[int(i)]])))
           for i in bundle.node_ids[np.unique(right)]}
    replay = at_offset(0, cap)
    if not (np.allclose(replay.displacement_m, base.displacement_m, rtol=1e-8, atol=1e-16)
            and math.isclose(replay.strain_energy_J, base.strain_energy_J, rel_tol=1e-8)):
        raise RuntimeError("Prescribed cap displacements did not recover the baseline")
    fields = base.cell_fields
    if not np.array_equal(fields.cell_ids, bundle.cell_ids):
        raise RuntimeError("Element output order differs from the mesh cell IDs")
    energy = .5 * np.einsum("c,cij,cij->c", fields.volume_m3, fields.strain, fields.stress_Pa)
    if not math.isclose(float(energy.sum()), base.strain_energy_J, rel_tol=1e-8):
        raise RuntimeError("Cell energies do not sum to the reported energy")
    share = float(energy[mask].sum() / base.strain_energy_J)

    def slope(control, expected):
        h = 0.003
        lower = at_offset(-h, control).strain_energy_J
        upper = at_offset(h, control).strain_energy_J
        finite_difference = (math.log(upper) - math.log(lower)) / (2 * h)
        if not math.isclose(finite_difference, expected, rel_tol=2e-5, abs_tol=1e-10):
            raise RuntimeError("Finite-difference slope differs from the energy-share prediction")
        return {"lower_energy_J": lower, "upper_energy_J": upper,
                "finite_difference_d_ln_U_d_ln_E": finite_difference,
                "predicted_d_ln_U_d_ln_E": expected}

    return {"nodes": len(bundle.points), "elements": len(bundle.cell_ids),
            "junction_elements": len(cell_ids), "junction_modulus_Pa": nominal,
            "bulk_modulus_Pa": bulk["young_modulus_pa"], "total_applied_force_N": load.tolist(),
            "log_modulus_half_step": 0.003, "baseline_energy_J": base.strain_energy_J,
            "junction_energy_share": share, "fixed_force": slope(None, -share),
            "fixed_displacement": slope(cap, share)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bundle", type=Path)
    cli.add_argument("boundary_manifest", type=Path)
    args = cli.parse_args()
    mesh, faces, phase, digest = load_named_mesh(args.bundle, args.boundary_manifest,
                                                 compare_junction=True)
    data = compare(mesh, faces, phase)
    print(json.dumps({"bundle_sha256": digest, **data}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
