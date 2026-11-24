"""The welded-junction example replays the reference state before perturbing."""
from dataclasses import replace

import numpy as np
import pytest

from examples.tet4_junction_load_control import compare
from examples.tet4_static_welded_junction import boundary, make_geometry
from fibernet.io.tet4_meshio import Tet4Bundle


def test_junction_material_derivative_switches_sign_at_matching_initial_state():
    # A deterministic low-resolution welded solid exercises the same solve
    # path without optional Meshio/Gmsh. The public example uses a fitted CAD mesh.
    n = 8
    grid, points, cells, core, _, _ = make_geometry(n)
    left, right, _ = boundary(grid, cells, n)
    faces = {"cap_neg_x": np.asarray(left), "cap_pos_x": np.asarray(right)}
    mat = {"young_modulus_pa": 1e9, "poisson_ratio": .25}
    phase = {"bulk": 7, "junction": 19}
    mesh = Tet4Bundle(
        points, cells, np.arange(len(points), dtype=np.uint64)*13 + np.uint64(2**63),
        np.arange(len(cells), dtype=np.uint64)*17 + np.uint64(2**62),
        np.where(core, 19, 7).astype(np.uint64), {7: dict(mat), 19: dict(mat)}, "mm",
    )
    result = compare(mesh, faces, phase)
    assert result["junction_elements"] == int(core.sum()) > 0
    assert result["baseline_energy_J"] > 0
    force = result["fixed_force"]
    motion = result["fixed_displacement"]
    assert force["upper_energy_J"] < result["baseline_energy_J"] < force["lower_energy_J"]
    assert motion["lower_energy_J"] < result["baseline_energy_J"] < motion["upper_energy_J"]
    assert 0.001 < result["junction_energy_share"] < 0.25
    slope = "finite_difference_d_ln_U_d_ln_E"
    assert force[slope] == pytest.approx(-motion[slope], rel=2e-5, abs=0)

    # Reversing the element rows must not move the junction material.
    rows = np.arange(len(cells))[::-1]
    permuted = compare(replace(mesh, tetrahedra=mesh.tetrahedra[rows],
                              cell_ids=mesh.cell_ids[rows],
                              material_ids=mesh.material_ids[rows]), faces, phase)
    assert permuted["baseline_energy_J"] == pytest.approx(result["baseline_energy_J"], rel=1e-8, abs=0)
    assert permuted["junction_energy_share"] == pytest.approx(result["junction_energy_share"], rel=1e-8, abs=0)
