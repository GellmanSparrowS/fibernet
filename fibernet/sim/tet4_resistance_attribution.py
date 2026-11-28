# SPDX-License-Identifier: MIT
"""Separate shape and scalar conductivity changes in two-terminal DC."""
from dataclasses import dataclass, replace
from math import fsum, isfinite, log, log1p

import numpy as np

from ..io.tet4_meshio import Tet4Bundle, _integers, _points, _validate
from . import tet4
from .tet4_dc import _table, solve_dc_tet4

__all__ = ["Tet4ResistanceAttribution", "compare_tet4_resistance"]


@dataclass(frozen=True)
class Tet4ResistanceAttribution:
    """Four terminal resistances and an additive log-resistance attribution."""

    reference_ohm: float
    shape_only_ohm: float
    material_only_ohm: float
    combined_ohm: float
    shape_log_ratio: float
    material_log_ratio: float
    interaction_log_ratio: float
    combined_log_ratio: float


def _log_ratio(value, base):
    difference = value - base
    if abs(difference) <= .5 * base:
        return log1p(difference / base)
    return log(value) - log(base)


def _orientations(points, cells):
    for start in range(0, len(cells), 4096):
        c = cells[start:start + 4096]
        edges = points[c[:, 1:]] - points[c[:, :1]]
        scale = np.max(np.abs(edges), axis=(1, 2))
        if np.any(scale == 0.):
            raise ValueError("DC geometry contains a collapsed cell")
        sign = np.linalg.slogdet(edges / scale[:, None, None])[0]
        if not np.all((sign == 1.) | (sign == -1.)):
            raise ValueError("DC geometry contains a collapsed cell")
        yield sign


def compare_tet4_resistance(
    bundle, displacement_m, *, node_ids, reference_conductivity_S_per_m_by_material,
    changed_conductivity_S_per_m_by_material, electrode_voltage_V_by_id, solver="banded",
):
    """Compare DC on one Tet4 topology across shape and conductivity changes.

    Displacements are in metres and may be supplied in any order with matching
    stable node IDs. Each conductivity table supplies a positive isotropic
    S/m value for every material. Exactly two electrode voltages are required.
    The geometry and material input can come from a mechanics solver and a
    constitutive law; neither is inferred or solved here. The interaction is
    the finite joint log change minus the separate shape and material changes.
    Contact topology and electrode membership remain fixed. The orientation
    check does not certify absence of intersections between different cells.
    """
    if not isinstance(bundle, Tet4Bundle):
        raise TypeError("expected a Tet4Bundle")
    if not isinstance(solver, str) or solver not in ("banded", "cg"):
        raise ValueError("DC solver must be banded or cg")
    n = tet4._row_count(bundle.points, "points")
    e = tet4._row_count(bundle.tetrahedra, "tetrahedra")
    max_n, max_e = (1024, 8000) if solver == "banded" else (60_000, 200_000)
    if not (4 <= n <= max_n and 1 <= e <= max_e):
        raise ValueError("DC attribution mesh limits")
    mesh = _validate(bundle.points, bundle.tetrahedra, bundle.node_ids,
                     bundle.cell_ids, bundle.material_ids, bundle.materials, bundle.length_unit)
    ids = _integers(node_ids, (n,), "displacement node_ids", unique=True).astype(np.uint64)
    # Validated IDs are nonnegative. Keep both lookup operands integral:
    # mixing uint64 with int64 promotes some NumPy comparisons to binary64.
    mesh_ids = mesh.node_ids.astype(np.uint64, copy=False)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    if not np.array_equal(sorted_ids, np.sort(mesh_ids)):
        raise ValueError("displacement node IDs must match the Tet4 mesh")
    if tet4._row_count(displacement_m, "displacement_m") != n:
        raise ValueError("displacement_m must contain one row per node")
    u = _points(displacement_m)[order[np.searchsorted(sorted_ids, mesh_ids)]]
    ref_sig = _table(reference_conductivity_S_per_m_by_material, set(mesh.materials),
                     1e-6, 1e6, "reference conductivity", True)
    new_sig = _table(changed_conductivity_S_per_m_by_material, set(mesh.materials),
                     1e-6, 1e6, "changed conductivity", True)
    electrodes = _table(electrode_voltage_V_by_id, set(map(int, mesh.node_ids)),
                        -100., 100., "electrode voltage")
    levels = sorted(set(electrodes.values()))
    if len(levels) != 2 or levels[1] - levels[0] < 1e-6:
        raise ValueError("DC attribution needs exactly two distinct electrode voltages")
    span = levels[1] - levels[0]
    high = np.asarray([electrodes.get(int(key)) == levels[1] for key in mesh.node_ids])
    scale = tet4._LENGTH_TO_M[mesh.length_unit]
    with np.errstate(over="ignore", invalid="ignore"):
        moved = mesh.points + u / scale
    if not np.isfinite(moved).all() or np.max(np.abs(moved * scale), initial=0.) > 1.:
        raise ValueError("deformed DC geometry exceeds the finite coordinate limit")
    shape_changed = not np.array_equal(moved, mesh.points)
    material_changed = ref_sig != new_sig
    if shape_changed:
        for a, b in zip(_orientations(mesh.points, mesh.tetrahedra),
                        _orientations(moved, mesh.tetrahedra)):
            if not np.array_equal(a, b):
                raise ValueError("deformation reverses a tetrahedron")
    deformed = replace(mesh, points=moved) if shape_changed else mesh

    def resistance(m, conductivity):
        dc = solve_dc_tet4(m, conductivity_S_per_m_by_material=conductivity,
                           electrode_voltage_V_by_id=electrodes, solver=solver)
        if not np.array_equal(dc.node_ids, mesh.node_ids):
            raise ValueError("DC output must preserve the mesh node IDs")
        current = fsum(dc.electrode_current_into_solid_A[high])
        if not current > 0.:
            raise ValueError("the two terminal levels do not carry a positive current")
        value = span / current
        if not isfinite(value) or value <= 0.:
            raise ValueError("terminal resistance is outside the finite range")
        return value

    r0 = resistance(mesh, ref_sig)
    rg = resistance(deformed, ref_sig) if shape_changed else r0
    rm = resistance(mesh, new_sig) if material_changed else r0
    rj = resistance(deformed, new_sig) if shape_changed and material_changed else (
        rg if shape_changed else rm
    )
    a, b, c = (_log_ratio(v, r0) for v in (rg, rm, rj))
    return Tet4ResistanceAttribution(r0, rg, rm, rj, a, b, fsum((c, -a, -b)), c)
