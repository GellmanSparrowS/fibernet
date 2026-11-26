# SPDX-License-Identifier: MIT
"""Rank equal-area load patches by the initial resistance response.

Run from a checkout with the solid-iterative extra:
python -m examples.tet4_mechanoelectric_patch_screen
Add --verify to recompute representative patches with forward solutions.
"""
import argparse
from collections import Counter
from dataclasses import replace
from itertools import combinations
import json
from math import fsum, isclose

import numpy as np

from fibernet.sim import tet4
from fibernet.sim.tet4_dc import solve_dc_tet4
from .tet4_mechanoelectric_looped_network import _bundle, _lattice


def _geometric_objective_gradient(mesh, dc):
    """Reference gradient of log resistance in inverse metres, not nodal forces."""
    if not (np.array_equal(mesh.node_ids, dc.node_ids) and
            np.array_equal(mesh.cell_ids, dc.cell_ids)):
        raise ValueError("The electrical fields must use the reference mesh IDs")
    power = dc.joule_power_W_by_cell
    total = fsum(power)
    if total <= 0. or not np.isfinite(total) or np.any(power < 0.):
        raise ValueError("Expected finite positive reference power")
    points = mesh.points * 1e-3  # The synthetic fixture is in millimetres.
    gradient = np.zeros_like(points)
    for first in range(0, len(mesh.tetrahedra), 4096):
        window = slice(first, first + 4096)
        c = mesh.tetrahedra[window]
        offsets = points[c[:, 1:]] - points[c[:, :1]]
        h = np.max(abs(offsets), axis=(1, 2))
        if np.any(h < 1e-12):
            raise ValueError("Outside the example's element scale")
        a = offsets / h[:, None, None]
        if np.any(np.linalg.cond(a) > 1e6):
            raise ValueError("Outside the example's element shape range")
        inverse = np.linalg.inv(a) / h[:, None, None]
        shape_gradient = np.concatenate((-inverse.sum(axis=2, keepdims=True), inverse), axis=2)
        current = dc.current_density_A_per_m2[window]
        current_scale = np.max(abs(current), axis=1, keepdims=True)
        scaled = np.divide(current, current_scale, out=np.zeros_like(current), where=current_scale > 0.)
        norm = np.linalg.norm(scaled, axis=1, keepdims=True)
        n = np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0.)
        if np.any(power[window][norm[:, 0] == 0.] != 0.):
            raise ValueError("Zero current must carry zero Joule power")
        local = (power[window] / total)[:, None, None] * (
            (2. * n[:, :, None] * n[:, None, :] - np.eye(3)) @ shape_gradient)
        np.add.at(gradient, c, local.transpose(0, 2, 1))
    if not np.isfinite(gradient).all():
        raise ValueError("Nonfinite geometric objective gradient")
    return gradient


def _adjoint(mesh, fixed, gradient):
    # Same checked numerical machinery as the iterative static solver.
    # Keep the geometric objective out of the public physical-force mapping.
    owned, parents = tet4._owned_amg_mesh(mesh)
    plan, k, factor = tet4._assemble_amg_system(owned, fixed, parents)
    scale, normal, norm, _, limit, solve = tet4._prepare_fixed_solve(k, plan, iterative_factor=factor)
    g = gradient.ravel()[plan.free]
    max_g = np.max(abs(g), initial=0.)
    result = np.zeros(gradient.size)
    if max_g:
        rhs = g / max_g
        scaled = solve(rhs)
        error = np.max(abs(normal @ scaled - rhs)) / (norm * np.max(abs(scaled)) + 1.)
        if not 0. <= error <= limit:
            raise ValueError("The adjoint failed the normalized residual check")
        result[plan.free] = (scaled / scale) * max_g
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reference adjoint")
    return result.reshape(gradient.shape)


def _equal_patches(points, cells, reference, direction, tip, adjoint, refinement):
    counts = Counter(face for tet in cells for face in combinations(sorted(map(int, tet)), 3))
    faces = np.asarray([f for f, count in counts.items() if count == 1 and all(tip[list(f)])])
    centroids = points[faces].mean(1)
    output = []
    for limb, (base_y, base_z) in enumerate(((.25, .25), (1.25, .25), (.25, 1.25)), start=1):
        for quadrant, (dy, dz) in enumerate(((0., 0.), (.25, 0.), (0., .25), (.25, .25)), start=1):
            y, z = base_y + dy, base_z + dz
            mask = ((centroids[:, 1] > y) & (centroids[:, 1] < y + .25) &
                    (centroids[:, 2] > z) & (centroids[:, 2] < z + .25))
            triangles = faces[mask]
            if len(triangles) != 2 * refinement**2:
                raise ValueError("Incomplete exterior square in the reference fixture")
            xyz = points[triangles] * 1e-3
            area = .5 * np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)
            total = fsum(area)
            if not isclose(total, 6.25e-8, rel_tol=1e-12):
                raise ValueError("Reference patches must all be 0.0625 mm²")
            loc = (area[:, None] * xyz.mean(axis=1)).sum(axis=0) / total
            score = fsum((area / total) * (adjoint[triangles] @ direction).mean(axis=1))
            report = dict(patch=f"limb-{limb}-quarter-{quadrant}", centroid_m=loc.tolist(),
                          area_mm2=total*1e6,
                          moment_per_N_m=np.cross(loc-reference, direction).tolist(),
                          relative_resistance_slope_per_N=score)
            output.append((report, triangles, area / total))
    return output


def _forward_check(mesh, patch, fixed_ids, direction, tip, dc_opt, r0):
    # Equal resultant; force is distributed by area, not by node count.
    sample, triangles, weights = patch
    amplitude_N = .004
    nodal = np.zeros(len(mesh.points))
    np.add.at(nodal, triangles, np.repeat((weights/3.)[:, None], 3, axis=1))
    np.testing.assert_allclose(fsum(nodal), 1., rtol=1e-13)
    load = {int(mesh.node_ids[i]): tuple((amplitude_N * nodal[i]) * direction)
            for i in np.flatnonzero(nodal)}
    out = tet4.solve_static_tet4(mesh, forces_by_id=load, fixed_dofs_by_id=fixed_ids, solver="amg-cg")
    np.testing.assert_allclose(out.reaction_N.sum(0), -amplitude_N*direction, atol=2e-11)
    points = mesh.points
    ref = points[mesh.tetrahedra]
    signs = np.sign(np.linalg.det(ref[:, 1:] - ref[:, :1]))
    resistance = []
    for sign in (-1., 1.):
        changed = points + 1000. * sign * out.displacement_m
        xyz = changed[mesh.tetrahedra]
        if not np.all((signs != 0.) & (np.sign(np.linalg.det(xyz[:, 1:] - xyz[:, :1])) == signs)):
            raise ValueError("A finite validation sample inverted or collapsed a cell")
        dc = solve_dc_tet4(replace(mesh, points=changed), **dc_opt)
        current = fsum(dc.electrode_current_into_solid_A[tip])
        if current <= 0.:
            raise ValueError("Expected positive current through the high electrode")
        resistance.append(1. / current)
    secant = (resistance[1] - resistance[0]) / (2. * amplitude_N * r0)
    np.testing.assert_allclose(secant, sample["relative_resistance_slope_per_N"], rtol=3e-4, atol=1e-8)
    return secant


def run(*, refinement=3, verify=False):
    points, cells = _lattice(refinement=refinement)
    mesh = _bundle(points, cells, "stiff_half")
    left = points[:, 0] == points[:, 0].min()
    tip = points[:, 0] == points[:, 0].max()
    fixed = np.broadcast_to(left[:, None], points.shape)
    fixed_ids = dict.fromkeys(map(int, mesh.node_ids[left]), ("x", "y", "z"))
    electrodes = {**dict.fromkeys(mesh.node_ids[left], -.2), **dict.fromkeys(mesh.node_ids[tip], .8)}
    low, high = sorted(mesh.materials)
    dc_kwargs = dict(electrode_voltage_V_by_id=electrodes,
                     conductivity_S_per_m_by_material={low: 1., high: 100.}, solver="cg")
    dc = solve_dc_tet4(mesh, **dc_kwargs)
    current = fsum(dc.electrode_current_into_solid_A[tip])
    if current <= 0. or not isclose(current, dc.total_joule_power_W, rel_tol=3e-8):
        raise ValueError("Reference current and Joule power disagree")
    r0 = 1. / current
    g = _geometric_objective_gradient(mesh, dc)
    adj = _adjoint(mesh, fixed, g)
    direction = np.array([.6, -.2, .77])
    direction /= np.linalg.norm(direction)
    origin = points[left].mean(0) * 1e-3
    patches = _equal_patches(points, cells, origin, direction, tip, adj, refinement)
    ordered = sorted(patches, key=lambda patch: patch[0]["relative_resistance_slope_per_N"], reverse=True)
    if verify:
        for index in (0, len(ordered)-1, len(ordered)//2):
            item = ordered[index]
            item[0]["centered_finite_check_per_N"] = _forward_check(
                mesh, item, fixed_ids, direction, tip, dc_kwargs, r0)
    return dict(tetrahedra=len(cells), reference_resistance_ohm=r0,
                force_direction=direction.tolist(), moment_origin_m=origin.tolist(),
                patches=[item[0] for item in ordered])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="replay three patches with independent forward solves")
    args = parser.parse_args()
    print(json.dumps(run(verify=args.verify), indent=2))


if __name__ == "__main__":
    main()
