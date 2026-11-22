"""Check a local nested refinement of the frozen same-stroke Tet4 junction."""
import argparse
from dataclasses import asdict
from itertools import combinations
from collections import defaultdict
import json

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission, solve_static_tet4

if __package__:
    from .tet4_static_fitted_placement import FIXTURE_SHA256, _prepare
else:
    from tet4_static_fitted_placement import FIXTURE_SHA256, _prepare


STROKE_M = 3.2918408360037934e-9
# Ordered endpoint pairs in the pinned public fixture's stable IDs. Freeze
# the geometry recipe rather than relying on host sorting of nearly tied lengths.
EDGE_ENDPOINT_IDS = (
    (108, 618), (7, 585), (78, 629), (168, 585), (19, 677), (30, 690),
    (12, 582), (167, 585), (7, 583), (173, 585), (5, 584), (28, 585),
    (92, 589), (163, 632), (14, 582), (223, 585), (91, 589), (34, 647),
    (578, 671), (240, 660), (30, 578), (671, 690), (81, 578), (149, 671),
    (80, 578), (178, 660), (12, 589), (23, 690), (78, 589), (173, 608),
    (9, 582), (167, 600), (618, 620), (154, 687), (13, 582), (175, 690),
    (3, 582), (168, 680), (18, 578), (231, 690), (34, 583), (171, 652),
    (60, 583), (23, 610), (31, 584), (574, 690), (61, 583), (574, 680),
    (32, 584), (574, 689), (93, 626), (198, 602), (107, 623), (610, 654),
    (93, 583), (233, 608), (92, 584), (201, 602), (108, 583), (188, 600),
    (94, 584), (238, 652), (79, 691), (147, 687), (79, 618), (148, 605),
    (18, 584), (170, 672), (11, 583), (199, 679), (6, 584), (610, 648),
)


def _faces(tets):
    out = defaultdict(int)
    for tet in tets:
        for face in combinations(sorted(map(int, tet)), 3):
            out[face] += 1
    if not out or max(out.values()) > 2:
        raise ValueError('nonconforming frozen-junction faces')
    return {face for face, count in out.items() if count == 1}


def _volume_mm3(x, t):
    p = x[t]
    volume = np.linalg.det(p[:, 1:] - p[:, :1]) / 6.
    if not np.all(volume > 0.):
        raise ValueError('expected positive tetrahedron orientation')
    return volume


def _nested(b, supports, grip):
    points, tets = b.points, b.tetrahedra
    rows = {int(v): i for i, v in enumerate(b.node_ids)}
    edges = np.array([tuple(rows[n] for n in pair) for pair in EDGE_ENDPOINT_IDS])
    use = {tuple(edge): i for i, edge in enumerate(edges)}
    outer = _faces(tets)
    outer_edges = {e for f in outer for e in combinations(f, 2)}
    cap_ids = supports.keys() | grip.keys()
    if len(use) != len(edges) or outer_edges.intersection(use):
        raise ValueError('invalid frozen interior edge recipe')
    mids = (points[edges[:, 0]] + points[edges[:, 1]]) * .5
    points_f = np.vstack((points, mids))
    fine_cells, parent_rows, fine_ids = [], [], []
    next_id = int(max(b.cell_ids)) + 1
    visited = set()
    for parent, tet in enumerate(tets):
        split = [edge for edge in combinations(sorted(map(int, tet)), 2) if edge in use]
        if not split:
            fine_cells.append(tet)
            fine_ids.append(b.cell_ids[parent])
            parent_rows.append(parent)
            continue
        if len(split) != 1 or any(int(b.node_ids[n]) in cap_ids for n in tet):
            raise ValueError('frozen edge stars overlap or touch a grip')
        edge = split[0]
        visited.add(edge)
        mid = len(points) + use[edge]
        for index, node in enumerate(edge):
            fine_cells.append(np.where(tet == node, mid, tet))
            fine_ids.append(b.cell_ids[parent] if index == 0 else next_id)
            parent_rows.append(parent)
        next_id += 1
    fine_cells = np.asarray(fine_cells)
    parents = np.asarray(parent_rows)
    if visited != set(use) or _faces(fine_cells) != outer:
        raise ValueError('frozen exterior or edge stars changed')
    np.testing.assert_allclose(
        np.bincount(parents, weights=_volume_mm3(points_f, fine_cells)),
        _volume_mm3(points, tets), rtol=1e-12, atol=0.)
    ids = np.r_[b.node_ids, np.arange(int(max(b.node_ids)) + 1,
                                     int(max(b.node_ids)) + 1 + len(edges), dtype=np.uint64)]
    return points_f, fine_cells, ids, np.asarray(fine_ids, dtype=np.uint64), parents, edges


def _strain(points_mm, tets, u):
    p = points_mm[tets] * 1e-3
    v = u[tets]
    grad = np.linalg.solve(p[:, 1:] - p[:, :1], v[:, 1:] - v[:, :1])
    return .5 * (grad + grad.transpose(0, 2, 1))


def _difference_energy(fine, delta):
    v = _volume_mm3(fine.points, fine.tetrahedra) * 1e-9
    e = _strain(fine.points, fine.tetrahedra, delta)
    moduli = np.array([fine.materials[int(n)]['young_modulus_pa'] for n in fine.material_ids])
    nu = .30  # both assignments in this pinned fixture share this value
    mu = moduli / (2 * (1 + nu))
    lam = moduli * nu / ((1 + nu) * (1 - 2 * nu))
    return .5 * float(np.sum(v * (2 * mu * np.einsum('nij,nij->n', e, e)
                                  + lam * np.trace(e, axis1=1, axis2=2)**2)))


def _measure(bundle, holds, motion):
    r = solve_static_tet4(bundle, fixed_dofs_by_id=holds, forces_by_id={},
                         prescribed_displacements_m_by_id=motion)
    np.testing.assert_array_equal(bundle.node_ids, r.node_ids)
    left, right = (np.isin(r.node_ids, list(v)) for v in (holds, motion))
    np.testing.assert_array_equal(r.displacement_m[left], 0.)
    np.testing.assert_array_equal(r.displacement_m[right, 0], STROKE_M)
    np.testing.assert_array_equal(r.fixed_mask[right],
                                  np.tile([True, False, False], (len(motion), 1)))
    actuator = float(r.reaction_N[right, 0].sum())
    support = float(r.reaction_N[left, 0].sum())
    if not actuator > 0. or not support < 0. or r.normalized_backward_error > 1e-10:
        raise ValueError('unexpected grip reaction or free residual')
    work = .5 * actuator * STROKE_M
    np.testing.assert_allclose([r.strain_energy_J, r.half_external_work_J],
                               work, rtol=1e-8, atol=0.)
    net_force = r.reaction_N.sum(axis=0)
    net_moment = np.cross(bundle.points * 1e-3, r.reaction_N).sum(axis=0)
    np.testing.assert_allclose(net_force, 0., rtol=0., atol=2e-13)
    np.testing.assert_allclose(net_moment, 0., rtol=0., atol=3e-16)
    return r, dict(actuator_force_x_N=actuator, support_force_x_N=support,
                   stiffness_N_per_m=actuator / STROKE_M, energy_J=r.strain_energy_J,
                   half_actuator_work_J=work, net_force_N=net_force.tolist(),
                   net_moment_Nm=net_moment.tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solve', action='store_true',
                        help='run four solves (two layouts, two levels); default is preflight only')
    args = parser.parse_args(argv)
    motion = {}
    coarse, holds, _, _, _ = _prepare(stroke_control=(STROKE_M, motion))
    x, t, ids, cell_ids, parent, edges = _nested(coarse['uniform'], holds, motion)
    output = dict(input_sha256=FIXTURE_SHA256, stroke_m=STROKE_M,
                  exterior_and_caps_unchanged=True, parent_materials_inherited=True,
                  bisected_interior_edges=len(edges), solve_performed=False,
                  interpretation='Local refinement on a fixed discrete model, not convergence.')
    decks = {}
    for name in ('junction', 'arm'):
        old = coarse[name]
        fine = Tet4Bundle(x, t, ids, cell_ids, old.material_ids[parent],
                          old.materials, old.length_unit)
        decks[name] = (old, fine)
    coarse_plan, fine_plan = (inspect_static_tet4_admission(
        b, fixed_dofs_by_id=holds, prescribed_displacements_m_by_id=motion)
        for b in decks['junction'])
    # A shared topology and holds give identical admission for both material rows.
    admitted = all(plan.within_factor_budget and
                   max(plan.factor_entries, plan.padded_factor_entries) <= .8 * plan.factor_entry_limit
                   for plan in (coarse_plan, fine_plan))
    output['admitted_with_margin'] = admitted
    output['preflight'] = {'coarse': asdict(coarse_plan), 'fine': asdict(fine_plan)}
    if args.solve:
        if not admitted:
            parser.error('frozen nested example exceeds its local factor-budget margin')
        results = {}
        for name, (old, fine) in decks.items():
            r0, m0 = _measure(old, holds, motion)
            r1, m1 = _measure(fine, holds, motion)
            u = r0.displacement_m
            ui = np.vstack((u, (u[edges[:, 0]] + u[edges[:, 1]]) * .5))
            delta = ui - r1.displacement_m
            np.testing.assert_array_equal(delta[r1.fixed_mask], 0.)
            loss = m0['energy_J'] - m1['energy_J']
            if loss < -1e-8 * m0['energy_J']:
                raise ValueError('energy unexpectedly increased under nested refinement')
            difference = _difference_energy(fine, delta)
            np.testing.assert_allclose(loss, difference, rtol=0., atol=1e-8 * m0['energy_J'])
            results[name] = dict(coarse=m0, fine=m1, energy_drop_J=loss,
                                 displacement_difference_energy_J=difference)
        output['results'] = results
        output['arm_stiffness_advantage_percent'] = {
            level: 100 * (results['arm'][level]['stiffness_N_per_m'] /
                          results['junction'][level]['stiffness_N_per_m'] - 1)
            for level in ('coarse', 'fine')}
        output['solve_performed'] = True
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
