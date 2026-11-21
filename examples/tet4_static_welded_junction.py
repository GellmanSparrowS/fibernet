#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproducible Tet4 static example: three orthogonal welded cylinders."""
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations, permutations, product
import json
import os
from pathlib import Path
import sys

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission, solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction


HALF_LENGTH_MM = 1.5
RADIUS_MM = .5
CORE_RADIUS_MM = .45
FORCE_N = 1e-3
E_BASE_PA, E_CORE_PA, POISSON = 1e9, 3e9, .30
BASE_ID, CORE_ID = 7, 2**64 - 1
TIERS = {'quick': 12, 'reference': 16}
CASE_NAMES = ('homogeneous', 'stiffer_core')


def make_geometry(intervals):
    """Voxel union of x/y/z cylinders, with one conforming six-tet split per cube."""
    origins = []
    for origin in product(range(intervals), repeat=3):
        a, b, c = (2*i + 1 - intervals for i in origin)
        # Squared radial distance at cube center <= (0.5 mm)^2.
        if 9*min(a*a+b*b, b*b+c*c, c*c+a*a) <= intervals**2:
            origins.append(origin)
    corners = {key for origin in origins for key in product(*(range(i, i+2) for i in origin))}
    grid = np.array(sorted(corners), dtype=np.int64)
    rows = {tuple(key): row for row, key in enumerate(grid)}
    axes = np.eye(3, dtype=int)
    paths = [np.vstack((np.zeros(3, dtype=int), np.cumsum(axes[list(d)], axis=0)))
             for d in permutations(range(3))]
    cells = np.array([[rows[tuple(key)] for key in start+path]
                      for start in origins for path in paths], dtype=np.int64)
    step = (2*HALF_LENGTH_MM) / intervals
    points = grid.astype(np.float64)*step - HALF_LENGTH_MM
    center4 = grid[cells].sum(axis=1) - 2*intervals
    # Integer arithmetic for cell centroid distance <= 0.45 mm.
    core = 25*np.sum(center4**2, axis=1) <= 9*intervals**2
    digest = sha256()
    for arr, dtype in ((grid,'<i8'), (cells,'<i8'), (core,'?')):
        digest.update(arr.astype(dtype, copy=False).tobytes())
    return grid, points, cells, core, len(origins), digest.hexdigest()


def boundary(grid, cells, intervals):
    owners = defaultdict(list)
    for cell_id, tet in enumerate(cells):
        for face in combinations(tet, 3):
            owners[tuple(sorted(face))].append(cell_id)
    assert len({tuple(sorted(tet)) for tet in cells}) == len(cells)
    assert max(map(len, owners.values())) == 2
    adjacent = defaultdict(list)
    exterior, held, loaded = [], [], []
    for face, attached in owners.items():
        if len(attached) == 2:
            a,b = attached
            adjacent[a].append(b); adjacent[b].append(a)
            continue
        exterior.append(face)
        column = grid[list(face), 0]
        if np.all(column == 0):
            held.append(face)
        if np.all(column == intervals):
            loaded.append(face)
    # Verify one solid meeting across full triangular faces.
    pending, seen = deque([0]), {0}
    while pending:
        for other in adjacent[pending.popleft()]:
            if other not in seen:
                seen.add(other); pending.append(other)
    assert len(seen) == len(cells)
    edge_counts = Counter(tuple(sorted(edge)) for face in exterior for edge in combinations(face, 2))
    assert set(edge_counts.values()) == {2}
    return held, loaded, len(exterior)


def cap_loading(mesh, held_faces, loaded_faces):
    # The public helper takes Pa, not a prescribed resultant. Normalize by
    # the selected cap's actual area so both tiers receive the same 1 mN.
    def areas(faces):
        p = mesh.points[np.array(faces)] * 1e-3
        return .5*np.linalg.norm(np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]), axis=1)
    a, b = areas(held_faces), areas(loaded_faces)
    area = float(b.sum())
    np.testing.assert_allclose(a.sum(), area, rtol=1e-14)
    traction = FORCE_N / area
    selected = [tuple(int(mesh.node_ids[i]) for i in face) for face in loaded_faces]
    loads = nodal_forces_from_traction(mesh, selected, traction_Pa=(traction, 0., 0.))
    return loads, area, traction


def strain_energy(mesh, displacement_m, modulus_pa):
    """Independent element integral: no access to solver internals or factorization."""
    x = mesh.points*1e-3
    t = mesh.tetrahedra
    edges = x[t[:,1:]] - x[t[:,:1]]
    volume_m3 = np.abs(np.linalg.det(edges))/6.
    assert np.min(volume_m3) > 0.
    u = displacement_m[t]
    grad = np.linalg.solve(edges, u[:,1:]-u[:,:1])
    e = .5*(grad+grad.transpose(0,2,1))
    trace = np.trace(e, axis1=1, axis2=2)
    mu = modulus_pa / (2*(1+POISSON))
    lam = modulus_pa*POISSON / ((1+POISSON)*(1-2*POISSON))
    energy = .5*np.sum(volume_m3*(lam*trace**2 + 2*mu*np.sum(e**2, axis=(1,2))))
    return float(energy), volume_m3


def run(tier, write_vtu_dir=None, *, preflight=False):
    n = TIERS[tier]
    outputs = None
    if write_vtu_dir is not None:
        requested_dir = Path(write_vtu_dir)
        try:
            directory = requested_dir.resolve(strict=True)
        except RuntimeError as exc:
            # Older Path.resolve versions report a symlink loop as RuntimeError.
            raise OSError(f'could not resolve output directory {requested_dir}: {exc}') from exc
        if not directory.is_dir():
            raise NotADirectoryError(f'VTU output must be an existing directory: {directory}')
        outputs = {label: directory / f'welded-{tier}-{label}.vtu' for label in CASE_NAMES}
        for target in outputs.values():
            if os.path.lexists(target):
                raise FileExistsError(f'VTU output already exists: {target}')
        from fibernet.io.tet4_results import solve_static_tet4_to_vtu
    grid, x, t, core, cubes, digest = make_geometry(n)
    left, right, exterior = boundary(grid, t, n)
    fixed = np.unique(left)
    node_ids = np.arange(len(x), dtype=np.uint64)*19 + np.uint64(2**63+23)
    material = {BASE_ID: {'young_modulus_pa': E_BASE_PA, 'poisson_ratio': POISSON}}
    mesh = Tet4Bundle(x, t, node_ids, np.arange(len(t), dtype=np.int64)+100,
                      np.full(len(t), BASE_ID, dtype=np.int64), material, 'mm')
    hard = replace(mesh, material_ids=np.where(core,np.uint64(CORE_ID),np.uint64(BASE_ID)),
                   materials={**material, CORE_ID:{'young_modulus_pa':E_CORE_PA,'poisson_ratio':POISSON}})
    holds = {int(node_ids[i]): ('x','y','z') for i in fixed}
    if preflight:
        # Both cases have the same geometry, topology and held axes.
        admission = inspect_static_tet4_admission(mesh, fixed_dofs_by_id=holds)
        print(f'Tet4 preflight: {admission.free_dofs} free DOFs; '
              f'{admission.factor_entries:,} of {admission.factor_entry_limit:,} '
              'structural factor entries (not a numerical solvability check).',
              file=sys.stderr)
        if not admission.within_factor_budget:
            raise ValueError('Tet4 preflight exceeds the factor-entry budget')
    input_loads, area, traction = cap_loading(mesh, left, right)
    force = np.array([input_loads.get(int(i), (0., 0., 0.)) for i in node_ids])
    np.testing.assert_allclose(force.sum(axis=0), [FORCE_N,0.,0.], atol=2e-18)
    np.testing.assert_allclose(np.cross(x*1e-3,force).sum(axis=0), 0., atol=3e-21)
    cases = {}
    for label, bundle, e in (('homogeneous',mesh,np.full(len(t),E_BASE_PA)),
                             ('stiffer_core',hard,np.where(core,E_CORE_PA,E_BASE_PA))):
        if outputs is None:
            s = solve_static_tet4(bundle, forces_by_id=input_loads, fixed_dofs_by_id=holds)
        else:
            s = solve_static_tet4_to_vtu(outputs[label], bundle,
                                       forces_by_id=input_loads, fixed_dofs_by_id=holds)
        np.testing.assert_array_equal(s.node_ids,node_ids)
        np.testing.assert_array_equal(s.displacement_m[fixed],0.)
        np.testing.assert_array_equal(s.reaction_N[~s.fixed_mask],0.)
        energy, volumes = strain_energy(bundle,s.displacement_m,e)
        work = float(np.sum(force*s.displacement_m))
        assert work > 0.
        np.testing.assert_allclose([s.strain_energy_J,s.half_external_work_J,.5*work],energy,rtol=3e-11)
        net = force+s.reaction_N
        force_balance = net.sum(axis=0)
        moment_balance = np.cross(x*1e-3,net).sum(axis=0)
        np.testing.assert_allclose(force_balance,0.,atol=2e-12*FORCE_N)
        np.testing.assert_allclose(moment_balance,0.,atol=2e-12*FORCE_N*HALF_LENGTH_MM*1e-3)
        cases[label] = {'compliance_m_per_N':work/FORCE_N**2,
                        'effective_stiffness_N_per_m':FORCE_N**2/work,
                        'load_weighted_cap_displacement_m':work/FORCE_N,
                        'reaction_resultant_N':s.reaction_N.sum(axis=0).tolist(),
                        'net_force_N':force_balance.tolist(), 'net_moment_N_m':moment_balance.tolist(),
                        'strain_energy_J':s.strain_energy_J,'independent_strain_energy_J':energy,
                        'half_external_work_J':s.half_external_work_J,
                        'free_residual_inf_N':s.free_residual_inf_N,
                        'normalized_backward_error':s.normalized_backward_error}
    a,b = (cases[name]['compliance_m_per_N'] for name in ('homogeneous','stiffer_core'))
    assert b<a, 'compliance should decrease when only E increases under fixed force'
    return {'tier':tier, 'runtime_dependencies':{'numpy':np.__version__,'scipy':version('scipy')},
            'units':{'bundle_coordinates':'mm','solver_displacements':'m','modulus':'Pa',
                     'forces_and_reactions':'N','moments':'N m','energy_and_work':'J'},
            'voxel_step_mm':2*HALF_LENGTH_MM/n, 'nominal_radius_mm':RADIUS_MM,
            'arm_half_length_mm':HALF_LENGTH_MM,'core_centroid_radius_mm':CORE_RADIUS_MM,
            'cubes':cubes,'points':len(x),'tetrahedra':len(t),'core_tetrahedra':int(core.sum()),
            'mesh_and_core_sha256':digest, 'total_volume_mm3':float(volumes.sum()*1e9),
            'assigned_core_volume_mm3':float(volumes[core].sum()*1e9),
            'exterior_faces':exterior,'fixed_cap_nodes':len(fixed),
            'loaded_cap_nodes':int(np.count_nonzero(force[:,0])),
            'cap_area_m2':area,'equivalent_end_traction_Pa':traction,'applied_resultant_N':FORCE_N,
            'poisson_ratio':POISSON,'base_modulus_Pa':E_BASE_PA,'core_modulus_Pa':E_CORE_PA,
            'relative_compliance_change':b/a-1.,'materials':cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=TIERS, default='quick')
    parser.add_argument('--write-vtu-dir', metavar='DIR',
                        help='write two new diagnostic VTUs to an existing directory (optional Meshio)')
    parser.add_argument('--preflight', action='store_true',
                        help='report the shared structural factor budget before solving (to stderr)')
    args = parser.parse_args()
    try:
        report = run(args.tier, args.write_vtu_dir, preflight=args.preflight)
    except (OSError, ImportError) as exc:
        if args.write_vtu_dir is not None:
            parser.exit(2, f'Cannot write welded-junction VTU results: {exc}\n')
        raise
    except ValueError as exc:
        if args.tier == 'reference' and 'factor-entry budget' in str(exc):
            parser.exit(2, f'Reference tier rejected by bounded solver: {exc}\n'
                           'Run --tier quick for the smaller, separately discretized case.\n')
        if args.preflight and 'factor-entry budget' in str(exc):
            parser.exit(2, f'Quick tier rejected by bounded solver: {exc}\n')
        raise
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
