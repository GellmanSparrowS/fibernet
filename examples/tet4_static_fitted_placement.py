"""Compare two equal-volume stiffening placements on one fitted Tet4 mesh."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from math import pi
from pathlib import Path

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission, solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'fitted_axial_B.json'
FIXTURE_SHA256 = '1b064ddc15aca50f2da24a8350fb68ede109a56732b10af12a8dbcc21aef8295'


def _fixture():
    raw = FIXTURE.read_bytes()
    if sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError('fitted axial fixture does not match the frozen input SHA256')
    return json.loads(raw)


def _y_cap_faces(points, tets, node_ids):
    # A boundary triangle occurs once in the frozen tetrahedral topology.
    corners = tuple(combinations(range(4), 3))
    triples = np.sort(tets[:, corners].reshape(-1, 3), axis=1)
    faces, counts = np.unique(triples, axis=0, return_counts=True)
    exterior = faces[counts == 1]
    caps = []
    for end in (-1.5, 1.5):
        # Both planar y caps in this fixture have exact ordinates in mm.
        cap = exterior[np.all(points[exterior, 1] == end, axis=1)]
        if not len(cap):
            raise ValueError('the frozen mesh is missing a y end cap')
        caps.append(node_ids[cap].tolist())
    return caps


def _prepare(*, load_axis='x'):
    data = _fixture()
    nodes, cells = data['nodes_id_xyz'], data['cells_id_node_ids']
    points = np.asarray([r[1:] for r in nodes], dtype=np.float64)
    node_ids = np.asarray([r[0] for r in nodes], dtype=np.uint64)
    cell_ids = np.asarray([r[0] for r in cells], dtype=np.uint64)
    node_rows = {r[0]: i for i, r in enumerate(nodes)}
    cell_rows = {r[0]: i for i, r in enumerate(cells)}
    tets = np.asarray([[node_rows[n] for n in r[1:]] for r in cells], dtype=np.int64)
    if load_axis == 'y':
        fixed_faces, loaded_faces = _y_cap_faces(points, tets, node_ids)
    else:
        fixed_faces, loaded_faces = data['fixed_minus_x_faces'], data['loaded_plus_x_faces']
    holds = {n: ('x', 'y', 'z') for face in fixed_faces for n in face}
    cap = points[[[node_rows[n] for n in face] for face in loaded_faces]] * 1e-3
    area_m2 = float(np.linalg.norm(
        np.cross(cap[:, 1] - cap[:, 0], cap[:, 2] - cap[:, 0]), axis=1).sum() / 2)
    total_force_N = data['total_force_N']
    traction_Pa = total_force_N / area_m2

    # Fractions are volume intersections, not centroid labels. Refuse any
    # accidental footprint on tets sharing a node with either end cap.
    cap_rows = [node_rows[n] for face in fixed_faces + loaded_faces for n in face]
    touches_cap = np.isin(tets, cap_rows).any(axis=1)
    corners = points[tets]
    volume_mm3 = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    if not np.all(volume_mm3 > 0.):
        raise ValueError('the frozen fixture must contain only nondegenerate tetrahedra')
    target_volume_mm3 = 4 * pi * data['sphere_radius_mm']**3 / 3
    base_E = data['base_E_Pa']
    contrast = data['limit_E_Pa'] / base_E - 1
    assignments = [{'name': 'uniform', 'center_mm': None, 'positive_cell_fractions': []},
                   *data['assignments']]
    bundles, placement = {}, {}
    for assignment in assignments:
        name = assignment['name']
        phi = np.zeros(len(tets))
        for tag, fraction in assignment['positive_cell_fractions']:
            row = cell_rows[tag]
            if not 0 < fraction <= 1 or phi[row] or touches_cap[row]:
                raise ValueError('invalid or repeated stiffened fraction, or overlap with an end cap')
            phi[row] = fraction
        assigned_mm3 = float(phi @ volume_mm3)
        if name != 'uniform':
            np.testing.assert_allclose(assigned_mm3, target_volume_mm3, rtol=0, atol=1e-8)
        moduli, material_ids = np.unique(base_E * (1 + contrast * phi), return_inverse=True)
        materials = {i: {'young_modulus_pa': float(e), 'poisson_ratio': data['poisson_ratio']}
                     for i, e in enumerate(moduli)}
        bundles[name] = Tet4Bundle(points, tets, node_ids, cell_ids, material_ids,
                                   materials, data['geometry_unit'])
        placement[name] = {'center_mm': assignment['center_mm'],
                           'stiffened_volume_mm3': assigned_mm3,
                           'cells_with_nonzero_fraction': int(np.count_nonzero(phi))}

    base = bundles['uniform']
    plan = inspect_static_tet4_admission(base, fixed_dofs_by_id=holds)
    if not plan.within_factor_budget:
        raise ValueError('fitted mesh exceeds the Tet4 structural factor-entry budget')
    traction = (0., traction_Pa, 0.) if load_axis == 'y' else (traction_Pa, 0., 0.)
    loads = nodal_forces_from_traction(base, loaded_faces, traction_Pa=traction)
    force = np.asarray([loads.get(int(n), (0., 0., 0.)) for n in node_ids])
    resultant = [0., total_force_N, 0.] if load_axis == 'y' else [total_force_N, 0., 0.]
    np.testing.assert_allclose(force.sum(axis=0), resultant, rtol=0, atol=2e-18)
    output = {'input_sha256': FIXTURE_SHA256, 'tet4_preflight': asdict(plan),
              'fixed_cap_nodes': len(holds), 'loaded_cap_nodes': len(loads),
              'loaded_cap_area_m2': area_m2, 'traction_Pa': traction_Pa,
              'total_force_N': total_force_N, 'poisson_ratio': data['poisson_ratio'],
              'base_E_Pa': base_E, 'fully_stiffened_E_Pa': data['limit_E_Pa'],
              'sphere_radius_mm': data['sphere_radius_mm'], 'placements': placement}
    if load_axis == 'y':
        output['load_axis'] = 'y'
    return bundles, holds, loads, force, output


def _solve(bundles, holds, loads, force, total_force_N):
    results = {}
    for name, bundle in bundles.items():
        result = solve_static_tet4(bundle, fixed_dofs_by_id=holds, forces_by_id=loads,
                                  return_cell_fields=True)
        fields = result.cell_fields
        np.testing.assert_array_equal(result.node_ids, bundle.node_ids)
        np.testing.assert_array_equal(fields.cell_ids, bundle.cell_ids)
        work_J = float(np.sum(force * result.displacement_m))
        if not work_J > 0.:
            raise ValueError('expected positive external work in a loaded example')
        recovered_J = float(np.sum(0.5 * fields.volume_m3 *
                                   np.einsum('cij,cij->c', fields.stress_Pa, fields.strain)))
        np.testing.assert_allclose([result.strain_energy_J, result.half_external_work_J, recovered_J],
                                   0.5 * work_J, rtol=1e-8, atol=0)
        net = force + result.reaction_N
        force_residual_N = net.sum(axis=0)
        moment_residual_Nm = np.cross(bundle.points * 1e-3, net).sum(axis=0)
        np.testing.assert_allclose(force_residual_N, 0., rtol=0, atol=2e-13)
        np.testing.assert_allclose(moment_residual_Nm, 0., rtol=0, atol=3e-16)
        results[name] = {'compliance_m_per_N': work_J / total_force_N**2,
                         'load_weighted_cap_displacement_m': work_J / total_force_N,
                         'strain_energy_J': result.strain_energy_J,
                         'recovered_cell_energy_J': recovered_J,
                         'half_external_work_J': result.half_external_work_J,
                         'net_force_N': force_residual_N.tolist(),
                         'net_moment_N_m': moment_residual_Nm.tolist()}
    base = results['uniform']['compliance_m_per_N']
    for label in ('junction', 'arm'):
        value = results[label]['compliance_m_per_N']
        if not value < base:
            raise ValueError('stiffening must reduce compliance for this fixed-force example')
        results[label]['relative_compliance_reduction'] = 1 - value / base
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true',
                        help='check input/loads/factor budget, without solving')
    parser.add_argument('--load-axis', choices=('x', 'y'), default='x',
                        help='clamp the negative cap and pull the positive cap on this axis (default: x)')
    args = parser.parse_args(argv)
    try:
        bundles, holds, loads, force, output = _prepare(load_axis=args.load_axis)
        if not args.preflight:
            output['results'] = _solve(bundles, holds, loads, force, output['total_force_N'])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output['solve_performed'] = not args.preflight
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
