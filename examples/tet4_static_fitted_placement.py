"""Compare two equal-volume stiffening placements on one fitted Tet4 mesh."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite, pi
from pathlib import Path

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission, solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction

LOAD_AXES = ('x', 'y', 'z')

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'fitted_axial_B.json'
FIXTURE_SHA256 = '1b064ddc15aca50f2da24a8350fb68ede109a56732b10af12a8dbcc21aef8295'


def _fixture():
    raw = FIXTURE.read_bytes()
    if sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError('fitted axial fixture does not match the frozen input SHA256')
    return json.loads(raw)


def _transverse_cap_faces(points, tets, node_ids, axis_index):
    # A boundary triangle occurs once in the frozen tetrahedral topology.
    corners = tuple(combinations(range(4), 3))
    triples = np.sort(tets[:, corners].reshape(-1, 3), axis=1)
    faces, counts = np.unique(triples, axis=0, return_counts=True)
    exterior = faces[counts == 1]
    caps = []
    for end in (-1.5, 1.5):
        # The transverse end caps in this fixture have exact ordinates in mm.
        cap = exterior[np.all(points[exterior, axis_index] == end, axis=1)]
        if not len(cap):
            raise ValueError(f'the frozen mesh is missing a {LOAD_AXES[axis_index]} end cap')
        caps.append(node_ids[cap].tolist())
    return caps


def _cap_loads(bundle, faces, traction_Pa, total_force_N, axis):
    traction = tuple(traction_Pa if i == axis else 0. for i in range(3))
    loads = nodal_forces_from_traction(bundle, faces, traction_Pa=traction)
    force = np.asarray([loads.get(int(n), (0., 0., 0.)) for n in bundle.node_ids])
    resultant = [total_force_N if i == axis else 0. for i in range(3)]
    np.testing.assert_allclose(force.sum(axis=0), resultant, rtol=0, atol=2e-18)
    return loads, force


def _prepare(*, load_axis='x', fractions_by_placement=None, force_bases=None, stroke_control=None):
    data = _fixture()
    nodes, cells = data['nodes_id_xyz'], data['cells_id_node_ids']
    points = np.asarray([r[1:] for r in nodes], dtype=np.float64)
    node_ids = np.asarray([r[0] for r in nodes], dtype=np.uint64)
    cell_ids = np.asarray([r[0] for r in cells], dtype=np.uint64)
    node_rows = {r[0]: i for i, r in enumerate(nodes)}
    cell_rows = {r[0]: i for i, r in enumerate(cells)}
    tets = np.asarray([[node_rows[n] for n in r[1:]] for r in cells], dtype=np.int64)
    axis_index = LOAD_AXES.index(load_axis)
    if load_axis == 'x':
        fixed_faces, loaded_faces = data['fixed_minus_x_faces'], data['loaded_plus_x_faces']
    else:
        fixed_faces, loaded_faces = _transverse_cap_faces(points, tets, node_ids, axis_index)
    holds = {n: ('x', 'y', 'z') for face in fixed_faces for n in face}
    cap = points[[[node_rows[n] for n in face] for face in loaded_faces]] * 1e-3
    area_m2 = float(np.linalg.norm(
        np.cross(cap[:, 1] - cap[:, 0], cap[:, 2] - cap[:, 0]), axis=1).sum() / 2)

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
        if fractions_by_placement is not None and name != 'uniform':
            fractions_by_placement[name] = phi
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
    if stroke_control is None:
        plan = inspect_static_tet4_admission(base, fixed_dofs_by_id=holds)
    else:
        stroke_m, motion = stroke_control
        motion.update({int(n): {'x': stroke_m} for face in loaded_faces for n in face})
        plan = inspect_static_tet4_admission(
            base, fixed_dofs_by_id=holds, prescribed_displacements_m_by_id=motion)
    if not plan.within_factor_budget:
        raise ValueError('fitted mesh exceeds the Tet4 structural factor-entry budget')
    if stroke_control is not None:
        output = {'input_sha256': FIXTURE_SHA256, 'tet4_preflight': asdict(plan),
                  'fixed_cap_nodes': len(holds), 'actuated_cap_nodes': len(motion),
                  'actuated_cap_area_m2': area_m2, 'stroke_m': stroke_m,
                  'poisson_ratio': data['poisson_ratio'], 'base_E_Pa': base_E,
                  'fully_stiffened_E_Pa': data['limit_E_Pa'],
                  'sphere_radius_mm': data['sphere_radius_mm'], 'placements': placement}
        return bundles, holds, {}, None, output
    total_force_N = data['total_force_N']
    traction_Pa = total_force_N / area_m2
    loads, force = _cap_loads(base, loaded_faces, traction_Pa, total_force_N, axis_index)
    if force_bases is not None:
        if load_axis != 'x':
            raise ValueError('fixed-cap bases require the x end caps')
        for column, key in enumerate(LOAD_AXES):
            pairs = (loads, force) if column == 0 else _cap_loads(
                base, loaded_faces, traction_Pa, total_force_N, column)
            _, dense = pairs
            np.testing.assert_array_equal(dense[:, column], force[:, 0])
            np.testing.assert_array_equal(np.delete(dense, column, axis=1), 0.)
            dense.setflags(write=False)
            force_bases[key] = pairs
    output = {'input_sha256': FIXTURE_SHA256, 'tet4_preflight': asdict(plan),
              'fixed_cap_nodes': len(holds), 'loaded_cap_nodes': len(loads),
              'loaded_cap_area_m2': area_m2, 'traction_Pa': traction_Pa,
              'total_force_N': total_force_N, 'poisson_ratio': data['poisson_ratio'],
              'base_E_Pa': base_E, 'fully_stiffened_E_Pa': data['limit_E_Pa'],
              'sphere_radius_mm': data['sphere_radius_mm'], 'placements': placement}
    if load_axis != 'x':
        output['load_axis'] = load_axis
    return bundles, holds, loads, force, output


def _solve(bundles, holds, loads, force, total_force_N, *,
           cell_energies=None, displacements=None):
    results = {}
    for name, bundle in bundles.items():
        result = solve_static_tet4(bundle, fixed_dofs_by_id=holds, forces_by_id=loads,
                                  return_cell_fields=True)
        fields = result.cell_fields
        np.testing.assert_array_equal(result.node_ids, bundle.node_ids)
        np.testing.assert_array_equal(fields.cell_ids, bundle.cell_ids)
        if displacements is not None:
            displacements[name] = result.displacement_m
        work_J = float(np.sum(force * result.displacement_m))
        if not work_J > 0.:
            raise ValueError('expected positive external work in a loaded example')
        energies_J = 0.5 * fields.volume_m3 * np.einsum(
            'cij,cij->c', fields.stress_Pa, fields.strain)
        recovered_J = float(np.sum(energies_J))
        if cell_energies is not None:
            cell_energies[name] = energies_J
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


def _compliance_sensitivity(fractions, cell_energies, results, final_ratio):
    # For fixed force, d(F·u)/dr = -u^T (dK/dr) u. Since E(r) is
    # affine at fixed Poisson ratio, dK_i/dr = phi_i K_i / (1+(r-1)phi_i).
    # The reference denominator is always the uniform strain energy at r=1.
    uniform_J = results['uniform']['recovered_cell_energy_J']
    report = {'initial_modulus_ratio': 1., 'final_modulus_ratio': final_ratio, 'placements': {}}
    for name, phi in fractions.items():
        initial = float(phi @ cell_energies['uniform'] / uniform_J)
        final = float((phi / (1. + (final_ratio - 1.) * phi)) @ cell_energies[name] / uniform_J)
        mean = results[name]['relative_compliance_reduction'] / (final_ratio - 1.)
        tolerance = 1e-8 * max(initial, mean, final)
        if not 0. <= final <= mean + tolerance or not mean <= initial + tolerance:
            raise ValueError('average compliance decrease lies outside endpoint slopes')
        report['placements'][name] = {'initial_normalized_slope_per_r': initial,
                                      'average_normalized_slope_per_r': mean,
                                      'final_normalized_slope_per_r': final}
    return report


def _fixed_cap_summary(work_matrices, force_N):
    # Work and compliance have distinct units: W_ij = F_i·u_j [J],
    # C_ij = W_ij/F0² [m/N]. Check reciprocity before symmetrizing.
    cases, matrices = {}, {}
    abs_skew = 0.
    for name, work in work_matrices.items():
        raw = work / force_N**2
        diag = raw.diagonal()
        if not np.isfinite(raw).all() or not np.all(diag > 0.):
            raise ValueError(f'{name}: nonfinite compliance or nonpositive basis work')
        skew = np.abs(raw - raw.T)
        reciprocal = float(np.max(skew / np.sqrt(np.outer(diag, diag))))
        if reciprocal > 1e-8:
            raise ValueError(f'{name}: fixed-cap work violates reciprocity')
        symmetric = (raw + raw.T) / 2
        eig = np.linalg.eigvalsh(symmetric)
        if eig[0] <= 0.:
            raise ValueError(f'{name}: fixed-cap compliance is not positive definite')
        abs_skew = max(abs_skew, float(skew.max()))
        matrices[name] = symmetric
        cases[name] = {'work_matrix_J': work.tolist(),
                       'compliance_matrix_m_per_N': symmetric.tolist(),
                       'max_scaled_reciprocity_error': reciprocal}
    reference = matrices['uniform']
    difference = matrices['junction'] - matrices['arm']
    # Generalized eigenvalues are extrema over all nonzero real directions.
    L = np.linalg.cholesky(reference)
    whitened = np.linalg.solve(L, np.linalg.solve(L, difference).T).T
    values, axes = np.linalg.eigh((whitened + whitened.T) / 2)
    directions = np.linalg.solve(L.T, axes)
    directions /= np.linalg.norm(directions, axis=0)
    quotients = (np.sum(directions * (difference @ directions), axis=0) /
                 np.sum(directions * (reference @ directions), axis=0))
    np.testing.assert_allclose(quotients, values, rtol=1e-8, atol=1e-12)
    residual = difference @ directions - (reference @ directions) * values
    scale = np.linalg.norm(difference, ord=2) + max(abs(values)) * np.linalg.norm(reference, ord=2)
    if np.linalg.norm(residual, ord=2) > 1e-8 * scale:
        raise ValueError('fixed-cap generalized eigenpair residual is too large')
    # This conservative numerical margin is not a continuum error bound.
    margin = max(1e-8, 1e3 * abs_skew / np.linalg.eigvalsh(reference)[0])
    lo, hi = map(float, values[[0, -1]])
    if lo > margin:
        verdict = 'offset_lower_for_all_directions'
    elif hi < -margin:
        verdict = 'center_lower_for_all_directions'
    elif lo < -margin and hi > margin:
        verdict = 'ranking_depends_on_direction'
    else:
        verdict = 'inconclusive_at_numerical_margin'
    return {'basis_order': list(LOAD_AXES), 'basis_force_N': force_N, 'placements': cases,
            'offset_advantage_extrema_percentage_points': [100 * lo, 100 * hi],
            'ranking_margin_fraction': float(margin), 'ranking': verdict}


def _fixed_cap_compliance(bundles, holds, force_bases, force_N):
    works = {name: np.zeros((3, 3)) for name in bundles}
    reference_ids = bundles['uniform'].node_ids
    for bundle in bundles.values():
        np.testing.assert_array_equal(bundle.node_ids, reference_ids)
    for j, key in enumerate(LOAD_AXES):
        loads, force = force_bases[key]
        displacements = {}
        results = _solve(bundles, holds, loads, force, force_N, displacements=displacements)
        for name, U in displacements.items():
            works[name][:, j] = [np.sum(force_bases[axis][1] * U) for axis in LOAD_AXES]
            np.testing.assert_allclose(works[name][j, j], 2 * results[name]['half_external_work_J'],
                                       rtol=1e-8, atol=0)
    return _fixed_cap_summary(works, force_N)


def _fixed_cap_stroke(bundles, holds, motion, stroke_m):
    results = {}
    for name, bundle in bundles.items():
        r = solve_static_tet4(bundle, fixed_dofs_by_id=holds, forces_by_id={},
                              prescribed_displacements_m_by_id=motion, return_cell_fields=True)
        np.testing.assert_array_equal(r.node_ids, bundle.node_ids)
        np.testing.assert_array_equal(r.cell_fields.cell_ids, bundle.cell_ids)
        actuator = np.isin(r.node_ids, list(motion))
        support = np.isin(r.node_ids, list(holds))
        np.testing.assert_array_equal(r.displacement_m[actuator, 0], stroke_m)
        np.testing.assert_array_equal(r.displacement_m[support], 0.)
        np.testing.assert_array_equal(r.fixed_mask[actuator], np.tile([True, False, False], (len(motion),1)))
        actuator_x_N = float(r.reaction_N[actuator, 0].sum())
        support_x_N = float(r.reaction_N[support, 0].sum())
        stiffness_N_per_m = actuator_x_N / stroke_m
        if not isfinite(stiffness_N_per_m) or not stiffness_N_per_m > 0.:
            raise ValueError('expected positive finite stiffness for this prescribed stroke')
        half_actuator_work_J = .5 * actuator_x_N * stroke_m
        fields = r.cell_fields
        reconstructed_J = .5 * float(np.einsum('i,ijk,ijk->', fields.volume_m3, fields.stress_Pa, fields.strain))
        np.testing.assert_allclose([r.strain_energy_J, r.half_external_work_J, reconstructed_J],
                                   half_actuator_work_J, rtol=1e-8, atol=0.)
        net_force_N = r.reaction_N.sum(axis=0)
        net_moment_N_m = np.cross(bundle.points*1e-3, r.reaction_N).sum(axis=0)
        # Scale residual tolerances to the actual force/stroke, not the old 1 mN.
        np.testing.assert_allclose(net_force_N, 0., atol=2e-10*abs(actuator_x_N), rtol=0.)
        np.testing.assert_allclose(net_moment_N_m, 0., atol=3e-13*abs(actuator_x_N), rtol=0.)
        results[name] = {'actuator_force_x_N': actuator_x_N, 'support_force_x_N': support_x_N,
                         'stiffness_N_per_m': stiffness_N_per_m,
                         'strain_energy_J': r.strain_energy_J, 'recovered_cell_energy_J': reconstructed_J,
                         'half_actuator_work_J': half_actuator_work_J, 'net_force_N': net_force_N.tolist(),
                         'net_moment_N_m': net_moment_N_m.tolist()}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true',
                      help='check input/loads/factor budget, without solving')
    mode.add_argument('--compliance-sensitivity', action='store_true',
                      help='include endpoint and average normalized compliance slopes')
    mode.add_argument('--fixed-cap-compliance', action='store_true',
                      help='solve three load bases per placement on the fixed x end caps')
    mode.add_argument('--fixed-cap-stroke-m', type=float, metavar='METRES',
                      help='impose one signed nonzero axial stroke on the positive x cap')
    parser.add_argument('--load-axis', choices=LOAD_AXES,
                        help='clamp the negative cap and pull the positive cap on this axis (default: x)')
    args = parser.parse_args(argv)
    if args.fixed_cap_compliance and args.load_axis is not None:
        parser.error('--fixed-cap-compliance cannot be combined with --load-axis')
    stroke = args.fixed_cap_stroke_m
    if stroke is not None and (not isfinite(stroke) or stroke == 0.):
        parser.error('--fixed-cap-stroke-m requires finite nonzero metres')
    if stroke is not None and args.load_axis is not None:
        parser.error('--fixed-cap-stroke-m cannot be combined with --load-axis')
    try:
        prepare_options = {'load_axis': args.load_axis or 'x'}
        if args.fixed_cap_compliance:
            prepare_options['force_bases'] = {}
        if args.compliance_sensitivity:
            prepare_options['fractions_by_placement'] = {}
        if stroke is not None:
            motion = {}
            prepare_options['stroke_control'] = (stroke, motion)
        bundles, holds, loads, force, output = _prepare(**prepare_options)
        if stroke is not None:
            output['results'] = _fixed_cap_stroke(bundles, holds, motion, stroke)
        elif args.fixed_cap_compliance:
            output['fixed_cap_compliance'] = _fixed_cap_compliance(
                bundles, holds, prepare_options['force_bases'], output['total_force_N'])
        elif not args.preflight:
            cell_energies = {} if args.compliance_sensitivity else None
            output['results'] = _solve(bundles, holds, loads, force, output['total_force_N'],
                                       cell_energies=cell_energies)
            if args.compliance_sensitivity:
                ratio = output['fully_stiffened_E_Pa'] / output['base_E_Pa']
                output['compliance_sensitivity'] = _compliance_sensitivity(
                    prepare_options['fractions_by_placement'], cell_energies, output['results'], ratio)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output['solve_performed'] = not args.preflight
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
