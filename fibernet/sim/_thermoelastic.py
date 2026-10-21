"""Two separate experimental prescribed-snapshot profiles; neither is a facade API.

The existing elementary profile retains its two-node axial-only, uniform-
temperature restrictions. The network profile admits bounded planar welded
Euler–Bernoulli frames with linearly interpolated nodal temperature snapshots.
No thermal solve, feedback, graph mutation, material inference, stability
qualification or numerical regularization is provided by either profile.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json
import math
import numbers
import numpy as np
from ._frame_linear_common import (
    _geometry, _histories, _check_supports, _solve_constrained_sparse,
)

PROFILE = 'thermoelastic_2d_elementary/2'
NETWORK_PROFILE = 'thermoelastic_2d_network/1'


@dataclass(frozen=True)
class ThermoelasticSnapshot:
    """Immutable JSON-owned snapshot; to_dict returns a fresh detached copy."""
    data_json: str

    def to_dict(self):
        return json.loads(self.data_json)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _number(value, label, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(label + ' must be a real non-bool number')
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(label + ' must be finite' + (' and positive' if positive else ''))
    return result


def _mapping(value, ids, label, *, complete, triples=False, positive=False):
    if not isinstance(value, Mapping):
        raise ValueError(label + ' must be an ID mapping')
    copied = {}
    for key, val in value.items():
        if (isinstance(key, (bool, np.bool_)) or not isinstance(key, numbers.Integral)
                or key not in ids):
            raise ValueError(label + ' has invalid ID')
        if triples:
            if not isinstance(val, (list, tuple, np.ndarray)) or len(val) != 3:
                raise ValueError(label + ' needs triples')
            copied[int(key)] = tuple(None if x is None and label == 'constraints'
                                     else _number(x, label) for x in val)
        else:
            copied[int(key)] = _number(val, label, positive)
    if complete and set(copied) != set(ids):
        raise ValueError(label + ' requires complete ID coverage')
    return copied


def _finite_derived(label, *values):
    if any(not np.isfinite(value).all() for value in values):
        raise FloatingPointError('nonfinite derived ' + label)


def assemble_axial_eigenstrain_load(pos, edges, radii, E, eigenstrain):
    """Equivalent axial pair loads only; no stiffness assembly or solve."""
    force = np.zeros((len(pos), 3))
    for row, ((a, b), radius) in enumerate(zip(edges, radii)):
        direction = pos[b] - pos[a]
        length = np.linalg.norm(direction)
        area = math.pi * radius**2
        pair = E * area * eigenstrain[row] * (direction / length)
        _finite_derived('thermal pair load', direction, length, area, pair)
        force[a, :2] -= pair
        force[b, :2] += pair
    _finite_derived('thermal load sum', force)
    return force.ravel()


def _accept_residual(K, u, thermal, applied, mask, pos, root, reaction):
    """Consistency checks, not an independent analytical/FEM reference."""
    residual = K @ u - thermal - applied
    scale = abs(K) @ np.abs(u) + np.abs(thermal) + np.abs(applied)
    absolute = np.tile([1e-12, 1e-12, 1e-13], len(pos))
    bound = absolute + 1e-9 * scale
    _finite_derived('residual and bounds', residual, scale, bound, reaction)
    if np.any(np.abs(residual[~mask]) > bound[~mask]):
        raise FloatingPointError('free equilibrium residual exceeded')
    support = np.where(mask, reaction, 0.).reshape(-1, 3)
    external = applied.reshape(-1, 3)
    loads = support + external
    offset = pos - pos[root]
    moment_terms = (offset[:, 0]*loads[:, 1] - offset[:, 1]*loads[:, 0]
                    + loads[:, 2])
    balance = np.array([loads[:, 0].sum(), loads[:, 1].sum(), moment_terms.sum()])
    magnitudes = np.abs(support) + np.abs(external)
    balance_scale = np.array([
        magnitudes[:, 0].sum(), magnitudes[:, 1].sum(),
        (np.abs(offset[:, 0])*magnitudes[:, 1]
         + np.abs(offset[:, 1])*magnitudes[:, 0] + magnitudes[:, 2]).sum()])
    balance_bound = np.array([1e-12, 1e-12, 1e-13]) + 1e-9*balance_scale
    _finite_derived('component balance', offset, loads, moment_terms, balance,
                    balance_scale, balance_bound)
    if np.any(np.abs(balance) > balance_bound):
        raise FloatingPointError('component force/moment balance exceeded')
    return residual, bound, balance, balance_bound


def solve_thermoelastic_snapshot(graph, *, length_unit, E_Pa,
        expansion_alpha_per_K, stress_free_temperature_K, temperature_K,
        constraints, loads=None):
    """Bounded axial-only snapshot; this elementary profile does not admit frames."""
    with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
        return _snapshot(graph, length_unit=length_unit, E_Pa=E_Pa,
            expansion_alpha_per_K=expansion_alpha_per_K,
            stress_free_temperature_K=stress_free_temperature_K,
            temperature_K=temperature_K, constraints=constraints, loads=loads)


def _snapshot(graph, *, length_unit, E_Pa, expansion_alpha_per_K,
              stress_free_temperature_K, temperature_K, constraints, loads):
    if len(graph.nodes) != 2 or not 1 <= len(graph.edges) <= 2:
        raise ValueError('elementary profile requires two nodes and one or two edges')
    E = _number(E_Pa, 'E_Pa', True)
    reference = _number(stress_free_temperature_K, 'stress_free_temperature_K', True)
    ids, eids, pos, edges, radii, dofs, B, metric = _geometry(graph, length_unit)
    temperatures = _mapping(temperature_K, ids, 'temperature_K', complete=True, positive=True)
    alpha = _mapping(expansion_alpha_per_K, eids, 'expansion_alpha_per_K', complete=True)
    prescribed = _mapping(constraints, ids, 'constraints', complete=False, triples=True)
    applied = _mapping({} if loads is None else loads, ids, 'loads', complete=False, triples=True)
    roots = [nid for nid, triple in prescribed.items() if triple == (0., 0., 0.)]
    if len(roots) != 1:
        raise ValueError('exactly one zero root clamp is required')
    root_id = roots[0]
    root = ids.index(root_id)
    tip = 1-root
    tip_id = ids[tip]
    direction = pos[tip]-pos[root]
    _finite_derived('geometry', pos, direction, radii, B, metric)
    axes = np.flatnonzero(direction != 0)
    if len(axes) != 1 or direction[axes[0]] <= 0:
        raise ValueError('root-to-tip axis must be positive x or y')
    axis = int(axes[0]); transverse = 1-axis
    tip_control = prescribed.get(tip_id, (None, None, None))
    if tip_control[transverse] is not None or tip_control[2] is not None:
        raise ValueError('tip transverse and rotational DOFs must be free')
    if any(set(pair) != {0, 1} for pair in edges):
        raise ValueError('only straight parallel root-tip members are admitted')
    for nid, triple in applied.items():
        if ((nid == root_id and any(x != 0 for x in triple))
                or triple[transverse] != 0 or triple[2] != 0):
            raise ValueError('only tip axial force is admitted')
    if len(set(temperatures.values())) != 1 or len(set(alpha.values())) != 1:
        raise ValueError('uniform temperature and equal parallel alpha are required')
    force, values, mask = _histories([0.], ids,
        {nid: [triple] for nid, triple in applied.items()},
        {nid: [triple] for nid, triple in prescribed.items()})
    _check_supports(pos, edges, mask)
    delta_temperature = np.float64(temperatures[root_id]) - np.float64(reference)
    thermal_strain = np.array([alpha[eid] for eid in eids]) * delta_temperature
    _finite_derived('thermal strain', thermal_strain)
    if np.any(np.abs(thermal_strain) > .001):
        raise ValueError('thermal strain exceeds elementary profile bound')
    return _snapshot_core(graph, length_unit=length_unit, E=E, reference=reference,
        ids=ids, eids=eids, pos=pos, edges=edges, radii=radii, dofs=dofs, B=B,
        temperatures=temperatures, alpha=alpha, force=force, values=values,
        mask=mask, root=root, thermal_strain=thermal_strain, profile=PROFILE)


def _snapshot_core(graph, *, length_unit, E, reference, ids, eids, pos, edges,
                   radii, dofs, B, temperatures, alpha, force, values, mask,
                   root, thermal_strain, profile):
    """One assembly/recovery/output body, entered after profile preflight."""
    from fibernet.ml.beam_frame_fem import BeamFrameFEM
    fem = BeamFrameFEM(E=E)
    thermal = assemble_axial_eigenstrain_load(pos, edges, radii, E, thermal_strain)
    K, selected = fem.build_stiffness_2d(edges.T, pos, radii, deduplicate=False)
    if list(selected) != list(range(len(eids))):
        raise ValueError('unexpected selected edge rows')
    _finite_derived('stiffness and RHS', K.data, thermal, force)
    u = _solve_constrained_sparse(K, force[0]+thermal, values[0], mask)
    displacement = u.reshape(-1, 3)
    total = np.einsum('ei,ei->e', B[:, 0, :], u[dofs])
    elastic = total-thermal_strain
    _finite_derived('displacement and strains', displacement, total, elastic)
    applicability = None
    if profile == PROFILE:
        if np.any(np.abs(total) > .005) or np.any(np.abs(displacement[:, 2]) > .005):
            raise FloatingPointError('candidate exceeds elementary strain/rotation bound')
    else:
        applicability = _network_applicability(pos, edges, radii, displacement,
            total, np.array([[temperatures[ids[a]], temperatures[ids[b]]]
                             for a, b in edges]),
            np.array([alpha[eid] for eid in eids]), reference)
    recovery = fem._compute_element_stress_2d(pos, displacement, edges.T,
        selected, radii, axial_eigenstrain=thermal_strain)
    _finite_derived('recovery', *recovery)
    reaction = K @ u-thermal-force[0]
    residual, bound, balance, balance_bound = _accept_residual(
        K, u, thermal, force[0], mask, pos, root, reaction)
    original_geometry = {
        'dimension': int(graph.dimension),
        'positions_xyz': [np.asarray(graph.nodes[nid].position).tolist() for nid in ids],
        'radii': [float(graph.edges[eid].radius) for eid in eids],
        'internal_points': [None if graph.edges[eid].internal_points is None
                            else np.asarray(graph.edges[eid].internal_points).tolist()
                            for eid in eids],
    }
    identity = {'profile': profile, 'node_ids': ids, 'edge_ids': eids,
        'original_geometry': original_geometry,
        'endpoints_by_id': [[ids[a], ids[b]] for a, b in edges],
        'positions_m': pos.tolist(), 'radii_m': radii.tolist(),
        'length_unit': length_unit, 'E_Pa': E, 'stress_free_temperature_K': reference,
        'temperature_K': [temperatures[i] for i in ids],
        'expansion_alpha_per_K': [alpha[i] for i in eids],
        'constraint_mask': mask.reshape(-1, 3).tolist(),
        'prescribed_values': values[0].reshape(-1, 3).tolist(),
        'loads': force[0].reshape(-1, 3).tolist()}
    result = dict(identity, input_sha256=hashlib.sha256(_json(identity).encode()).hexdigest(),
        units={'length': 'm', 'rotation': 'rad', 'force': 'N', 'moment': 'N*m',
               'stress': 'Pa', 'temperature': 'K', 'alpha': '1/K'},
        interpolation='linear_endpoint_temperature_mean_axial/1',
        numerical_regularization=False, total_axial_strain=total.tolist(),
        thermal_axial_strain=thermal_strain.tolist(), elastic_axial_strain=elastic.tolist(),
        displacement=displacement.tolist(), sigma_axial=recovery[0].tolist(),
        sigma_bending=recovery[1].tolist(), sigma_total=recovery[2].tolist(),
        moments=recovery[3].tolist(), edge_forces=recovery[4].tolist(),
        reactions=reaction.reshape(-1, 3).tolist(),
        free_dof_indices=np.flatnonzero(~mask).tolist(), free_residual=residual[~mask].tolist(),
        free_acceptance_bound=bound[~mask].tolist(), component_balance=balance.tolist(),
        component_acceptance_bound=balance_bound.tolist())
    if applicability is not None:
        result['applicability'] = applicability
        result['balance_origin_node_id'] = ids[root]
    return ThermoelasticSnapshot(_json(result))


def solve_thermoelastic_network_snapshot(graph, *, length_unit, E_Pa,
        expansion_alpha_per_K, stress_free_temperature_K, temperature_K,
        constraints, loads=None):
    """Experimental planar welded EB network under prescribed nodal temperatures.

    Admission is not material, junction, shear, mesh or stability qualification.
    There is no thermal solve, lineage inference, coordinate-based identity,
    geometric stiffness, feedback, public facade or numerical regularization.
    """
    with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
        return _network_snapshot(graph, length_unit=length_unit, E_Pa=E_Pa,
            expansion_alpha_per_K=expansion_alpha_per_K,
            stress_free_temperature_K=stress_free_temperature_K,
            temperature_K=temperature_K, constraints=constraints, loads=loads)


def _network_snapshot(graph, *, length_unit, E_Pa, expansion_alpha_per_K,
                      stress_free_temperature_K, temperature_K, constraints, loads):
    if not 2 <= len(graph.nodes) <= 192 or not 1 <= len(graph.edges) <= 256:
        raise ValueError('network profile requires 2..192 nodes and 1..256 edges')
    for mapping in (graph.nodes, graph.edges):
        if any(isinstance(key, (bool, np.bool_))
               or not isinstance(key, numbers.Integral) or key < 0 for key in mapping):
            raise ValueError('stable IDs must be nonnegative integers')
    for edge in graph.edges.values():
        _number(edge.radius, 'radius', True)
        for endpoint in (edge.node_i, edge.node_j):
            if isinstance(endpoint, (bool, np.bool_)) or not isinstance(endpoint, numbers.Integral):
                raise ValueError('edge endpoints require integer stable IDs')
    E = _number(E_Pa, 'E_Pa', True)
    reference = _number(stress_free_temperature_K, 'stress_free_temperature_K', True)
    ids, eids, pos, edges, radii, dofs, B, metric = _geometry(graph, length_unit)
    _finite_derived('network geometry', pos, radii, B, metric)
    temperatures = _mapping(temperature_K, ids, 'temperature_K', complete=True, positive=True)
    alpha = _mapping(expansion_alpha_per_K, eids, 'expansion_alpha_per_K', complete=True)
    prescribed = _mapping(constraints, ids, 'constraints', complete=False, triples=True)
    applied = _mapping({} if loads is None else loads, ids, 'loads', complete=False, triples=True)
    # _check_supports intentionally admits separate supported components for
    # other users. This profile additionally requires exactly one component.
    adjacency = [set() for _ in ids]
    for a, b in edges:
        adjacency[a].add(int(b)); adjacency[b].add(int(a))
    visited, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node not in visited:
            visited.add(node); pending.extend(adjacency[node] - visited)
    if len(visited) != len(ids):
        raise ValueError('network profile requires one connected component without isolated nodes')
    force, values, mask = _histories([0.], ids,
        {nid: [triple] for nid, triple in applied.items()},
        {nid: [triple] for nid, triple in prescribed.items()})
    _check_supports(pos, edges, mask)
    endpoints = np.array([[temperatures[ids[a]], temperatures[ids[b]]] for a, b in edges])
    coefficients = np.array([alpha[eid] for eid in eids])
    endpoint_thermal = coefficients[:, None] * (endpoints - np.float64(reference))
    # The guarded sum is deliberate: nonfinite intermediates are rejected even
    # if a differently arranged mathematical expression would remain finite.
    mean_temperature = (endpoints[:, 0] + endpoints[:, 1]) / 2.
    thermal_strain = coefficients * (mean_temperature - np.float64(reference))
    _finite_derived('network thermal strain', endpoint_thermal, mean_temperature, thermal_strain)
    if np.any(np.abs(endpoint_thermal) > 1e-5):
        raise ValueError('thermal strain exceeds network profile bound')
    zero_roots = [nid for nid in ids if prescribed.get(nid) == (0., 0., 0.)]
    root = ids.index(zero_roots[0]) if zero_roots else 0
    return _snapshot_core(graph, length_unit=length_unit, E=E, reference=reference,
        ids=ids, eids=eids, pos=pos, edges=edges, radii=radii, dofs=dofs, B=B,
        temperatures=temperatures, alpha=alpha, force=force, values=values,
        mask=mask, root=root, thermal_strain=thermal_strain, profile=NETWORK_PROFILE)


def _network_applicability(pos, edges, radii, displacement, total,
                           endpoint_temperatures, alpha, reference):
    """Exact cubic-interpolation diagnostics, not a stability criterion.

    Polynomial coefficients describe v(t), t=x/L. Slopes are dv/dx and
    endpoint curvature is d2v/dx2. All returned arrays follow input edge rows.
    """
    _finite_derived('network applicability inputs', pos, radii, displacement,
                    total, endpoint_temperatures, alpha, reference)
    if np.any(np.abs(total) > 1e-4):
        raise FloatingPointError('network mean total axial strain bound exceeded')
    if np.any(np.abs(displacement[:, 2]) > 1e-3):
        raise FloatingPointError('network nodal rotation bound exceeded')
    thermal = alpha[:, None] * (endpoint_temperatures - reference)
    _finite_derived('network endpoint thermal strain', thermal)
    if np.any(np.abs(thermal) > 1e-5):
        raise FloatingPointError('network endpoint thermal strain bound exceeded')
    rows = []
    for row, ((a, b), radius) in enumerate(zip(edges, radii)):
        direction = pos[b] - pos[a]
        length = np.linalg.norm(direction)
        ex = direction / length
        ey = np.array([-ex[1], ex[0]])
        vi, vj = displacement[[a, b], :2] @ ey
        thi, thj = displacement[[a, b], 2]
        delta = vj - vi
        coefficients = np.array([vi, length*thi,
            3.*delta - length*(2.*thi + thj),
            -2.*delta + length*(thi + thj)])
        _finite_derived('network Hermite polynomial coefficients', length, ex, ey, coefficients)
        locations = [0., 1.]
        if coefficients[3] != 0.:
            extremum = -coefficients[2] / (3.*coefficients[3])
            _finite_derived('network Hermite slope extremum', extremum)
            if 0. < extremum < 1.:
                locations.append(float(extremum))
        locations = np.array(sorted(locations))
        slopes = (coefficients[1] + 2.*coefficients[2]*locations
                  + 3.*coefficients[3]*locations**2) / length
        curvature = (2.*coefficients[2] + 6.*coefficients[3]*np.array([0., 1.])) / length**2
        chord_slope = delta / length
        fiber_bending = radius * curvature[:, None] * np.array([-1., 1.])[None, :]
        total_fiber = total[row] + fiber_bending
        elastic_fiber = total[row] - thermal[row, :, None] + fiber_bending
        _finite_derived('network Hermite applicability', locations, slopes, curvature,
                        chord_slope, total_fiber, elastic_fiber)
        if np.any(np.abs(slopes) > 1e-3):
            raise FloatingPointError('network Hermite slope bound exceeded')
        if np.any(np.abs(total_fiber) > 1e-4):
            raise FloatingPointError('network total extreme-fiber strain bound exceeded')
        if np.any(np.abs(elastic_fiber) > 1e-4):
            raise FloatingPointError('network elastic extreme-fiber strain bound exceeded')
        rows.append({'transverse_polynomial_normalized_x_m': coefficients.tolist(),
            'slope_sample_x_over_L': locations.tolist(), 'slope_rad': slopes.tolist(),
            'transverse_chord_slope_rad': float(chord_slope),
            'endpoint_curvature_per_m': curvature.tolist(),
            'endpoint_thermal_strain': thermal[row].tolist(),
            'endpoint_total_fiber_strain': total_fiber.tolist(),
            'endpoint_elastic_fiber_strain': elastic_fiber.tolist()})
    return {'model': 'linear_planar_welded_Euler_Bernoulli',
            'stability_qualified': False, 'fiber_signs': [-1, 1],
            'endpoint_order': ['node_i', 'node_j'], 'edge_rows': rows,
            'ceilings': {'thermal_strain': 1e-5, 'mean_total_axial_strain': 1e-4,
                         'rotation_and_slope_rad': 1e-3, 'extreme_fiber_strain': 1e-4}}
