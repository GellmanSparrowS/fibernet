"""History extension of BeamFrameFEM's existing 2-D sparse assembly.

No alternative graph, element solver, inertia or hidden regularization.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from fibernet.core.viscoelastic import LinearViscoelasticMaterial
from ._frame_linear_common import (
    _finite, _geometry, _histories, _check_supports, _solve_constrained_sparse,
    _reject_masked_values,
)

_STATE_SCHEMA = 'fibernet.viscoelastic_state/1'
_RESULT_SCHEMA = 'fibernet.viscoelastic_result/1'
_MAPPED_STATE = 'fibernet.viscoelastic_state/2'
_MAPPED_RESULT = 'fibernet.viscoelastic_result/2'
_MAPPED_MAX_EDGES = 256
_MAPPED_MAX_SAMPLES = 513  # Initial sample plus at most 512 advances.
_MAPPED_MAX_EDGE_SAMPLES = 20_000
_MAPPED_MAX_BRANCHES = 8
_LEDGER_KEYS = frozenset(('physical_dissipation', 'numerical_dissipation', 'work'))
_STATE_KEYS = frozenset(('schema', 'fingerprint', 'node_ids', 'edge_ids', 'units', 'time', 'step',
    'displacement', 'section_strain', 'viscous_strain', 'last_load', 'prescribed_values',
    'constraint_mask', 'cumulative', 'stored_energy', 'origin_energy',
    'last_stress', 'last_strain_rate', 'checksum'))
_UNITS = {'length': 'm', 'force': 'N', 'moment': 'N*m', 'time': 's',
          'rotation': 'rad', 'energy': 'J', 'axial_stress': 'Pa',
          'curvature': '1/m', 'curvature_stress': 'Pa/m'}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()




class ViscoelasticResult:
    """SI history result with detached data and a restartable final_state.

    displacements/reactions have node order node_ids and components
    ux/uy/rz and Fx/Fy/Mz. section_strain has edge order edge_ids and
    epsilon/kappa_at_Gauss1/kappa_at_Gauss2. Bending moments are at those
    Gauss points, not endpoint maxima; M=-I*sigma_kappa follows the existing
    BeamFrameFEM sign convention. A load result is not a material fit.
    """
    _arrays = ('times', 'displacements', 'reactions', 'section_strain',
               'axial_stress', 'axial_force', 'bending_moment', 'free_residual')

    def __init__(self, data):
        if not isinstance(data, dict) or data.get('schema') not in (_RESULT_SCHEMA, _MAPPED_RESULT):
            raise ValueError('unsupported viscoelastic result schema')
        mapped = data['schema'] == _MAPPED_RESULT
        required = set(self._arrays) | {'schema', 'node_ids', 'edge_ids', 'energies', 'metadata', 'final_state'}
        if set(data) != required:
            raise ValueError('invalid viscoelastic result fields')
        if mapped:
            _admit_mapped_result(data)
        # JSON validation rejects NaN and arbitrary objects; detach all values.
        self._data = json.loads(_json(data))
        t = np.asarray(data['times'], dtype=float)
        n, e = len(data['node_ids']), len(data['edge_ids'])
        if t.ndim != 1 or len(t) < 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('result times must be finite and increasing')
        shapes = _history_shapes(len(t), n, e)
        for key, shape in shapes.items():
            _finite(data[key], shape, key)
        if set(data['energies']) != {'stored', 'physical_dissipation', 'numerical_dissipation', 'work', 'balance_residual'}:
            raise ValueError('invalid energy fields')
        for key, values in data['energies'].items():
            _finite(values, (len(t),), key)
        state = data['final_state']
        if not isinstance(state, dict) or state.get('schema') != (_MAPPED_STATE if mapped else _STATE_SCHEMA):
            raise ValueError('invalid final state')
        check = dict(state)
        digest = check.pop('checksum', None)
        if digest != _hash(check) or state['time'] != float(t[-1]):
            raise ValueError('final state checksum/time mismatch')
        if (state['node_ids'] != data['node_ids'] or state['edge_ids'] != data['edge_ids'] or
                state['units'] != _UNITS or data['metadata'].get('units') != _UNITS or
                data['metadata'].get('fingerprint') != state['fingerprint']):
            raise ValueError('result/state identity mismatch')
        if not np.array_equal(_finite(state['displacement'],(n*3,),'final displacement').reshape(n,3),data['displacements'][-1]):
            raise ValueError('result/final displacement mismatch')
        if not np.array_equal(_finite(state['section_strain'],(e,3),'final strain'),data['section_strain'][-1]):
            raise ValueError('result/final strain mismatch')
        if state['stored_energy'] != data['energies']['stored'][-1] or any(
                state['cumulative'][key] != data['energies'][key][-1] for key in ('physical_dissipation','numerical_dissipation','work')):
            raise ValueError('result/final ledger mismatch')

    def __getattr__(self, name):
        if name in self._arrays:
            return np.array(self._data[name], dtype=float, copy=True)
        if name in ('node_ids', 'edge_ids', 'energies', 'metadata', 'final_state'):
            return copy.deepcopy(self._data[name])
        raise AttributeError(name)

    def to_dict(self):
        return copy.deepcopy(self._data)

    def save(self, path):
        """Atomically replace uniform-result JSON, creating missing parent directories."""
        if self._data['schema'] == _MAPPED_RESULT:
            raise ValueError('mapped history persistence is unsupported; use in-memory continuation')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=path.parent, prefix='.'+path.name+'.', delete=False) as stream:
                temporary = stream.name
                stream.write(_json(self._data)+'\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        # Legacy reader still parses as before; there is no supported mapped file format.
        if isinstance(data, dict) and data.get('schema') == _MAPPED_RESULT:
            raise ValueError('mapped history persistence is unsupported; use in-memory continuation')
        return cls(data)




def _assemble(dofs, B, metric, stress, ndof):
    local = np.einsum('eji,ej->ei', B, metric * stress)
    force = np.zeros(ndof)
    np.add.at(force, dofs.ravel(), local.ravel())
    return force


def _section(B, dofs, u):
    return np.einsum('eij,ej->ei', B, u[dofs])


def _residual(reaction, internal, applied, free, length_scale):
    """Normalize moment rows by specimen member length to compare forces."""
    factors = np.tile([1.,1.,length_scale], len(reaction)//3)
    scale = max(float(np.max(np.abs(internal/factors))),
                float(np.max(np.abs(applied/factors))), 1e-30)
    return float(np.max(np.abs((reaction/factors)[free]))/scale) if free.size else 0.


def _stored(law, s, v, metric):
    density = law.equilibrium_modulus * s**2
    for j, (e, _) in enumerate(law.branches):
        density = density + e*(s-v[j])**2
    return float(0.5*np.sum(metric*density))


def _history_shapes(samples, nodes, edges):
    return {'times': (samples,), 'displacements': (samples, nodes, 3),
            'reactions': (samples, nodes, 3), 'section_strain': (samples, edges, 3),
            'axial_stress': (samples, edges), 'axial_force': (samples, edges),
            'bending_moment': (samples, edges, 2), 'free_residual': (samples,)}


def _mapped_numbers(value, shape, name):
    if shape:
        if type(value) is not list or len(value) != shape[0]:
            raise ValueError(f'{name} requires a list with shape {shape}')
        for item in value:
            _mapped_numbers(item, shape[1:], name)
    else:
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f'{name} requires finite real numbers')


def _mapped_material_records(records, edge_ids):
    if type(records) is not list or len(records) != len(edge_ids):
        raise ValueError('invalid mapped material result identity')
    laws = []
    for pair, edge_id in zip(records, edge_ids):
        if (type(pair) is not list or len(pair) != 2 or type(pair[0]) is not int
                or pair[0] != edge_id or type(pair[1]) is not dict):
            raise ValueError('invalid mapped material result identity')
        law = pair[1]
        branches = law.get('branches')
        if type(branches) is not list or len(branches) > _MAPPED_MAX_BRANCHES:
            raise ValueError('mapped law allows at most eight branches')
        _mapped_numbers(branches, (len(branches), 2), 'mapped law branches')
        for key in ('equilibrium_modulus', 'parallel_viscosity'):
            _mapped_numbers(law.get(key), (), 'mapped law ' + key)
        checked = LinearViscoelasticMaterial.from_dict(law)
        if checked.to_dict() != law:
            raise ValueError('noncanonical mapped material')
        laws.append(checked)
    return _law_groups(laws)


def _admit_mapped_state(state, node_ids, edge_ids, material):
    if (type(state) is not dict or state.keys() != _STATE_KEYS | {'material_by_edge'}
            or state['schema'] != _MAPPED_STATE):
        raise ValueError('invalid mapped restart fields/schema')
    if (type(state['step']) is not int or state['step'] < 0
            or any(type(state[key]) is not str or len(state[key]) != 64
                   for key in ('checksum', 'fingerprint'))):
        raise ValueError('invalid mapped restart time/step or checksum fields')
    for key, ids in (('node_ids', node_ids), ('edge_ids', edge_ids)):
        column = state[key]
        if (type(column) is not list or len(column) != len(ids)
                or any(type(i) is not int for i in column) or column != ids):
            raise ValueError('invalid mapped restart stable IDs')
    # Validate the state's records before equality: bool/int equality alone
    # cannot prove the field is a real number in a canonical material.
    groups = _mapped_material_records(state['material_by_edge'], edge_ids)
    if type(state['units']) is not dict or state['units'] != _UNITS or state['material_by_edge'] != material:
        raise ValueError('restart material/geometry/identity/support mismatch')
    ndof, ne = 3*len(node_ids), len(edge_ids)
    for key in ('time', 'stored_energy', 'origin_energy'):
        _mapped_numbers(state[key], (), f'mapped restart {key}')
    mask = state['constraint_mask']
    if type(mask) is not list or len(mask) != ndof or any(type(x) is not bool for x in mask):
        raise ValueError('invalid mapped restart constraint mask')
    for key in ('displacement', 'last_load', 'prescribed_values'):
        _mapped_numbers(state[key], (ndof,), 'mapped restart ' + key)
    for key in ('section_strain', 'last_stress', 'last_strain_rate'):
        _mapped_numbers(state[key], (ne, 3), 'mapped restart ' + key)
    cumulative = state['cumulative']
    if type(cumulative) is not dict or cumulative.keys() != _LEDGER_KEYS:
        raise ValueError('invalid mapped restart energy fields')
    for key in _LEDGER_KEYS:
        _mapped_numbers(cumulative[key], (), 'mapped restart ' + key)
    values = state['viscous_strain']
    if type(values) is not list or len(values) != len(groups):
        raise ValueError('invalid mapped viscous-state groups')
    for (law, rows), value in zip(groups, values):
        _mapped_numbers(value, (len(law.branches), len(rows), 3), 'mapped restart viscous strain')


def _admit_mapped_result(data):
    metadata = data['metadata']
    text_fields = ('backend', 'solver', 'kinematics', 'quadrature', 'integration',
                   'bending_moment_sign', 'work_convention', 'graph_length_unit',
                   'material_assignment', 'initialization', 'fingerprint')
    if (type(metadata) is not dict or metadata.keys() != set(text_fields) | {
            'units', 'material', 'residual_moment_normalization_length_m',
            'graph_static_moduli_used', 'numerical_regularization', 'assumptions'}
            or any(type(metadata[key]) is not str or len(metadata[key]) > 128 for key in text_fields)
            or type(metadata['units']) is not dict or metadata['units'] != _UNITS
            or type(metadata['graph_static_moduli_used']) is not bool
            or type(metadata['numerical_regularization']) is not bool
            or type(metadata['assumptions']) is not list
            or metadata['assumptions'] != ['small_strain', 'small_rotation', 'straight_members', 'welded_joints']):
        raise ValueError('invalid mapped result metadata')
    _mapped_numbers(metadata['residual_moment_normalization_length_m'], (),
                    'mapped result residual moment normalization length')
    nodes, edges, times = (data[key] for key in ('node_ids', 'edge_ids', 'times'))
    if not all(type(column) is list for column in (nodes, edges, times)):
        raise ValueError('invalid mapped result dimensions')
    n, e, nt = len(nodes), len(edges), len(times)
    _admit_mapped_size(n, e, nt)
    for ids in (nodes, edges):
        if any(type(i) is not int or i < 0 for i in ids) or ids != sorted(set(ids)):
            raise ValueError('invalid mapped result stable IDs')
    energies = data['energies']
    if type(energies) is not dict or energies.keys() != _LEDGER_KEYS | {'stored', 'balance_residual'}:
        raise ValueError('invalid mapped result energy fields')
    records = metadata.get('material')
    if metadata.get('material_assignment') != 'stable_edge_ids':
        raise ValueError('invalid mapped material result identity')
    _mapped_material_records(records, edges)
    _admit_mapped_state(data['final_state'], nodes, edges, records)
    for key, shape in _history_shapes(nt, n, e).items():
        _mapped_numbers(data[key], shape, 'mapped result ' + key)
    for key, value in energies.items():
        _mapped_numbers(value, (nt,), 'mapped result ' + key)


def _admit_mapped_size(nodes, edges, samples):
    if (not 1 <= edges <= _MAPPED_MAX_EDGES
            or not 2 <= samples <= _MAPPED_MAX_SAMPLES
            or edges*samples > _MAPPED_MAX_EDGE_SAMPLES):
        raise ValueError('mapped history exceeds experimental edge or sample budget')
    if not 2 <= nodes <= 2*edges:
        raise ValueError('invalid mapped node count; isolated nodes are unsupported')


def _law_groups(laws):
    groups = {}
    for edge, law in enumerate(laws):
        groups.setdefault(_json(law.to_dict()), (law, []))[1].append(edge)
    return tuple((law, np.array(edges, dtype=np.intp)) for law, edges in groups.values())


def _decode_group_state(groups, state):
    if not isinstance(state, list) or len(state) != len(groups):
        raise ValueError('invalid mapped viscous-state groups')
    return [_finite(value, (len(law.branches), len(rows), 3), 'restart viscous strain')
            for (law, rows), value in zip(groups, state)]


class _MappedLaws:
    def __init__(self, mapping, edge_ids, dofs, B, metric):
        if any(isinstance(e, (bool, np.bool_, np.timedelta64))
               or not isinstance(e, (int, np.integer)) or e < 0 for e in mapping):
            raise ValueError('material keys must be stable nonnegative integer edge IDs')
        if set(mapping) != set(edge_ids):
            raise ValueError('material mapping must cover each stable edge ID exactly')
        laws = [mapping[e] for e in edge_ids]
        if any(type(law) is not LinearViscoelasticMaterial for law in laws):
            raise ValueError('each edge law must be a LinearViscoelasticMaterial')
        if any(len(law.branches) > _MAPPED_MAX_BRANCHES for law in laws):
            raise ValueError('prototype admits at most eight branches per edge law')
        self.record = [[e, law.to_dict()] for e, law in zip(edge_ids, laws)]
        self.groups = _law_groups(laws)
        self.gram = B.transpose(0, 2, 1) @ (metric[:, :, None] * B)
        if not np.isfinite(self.gram).all():
            raise ValueError('nonfinite mapped section Gram')
        shape = self.gram.shape
        self.row = np.broadcast_to(dofs[:, :, None], shape).ravel()
        self.col = np.broadcast_to(dofs[:, None, :], shape).ravel()

    def zero(self):
        return [np.zeros((len(law.branches), len(rows), 3)) for law, rows in self.groups]

    def linearize(self, s, v, dt, ndof):
        memory = np.empty_like(s)
        tangent = np.empty(len(s))
        for (law, rows), state in zip(self.groups, v):
            a, tangent[rows] = law.coefficients(dt)
            local = law.parallel_viscosity / dt * s[rows]
            for j, (e, _) in enumerate(law.branches):
                local = local + e*a[j]*state[j]
            memory[rows] = local
        values = (tangent[:, None, None] * self.gram).ravel()
        return coo_matrix((values, (self.row, self.col)), shape=(ndof, ndof)).tocsr(), memory

    def trial(self, next_s, s, v, dt):
        fields = {key: np.empty_like(s) for key in ('stress', 'stored_energy_density',
                  'physical_dissipation_density', 'numerical_dissipation_density')}
        memory = []
        for (law, rows), previous in zip(self.groups, v):
            t = law.trial(next_s[rows], s[rows], previous, dt)
            for key in fields:
                fields[key][rows] = t[key]
            memory.append(t['viscous_strain'])
        return dict(fields, viscous_strain=memory)

    def check_restart(self, s, v, strain_rate, metric):
        stress = np.empty_like(s)
        stored = 0.
        for (law, rows), state in zip(self.groups, v):
            local = s[rows]
            stress[rows] = law.equilibrium_modulus*local + law.parallel_viscosity*strain_rate[rows]
            for j, (e, _) in enumerate(law.branches):
                stress[rows] += e*(local-state[j])
            stored += _stored(law, local, state, metric[rows])
        return stress, stored


def solve_history(solver, graph, *, material, times, constraints, loads=None,
                  length_unit, initial_state=None):
    """Implementation of BeamFrameFEM.history; see VISCOELASTIC_NETWORK.md."""
    if not isinstance(material, (LinearViscoelasticMaterial, dict)):
        raise ValueError('material must be a LinearViscoelasticMaterial or stable edge ID dictionary')
    is_mapped = isinstance(material, dict)
    if is_mapped:
        if not isinstance(times, (list, tuple, np.ndarray)) or (
                isinstance(times, np.ndarray) and times.ndim != 1):
            raise ValueError('mapped history times must be a 1-D list, tuple or array')
        _admit_mapped_size(len(graph.nodes), len(graph.edges), len(times))
        _reject_masked_values(times, 'times')
        samples = []
        for value in times:
            if (not isinstance(value, (int, float, np.integer, np.floating))
                    or isinstance(value, (bool, np.bool_, np.timedelta64))):
                raise ValueError('mapped history times require finite real scalar samples')
            try:
                sample = float(value)
            except OverflowError as exc:
                raise ValueError('mapped history times require finite real scalar samples') from exc
            if not math.isfinite(sample):
                raise ValueError('mapped history times require finite real scalar samples')
            samples.append(sample)
        t = np.array(samples, dtype=float)
    else:
        _reject_masked_values(times, 'times')
        t = np.asarray(times)
        t = _finite(times, t.shape, 'times')
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('times must contain at least two finite strictly increasing samples')
    node_ids, edge_ids, pos, edges, radii, dofs, B, metric = _geometry(graph, length_unit)
    mapped = _MappedLaws(material, edge_ids, dofs, B, metric) if is_mapped else None
    state_schema = _MAPPED_STATE if is_mapped else _STATE_SCHEMA
    result_schema = _MAPPED_RESULT if is_mapped else _RESULT_SCHEMA
    material_record = mapped.record if is_mapped else material.to_dict()
    force, prescribed, mask = _histories(t, node_ids, loads, constraints)
    _check_supports(pos, edges, mask)
    n, ne, ndof = len(node_ids), len(edge_ids), 3*len(node_ids)
    free = np.flatnonzero(~mask); fixed = np.flatnonzero(mask)
    length_scale=float(np.max(np.linalg.norm(pos[edges[:,1]]-pos[edges[:,0]],axis=1)))
    # Reuse the original sparse assembler. E=1 yields a geometry/section matrix.
    if not is_mapped:
        assembler = type(solver)(E=1.0, nu=solver.nu)
        K0, selected = assembler.build_stiffness_2d(edges.T, pos, radii, deduplicate=False)
        if list(selected) != list(range(ne)) or not np.isfinite(K0.data).all():
            raise ValueError('unsupported element assembly')
    identity = {'node_ids': node_ids, 'edge_ids': edge_ids, 'positions_m': pos.tolist(),
                'edges': edges.tolist(), 'radii_m': radii.tolist(),
                'material': material_record, 'constraint_mask': mask.tolist(),
                'kinematics': 'linear_2d_euler_bernoulli_two_gauss/1'}
    fingerprint = _hash(identity)
    u = np.zeros(ndof); s = np.zeros((ne,3))
    v = mapped.zero() if is_mapped else np.zeros((len(material.branches),ne,3))
    cumulative = {'physical_dissipation': 0., 'numerical_dissipation': 0., 'work': 0.}
    step = 0; stored = 0.; origin_energy = 0.
    initial_stress = np.zeros((ne,3)); initial_reaction = np.zeros(ndof); initial_residual=0.
    last_rate = np.zeros((ne,3))
    if initial_state is None:
        if np.any(force[0] != 0) or np.any(prescribed[0] != 0):
            raise ValueError('fresh history starts at zero load and zero prescribed displacement')
    else:
        if is_mapped:
            _admit_mapped_state(initial_state, node_ids, edge_ids, material_record)
        state = copy.deepcopy(initial_state)
        fields = _STATE_KEYS | {'material_by_edge'} if is_mapped else _STATE_KEYS
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError('invalid restart fields')
        check = dict(state); checksum = check.pop('checksum')
        if checksum != _hash(check) or state['schema'] != state_schema:
            raise ValueError('restart checksum/schema mismatch')
        if (state['fingerprint'] != fingerprint or state['node_ids'] != node_ids or
                state['edge_ids'] != edge_ids or state['units'] != _UNITS or state['constraint_mask'] != mask.tolist()
                or (is_mapped and state['material_by_edge'] != material_record)):
            raise ValueError('restart material/geometry/identity/support mismatch')
        if state['time'] != float(t[0]) or isinstance(state['step'], bool) or not isinstance(state['step'], int) or state['step'] < 0:
            raise ValueError('restart time/step mismatch')
        u = _finite(state['displacement'], (ndof,), 'restart displacement')
        s = _finite(state['section_strain'], (ne,3), 'restart strain')
        initial_stress = _finite(state['last_stress'], (ne,3), 'restart stress')
        last_rate = _finite(state['last_strain_rate'], (ne,3), 'restart strain rate')
        if is_mapped:
            v = _decode_group_state(mapped.groups, state['viscous_strain'])
            reconstructed, mapped_stored = mapped.check_restart(s, v, last_rate, metric)
        else:
            v = _finite(state['viscous_strain'], (len(material.branches),ne,3), 'restart viscous strain')
            reconstructed = material.equilibrium_modulus*s + material.parallel_viscosity*last_rate
            for j,(e,_) in enumerate(material.branches):
                reconstructed = reconstructed + e*(s-v[j])
        if not np.allclose(reconstructed,initial_stress,rtol=1e-12,atol=1e-24):
            raise ValueError('restart stress is incompatible with material history')
        if (not np.array_equal(_finite(state['last_load'], (ndof,), 'restart load'),force[0]) or
                not np.array_equal(_finite(state['prescribed_values'], (ndof,), 'restart prescribed values'),prescribed[0]) or
                not np.array_equal(u[fixed], prescribed[0,fixed])):
            raise ValueError('restart initial controls differ from committed controls')
        if not np.allclose(s, _section(B,dofs,u), rtol=1e-12, atol=1e-14):
            raise ValueError('restart strain is incompatible with displacement')
        if not isinstance(state['cumulative'],dict) or set(state['cumulative']) != set(cumulative):
            raise ValueError('invalid cumulative restart ledger')
        cumulative = {key: float(value) for key,value in state['cumulative'].items()}
        stored, origin_energy = float(state['stored_energy']), float(state['origin_energy'])
        if not all(math.isfinite(x) for x in [stored,origin_energy,*cumulative.values()]):
            raise ValueError('nonfinite restart ledger')
        if stored < 0 or origin_energy < 0 or min(cumulative['physical_dissipation'],cumulative['numerical_dissipation']) < 0:
            raise ValueError('negative restart energy')
        reconstructed_stored = mapped_stored if is_mapped else _stored(material,s,v,metric)
        if not math.isclose(stored,reconstructed_stored,rel_tol=1e-10,abs_tol=1e-24):
            raise ValueError('restart stored energy mismatch')
        balance=cumulative['work']-(stored-origin_energy)-cumulative['physical_dissipation']-cumulative['numerical_dissipation']
        escale=max(abs(cumulative['work']),stored,origin_energy,cumulative['physical_dissipation'],cumulative['numerical_dissipation'],1e-30)
        if abs(balance)>1e-8*escale:
            raise ValueError('restart cumulative ledger is inconsistent')
        step = state['step']
        initial_internal = _assemble(dofs,B,metric,initial_stress,ndof)
        initial_reaction = initial_internal-force[0]
        initial_residual = _residual(initial_reaction,initial_internal,force[0],free,length_scale)
        if initial_residual > 1e-8:
            raise ValueError('restart is not in equilibrium for its committed loads')
    history_u=[u.reshape(n,3).copy()]; history_r=[initial_reaction.reshape(n,3).copy()]
    history_s=[s.copy()]; history_stress=[initial_stress.copy()]; residuals=[initial_residual]
    energy = {'stored':[stored], 'physical_dissipation':[cumulative['physical_dissipation']],
              'numerical_dissipation':[cumulative['numerical_dissipation']],
              'work':[cumulative['work']],
              'balance_residual':[cumulative['work']-(stored-origin_energy)-cumulative['physical_dissipation']-cumulative['numerical_dissipation']]}
    for index in range(1,len(t)):
        dt=float(t[index]-t[index-1])
        if is_mapped:
            K, memory = mapped.linearize(s, v, dt, ndof)
        else:
            a, tangent=material.coefficients(dt)
            memory = material.parallel_viscosity/dt*s
            for j,(e,_) in enumerate(material.branches):
                memory = memory + e*a[j]*v[j]
            K = tangent*K0
        rhs = force[index] + _assemble(dofs,B,metric,memory,ndof)
        if not np.isfinite(K.data).all() or not np.isfinite(rhs).all():
            raise FloatingPointError('nonfinite history system')
        candidate = _solve_constrained_sparse(
            K, rhs, prescribed[index], mask, solve=spsolve)
        next_s = _section(B,dofs,candidate)
        trial = (mapped.trial(next_s,s,v,dt) if is_mapped
                 else material.trial(next_s,s,v,dt))
        internal = _assemble(dofs,B,metric,trial['stress'],ndof)
        reaction = internal-force[index]
        residual=_residual(reaction,internal,force[index],free,length_scale)
        if residual > 1e-8:
            raise FloatingPointError('free-DOF equilibrium residual exceeds 1e-8')
        du=candidate-u
        work=float(np.dot(force[index],du)+np.dot(reaction[fixed],du[fixed]))
        new_stored=float(np.sum(metric*trial['stored_energy_density']))
        physical=float(np.sum(metric*trial['physical_dissipation_density']))
        numerical=float(np.sum(metric*trial['numerical_dissipation_density']))
        balance=work-(new_stored-stored)-physical-numerical
        escale=max(abs(work),abs(new_stored),abs(stored),physical,numerical,1e-30)
        if not all(math.isfinite(x) for x in (work,new_stored,physical,numerical,balance)) or abs(balance)>1e-8*escale:
            raise FloatingPointError('discrete history energy identity failed')
        # Transaction boundary: no input state or graph has been changed.
        last_rate=(next_s-s)/dt
        u,s,v,stored=candidate,next_s,trial['viscous_strain'],new_stored
        cumulative['work']+=work; cumulative['physical_dissipation']+=physical
        cumulative['numerical_dissipation']+=numerical; step+=1
        history_u.append(u.reshape(n,3).copy()); history_r.append(reaction.reshape(n,3).copy())
        history_s.append(s.copy()); history_stress.append(trial['stress'].copy()); residuals.append(residual)
        energy['stored'].append(stored)
        for key in cumulative: energy[key].append(cumulative[key])
        energy['balance_residual'].append(cumulative['work']-(stored-origin_energy)-cumulative['physical_dissipation']-cumulative['numerical_dissipation'])
    state={'schema':state_schema,'fingerprint':fingerprint,'node_ids':node_ids,'edge_ids':edge_ids,
           'units':_UNITS,'time':float(t[-1]),'step':step,'displacement':u.tolist(),
           'section_strain':s.tolist(),'viscous_strain':([x.tolist() for x in v] if is_mapped else v.tolist()),
           'last_load':force[-1].tolist(),
           'prescribed_values':prescribed[-1].tolist(),'constraint_mask':mask.tolist(),
           'cumulative':cumulative,'stored_energy':stored,'origin_energy':origin_energy,
           'last_stress':history_stress[-1].tolist(),'last_strain_rate':last_rate.tolist()}
    if is_mapped:
        state['material_by_edge'] = material_record
    state['checksum']=_hash(state)
    stress=np.array(history_stress); A=math.pi*radii**2; I=math.pi*radii**4/4
    return ViscoelasticResult({'schema':result_schema,'times':t.tolist(),'node_ids':node_ids,
        'edge_ids':edge_ids,'displacements':np.array(history_u).tolist(),
        'reactions':np.array(history_r).tolist(),'section_strain':np.array(history_s).tolist(),
        'axial_stress':stress[:,:,0].tolist(),'axial_force':(stress[:,:,0]*A).tolist(),
        'bending_moment':(-stress[:,:,1:]*I[None,:,None]).tolist(),
        'free_residual':residuals,'energies':energy,'final_state':state,
        'metadata':{'units':_UNITS,'material':material_record,'backend':'fem',
            'solver':'BeamFrameFEM.history','kinematics':'linear_2d_euler_bernoulli',
            'quadrature':'two_point_Gauss_bending','integration':'backward_euler',
            'bending_moment_sign':'M=-I*generalized_curvature_stress',
            'residual_moment_normalization_length_m':length_scale,
            'work_convention':'right_endpoint_load_plus_constraint_work',
            'graph_length_unit':length_unit,
            'material_assignment':('stable_edge_ids' if is_mapped else 'explicit_uniform'),
            'graph_static_moduli_used':False,'numerical_regularization':False,
            'assumptions':['small_strain','small_rotation','straight_members','welded_joints'],
            'initialization':'restart' if initial_state is not None else 'zero_unloaded',
            'fingerprint':fingerprint}})
