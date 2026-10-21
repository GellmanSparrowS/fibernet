"""Host-owned axial material history for the existing Taichi integrator.

There is exactly one constitutive update: LinearViscoelasticMaterial.trial.
This module does not integrate nodes, infer properties, or implement a second law.
"""
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
import copy
import math
import numpy as np
from fibernet.core.viscoelastic import LinearViscoelasticMaterial

SCHEMA = 'native_axial_viscoelastic/1'
SI = dict(length='m', time='s', mass='kg', frame='world_xyz')
DYNAMIC_FIELDS = ('external_force_work_J', 'numerical_drag_energy_change_J',
    'projection_kinetic_energy_change_J', 'projection_force_work_J',
    'kick_drift_residual_J', 'old_stress_endpoint_residual_J')
ENERGY_FIELDS = ('stored_energy_J', 'physical_dissipation_J',
                'be_numerical_dissipation_J', 'endpoint_work_J')


def material_options(method):
    """Reject even an explicitly supplied legacy default stiffness in this mode."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if kwargs.get('material') is not None and ('stiffness' in kwargs or len(args) > 4):
            raise ValueError('material conflicts with explicit stiffness; use reference_areas_by_id')
        return method(self, *args, **kwargs)
    return wrapped


def _array(value, shape, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(name + ' must be unmasked')
    a = np.asarray(value)
    if shape and shape[0] == 0 and a.shape == (0,):
        a = np.empty(shape, dtype=np.float64)
    if a.shape != shape or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ValueError(name + ' must have finite numeric shape ' + str(shape))
    a = np.array(a, dtype=np.float64, copy=True)
    if not np.isfinite(a).all():
        raise ValueError(name + ' exceeds float64 range')
    return a


def _ids(ids, name):
    values = list(ids)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError(name + ' must be strict integer IDs')
    if len(set(values)) != len(values):
        raise ValueError(name + ' must be unique')
    return [int(x) for x in values]


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + ' must be a positive finite real number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be positive and finite')
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class AxialMaterialObservation:
    """Completed x/v and committed material state at the same time level."""
    schema: str
    step: int
    time: float
    node_ids: tuple
    units: tuple
    positions: np.ndarray
    velocities: np.ndarray
    material_state: object


def material_observation(event, state):
    if state['step'] != event.step or state['time'] != event.time:
        raise ValueError('material observation must match the completed native state')
    return AxialMaterialObservation('native_axial_viscoelastic_observer/1',
        event.step, event.time, event.node_ids, event.units,
        event.positions, event.velocities, _freeze(state))


class DynamicAxialMaterial:
    """One transaction owner; observations and exports never advance history."""
    def __init__(self, graph, node_ids, elements, rest, areas_by_id, material,
                 max_abs_strain, units, masses, dt, steps, save_interval, positions, velocities):
        if type(material) is not LinearViscoelasticMaterial:
            raise TypeError('material must be a uniform LinearViscoelasticMaterial')
        if material.parallel_viscosity != 0:
            raise ValueError('dynamic parallel_viscosity > 0 is not qualified; use Maxwell/SLS branches')
        if units != SI:
            raise ValueError('dynamic material requires SI state_units: m, s, kg, world_xyz')
        self.node_ids = _ids(node_ids, 'node IDs')
        self.edge_ids = _ids(graph.edges, 'edge IDs')
        n = len(self.edge_ids)
        if not n:
            raise ValueError('dynamic material requires at least one edge')
        if type(areas_by_id) is not dict or set(_ids(areas_by_id, 'area IDs')) != set(self.edge_ids):
            raise ValueError('reference_areas_by_id must cover exactly the stable edge IDs')
        self.area = np.array([_positive(areas_by_id[e], 'reference area') for e in self.edge_ids])
        self.rest = _array(rest, (n,), 'rest lengths')
        if np.any(self.rest <= 1e-12):
            raise ValueError('material rest lengths must exceed the native direction cutoff 1e-12 m')
        with np.errstate(over='raise', under='ignore', invalid='raise'):
            self.volume = self.area * self.rest
        if not np.isfinite(self.volume).all() or np.any(self.volume <= 0):
            raise ValueError('reference volumes A0 L0 must remain finite and positive')
        self.elements = np.array(elements, dtype=np.int64, copy=True)
        self.material = material
        self.bound = _positive(max_abs_strain, 'max_abs_strain')
        self.dt = _positive(dt, 'dt')
        if not math.isfinite(self.dt * steps):
            raise ValueError('material horizon must be finite')
        material.coefficients(self.dt)  # preflight only, no state/history advance
        # Bound stored numeric values before any native allocation/iteration.
        frames = 3 + steps // save_interval  # initial/sampled/final plus duplicated final record
        self.planned_output_values = (n*(len(material.branches)+12)+64+9*len(node_ids))*frames + n*8 + len(node_ids)*16
        if self.planned_output_values > 1000000:
            raise ValueError('dynamic material sampled output exceeds 1,000,000 numeric values')
        self.masses = _array(masses, (len(node_ids),), 'masses')
        if np.any(self.masses <= 0):
            raise ValueError('material masses must be positive')
        self.x = _array(positions, (len(node_ids), 3), 'initial positions')
        self.initial_x = self.x.copy()
        self.v = _array(velocities, self.x.shape, 'initial velocities')
        self.strain = self.strains(self.x)
        if np.any(self.strain != 0):
            raise ValueError('initial material state must be relaxed at exact geometric rest')
        self.viscous = np.zeros((len(material.branches), n))
        self.stress = np.zeros(n)
        self.stored = np.zeros(n)
        self.step = 0
        self.cumulative = {k: np.zeros(n) for k in ENERGY_FIELDS[1:]}
        self.dynamic_cumulative = {k: 0. for k in DYNAMIC_FIELDS}
        self.initial_kinetic = self.kinetic(self.v)
        self.current = self._state(0., {k: np.zeros(n) for k in ENERGY_FIELDS[1:]}, None)
        self.samples = [copy.deepcopy(self.current)]

    def strains(self, positions):
        x = _array(positions, self.x.shape, 'completed positions')
        lengths = np.array([math.dist(x[i], x[j]) for i, j in self.elements])
        if np.any(lengths <= 1e-12):
            raise ValueError('dynamic material edge crossed the native direction cutoff')
        strain = (lengths - self.rest) / self.rest
        if not np.isfinite(strain).all() or np.any(np.abs(strain) > self.bound):
            raise ValueError('completed engineering axial strain exceeds max_abs_strain')
        return strain

    def kinetic(self, velocity):
        # Reuse the backend's exponent-scaled kinetic diagnostic; avoid false
        # zeros or overflow from m*v*v ordering. No additional physical law.
        from .accelerated import _checked_kinetic_energy
        return _checked_kinetic_energy(self.masses, velocity)

    def _state(self, interval_dt, increments, dynamic):
        return dict(schema='native_axial_material_state/1', step=self.step, time=self.step*self.dt,
            interval_dt=interval_dt, time_level='completed_projected_t_n',
            edge_ids=list(self.edge_ids), strain=self.strain.tolist(),
            viscous_strain=self.viscous.tolist(), stress_Pa=self.stress.tolist(),
            stored_energy_J=self.stored.tolist(),
            increments={k: v.tolist() for k, v in increments.items()},
            cumulative={k: v.tolist() for k, v in self.cumulative.items()},
            kinetic_energy_J=self.kinetic(self.v), dynamic_interval=dynamic,
            dynamic_cumulative=dict(self.dynamic_cumulative))

    def prepare(self, positions, velocities, kick_velocity, drag_velocity, drift_positions, external_force):
        """Pure candidate construction; caller uploads force then commits this record."""
        x = _array(positions, self.x.shape, 'completed positions')
        v = _array(velocities, self.v.shape, 'completed velocities')
        vk = _array(kick_velocity, self.v.shape, 'native kick velocities')
        vd = _array(drag_velocity, self.v.shape, 'native drag velocities')
        xd = _array(drift_positions, self.x.shape, 'native drift positions')
        ext = _array(external_force, self.x.shape, 'applied external forces')
        strain = self.strains(x)
        # The only history update, exactly once per actual completed interval.
        trial = self.material.trial(strain, self.strain, self.viscous, self.dt)
        increments = {
            'physical_dissipation_J': trial['physical_dissipation_density'] * self.volume,
            'be_numerical_dissipation_J': trial['numerical_dissipation_density'] * self.volume,
            'endpoint_work_J': trial['endpoint_work_density'] * self.volume}
        stored = trial['stored_energy_density'] * self.volume
        force = self.area * trial['stress']
        # Native stage snapshots give the actual combined kick impulse. This is
        # a signed binary64 diagnostic, not a continuum or exact energy theorem.
        impulse_force = self.masses[:, None] * (vk-self.v) / self.dt
        drift_dx, projected_dx = xd-self.x, x-xd
        total_work = float(np.sum(impulse_force * drift_dx))
        external_work = float(np.sum(ext * (x-self.x)))
        internal_work = float(np.sum((impulse_force-ext) * (x-self.x)))
        dynamic = dict(
            external_force_work_J=external_work,
            numerical_drag_energy_change_J=self.kinetic(vd)-self.kinetic(vk),
            projection_kinetic_energy_change_J=self.kinetic(v)-self.kinetic(vd),
            projection_force_work_J=float(np.sum(impulse_force * projected_dx)),
            kick_drift_residual_J=self.kinetic(vk)-self.kinetic(self.v)-total_work,
            old_stress_endpoint_residual_J=internal_work+float(np.sum(increments['endpoint_work_J'])))
        dynamic['constitutive_identity_residual_J'] = float(np.sum(
            stored-self.stored+increments['physical_dissipation_J']+
            increments['be_numerical_dissipation_J']-increments['endpoint_work_J']))
        balance = (external_work+dynamic['numerical_drag_energy_change_J']+
            dynamic['projection_kinetic_energy_change_J']-dynamic['projection_force_work_J']+
            dynamic['kick_drift_residual_J']+dynamic['old_stress_endpoint_residual_J'])
        dynamic['dynamic_identity_residual_J'] = (self.kinetic(v)-self.kinetic(self.v)+
            float(np.sum(stored-self.stored+increments['physical_dissipation_J']+
                         increments['be_numerical_dissipation_J']))-balance)
        values = [stored, force, trial['stress'], trial['viscous_strain'], *increments.values(), list(dynamic.values())]
        if not all(np.isfinite(a).all() for a in values):
            raise FloatingPointError('nonfinite dynamic material transaction')
        return dict(x=x, v=v, strain=strain, viscous=trial['viscous_strain'],
                    stress=trial['stress'], stored=stored, increments=increments, dynamic=dynamic, force=force)

    def commit(self, candidate):
        # No user callbacks or calculations that can fail between assignments.
        cumulative = {k: self.cumulative[k] + v for k, v in candidate['increments'].items()}
        dynamic_cumulative = {k: v + candidate['dynamic'][k] for k, v in self.dynamic_cumulative.items()}
        if not all(np.isfinite(a).all() for a in [*cumulative.values(), list(dynamic_cumulative.values())]):
            raise FloatingPointError('nonfinite cumulative material transaction')
        for name in ('x', 'v', 'strain', 'viscous', 'stress', 'stored'):
            setattr(self, name, candidate[name])
        self.cumulative, self.dynamic_cumulative = cumulative, dynamic_cumulative
        self.step += 1
        self.current = self._state(self.dt, candidate['increments'], candidate['dynamic'])

    def snapshot(self):
        return copy.deepcopy(self.current)

    def sample(self):
        if self.samples[-1]['step'] != self.step:
            self.samples.append(self.snapshot())

    def result(self, graph, fixed_rows, external_force, drag):
        self.sample()
        return dict(schema=SCHEMA, material=self.material.to_dict(),
            units=dict(SI, stress='Pa', area='m^2', energy='J', strain='1'),
            node_ids=list(self.node_ids), edge_ids=list(self.edge_ids),
            edge_nodes=[[self.node_ids[i], self.node_ids[j]] for i, j in self.elements],
            initial_positions=self.initial_x.tolist(),
            reference_areas_m2=self.area.tolist(), rest_lengths_m=self.rest.tolist(),
            masses_kg=self.masses.tolist(), fixed_node_ids=list(dict.fromkeys(self.node_ids[i] for i in fixed_rows)),
            constant_external_force_N=np.asarray(external_force).tolist(), drag_per_s=float(drag),
            max_abs_strain=self.bound, dt=self.dt, completed_steps=self.step,
            initialization='relaxed_zero_strain_zero_branch_history_at_geometric_rest',
            force_time_level='cached_sigma_n_at_current_direction_before_kick',
            update_time_level='one_trial_after_drift_and_projection_before_completed_observation',
            energy_scope='stored axial material energy only; contact potential and kinetic energy separate',
            dynamic_work_scope='endpoint work is constitutive; external work includes contact force work when present; signed residuals are not dissipation',
            restart_scope='full material state serialized; this version refuses resume',
            samples=copy.deepcopy(self.samples), final=self.snapshot(),
            final_positions_m=self.x.tolist(), final_velocities_m_per_s=self.v.tolist())


def populate_result(result, record, graph=None, stiffness=None):
    """Use committed stress/U; never reconstruct an elastic material state."""
    if stiffness is not None:
        raise ValueError('dynamic material result cannot be recomputed with stiffness')
    if record.get('schema') != SCHEMA:
        raise ValueError('unsupported dynamic material result schema')
    if graph is not None:
        if set(graph.nodes) != set(record['node_ids']) or set(graph.edges) != set(record['edge_ids']):
            raise ValueError('material result graph IDs differ')
        if [np.asarray(graph.nodes[n].position).tolist() for n in record['node_ids']] != record['initial_positions']:
            raise ValueError('material result graph geometry differs')
        for eid, pair in zip(record['edge_ids'], record['edge_nodes']):
            if [graph.edges[eid].node_i, graph.edges[eid].node_j] != pair:
                raise ValueError('material result graph topology differs')
    state = record['final']
    n = len(record['edge_ids'])
    stress = _array(state['stress_Pa'], (n,), 'record stress')
    strain = _array(state['strain'], (n,), 'record strain')
    stored = _array(state['stored_energy_J'], (n,), 'record energy')
    area = _array(record['reference_areas_m2'], (n,), 'record area')
    if np.any(stored < 0) or np.any(area <= 0):
        raise ValueError('invalid dynamic material energy or area')
    result.edge_forces = stress * area
    result.edge_stretches = 1 + strain
    result.energy = float(np.sum(stored))
    result.n_nodes, result.n_edges = len(record['node_ids']), n
    result.max_force = float(np.max(np.abs(result.edge_forces)))
    result.max_stretch = float(np.max(result.edge_stretches))
    result.mean_stretch = float(np.mean(result.edge_stretches))
    result.std_stretch = float(np.std(result.edge_stretches))
    if result.displacements is not None:
        result.max_displacement = float(np.max(np.linalg.norm(result.displacements, axis=1)))
    return result


def validate_record(record):
    """Read-only validation of the persisted, versioned material history."""
    if type(record) is not dict or record.get('schema') != SCHEMA:
        raise ValueError('unsupported dynamic material record')
    def finite_tree(value, depth=0):
        if depth > 32:
            raise ValueError('material record nesting exceeds supported depth')
        if isinstance(value, dict):
            for v in value.values(): finite_tree(v, depth+1)
        elif isinstance(value, (list, tuple)):
            for v in value: finite_tree(v, depth+1)
        elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
            raise ValueError('nonfinite material record value')
    finite_tree(record)
    material = LinearViscoelasticMaterial.from_dict(record['material'])
    if material.parallel_viscosity != 0 or record['units'] != dict(SI, stress='Pa', area='m^2', energy='J', strain='1'):
        raise ValueError('unsupported dynamic material units/law')
    node_ids, edge_ids = _ids(record['node_ids'], 'record node IDs'), _ids(record['edge_ids'], 'record edge IDs')
    n, m, b = len(node_ids), len(edge_ids), len(material.branches)
    if not n or not m:
        raise ValueError('material record requires nodes and edges')
    area = _array(record['reference_areas_m2'], (m,), 'reference area')
    rest = _array(record['rest_lengths_m'], (m,), 'rest lengths')
    masses = _array(record['masses_kg'], (n,), 'masses')
    if np.any(area <= 0) or np.any(rest <= 1e-12) or np.any(masses <= 0):
        raise ValueError('material record positive geometry/mass required')
    fixed = _ids(record['fixed_node_ids'], 'fixed node IDs')
    if not set(fixed) <= set(node_ids):
        raise ValueError('material fixed node IDs not in graph')
    drag = _array(record['drag_per_s'], (), 'drag')
    if drag < 0:
        raise ValueError('material drag must be nonnegative')
    dt = _positive(record['dt'], 'record dt')
    bound = _positive(record['max_abs_strain'], 'record max_abs_strain')
    steps = record['completed_steps']
    if type(steps) is not int or steps < 1:
        raise ValueError('material record completed_steps must be positive integer')
    initial_x = _array(record['initial_positions'], (n, 3), 'initial positions')
    final_x = _array(record['final_positions_m'], (n, 3), 'final positions')
    final_v = _array(record['final_velocities_m_per_s'], (n, 3), 'final velocities')
    _array(record['constant_external_force_N'], (n, 3), 'external force')
    rows = {nid: i for i, nid in enumerate(node_ids)}
    if len(record['edge_nodes']) != m:
        raise ValueError('material topology length mismatch')
    pairs = []
    for pair in record['edge_nodes']:
        if len(pair) != 2 or any(type(x) is not int or x not in rows for x in pair):
            raise ValueError('material topology references invalid node ID')
        pairs.append((rows[pair[0]], rows[pair[1]]))
    if not np.array_equal(rest, [math.dist(initial_x[i], initial_x[j]) for i, j in pairs]):
        raise ValueError('material initial geometry/rest mismatch')
    samples = record['samples']
    if type(samples) is not list or not samples or samples[0]['step'] != 0 or samples[-1] != record['final']:
        raise ValueError('material sampled endpoints mismatch')
    previous_step = -1
    previous_cumulative = {k: np.zeros(m) for k in ENERGY_FIELDS[1:]}
    for state in samples:
        if (set(state['increments']) != set(ENERGY_FIELDS[1:]) or
            set(state['cumulative']) != set(ENERGY_FIELDS[1:]) or
            set(state['dynamic_cumulative']) != set(DYNAMIC_FIELDS)):
            raise ValueError('material energy field keys mismatch')
        _array(list(state['dynamic_cumulative'].values()), (len(DYNAMIC_FIELDS),), 'dynamic cumulative')
        step = state['step']
        if step == 0:
            if state['dynamic_interval'] is not None or any(state['dynamic_cumulative'].values()):
                raise ValueError('initial dynamic diagnostics must be zero')
        else:
            expected = set(DYNAMIC_FIELDS) | {'constitutive_identity_residual_J', 'dynamic_identity_residual_J'}
            if type(state['dynamic_interval']) is not dict or set(state['dynamic_interval']) != expected:
                raise ValueError('material dynamic interval keys mismatch')
            _array(list(state['dynamic_interval'].values()), (len(expected),), 'dynamic interval')
        if (type(step) is not int or not previous_step < step <= steps or
            state['schema'] != 'native_axial_material_state/1' or state['edge_ids'] != edge_ids or
            state['time'] != step*dt or state['interval_dt'] != (0. if step == 0 else dt) or
            state['time_level'] != 'completed_projected_t_n'):
            raise ValueError('invalid material sample alignment')
        strain = _array(state['strain'], (m,), 'sample strain')
        viscous = _array(state['viscous_strain'], (b, m), 'sample branch history')
        stress = _array(state['stress_Pa'], (m,), 'sample stress')
        energy = _array(state['stored_energy_J'], (m,), 'sample energy')
        if np.any(np.abs(strain) > bound) or np.any(energy < 0):
            raise ValueError('material sample violates strain/energy domain')
        # Algebraic state consistency, not a history update or second integrator.
        moduli = np.array([e for e, _ in material.branches])[:, None]
        branch_stress = moduli * (strain[None, :] - viscous)
        expected_stress = material.equilibrium_modulus * strain + branch_stress.sum(axis=0)
        expected_energy = .5*(material.equilibrium_modulus*strain**2 +
            (moduli*(strain[None, :]-viscous)**2).sum(axis=0))*area*rest
        if not np.allclose(stress, expected_stress, rtol=2e-13, atol=0) or not np.allclose(energy, expected_energy, rtol=2e-13, atol=0):
            raise ValueError('material stress/energy inconsistent with committed branch history')
        for key in ENERGY_FIELDS[1:]:
            inc = _array(state['increments'][key], (m,), 'material increment')
            cumulative = _array(state['cumulative'][key], (m,), 'material cumulative')
            if key != 'endpoint_work_J' and (np.any(inc < 0) or np.any(cumulative < previous_cumulative[key])):
                raise ValueError('material dissipation must be nonnegative and cumulative')
            if step == 0 and (np.any(inc) or np.any(cumulative)):
                raise ValueError('initial material work/loss must be zero')
            previous_cumulative[key] = cumulative
        if step == 0 and (np.any(strain) or np.any(viscous) or np.any(stress) or np.any(energy)):
            raise ValueError('material initial state must be relaxed zero history')
        if not math.isfinite(state['kinetic_energy_J']) or state['kinetic_energy_J'] < 0:
            raise ValueError('invalid material kinetic energy')
        previous_step = step
    if previous_step != steps:
        raise ValueError('material final step mismatch')
    expected_strain = (np.array([math.dist(final_x[i], final_x[j]) for i, j in pairs])-rest)/rest
    if not np.array_equal(expected_strain, record['final']['strain']):
        raise ValueError('material final geometry/strain mismatch')
    from .accelerated import _checked_kinetic_energy
    if not math.isclose(_checked_kinetic_energy(masses, final_v), record['final']['kinetic_energy_J'], rel_tol=2e-13, abs_tol=0):
        raise ValueError('material final kinetic energy mismatch')
    for nid in fixed:
        row = rows[nid]
        if not np.array_equal(final_x[row], initial_x[row]) or np.any(final_v[row]):
            raise ValueError('material final fixed support changed')
    if 'sampled_native_state' in record:
        native_samples = record['sampled_native_state']
        if type(native_samples) is not list or len(native_samples) != len(samples):
            raise ValueError('sampled native/material state count mismatch')
        for native, state in zip(native_samples, samples):
            if native['step'] != state['step'] or native['time'] != state['time']:
                raise ValueError('sampled native/material state alignment mismatch')
            x = _array(native['positions'], (n,3), 'sampled native positions')
            v = _array(native['velocities'], (n,3), 'sampled native velocities')
            epsilon = (np.array([math.dist(x[i], x[j]) for i,j in pairs])-rest)/rest
            if not np.array_equal(epsilon, state['strain']):
                raise ValueError('sampled geometry/material strain mismatch')
            from .accelerated import _checked_kinetic_energy
            kinetic = _checked_kinetic_energy(masses, v)
            stored = float(np.sum(state['stored_energy_J']))
            if (native['kinetic_energy'] != kinetic or state['kinetic_energy_J'] != kinetic or
                native['potential_energy'] != stored or native['total_energy'] != kinetic+stored):
                raise ValueError('sampled native/material energy mismatch')
        if not np.array_equal(native_samples[-1]['positions'], final_x) or not np.array_equal(native_samples[-1]['velocities'], final_v):
            raise ValueError('sampled final native state mismatch')
    return record


def validate_result(result):
    record = validate_record(result.metadata['native_axial_viscoelastic'])
    if (result.mode != 'dynamics' or result.metadata.get('node_ids') != record['node_ids'] or
        result.metadata.get('edge_ids') != record['edge_ids'] or
        result.metadata.get('num_steps') != record['completed_steps'] or
        result.metadata.get('dt') != record['dt'] or
        result.n_nodes != len(record['node_ids']) or result.n_edges != len(record['edge_ids'])):
        raise ValueError('material/result identity or timing mismatch')
    inputs = result.metadata.get('native_axial_inputs')
    if (not isinstance(inputs, dict) or inputs.get('node_ids') != record['node_ids'] or
        inputs.get('masses') != record['masses_kg'] or inputs.get('units') != SI):
        raise ValueError('material/native mass or unit provenance mismatch')
    energy = float(sum(record['final']['stored_energy_J']))
    if not math.isclose(result.energy, energy, rel_tol=2e-13, abs_tol=0):
        raise ValueError('result energy differs from committed material state')
    if result.deformed_positions is not None and not np.array_equal(result.deformed_positions, record['final_positions_m']):
        raise ValueError('result positions differ from committed material state')
    if result.edge_forces is not None and not np.array_equal(result.edge_forces,
            np.asarray(record['final']['stress_Pa'])*record['reference_areas_m2']):
        raise ValueError('result force differs from committed material state')
    if result.algorithmic_energy_ledger is not None or result.dynamics_state is not None:
        raise ValueError('dynamic material must not be labeled memoryless axial state/ledger')
    return record


def source_capabilities():
    """Static capability declaration, never a runtime or accuracy certificate."""
    return dict(schema='native_axial_viscoelastic_capabilities/1',
        backend='taichi', constitutive_source='LinearViscoelasticMaterial.trial',
        supported=['Maxwell', 'standard_linear_solid', 'generalized_Maxwell', 'elastic_limit'],
        geometry='small_engineering_axial_strain_corotational_truss',
        units=dict(SI, modulus='Pa', relaxation_time='s', reference_area='m^2', energy='J'),
        initial_state='relaxed_at_geometric_rest; optional explicit nodal velocity',
        history='one backward-Euler material trial per native interval, after projection',
        contact='not exposed by this public dynamic material slice',
        unsupported=['parallel_viscosity', 'material_restart', 'displacement_schedule',
            'finite_strain_polymer', 'bending', 'torsion', 'plasticity'],
        numerical_damping='dashpot refused; drag recorded separately',
        implementation='host NumPy material history and native cached edge-force upload',
        qualification='synthetic member/branch and 105-node three-grid network evidence',
        performance='no GPU or throughput claim')
