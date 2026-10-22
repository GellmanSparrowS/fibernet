"""Explicit CLI harmonic-bead adapter; no engine discovery or material inference."""
import copy
import hashlib
import json
import time

import numpy as np
from fibernet.integrations import _harmonic_profile as _profiles
from fibernet.integrations import _cg_network_profile as _network


SCHEMA = 'lammps_harmonic_state/1'
CG_SCHEMA = 'lammps_cg_state/1'
CG_MODEL = 'conservative-ball-cosine-strand-cg/1'
UNITS = {'positions': 'm', 'velocities': 'm/s', 'nodal_forces': 'N',
         'time': 's', 'energy': 'J', 'pressure_tensor': 'Pa'}
CONVENTIONS = {
    'nodal_forces': 'native net nodal forces including constant loads; not general reactions',
    'pressure_tensor': 'global box pressure; compression-positive; xx,yy,zz,xy,xz,yz; not material stress',
    'total_energy': 'potential plus kinetic; external load work not included',
}
WORK = {'status': 'unavailable', 'value': None, 'reason': 'not_collected'}
NETWORK_FIELDS = ('sample_every', 'thermo_every', 'output_frames', 'thermo_frames',
                  'integration_steps', 'observation_scope')


def _network_schedule(value, steps):
    """Validate persisted observation counts independently of geometry arrays."""
    profile = _network.record(value.get('profile'))
    dumps, thermos = _network.schedule(steps, value.get('sample_every'), value.get('thermo_every'))
    expected = {'integration_steps': steps, 'output_frames': dumps, 'thermo_frames': thermos}
    if any(type(value.get(k)) is not int or value[k] != v for k, v in expected.items()):
        raise ValueError('CG network observation counts mismatch')
    if value.get('observation_scope') != 'sampled states only; no all-step geometry validation':
        raise ValueError('CG network observation scope mismatch')
    return profile, dumps



def source_capabilities():
    """Descriptive source support only, never installed/runtime/physics readiness."""
    return {'backend': 'lammps', 'profile': 'explicit-harmonic-beads',
            'dimension': 3, 'execution': 'owned CLI child',
            'runtime_modes': ['cli', 'cli_serial_static'],
            'runtime_readiness': 'unknown', 'physics_qualification': 'not_established_by_dispatch',
            'constraints': ['straight nonparallel edges', 'nonperiodic box',
                            'whole-node fixity', 'constant nodal loads',
                            'explicit SI masses and bond coefficients',
                            'bounded domain; every integration step retained'],
            'harmonic_profiles': {
                p['name']: {key: p[key] for key in ('max_nodes','max_edges',
                    'steps','max_steps','sample_every','output_stride','frames',
                    'max_output_frames','max_mass_types','max_loaded_nodes') if key in p}
                for p in (_profiles.PROFILE, _profiles.SAMPLED_PROFILE,
                          _profiles.FULL_HISTORY_PROFILE)},
            'dependency': 'explicit independently admitted RuntimeSpec',
            'cancellation': 'hard child cancellation via direct LAMMPSTask',
            'unsupported': ['material calibration', 'material stress', 'general reactions',
                            'work ledger', 'linked Python engine']}


def _real_array(value, name, shape):
    from .accelerated import _finite_real_array
    def check(item):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(name + ' must not contain booleans')
        if isinstance(item, (list, tuple)):
            for child in item: check(child)
        elif isinstance(item, np.ndarray):
            for child in item.flat: check(child)
    check(value)
    return _finite_real_array(value, name, shape)


def validate_state(value, n_nodes=None, node_ids=None):
    """Validate and detach this backend's records without accepting native-axial labels."""
    from .accelerated import _strict_state_ids
    if not isinstance(value, dict) or value.get('schema') not in (SCHEMA, CG_SCHEMA):
        raise ValueError('unsupported LAMMPS state schema')
    cg = value['schema'] == CG_SCHEMA
    sampled = value.get('representation') == 'sampled'
    network = cg and 'profile' in value
    if cg and not network and (value.get('final',{}).get('step',1001) > 1000 or (sampled and not 2 <= len(value.get('node_ids',())) <= 64)):
        raise ValueError('CG state exceeds model bounds')
    if value.get('representation') not in ('sampled', 'summary'):
        raise ValueError('invalid LAMMPS state representation')
    common = {'schema', 'backend', 'representation', 'availability', 'units',
              'conventions', 'dt', 'final', 'work_ledger', 'source_identity'}
    arrays = {'node_ids', 'steps', 'times', 'positions', 'velocities', 'nodal_forces',
              'potential_energy', 'kinetic_energy', 'total_energy', 'pressure_tensor'}
    if sampled and 'profile' not in value and not 2 <= len(value.get('node_ids',())) <= 256:
        raise ValueError('legacy LAMMPS state node bound exceeded')
    if network:
        p, count = _network_schedule(value, value.get('final', {}).get('step'))
        common.update(('profile',) + NETWORK_FIELDS)
        _profiles.shape([value.get('dt')], (1,), 'dt')
        for key in ('step', 'time', 'potential_energy', 'kinetic_energy', 'total_energy'):
            _profiles.shape([value.get('final', {}).get(key)], (1,), 'final ' + key)
        if sampled:
            n = len(value.get('node_ids', ()))
            if not 2 <= n <= p['max_nodes']:
                raise ValueError('CG network state node bound exceeded')
            for key in ('positions', 'velocities', 'nodal_forces'):
                _profiles.shape(value.get(key), (count, n, 3), key)
                _profiles.shape(value.get('final', {}).get(key), (n, 3), 'final ' + key)
            for key in ('steps', 'times', 'potential_energy', 'kinetic_energy', 'total_energy'):
                _profiles.shape(value.get(key), (count,), key)
            _profiles.shape(value.get('pressure_tensor'), (count, 6), 'pressure_tensor')
            _profiles.shape(value.get('final', {}).get('pressure_tensor'), (6,), 'final pressure_tensor')
        _profiles.json_bytes(value, p['export_bytes'])
    elif 'profile' in value:
        p = _profiles.record(value['profile'])
        common.add('profile')
        if _profiles.variable_steps(p):
            common.update(('output_stride','native_frames','output_frames'))
        count = _profiles.schedule(p,value.get('final',{}).get('step'),1,value.get('output_stride'))
        if _profiles.variable_steps(p) and (type(value.get('native_frames')) is not int or value['native_frames'] != value['final']['step']+1 or type(value.get('output_frames')) is not int or value['output_frames'] != count):
            raise ValueError('sampled state frame counts mismatch')
        _profiles.shape([value.get('dt')],(1,),'dt')
        for key in ('step','time','potential_energy','kinetic_energy','total_energy'):
            _profiles.shape([value.get('final',{}).get(key)],(1,),'final '+key)
        if sampled:
            n = len(value.get('node_ids',()))
            if not 2 <= n <= p['max_nodes']: raise ValueError('profile state node bound exceeded')
            for key in ('positions','velocities','nodal_forces'):
                _profiles.shape(value.get(key),(count,n,3),key)
                _profiles.shape(value.get('final',{}).get(key),(n,3),'final '+key)
            for key in ('steps','times','potential_energy','kinetic_energy','total_energy'):
                _profiles.shape(value.get(key),(count,),key)
            _profiles.shape(value.get('pressure_tensor'),(count,6),'pressure_tensor')
            _profiles.shape(value.get('final',{}).get('pressure_tensor'),(6,),'final pressure_tensor')
        _profiles.json_bytes(value,p['export_bytes'])
    if set(value) != common | (arrays if sampled else set()):
        raise ValueError('LAMMPS state has missing or unknown fields')
    if (value['backend'] != 'lammps' or value['units'] != UNITS or
            value['conventions'] != CONVENTIONS or value['work_ledger'] != WORK or
            value['availability'] != ('available' if sampled else 'summary_only')):
        raise ValueError('invalid LAMMPS state declarations')
    identity = value['source_identity']
    if (not isinstance(identity, dict) or set(identity) != {'prepared_sha256', 'graph_sha256'} or
            any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in identity.values())):
        raise ValueError('invalid LAMMPS source identity')
    dt = float(_real_array(value['dt'], 'dt', ()))
    if dt <= 0: raise ValueError('dt must be positive')
    scalar_names = {'step', 'time', 'potential_energy', 'kinetic_energy', 'total_energy'}
    vector_names = {'positions', 'velocities', 'nodal_forces', 'pressure_tensor'}
    final = value['final']
    if not isinstance(final, dict) or set(final) != scalar_names | (vector_names if sampled else set()):
        raise ValueError('invalid LAMMPS final fields')
    steps = _strict_state_ids([final['step']], 'final step', nonnegative=True)
    if final['step'] > (_network.PROFILE['max_steps'] if network else 100000): raise ValueError('LAMMPS step count exceeds compiler bound')
    for key in scalar_names - {'step'}: _real_array(final[key], key, ())
    if (not np.isclose(final['time'], steps[0]*dt, rtol=1e-12, atol=0) or
            (not cg and final['potential_energy'] < 0) or final['kinetic_energy'] < 0 or
            not np.isclose(final['total_energy'], final['potential_energy']+final['kinetic_energy'], rtol=1e-12, atol=0)):
        raise ValueError('inconsistent LAMMPS final time or energy')
    if sampled:
        ids = _strict_state_ids(value['node_ids'])
        if ((n_nodes is not None and len(ids) != n_nodes) or
                (node_ids is not None and ids != list(node_ids))):
            raise ValueError('LAMMPS node identity mismatch')
        steps = _strict_state_ids(value['steps'], 'steps', nonnegative=True)
        stride = value['sample_every'] if network else value.get('output_stride',1)
        if steps != list(range(0,final['step']+1,stride)):
            raise ValueError('LAMMPS state kept indices differ from output schedule')
        count, n = len(steps), len(ids)
        times = _real_array(value['times'], 'times', (count,))
        if not np.allclose(times, np.asarray(steps)*dt, rtol=1e-12, atol=0):
            raise ValueError('LAMMPS physical time mismatch')
        for key in ('positions', 'velocities', 'nodal_forces'):
            _real_array(value[key], key, (count, n, 3))
        _real_array(value['pressure_tensor'], 'pressure_tensor', (count, 6))
        energies = {k: _real_array(value[k], k, (count,)) for k in ('potential_energy', 'kinetic_energy', 'total_energy')}
        if ((not cg and np.any(energies['potential_energy'] < 0)) or np.any(energies['kinetic_energy'] < 0) or
                not np.allclose(energies['total_energy'], energies['potential_energy']+energies['kinetic_energy'], rtol=1e-12, atol=0)):
            raise ValueError('inconsistent LAMMPS sampled energies')
        for key in ('positions', 'velocities', 'nodal_forces'):
            _real_array(final[key], 'final '+key, (n, 3))
        _real_array(final['pressure_tensor'], 'final pressure_tensor', (6,))
        for key in vector_names | {'potential_energy', 'kinetic_energy', 'total_energy'}:
            if not np.array_equal(final[key], value[key][-1]):
                raise ValueError('LAMMPS final/sample mismatch')
        if final['time'] != value['times'][-1]: raise ValueError('LAMMPS final time mismatch')
    return json.loads(json.dumps(value, allow_nan=False))


def state_summary(value):
    return _state_summary_from_validated(validate_state(value))


def _state_summary_from_validated(record):
    """Project the detached record validated within this operation only."""
    if record['representation'] == 'summary': return record
    keys = ('schema', 'backend', 'units', 'conventions', 'dt', 'work_ledger', 'source_identity')
    if 'profile' in record: keys += ('profile',)
    if 'output_stride' in record: keys += ('output_stride','native_frames','output_frames')
    if record['schema'] == CG_SCHEMA and 'profile' in record: keys += NETWORK_FIELDS
    return dict({k: record[k] for k in keys}, representation='summary', availability='summary_only',
                final={k: record['final'][k] for k in ('step', 'time', 'potential_energy', 'kinetic_energy', 'total_energy')})


def _plain_json_tree(value, depth=0, budget=None):
    """Conservative hook-free fast path; every other accepted type falls back."""
    if budget is not None:
        budget[0] -= 1
        if budget[0] < 0:
            return False
    if depth > 64:
        return False
    if type(value) in (str, int, float, bool, type(None)):
        return True
    if type(value) in (list, tuple):
        return all(_plain_json_tree(item, depth + 1, budget) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain_json_tree(item, depth + 1, budget)
                   for key, item in value.items())
    return False



def _plain_full_history_export(result, detailed):
    """Eligibility only, never validation or a cache across export calls.

    The detailed projection must not invoke user coercion or mutation hooks
    between full validation and use of its detached record. Other types retain
    the original validation order, including the post-projection guard.
    """
    from .accelerated import SimResult, _SPRING_PROVENANCE
    if (type(result) is not SimResult or detailed is not True
            or type(vars(result)) is not dict
            or len(vars(result)) != len(SimResult.__dataclass_fields__)
            or any(type(key) is not str for key in vars(result))
            or set(vars(result)) != set(SimResult.__dataclass_fields__)
            or type(result.metadata) is not dict
            or type(result.dynamics_state) is not dict
            or type(result.history) is not list
            or type(result.positions_trajectory) is not list
            or len(result.positions_trajectory) > _profiles.FULL_HISTORY_PROFILE['max_output_frames']
            or result.algorithmic_energy_ledger is not None):
        return False
    arrays = ('edge_forces', 'edge_stretches', 'displacements', 'deformed_positions')
    def plain_array(value):
        return type(value) is np.ndarray and value.dtype.kind in 'fiu'
    if (any(value is not None and not plain_array(value)
            for value in (getattr(result, key) for key in arrays))
            or any(not plain_array(frame) for frame in result.positions_trajectory)):
        return False
    remaining = set(SimResult.__dataclass_fields__) - set(arrays) - {'positions_trajectory'}
    # Eligibility must itself terminate on wide/aliased JSON trees before the
    # authoritative byte guard. Exhaustion merely selects the original path.
    budget = [2*1024*1024]
    if not all(_plain_json_tree(getattr(result, key), budget=budget) for key in remaining):
        return False
    # Do not look up user-controlled mapping keys before proving exact strings.
    if (type(result.dynamics_state.get('schema')) is not str
            or result.dynamics_state['schema'] != SCHEMA
            or type(result.metadata.get('profile')) is not dict
            or type(result.metadata['profile'].get('name')) is not str
            or result.metadata['profile']['name'] != _profiles.FULL_HISTORY_NAME
            or any(k in result.metadata for k in
                   ('native_axial_inputs', 'native_axial_viscoelastic',
                    _SPRING_PROVENANCE))):
        return False
    return True


def validate_result(result):
    return _validate_result(result)


def _validate_result(result, *, with_summary=False):
    _profiles.result_guard(result.metadata,result.n_nodes,result.n_edges,result.dynamics_state,vars(result))
    from .accelerated import SimResult
    plain_state = (with_summary and type(result) is SimResult
                   and type(result.n_nodes) is int
                   and type(result.n_edges) is int and type(result.metadata) is dict
                   and _plain_json_tree(result.metadata)
                   and _plain_json_tree(result.dynamics_state))
    record = validate_state(result.dynamics_state, result.n_nodes, result.metadata.get('node_ids'))
    if (result.metadata.get('backend') != 'lammps' or
            result.metadata.get('source_identity') != record['source_identity'] or
            not np.isclose(result.energy, record['final']['potential_energy'], rtol=1e-12, atol=0)):
        raise ValueError('LAMMPS state/result provenance or energy mismatch')
    if result.algorithmic_energy_ledger is not None:
        raise ValueError('LAMMPS does not provide a native-axial algorithmic ledger')
    from fibernet.integrations.lammps_backend import LENGTH_FACTORS
    contract = result.metadata.get('model_contract', {})
    if (record['schema'] == CG_SCHEMA) != (contract.get('model') == CG_MODEL):
        raise ValueError('LAMMPS state/model schema mismatch')
    if record['schema'] == CG_SCHEMA and 'profile' in record:
        profile, _ = _network_schedule(contract, contract.get('steps'))
        if _network.record(record['profile']) != profile or any(record[k] != contract.get(k) for k in NETWORK_FIELDS):
            raise ValueError('CG network state/model observation schedule mismatch')
        if _network.record(result.metadata.get('profile')) != profile:
            raise ValueError('CG network result/profile mismatch')
    unit = result.metadata.get('original_graph_length_unit')
    scale = result.metadata.get('original_graph_length_to_m')
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid graph-to-SI length scale')
    if (unit not in LENGTH_FACTORS or scale != LENGTH_FACTORS[unit] or
            contract.get('length_unit') != unit or contract.get('length_to_m') != scale or
            result.metadata.get('field_units') != field_units(unit)):
        raise ValueError('inconsistent LAMMPS geometry units')
    if (contract.get('timestep_s') != record['dt'] or
            contract.get('steps') != record['final']['step']):
        raise ValueError('LAMMPS state/model time contract mismatch')
    if result.displacements is not None:
        _real_array(result.displacements, 'displacements', (result.n_nodes, 3))
    if record['representation'] == 'sampled':
        if record['schema'] == CG_SCHEMA:
            import math
            expected_norm = max(math.hypot(*v) for v in record['final']['nodal_forces'])
            if not math.isfinite(expected_norm) or result.metadata['final_max_net_nodal_force_N'] != expected_norm:
                raise ValueError('CG net nodal force metric mismatch')
        if result.deformed_positions is not None:
            _real_array(result.deformed_positions, 'deformed_positions', (result.n_nodes, 3))
        if result.positions_trajectory:
            _real_array(result.positions_trajectory, 'positions_trajectory', (len(record['steps']), result.n_nodes, 3))
        if result.deformed_positions is not None and not np.allclose(result.deformed_positions*scale, record['final']['positions'], rtol=1e-12, atol=0):
            raise ValueError('LAMMPS final positions disagree with result')
        if result.positions_trajectory and not np.allclose(np.asarray(result.positions_trajectory)*scale, record['positions'], rtol=1e-12, atol=0):
            raise ValueError('LAMMPS trajectory disagrees with result')
    if with_summary:
        summary = (_state_summary_from_validated(record) if plain_state
                   else state_summary(record))
        return record, summary
    return record


def field_units(length_unit):
    return {'displacements':length_unit, 'deformed_positions':length_unit,
            'positions_trajectory':length_unit, 'max_displacement':length_unit,
            'edge_forces':'N', 'edge_stretches':'1', 'energy':'J', 'time_seconds':'s wall time'}


def to_sim_result(prepared, raw, runtime, wall_seconds, record_state=False):
    """Map completed native output; geometry-facing arrays retain original graph units."""
    from .accelerated import SimResult
    from fibernet.integrations.lammps_backend import validate_prepared
    validate_prepared(prepared)
    if raw.identity != prepared.identity:
        raise ValueError('LAMMPS result/model identity mismatch')
    if not np.isfinite(wall_seconds) or wall_seconds < 0: raise ValueError('invalid wall time')
    contract, graph = json.loads(prepared.contract_json), json.loads(prepared.graph_json)
    cg = contract.get('model') == CG_MODEL
    network = cg and 'profile' in contract
    selected = _network.record(contract['profile']) if network else _profiles.contract_profile(contract)
    if network:
        _, output_count = _network_schedule(contract, prepared.steps)
    elif selected:
        native_count, output_count = _profiles.frame_counts(contract)
        if len(raw.trajectory)!=native_count or len(raw.thermo)!=native_count:
            raise ValueError('profile raw result requires every native frame')
    ids, edges = list(prepared.node_ids), list(prepared.edge_ids)
    expected = list(range(0, prepared.steps+1, contract['sample_every'] if network else 1))
    thermo_expected = list(range(0, prepared.steps+1, contract['thermo_every'] if network else 1))
    if ([f.step for f in raw.trajectory] != expected or [f.step for f in raw.thermo] != thermo_expected or
            any(list(f.node_ids) != ids for f in raw.trajectory)):
        raise ValueError('LAMMPS incomplete or misidentified result')
    # Validate every observed geometry and thermo row before projection. Network
    # geometry checks cover sampled states only, never unobserved integration steps.
    for frame in raw.trajectory:
        if type(frame.step) is not int:
            raise ValueError('native steps must be integers')
        _profiles.shape([frame.time_s],(1,),'native time')
        if frame.time_s != frame.step*prepared.timestep_s:
            raise ValueError('native frame physical time mismatch')
        for attr in ('positions_m','velocities_m_s','forces_N'):
            _profiles.shape(getattr(frame,attr),(len(ids),3),attr)
    for thermo in raw.thermo:
        if type(thermo.step) is not int:
            raise ValueError('native steps must be integers')
        for attr in ('potential_J','kinetic_J','total_J'):
            _profiles.shape([getattr(thermo,attr)],(1,),attr)
        _profiles.shape(thermo.pressure_tensor_Pa,(6,),'native pressure')
        if (not cg and thermo.potential_J < 0) or thermo.kinetic_J < 0 or not np.isclose(thermo.total_J,thermo.potential_J+thermo.kinetic_J,rtol=1e-12,atol=0):
            raise ValueError('inconsistent native frame energy')
    # Typed results can bypass the parser. Held state is a compiler invariant,
    # independent of the all-node initial/domain checks below.
    if contract['fixed_node_ids']:
        import math
        rows = {nid:i for i,nid in enumerate(ids)}
        original = {n['id']:tuple(x*contract['length_to_m'] for x in n['position']) for n in graph['nodes']}
        for frame in raw.trajectory:
            for nid in contract['fixed_node_ids']:
                i = rows[nid]
                if any(not math.isclose(x,y,rel_tol=1e-13,abs_tol=0) for x,y in zip(frame.positions_m[i],original[nid])) or any(frame.velocities_m_s[i]):
                    raise ValueError('native initial/fixed state mismatch')
    # All compiler profiles share these checks, even without recorded state.
    rows = {nid:i for i,nid in enumerate(ids)}
    original = {n['id']:tuple(x*contract['length_to_m'] for x in n['position']) for n in graph['nodes']}
    initial_velocity = dict(contract['velocities_m_s'])
    import math
    for frame in raw.trajectory:
        for nid,point,velocity in zip(ids,frame.positions_m,frame.velocities_m_s):
            if any(not contract['box_m'][axis][0] < point[axis] < contract['box_m'][axis][1] for axis in range(3)):
                raise ValueError('native site left declared box')
            if frame.step == 0 or nid in contract['fixed_node_ids']:
                expected_velocity = initial_velocity.get(nid,(0,0,0)) if frame.step == 0 else (0,0,0)
                if any(not math.isclose(x,y,rel_tol=1e-13,abs_tol=0) for x,y in zip(point,original[nid])) or any(not math.isclose(x,y,rel_tol=1e-13,abs_tol=0) for x,y in zip(velocity,expected_velocity)):
                    raise ValueError('native initial/fixed state mismatch')
        if cg:
            for nid,_ in contract['projectile_pairs']:
                if math.dist(frame.positions_m[rows[nid]],frame.positions_m[rows[contract['projectile_node_id']]]) == 0:
                    raise ValueError('coincident selected CG pair')
        for edge in graph['edges']:
            length = math.dist(frame.positions_m[rows[edge['node_i']]],frame.positions_m[rows[edge['node_j']]])
            if not 0 < length <= contract['maximum_bond_length_m']:
                raise ValueError('native bond left declared reach')
    kept = expected if network else list(range(0,prepared.steps+1,contract.get('output_stride',1)))
    trajectory_by_step = {f.step: f for f in raw.trajectory}
    thermo_by_step = {f.step: f for f in raw.thermo}
    trajectory = [trajectory_by_step[i] for i in kept]
    thermo = [thermo_by_step[i] for i in kept]
    identity = {'prepared_sha256': prepared.identity,
                'graph_sha256': hashlib.sha256(prepared.graph_json.encode()).hexdigest()}
    state = dict(schema=CG_SCHEMA if cg else SCHEMA, backend='lammps', representation='sampled', availability='available',
                 units=copy.deepcopy(UNITS), conventions=copy.deepcopy(CONVENTIONS), dt=prepared.timestep_s,
                 work_ledger=copy.deepcopy(WORK), source_identity=identity, node_ids=ids,
                 steps=kept, times=[f.time_s for f in trajectory])
    if selected:
        state['profile'] = selected
        if network:
            state.update({k: contract[k] for k in NETWORK_FIELDS})
        elif _profiles.variable_steps(selected):
            state.update({k:contract[k] for k in ('output_stride','native_frames','output_frames')})
    for key, attr in [('positions','positions_m'), ('velocities','velocities_m_s'), ('nodal_forces','forces_N')]:
        state[key] = [np.asarray(getattr(f, attr)).tolist() for f in trajectory]
    for key, attr in [('potential_energy','potential_J'), ('kinetic_energy','kinetic_J'), ('total_energy','total_J'), ('pressure_tensor','pressure_tensor_Pa')]:
        state[key] = [getattr(f, attr) for f in thermo]
    state['pressure_tensor'] = [list(v) for v in state['pressure_tensor']]
    state['final'] = dict(step=expected[-1], time=state['times'][-1], **{k: state[k][-1] for k in ('positions','velocities','nodal_forces','potential_energy','kinetic_energy','total_energy','pressure_tensor')})
    state = validate_state(state, len(ids), ids)
    original = {n['id']: np.asarray(n['position'])*contract['length_to_m'] for n in graph['nodes']}
    positions = np.asarray(state['positions'][-1]); displacement = positions-np.asarray([original[n] for n in ids])
    row = {n:i for i,n in enumerate(ids)}; topology = {e['id']:e for e in graph['edges']}; bonds = dict(contract['bonds'])
    lengths = np.asarray([np.linalg.norm(positions[row[topology[e]['node_j']]]-positions[row[topology[e]['node_i']]]) for e in edges])
    coefficients = np.asarray([bonds[e] for e in edges]).reshape((-1,2)); forces = coefficients[:,0]*(lengths-coefficients[:,1]); stretches = lengths/coefficients[:,1]
    scale = contract['length_to_m']
    result = SimResult(displacements=displacement/scale, deformed_positions=positions/scale,
        energy=float(raw.thermo[-1].potential_J), time_seconds=float(wall_seconds), mode='dynamics',
        edge_forces=forces, edge_stretches=stretches, max_force=float(np.max(np.abs(forces))) if len(edges) else 0.0,
        max_stretch=float(np.max(stretches)) if len(edges) else 1.0, mean_stretch=float(np.mean(stretches)) if len(edges) else 1.0, std_stretch=float(np.std(stretches)) if len(edges) else 0.0,
        max_displacement=float(np.max(np.linalg.norm(displacement/scale,axis=1))), n_nodes=len(ids), n_edges=len(edges),
        metadata=_profiles.export_metadata(prepared,runtime),
        dynamics_state=state if record_state else None)
    if cg:
        from fibernet.integrations._lammps_cg import FORCE_ROLES
        from fibernet.integrations.lammps_backend import _number
        import math
        result.metadata['model_contract_sha256'] = hashlib.sha256(prepared.contract_json.encode()).hexdigest()
        result.metadata['force_roles'] = dict(FORCE_ROLES)
        result.metadata['final_max_net_nodal_force_N'] = _number(max(math.hypot(*v) for v in raw.trajectory[-1].forces_N),'net nodal force norm')
        result.metadata['edge_force_origin'] = 'harmonic bond contribution only: k*(length-equilibrium_length); tension-positive; not total nodal force'
        result.metadata['initial_bending_energy_J'] = contract['initial_bending_energy_J']
    if selected: result.metadata['profile'] = selected
    if record_state: result.positions_trajectory = [np.asarray(p)/scale for p in state['positions']]
    return result


def simulate_lammps(graph, mode, strain, **kwargs):
    """Synchronous convenience over the existing owned task; no alternate launcher."""
    from fibernet.integrations.lammps_backend import prepare_harmonic_beads, LAMMPSTask, TaskState, LAMMPSContractError
    cg_requested = 'cg_request' in kwargs
    cg_request = kwargs.pop('cg_request', None)
    required = {'length_unit','maximum_bond_length_m','masses_kg','bonds','run','runtime','workdir','limits'}
    allowed = required | {'fixed_node_ids','forces_N','velocities_m_s','record_state','profile','output_stride'}
    if cg_requested:
        required = {'runtime','workdir','limits'}
        allowed = required | {'record_state'}
    if mode != 'dynamics': raise ValueError("LAMMPS supports only mode='dynamics'")
    if strain != 2.0: raise ValueError('strain is not a LAMMPS dynamics parameter')
    if set(kwargs)-allowed: raise TypeError('unsupported LAMMPS options: '+', '.join(sorted(set(kwargs)-allowed)))
    if required-set(kwargs): raise TypeError('missing explicit LAMMPS options: '+', '.join(sorted(required-set(kwargs))))
    record_state = kwargs.pop('record_state', False)
    if type(record_state) is not bool: raise TypeError('record_state must be boolean')
    runtime, workdir, limits = (kwargs.pop(k) for k in ('runtime','workdir','limits'))
    if cg_requested:
        from fibernet.integrations._lammps_cg import prepare_cg
        prepared = prepare_cg(graph, cg_request)
    else:
        prepared = prepare_harmonic_beads(graph, **kwargs)
    task = LAMMPSTask(prepared, runtime, workdir, limits)
    started = time.monotonic()
    try:
        task.start()
        status = task.wait()
        if status != TaskState.SUCCEEDED or task.result is None:
            raise LAMMPSContractError('LAMMPS task '+str(status)+': '+str(task.error))
        return to_sim_result(prepared, task.result, runtime, time.monotonic()-started, record_state)
    except BaseException:
        task.cancel()
        task.wait()
        raise
