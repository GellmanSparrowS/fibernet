"""Explicit CLI harmonic-bead adapter; no engine discovery or material inference."""
import copy
import hashlib
import json
import time

import numpy as np


SCHEMA = 'lammps_harmonic_state/1'
UNITS = {'positions': 'm', 'velocities': 'm/s', 'nodal_forces': 'N',
         'time': 's', 'energy': 'J', 'pressure_tensor': 'Pa'}
CONVENTIONS = {
    'nodal_forces': 'native net nodal forces including constant loads; not general reactions',
    'pressure_tensor': 'global box pressure; compression-positive; xx,yy,zz,xy,xz,yz; not material stress',
    'total_energy': 'potential plus kinetic; external load work not included',
}
WORK = {'status': 'unavailable', 'value': None, 'reason': 'not_collected'}


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
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('unsupported LAMMPS state schema')
    sampled = value.get('representation') == 'sampled'
    if value.get('representation') not in ('sampled', 'summary'):
        raise ValueError('invalid LAMMPS state representation')
    common = {'schema', 'backend', 'representation', 'availability', 'units',
              'conventions', 'dt', 'final', 'work_ledger', 'source_identity'}
    arrays = {'node_ids', 'steps', 'times', 'positions', 'velocities', 'nodal_forces',
              'potential_energy', 'kinetic_energy', 'total_energy', 'pressure_tensor'}
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
    if final['step'] > 100000: raise ValueError('LAMMPS step count exceeds compiler bound')
    for key in scalar_names - {'step'}: _real_array(final[key], key, ())
    if (not np.isclose(final['time'], steps[0]*dt, rtol=1e-12, atol=0) or
            final['potential_energy'] < 0 or final['kinetic_energy'] < 0 or
            not np.isclose(final['total_energy'], final['potential_energy']+final['kinetic_energy'], rtol=1e-12, atol=0)):
        raise ValueError('inconsistent LAMMPS final time or energy')
    if sampled:
        ids = _strict_state_ids(value['node_ids'])
        if ((n_nodes is not None and len(ids) != n_nodes) or
                (node_ids is not None and ids != list(node_ids))):
            raise ValueError('LAMMPS node identity mismatch')
        steps = _strict_state_ids(value['steps'], 'steps', nonnegative=True)
        if len(steps) != final['step']+1 or any(step != i for i,step in enumerate(steps)):
            raise ValueError('LAMMPS state requires every integration step')
        count, n = len(steps), len(ids)
        times = _real_array(value['times'], 'times', (count,))
        if not np.allclose(times, np.asarray(steps)*dt, rtol=1e-12, atol=0):
            raise ValueError('LAMMPS physical time mismatch')
        for key in ('positions', 'velocities', 'nodal_forces'):
            _real_array(value[key], key, (count, n, 3))
        _real_array(value['pressure_tensor'], 'pressure_tensor', (count, 6))
        energies = {k: _real_array(value[k], k, (count,)) for k in ('potential_energy', 'kinetic_energy', 'total_energy')}
        if (np.any(energies['potential_energy'] < 0) or np.any(energies['kinetic_energy'] < 0) or
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
    record = validate_state(value)
    if record['representation'] == 'summary': return record
    keys = ('schema', 'backend', 'units', 'conventions', 'dt', 'work_ledger', 'source_identity')
    return dict({k: record[k] for k in keys}, representation='summary', availability='summary_only',
                final={k: record['final'][k] for k in ('step', 'time', 'potential_energy', 'kinetic_energy', 'total_energy')})


def validate_result(result):
    record = validate_state(result.dynamics_state, result.n_nodes, result.metadata.get('node_ids'))
    if (result.metadata.get('backend') != 'lammps' or
            result.metadata.get('source_identity') != record['source_identity'] or
            not np.isclose(result.energy, record['final']['potential_energy'], rtol=1e-12, atol=0)):
        raise ValueError('LAMMPS state/result provenance or energy mismatch')
    if result.algorithmic_energy_ledger is not None:
        raise ValueError('LAMMPS does not provide a native-axial algorithmic ledger')
    from fibernet.integrations.lammps_backend import LENGTH_FACTORS
    contract = result.metadata.get('model_contract', {})
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
        if result.deformed_positions is not None:
            _real_array(result.deformed_positions, 'deformed_positions', (result.n_nodes, 3))
        if result.positions_trajectory:
            _real_array(result.positions_trajectory, 'positions_trajectory', (len(record['steps']), result.n_nodes, 3))
        if result.deformed_positions is not None and not np.allclose(result.deformed_positions*scale, record['final']['positions'], rtol=1e-12, atol=0):
            raise ValueError('LAMMPS final positions disagree with result')
        if result.positions_trajectory and not np.allclose(np.asarray(result.positions_trajectory)*scale, record['positions'], rtol=1e-12, atol=0):
            raise ValueError('LAMMPS trajectory disagrees with result')
    return record


def field_units(length_unit):
    return {'displacements':length_unit, 'deformed_positions':length_unit,
            'positions_trajectory':length_unit, 'max_displacement':length_unit,
            'edge_forces':'N', 'edge_stretches':'1', 'energy':'J', 'time_seconds':'s wall time'}


def to_sim_result(prepared, raw, runtime, wall_seconds, record_state=False):
    """Map completed native output; geometry-facing arrays retain original graph units."""
    from .accelerated import SimResult
    if raw.identity != prepared.identity:
        raise ValueError('LAMMPS result/model identity mismatch')
    if not np.isfinite(wall_seconds) or wall_seconds < 0: raise ValueError('invalid wall time')
    contract, graph = json.loads(prepared.contract_json), json.loads(prepared.graph_json)
    ids, edges = list(prepared.node_ids), list(prepared.edge_ids)
    expected = list(range(prepared.steps+1))
    if ([f.step for f in raw.trajectory] != expected or [f.step for f in raw.thermo] != expected or
            any(list(f.node_ids) != ids for f in raw.trajectory)):
        raise ValueError('LAMMPS incomplete or misidentified result')
    identity = {'prepared_sha256': prepared.identity,
                'graph_sha256': hashlib.sha256(prepared.graph_json.encode()).hexdigest()}
    state = dict(schema=SCHEMA, backend='lammps', representation='sampled', availability='available',
                 units=copy.deepcopy(UNITS), conventions=copy.deepcopy(CONVENTIONS), dt=prepared.timestep_s,
                 work_ledger=copy.deepcopy(WORK), source_identity=identity, node_ids=ids,
                 steps=expected, times=[f.time_s for f in raw.trajectory])
    for key, attr in [('positions','positions_m'), ('velocities','velocities_m_s'), ('nodal_forces','forces_N')]:
        state[key] = [np.asarray(getattr(f, attr)).tolist() for f in raw.trajectory]
    for key, attr in [('potential_energy','potential_J'), ('kinetic_energy','kinetic_J'), ('total_energy','total_J'), ('pressure_tensor','pressure_tensor_Pa')]:
        state[key] = [getattr(f, attr) for f in raw.thermo]
    state['pressure_tensor'] = [list(v) for v in state['pressure_tensor']]
    state['final'] = dict(step=expected[-1], time=state['times'][-1], **{k: state[k][-1] for k in ('positions','velocities','nodal_forces','potential_energy','kinetic_energy','total_energy','pressure_tensor')})
    state = validate_state(state, len(ids), ids)
    original = {n['id']: np.asarray(n['position'])*contract['length_to_m'] for n in graph['nodes']}
    positions = np.asarray(state['positions'][-1]); displacement = positions-np.asarray([original[n] for n in ids])
    row = {n:i for i,n in enumerate(ids)}; topology = {e['id']:e for e in graph['edges']}; bonds = dict(contract['bonds'])
    lengths = np.asarray([np.linalg.norm(positions[row[topology[e]['node_j']]]-positions[row[topology[e]['node_i']]]) for e in edges])
    coefficients = np.asarray([bonds[e] for e in edges]); forces = coefficients[:,0]*(lengths-coefficients[:,1]); stretches = lengths/coefficients[:,1]
    pins = dict(runtime.file_sha256)
    receipt = {'mode': runtime.mode, 'expected_version': runtime.expected_version,
               'expected_banner': runtime.expected_banner,
               'python_sha256': pins.get(runtime.python_executable),
               'lammps_sha256': pins.get(runtime.lammps_executable),
               'inventory_sha256': sorted(pins.values())}
    scale = contract['length_to_m']
    result = SimResult(displacements=displacement/scale, deformed_positions=positions/scale,
        energy=float(raw.thermo[-1].potential_J), time_seconds=float(wall_seconds), mode='dynamics',
        edge_forces=forces, edge_stretches=stretches, max_force=float(np.max(np.abs(forces))),
        max_stretch=float(np.max(stretches)), mean_stretch=float(np.mean(stretches)), std_stretch=float(np.std(stretches)),
        max_displacement=float(np.max(np.linalg.norm(displacement/scale,axis=1))), n_nodes=len(ids), n_edges=len(edges),
        metadata={'backend':'lammps', 'node_ids':ids, 'edge_ids':edges, 'source_identity':identity,
                  'model_contract':contract, 'runtime_receipt':receipt,
                  'field_units': field_units(contract['length_unit']),
                  'original_graph_length_unit':contract['length_unit'], 'original_graph_length_to_m':contract['length_to_m'],
                  'edge_force_origin':'derived k*(length-equilibrium_length); tension-positive',
                  'stress':None, 'reactions':None, 'work':None},
        dynamics_state=state if record_state else None)
    if record_state: result.positions_trajectory = [np.asarray(p)/scale for p in state['positions']]
    return result


def simulate_lammps(graph, mode, strain, **kwargs):
    """Synchronous convenience over the existing owned task; no alternate launcher."""
    from fibernet.integrations.lammps_backend import prepare_harmonic_beads, LAMMPSTask, TaskState, LAMMPSContractError
    required = {'length_unit','maximum_bond_length_m','masses_kg','bonds','run','runtime','workdir','limits'}
    allowed = required | {'fixed_node_ids','forces_N','velocities_m_s','record_state'}
    if mode != 'dynamics': raise ValueError("LAMMPS supports only mode='dynamics'")
    if strain != 2.0: raise ValueError('strain is not a LAMMPS dynamics parameter')
    if set(kwargs)-allowed: raise TypeError('unsupported LAMMPS options: '+', '.join(sorted(set(kwargs)-allowed)))
    if required-set(kwargs): raise TypeError('missing explicit LAMMPS options: '+', '.join(sorted(required-set(kwargs))))
    record_state = kwargs.pop('record_state', False)
    if type(record_state) is not bool: raise TypeError('record_state must be boolean')
    runtime, workdir, limits = (kwargs.pop(k) for k in ('runtime','workdir','limits'))
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
