"""Explicit rotational BPM models for the pinned 30 September 2026 CLI."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Optional, Tuple

from fibernet.core.structure_graph import StructureGraph
from fibernet.units import LENGTH_FACTORS
from ._harmonic_profile import json_bytes, string_bytes
from .lammps_backend import (
    LAMMPSContractError, PreparedRun, RunSpec, RuntimeSpec, _BPM_MODEL, _digest, _integer, _json, _number, _vector,
)


_NATIVE_SHA256 = '599d046dde36bcf7666dacd45a0a6feb2265743058dea5c1e0d5970a57805784'
_NATIVE_BANNER = 'LAMMPS (30 Sep 2026)'
_LENGTH = 1e-6
_TIME = 1e-6
_MASS = 1e-15
_FORCE = 1e-9
_TORQUE = 1e-15
_MAX_PREPARED_BYTES = 256 * 1024
_MIN_BOND_MICRO = 1.
_CHECKPOINT = 'checkpoint.restart'
_STYLE = ('bond_style bpm/rotational frame average damping derivative smooth no normalize no '
          'break yes overlay/pair no store/local ruptures 1 time id1 id2')
_ATOM_COLUMNS = ('id type x y z vx vy vz c_orient[1] c_orient[2] c_orient[3] c_orient[4] '
                 'omegax omegay omegaz fx fy fz tqx tqy tqz mass diameter')
_EDGE_COLUMNS = 'index c_edges[1] c_edges[2] c_edges[3]'
_BREAK_COLUMNS = 'index f_ruptures[1] f_ruptures[2] f_ruptures[3]'


@dataclass(frozen=True)
class BPMParticle:
    """Mass in kg and sphere diameter in metres, set independently."""

    mass_kg: float
    diameter_m: float


@dataclass(frozen=True)
class RotationalBPMBond:
    """Twelve SI coefficients in native radial, shear, twist, bend order."""

    radial_stiffness_N_per_m: float
    shear_stiffness_N_per_m: float
    twist_stiffness_N_m_per_rad: float
    bend_stiffness_N_m_per_rad: float
    radial_limit_N: float
    shear_limit_N: float
    twist_limit_N_m: float
    bend_limit_N_m: float
    radial_damping_N_s_per_m: float = 0.0
    shear_damping_N_s_per_m: float = 0.0
    twist_damping_N_m_s_per_rad: float = 0.0
    bend_damping_N_m_s_per_rad: float = 0.0


@dataclass(frozen=True)
class BPMFrame:
    """One native step in SI and in the result's stable node-ID order."""

    step: int
    time_s: float
    positions_m: tuple
    velocities_m_s: tuple
    quaternions_wxyz: tuple
    angular_velocities_rad_s: tuple
    forces_N: tuple
    torques_N_m: tuple
    active_edge_ids: Tuple[int, ...]


@dataclass(frozen=True)
class BPMBreak:
    """A rupture at the native event step, using original graph IDs."""

    step: int
    time_s: float
    edge_id: int
    node_ids: Tuple[int, int]


@dataclass(frozen=True)
class BPMCheckpoint:
    """Owned binary restart generated and consumed within this run."""

    step: int
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class BPMRunResult:
    """SI motion, net forces/torques, and rupture history for the BPM model."""

    identity: str
    node_ids: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    frames: Tuple[BPMFrame, ...]
    breaks: Tuple[BPMBreak, ...]
    workdir: str
    checkpoint: Optional[BPMCheckpoint]


def _scale(value, unit, label, positive=False):
    return _number(value / unit, label, positive=positive)


def _record(row, cls, label):
    if type(row) is not cls:
        raise LAMMPSContractError(f'{label} requires {cls.__name__}')
    return tuple(_number(getattr(row, f.name), f.name) for f in fields(cls))


def _pairs(records, label):
    rows = tuple(records)
    values = dict(rows)
    if len(values) != len(rows):
        raise LAMMPSContractError(f'duplicate {label} IDs')
    return values


def _mapped(mapping, ids, label, complete):
    if mapping is None and not complete:
        return {}
    if not isinstance(mapping, Mapping) or len(mapping) > len(ids):
        raise LAMMPSContractError(f'{label} requires a mapping bounded by the graph IDs')
    result = {_integer(k, f'{label} ID'): v for k, v in mapping.items()}
    if (complete and set(result) != set(ids)) or not set(result) <= set(ids):
        raise LAMMPSContractError(f'{label} must match known graph IDs')
    return result


def prepare_rotational_bpm(graph: StructureGraph, *, length_unit: str,
                           maximum_bond_length_m: float, particles, bonds, run: RunSpec,
                           velocities_m_s=None, angular_velocities_rad_s=None,
                           checkpoint_step: Optional[int] = None) -> PreparedRun:
    """Compile an explicit rotational BPM network for use with LAMMPSTask.

    The pilot supports 2–32 mobile spheres, 1–64 straight unique bonds, and
    1–256 steps saved at every step. Particle and bond properties, the box,
    and velocities use SI. The length unit applies to graph positions.
    Initial orientations are identity and all damping is zero. Bonds
    must be at least one micrometre long. The optional checkpoint step
    splits a run around an owned binary restart. Graph materials and
    metadata are retained; the explicit records define the native model.
    """
    if not isinstance(graph, StructureGraph) or graph.dimension != 3:
        raise LAMMPSContractError('BPM requires a three-dimensional StructureGraph')
    if not 2 <= len(graph.nodes) <= 32 or not 1 <= len(graph.edges) <= 64:
        raise LAMMPSContractError('BPM requires 2–32 particles and 1–64 physical bonds')
    nodes, edges = [], []
    for key, node in sorted(graph.nodes.items()):
        key = _integer(key, 'node ID')
        if key != node.node_id:
            raise LAMMPSContractError('BPM node key differs from stored ID')
        nodes.append([key, _vector(node.position, 'site position')])
    for key, edge in sorted(graph.edges.items()):
        key = _integer(key, 'edge ID')
        if key != edge.edge_id or (edge.internal_points is not None and len(edge.internal_points)):
            raise LAMMPSContractError('BPM requires stable IDs and straight endpoint edges')
        edges.append([key, _integer(edge.node_i, 'edge endpoint'), _integer(edge.node_j, 'edge endpoint')])
    try:
        metadata = [graph.metadata] + [n.metadata for n in graph.nodes.values()]
        for edge in graph.edges.values():
            material = edge.material
            tensor = material.stiffness_tensor
            if tensor is not None and getattr(tensor, 'size', 37) > 36:
                raise ValueError('BPM graph provenance tensor exceeds the bound')
            metadata.extend((edge.metadata, material.to_dict()))
        json_bytes(metadata, _MAX_PREPARED_BYTES)
        full_graph = graph.to_dict()
        json_bytes(full_graph, _MAX_PREPARED_BYTES)
    except (AttributeError, OverflowError, TypeError, ValueError) as exc:
        raise LAMMPSContractError('BPM requires bounded native graph provenance') from exc
    return _compile(dict(nodes=nodes, edges=edges), _json(full_graph), length_unit,
                    maximum_bond_length_m, particles, bonds, run, velocities_m_s,
                    angular_velocities_rad_s, checkpoint_step)


def _compile(geometry, graph_json, length_unit, reach, particles, bonds, run, velocities, spins, checkpoint):
    if not isinstance(length_unit, str) or length_unit not in LENGTH_FACTORS:
        raise LAMMPSContractError('explicit graph length unit required for BPM')
    if type(run) is not RunSpec:
        raise LAMMPSContractError('BPM requires RunSpec')
    dt, steps, stride, box = run.checked(max_steps=256)
    if steps < 1 or stride != 1:
        raise LAMMPSContractError('BPM requires 1–256 steps sampled at every step')
    if checkpoint is not None:
        checkpoint = _integer(checkpoint, 'checkpoint step', 1, steps-1)
    box_native = tuple(tuple(_scale(x, _LENGTH, 'native box extent') for x in row) for row in box)
    if any(not math.isfinite(hi-lo) or lo >= hi for lo,hi in box_native):
        raise LAMMPSContractError('BPM native box extents must remain distinct and finite')
    dt_native = _scale(dt, _TIME, 'native timestep', True)
    _number(dt_native * steps, 'native integration duration', True)
    reach = _number(reach, 'maximum bond length (m)', True)
    reach_native = _scale(reach, _LENGTH, 'native bond reach', True)
    node_rows, edge_rows = geometry['nodes'], geometry['edges']
    node_ids, edge_ids = tuple(row[0] for row in node_rows), tuple(row[0] for row in edge_rows)
    if (not 2 <= len(node_ids) <= 32 or not 1 <= len(edge_ids) <= 64
            or any(tuple(sorted({_integer(i, 'stable ID') for i in ids})) != ids for ids in (node_ids, edge_ids))):
        raise LAMMPSContractError('BPM stable site and edge IDs must be unique and ordered')
    particles = _mapped(particles, node_ids, 'particles', True)
    bonds = _mapped(bonds, edge_ids, 'bonds', True)
    velocities = _mapped(velocities, node_ids, 'linear velocity', False)
    spins = _mapped(spins, node_ids, 'angular velocity', False)
    geometry_scale = LENGTH_FACTORS[length_unit]
    atom = {node: i for i,node in enumerate(node_ids,1)}
    points, particle_values, vel, omega = {}, {}, {}, {}
    for node, position in node_rows:
        points[node] = tuple(_scale(p * geometry_scale, _LENGTH, 'native position')
                             for p in _vector(position, 'site position'))
        if any(not lo < p < hi for p,(lo,hi) in zip(points[node],box_native)):
            raise LAMMPSContractError('every BPM particle must start strictly inside the declared box')
        mass, diameter = _record(particles[node], BPMParticle, 'each particle')
        native_mass = _scale(mass, _MASS, 'native particle mass', True)
        native_radius = _scale(diameter, _LENGTH, 'native particle diameter', True)/2
        _number(.4 * native_mass * native_radius * native_radius, 'native particle inertia', True)
        particle_values[node] = (mass, diameter)
        vel[node] = _vector(velocities.get(node, (0.,0.,0.)), 'initial velocity')
        omega[node] = _vector(spins.get(node, (0.,0.,0.)), 'initial angular velocity')
    seen, coefficients = set(), {}
    for eid, i, j in edge_rows:
        i, j = _integer(i, 'edge endpoint'), _integer(j, 'edge endpoint')
        pair = tuple(sorted((i,j)))
        if i not in atom or j not in atom or i == j or pair in seen:
            raise LAMMPSContractError('BPM requires unique non-self endpoint pairs')
        seen.add(pair)
        length = math.dist(points[i], points[j])
        if not _MIN_BOND_MICRO <= length <= reach_native:
            raise LAMMPSContractError('BPM initial bond length must be at least one micrometre and fit the declared reach')
        values = _record(bonds[eid], RotationalBPMBond, 'each bond')
        if values[0] <= 0 or any(x < 0 for x in values[1:4]) or any(x <= 0 for x in values[4:8]):
            raise LAMMPSContractError('BPM radial stiffness and thresholds must be positive; other stiffnesses nonnegative')
        if any(values[8:]):
            raise LAMMPSContractError('this BPM profile requires all four damping coefficients to be zero')
        coefficients[eid] = values

    contract = dict(model=_BPM_MODEL, units='micro', length_unit=length_unit, length_to_m=geometry_scale,
                    maximum_bond_length_m=reach, box_m=box, timestep_s=dt, steps=steps, sample_every=1,
                    checkpoint_step=checkpoint, particles=sorted(particle_values.items()),
                    bonds=sorted(coefficients.items()), velocities_m_s=sorted(vel.items()),
                    angular_velocities_rad_s=sorted(omega.items()), type_pair_cutoff_payload_lower_bound_bytes=64)
    contract_json = _json(contract)
    data_text = 'FiberNet rotational BPM creates sites and bonds in in.model.\n'
    native_script = _script(atom, points, particle_values, vel, omega, edge_rows, coefficients,
                            box_native, dt_native, reach_native, steps, checkpoint)
    marker = 'FIBERNET_LAMMPS_COMPLETE_' + _digest(_json([graph_json,contract_json,data_text]))
    native_script += [f'print "{marker}"']
    input_text = '\n'.join(native_script)+'\n'
    string_bytes((graph_json,contract_json,data_text,input_text),_MAX_PREPARED_BYTES)
    return PreparedRun(graph_json, contract_json, data_text, input_text,
                       node_ids, edge_ids, dt, steps, 1)


def _fmt(values):
    return ' '.join(format(x, '.17g') for x in values)


def _script(atom, points, particles, velocities, spins, edges, coefficients, box, dt, reach, steps, checkpoint):
    # One atom type with explicit per-particle mass and diameter.
    script = ['units micro','dimension 3','boundary f f f','atom_style bpm/sphere',
              'atom_modify map array','newton on off',
              'region cell block ' + _fmt(x for row in box for x in row) + ' units box',
              f'create_box 1 cell bond/types {len(edges)} extra/bond/per/atom {len(atom)-1}'
              f' extra/special/per/atom {len(atom)-1}']
    script += ['create_atoms 1 single ' + _fmt(p) + ' units box' for p in points.values()]
    for node, index in atom.items():
        mass, diameter = particles[node]
        script += [f'set atom {index} diameter {_fmt((diameter/_LENGTH,))} mass {_fmt((mass/_MASS,))}'
                   ' quat 1 0 0 0',
                   f'set atom {index} vx {velocities[node][0]:.17g} vy {velocities[node][1]:.17g}'
                   f' vz {velocities[node][2]:.17g} omega ' + _fmt(w * _TIME for w in spins[node])]
    pair = ['pair_style none','special_bonds lj 0 1 1 coul 1 1 1', _STYLE]
    units = (_FORCE/_LENGTH, _FORCE/_LENGTH, _TORQUE, _TORQUE, _FORCE, _FORCE, _TORQUE, _TORQUE)
    bond_coeff = [f'bond_coeff {row} ' + _fmt(_scale(x, u, 'native BPM coefficient', x > 0)
                   for x,u in zip(coefficients[eid],units)) + ' 0 0 0 0'
                  for row,(eid,_,_) in enumerate(edges,1)]
    script += pair + bond_coeff
    script += [f'create_bonds single/bond {row} {atom[i]} {atom[j]}'
               for row,(_,i,j) in enumerate(edges,1)]
    request = [f'comm_modify cutoff {reach:.17g} vel yes','neighbor 0.0 nsq',
               'neigh_modify every 1 delay 0 check no','fix integrate all nve/bpm/sphere',
               f'timestep {dt:.17g}','thermo 1','thermo_style custom step atoms bonds',
               'compute orient all property/atom quatw quati quatj quatk',
               'compute edges all property/local batom1 batom2 btype']
    script += request + _dumps('') + [f'run {steps if checkpoint is None else checkpoint}']
    if checkpoint is not None:
        script += [f'write_restart {_CHECKPOINT}','clear','newton on off',f'read_restart {_CHECKPOINT}']
        script += pair + bond_coeff + request + _dumps('-after') + [f'run {steps-checkpoint}']
    return script


def _dumps(suffix):
    return [f'dump trajectory all custom 1 trajectory{suffix}.dump {_ATOM_COLUMNS}',
            'dump_modify trajectory sort id units yes time yes format float %.17g',
            f'dump active all local 1 active{suffix}.dump {_EDGE_COLUMNS}',
            'dump_modify active format float %.17g',
            f'dump breaks all local 1 breaks{suffix}.dump {_BREAK_COLUMNS}',
            'dump_modify breaks format float %.17g']


def regenerate(prepared, contract):
    string_bytes((prepared.graph_json, prepared.contract_json, prepared.data_text, prepared.input_text),
                 _MAX_PREPARED_BYTES)
    particles = {n:BPMParticle(*p) for n,p in _pairs(contract['particles'],'particle').items()}
    bonds = {e:RotationalBPMBond(*b) for e,b in _pairs(contract['bonds'],'edge').items()}
    return prepare_rotational_bpm(StructureGraph.from_dict(json.loads(prepared.graph_json)),
        length_unit=contract['length_unit'], maximum_bond_length_m=contract['maximum_bond_length_m'],
        particles=particles, bonds=bonds,
        run=RunSpec(contract['timestep_s'],contract['steps'],contract['sample_every'],contract['box_m']),
        velocities_m_s=_pairs(contract['velocities_m_s'],'velocity'),
        angular_velocities_rad_s=_pairs(contract['angular_velocities_rad_s'],'angular velocity'),
        checkpoint_step=contract['checkpoint_step'])


def manifest_fields(prepared):
    return dict(schema='fibernet.lammps.rotational-bpm/1', identity=prepared.identity,
                graph_sha256=_digest(prepared.graph_json),
                files={'model.data':_digest(prepared.data_text),'in.model':_digest(prepared.input_text)},
                atom_to_node_id={str(i):n for i,n in enumerate(prepared.node_ids,1)},
                bond_type_to_edge_id={str(i):e for i,e in enumerate(prepared.edge_ids,1)},
                contract=json.loads(prepared.contract_json))


def check_runtime(runtime):
    if (type(runtime) is not RuntimeSpec or runtime.mode != 'cli_serial_static'
            or runtime.expected_version != 20260930 or runtime.expected_banner != _NATIVE_BANNER
            or dict(runtime.file_sha256).get(runtime.lammps_executable) != _NATIVE_SHA256):
        raise LAMMPSContractError('BPM requires the pinned official 30Sep2026 x86_64 static CLI')


def _native_real(text):
    try:
        return _number(float(text), 'BPM native field')
    except (OverflowError, ValueError) as exc:
        raise LAMMPSContractError('invalid or nonfinite BPM native field') from exc


def _native_int(text, label, lo=0, hi=2**31-1):
    try:
        return _integer(int(text), label, lo, hi)
    except ValueError as exc:
        raise LAMMPSContractError(f'invalid BPM native {label}') from exc


def _read_text(path, limit):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limit:
        raise LAMMPSContractError(f'invalid or oversized BPM artifact {path.name}')
    with path.open('rb') as file:
        payload = file.read(limit+1)
    if len(payload) > limit:
        raise LAMMPSContractError('BPM artifact exceeds output budget')
    return payload.decode('utf-8')


def _rows(text, steps, names, max_rows, contract, *, atoms=False):
    lines = iter(text.splitlines())

    def take():
        value = next(lines, None)
        if value is None:
            raise LAMMPSContractError('truncated BPM dump')
        return value.strip()

    def require(token):
        if take() != token:
            raise LAMMPSContractError(f'BPM dump expected {token}')

    if atoms:
        require('ITEM: UNITS')
        require('micro')
    for step in steps:
        if atoms:
            require('ITEM: TIME')
            time = _native_real(take()) * _TIME
            if not math.isclose(time, step*contract['timestep_s'], rel_tol=1e-12):
                raise LAMMPSContractError('BPM native time does not match the declared timestep')
        require('ITEM: TIMESTEP')
        if _native_int(take(), 'step', 0, contract['steps']) != step:
            raise LAMMPSContractError('BPM dump steps are missing or out of order')
        require('ITEM: NUMBER OF ATOMS' if atoms else 'ITEM: NUMBER OF ENTRIES')
        count = _native_int(take(), 'row count', max_rows if atoms else 0, max_rows)
        require('ITEM: BOX BOUNDS ff ff ff')
        for expected in contract['box_m']:
            given = take().split()
            if len(given)!=2 or any(not math.isclose(_native_real(x)*_LENGTH,y,rel_tol=1e-14)
                                    for x,y in zip(given,expected)):
                raise LAMMPSContractError('BPM native box differs from declared bounds')
        require(('ITEM: ATOMS ' if atoms else 'ITEM: ENTRIES ')+names)
        table = [take().split() for _ in range(count)]
        if any(len(row) != len(names.split()) for row in table):
            raise LAMMPSContractError('BPM native row width differs from the declared columns')
        yield step, table
    if next(lines, None) is not None:
        raise LAMMPSContractError('unexpected trailing BPM dump data')


def _atom_frame(step, table, active, prepared, contract):
    positions, native_positions = [], []
    velocities, quaternions, omega, forces, torques = [], [], [], [], []
    for idx,(row,(_, (mass,diameter))) in enumerate(zip(table,contract['particles']),1):
        if _native_int(row[0],'atom ID') != idx or _native_int(row[1],'atom type') != 1:
            raise LAMMPSContractError('BPM atom identity/order changed')
        values = tuple(_native_real(x) for x in row[2:])
        p = tuple(x*_LENGTH for x in values[:3])
        v = values[3:6]
        q = values[6:10]
        w = tuple(_number(x/_TIME,'BPM angular velocity in SI') for x in values[10:13])
        if (not math.isclose(values[19]*_MASS,mass,rel_tol=1e-12)
                or not math.isclose(values[20]*_LENGTH,diameter,rel_tol=1e-12)):
            raise LAMMPSContractError('BPM native mass or diameter differs from the compiler')
        if not math.isclose(math.hypot(*q),1.,rel_tol=1e-10):
            raise LAMMPSContractError('BPM quaternion is not normalized')
        if any(not lo/_LENGTH<x<hi/_LENGTH for x,(lo,hi) in zip(values[:3],contract['box_m'])):
            raise LAMMPSContractError('BPM node left the declared open box')
        positions.append(p)
        native_positions.append(values[:3])
        velocities.append(v)
        quaternions.append(q)
        omega.append(w)
        forces.append(tuple(x*_FORCE for x in values[13:16]))
        torques.append(tuple(x*_TORQUE for x in values[16:19]))
    frame = BPMFrame(step,step*prepared.timestep_s,tuple(positions),tuple(velocities),
                     tuple(quaternions),tuple(omega),tuple(forces),tuple(torques),tuple(active))
    return frame, native_positions


def _close_rows(a,b):
    return all(math.isclose(x,y,rel_tol=1e-12) for ra,rb in zip(a,b) for x,y in zip(ra,rb))


def _same_restart_atoms(before, after, active, stable_edges):
    degrees = [0] * len(before)
    for edge_id in active:
        i, j = stable_edges[edge_id]
        degrees[i] += 1
        degrees[j] += 1
    for left, right, degree in zip(before, after, degrees):
        left = tuple(map(float, left))
        right = tuple(map(float, right))
        # Atom IDs, position/velocity, orientation/spin and mass/diameter
        # are native stored fields. Compare them before any SI conversion.
        if left[:15] != right[:15] or left[21:] != right[21:]:
            return False
        # Restart rebuilds the neighbor order and can round the sum of
        # bond forces and torques differently. Bound each vector in its own
        # native units using its local scale and the incident bond count.
        for start in (15, 18):
            a, b = left[start:start+3], right[start:start+3]
            scale = max(map(abs, a+b))
            tolerance = 4 * degree * math.ulp(scale)
            if any(abs(x-y) > tolerance for x, y in zip(a, b)):
                return False
    return True


def parse_task_output(prepared, workdir, runtime, limit):
    """Read the fixed BPM task files after the shared watcher and worker finish."""
    def checkpoint_artifact(step):
        path = workdir/_CHECKPOINT
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise LAMMPSContractError('invalid owned BPM checkpoint artifact')
        with path.open('rb') as file:
            payload = file.read(limit+1)
        if len(payload)!=info.st_size:
            raise LAMMPSContractError('BPM checkpoint artifact changed while reading')
        return BPMCheckpoint(step,path.name,info.st_size,hashlib.sha256(payload).hexdigest())

    return _parse_task_text(prepared, runtime.expected_banner,
                            lambda name: _read_text(workdir/name, limit),
                            workdir=str(workdir), read_checkpoint=checkpoint_artifact)


def _parse_task_text(prepared, expected_banner, read_text, *, workdir='', read_checkpoint=None):
    """One evidence parser for managed output and captured offline text."""
    contract = json.loads(prepared.contract_json)
    checkpoint = contract['checkpoint_step']
    if checkpoint is not None and read_checkpoint is None:
        raise LAMMPSContractError('saved BPM readback does not support a restart')
    graph = json.loads(prepared.graph_json)
    edge_rows = [(e['id'],e['node_i'],e['node_j']) for e in graph['edges']]
    log = read_text('log.lammps').splitlines()
    if not log or log[0] != expected_banner or log.count(prepared.completion_marker) != 1:
        raise LAMMPSContractError('BPM native banner or completion marker missing')
    atom_ids = {n:i for i,n in enumerate(prepared.node_ids,1)}
    endpoint_records = {
        tuple(sorted((atom_ids[i],atom_ids[j]))):(ordinal,eid,(i,j))
        for ordinal,(eid,i,j) in enumerate(edge_rows,1)
    }
    stable_edges = {eid:(atom_ids[i]-1,atom_ids[j]-1) for eid,i,j in edge_rows}
    segments = [('',range(prepared.steps+1))] if checkpoint is None else [
        ('',range(checkpoint+1)),('-after',range(checkpoint,prepared.steps+1)),
    ]
    frames, events = [], {}
    for suffix, steps in segments:
        read = lambda name: read_text(f'{name}{suffix}.dump')
        atoms = _rows(read('trajectory'),steps,_ATOM_COLUMNS,len(atom_ids),contract,atoms=True)
        edges = _rows(read('active'),steps,_EDGE_COLUMNS,len(stable_edges),contract)
        breaks = _rows(read('breaks'),steps,_BREAK_COLUMNS,len(stable_edges),contract)
        for (step,ar),(_,er),(_,br) in zip(atoms,edges,breaks):
            active = set()
            for idx,row in enumerate(er,1):
                ordinal,i,j,btype = (_native_int(x,'bond field',1) for x in row)
                record = endpoint_records.get(tuple(sorted((i,j))))
                if record is None or ordinal!=idx or record[0]!=btype or record[1] in active:
                    raise LAMMPSContractError('unknown or duplicate BPM active bond')
                active.add(record[1])
            frame, native_positions = _atom_frame(step,ar,sorted(active),prepared,contract)
            for e in active:
                i,j = stable_edges[e]
                length = math.dist(native_positions[i],native_positions[j])
                if not _MIN_BOND_MICRO <= length <= contract['maximum_bond_length_m']/_LENGTH:
                    raise LAMMPSContractError('BPM active bond left declared length domain')
            if frames and frames[-1].step == step:
                if (frames[-1].active_edge_ids != frame.active_edge_ids
                        or not _same_restart_atoms(previous_atoms, ar, active, stable_edges)):
                    raise LAMMPSContractError('BPM restart boundary state differs from its owned checkpoint')
            else:
                frames.append(frame)
                previous_atoms = ar
            for idx,row in enumerate(br,1):
                ordinal,event_step,i,j = (_native_int(x,'rupture field') for x in row)
                record = endpoint_records.get(tuple(sorted((i,j))))
                if (record is None or ordinal!=idx or not steps.start <= event_step <= step
                        or record[1] in events):
                    raise LAMMPSContractError('unknown, duplicate or misdated BPM rupture')
                events[record[1]] = BPMBreak(event_step,event_step*prepared.timestep_s,record[1],record[2])
        # zip stops after the first producer's StopIteration. Finish their strict EOF checks.
        for generator in (edges,breaks):
            if next(generator,None) is not None:
                raise LAMMPSContractError('inconsistent BPM frame counts')
    initial = frames[0]
    expected_pos = [tuple(p*contract['length_to_m'] for p in row['position']) for row in graph['nodes']]
    if (initial.active_edge_ids != prepared.edge_ids
            or any(q != (1.,0.,0.,0.) for q in initial.quaternions_wxyz)
            or not _close_rows(initial.positions_m,expected_pos)
            or not _close_rows(initial.velocities_m_s,[v for _,v in contract['velocities_m_s']])
            or not _close_rows(initial.angular_velocities_rad_s,[v for _,v in contract['angular_velocities_rad_s']])):
        raise LAMMPSContractError('BPM initial state does not match the prepared input')
    prior = set(prepared.edge_ids)
    for frame in frames[1:]:
        active = set(frame.active_edge_ids)
        if not active <= prior or any(e not in events or events[e].step!=frame.step for e in prior-active):
            raise LAMMPSContractError('BPM rupture events do not match active topology by native step')
        prior = active
    if set(events) != set(prepared.edge_ids)-prior:
        raise LAMMPSContractError('BPM rupture events do not match final topology')
    owned = None if checkpoint is None else read_checkpoint(checkpoint)
    return BPMRunResult(prepared.identity,prepared.node_ids,prepared.edge_ids,tuple(frames),
                        tuple(sorted(events.values(),key=lambda e:(e.step,e.edge_id))),workdir,owned)
