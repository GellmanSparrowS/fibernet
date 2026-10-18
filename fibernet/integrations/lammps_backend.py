"""Bounded CLI adapter for explicit SI bead-and-harmonic-bond models.

StructureGraph is the sole source of topology and geometry. No chemistry,
continuum-to-molecular parameter conversion, resampling or physics solver lives
here. Native execution is explicit and always isolated in a bounded child.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real
import os
import platform
from pathlib import Path
import signal
import subprocess
import struct
import sys
import threading
import time
from typing import Mapping, Optional, Tuple

from fibernet.core.structure_graph import StructureGraph
from fibernet.units import LENGTH_FACTORS


class LAMMPSContractError(ValueError):
    """Invalid or unsupported input/output; never a successful empty result."""


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LAMMPSContractError(f'{label} must be a finite real number')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise LAMMPSContractError(f'{label} must be finite' + (' and positive' if positive else ''))
    return value


def _integer(value, label, minimum=0, maximum=2**31-1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise LAMMPSContractError(f'{label} must be an integer in [{minimum}, {maximum}]')
    return int(value)


def _vector(value, label):
    if len(value) != 3:
        raise LAMMPSContractError(f'{label} must have three components')
    return tuple(_number(x, label) for x in value)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise LAMMPSContractError('inputs must have finite JSON provenance') from exc


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class HarmonicBond:
    stiffness_N_per_m: float
    equilibrium_length_m: float

    def checked(self):
        return (_number(self.stiffness_N_per_m, 'stiffness_N_per_m', True),
                _number(self.equilibrium_length_m, 'equilibrium_length_m', True))


@dataclass(frozen=True)
class RunSpec:
    timestep_s: float
    steps: int
    sample_every: int
    box_m: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

    def checked(self):
        dt = _number(self.timestep_s, 'timestep_s', True)
        steps = _integer(self.steps, 'steps', 0, 100000)
        every = _integer(self.sample_every, 'sample_every', 1, max(1, steps))
        if steps % every:
            raise LAMMPSContractError('steps must be divisible by sample_every')
        if len(self.box_m) != 3:
            raise LAMMPSContractError('box_m must contain three lo/hi pairs')
        box = tuple(tuple(_number(x, 'box_m') for x in row) for row in self.box_m)
        if any(len(row) != 2 or row[0] >= row[1] for row in box):
            raise LAMMPSContractError('box_m requires finite lo < hi on each axis')
        _number(dt*steps, 'physical duration')
        spans = [_number(hi-lo, 'box span', True) for lo,hi in box]
        _number(math.hypot(*spans), 'box diagonal', True)
        return dt, steps, every, box


@dataclass(frozen=True)
class PreparedRun:
    graph_json: str
    contract_json: str
    data_text: str
    input_text: str
    node_ids: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    timestep_s: float
    steps: int
    sample_every: int

    @property
    def identity(self):
        return _digest(_json([self.graph_json, self.contract_json, self.data_text, self.input_text]))

    @property
    def completion_marker(self):
        return 'FIBERNET_LAMMPS_COMPLETE_' + _digest(_json([self.graph_json,self.contract_json,self.data_text]))

    def manifest(self):
        return dict(schema='fibernet.lammps.explicit-beads-exact-mass-types.v3', identity=self.identity,
                    graph_sha256=_digest(self.graph_json),
                    files={'model.data': _digest(self.data_text), 'in.model': _digest(self.input_text)},
                    atom_to_node_id={str(i+1): n for i, n in enumerate(self.node_ids)},
                    bond_to_edge_id={str(i+1): n for i, n in enumerate(self.edge_ids)},
                    atom_to_type_id={str(a):t for a,t in json.loads(self.contract_json)['atom_to_type_id']},
                    type_to_mass_kg={str(t):m for t,m in json.loads(self.contract_json)['type_to_mass_kg']},
                    contract=json.loads(self.contract_json))


def prepare_harmonic_beads(graph: StructureGraph, *, length_unit: str,
                           maximum_bond_length_m: float,
                           masses_kg: Mapping[int, float],
                           bonds: Mapping[int, HarmonicBond], run: RunSpec,
                           fixed_node_ids=(), forces_N=None, velocities_m_s=None) -> PreparedRun:
    """Compile explicit synthetic/CG beads, not atomistic chemistry or beam FEM.

    Caller supplies every bead mass and every edge's equilibrium length and
    physical spring stiffness. LAMMPS harmonic K is stiffness/2. Material records
    are retained as provenance, never converted to molecular force fields.
    All geometric boundary flags are retained; fixed constraints are explicit.
    Only 3-D nonperiodic boxes, straight edges, whole-node fixed constraints,
    constant forces and zero/default or explicit velocities are supported.
    A positive maximum_bond_length_m is mandatory for pair-free communication;
    every integration step must be retained and remain in this declared domain.
    This source revision is not distributed/MPI or arbitrary-strain qualification.
    """
    if not isinstance(graph, StructureGraph):
        raise LAMMPSContractError('expected the authoritative StructureGraph')
    if graph.dimension != 3:
        raise LAMMPSContractError('first qualification supports dimension=3 only')
    if length_unit not in LENGTH_FACTORS:
        raise LAMMPSContractError('explicit supported geometry length_unit required')
    scale = LENGTH_FACTORS[length_unit]
    dt, steps, every, box = run.checked()
    reach = _number(maximum_bond_length_m, 'maximum_bond_length_m', True)
    if reach > math.hypot(*(hi-lo for lo,hi in box)):
        raise LAMMPSContractError('declared maximum bond length exceeds box diagonal')
    if every != 1:
        raise LAMMPSContractError('pair-none qualification requires every integration step retained')
    node_ids = tuple(sorted(graph.nodes))
    edge_ids = tuple(sorted(graph.edges))
    if not 2 <= len(node_ids) <= 256 or not 1 <= len(edge_ids) <= 1024:
        raise LAMMPSContractError('bounded pilot requires 2..256 beads and 1..1024 bonds')
    for key in (*node_ids, *edge_ids):
        _integer(key, 'graph stable ID')
    for mapping in (masses_kg, bonds, forces_N or {}, velocities_m_s or {}):
        for key in mapping: _integer(key, 'mapping stable ID')
    if set(masses_kg) != set(node_ids) or set(bonds) != set(edge_ids):
        raise LAMMPSContractError('mass/bond mappings must exactly cover stable graph IDs')
    atom = {nid: i+1 for i, nid in enumerate(node_ids)}
    fixed = tuple(sorted(_integer(x, 'fixed node ID') for x in fixed_node_ids))
    if len(set(fixed)) != len(fixed) or not set(fixed) <= set(node_ids) or len(fixed) == len(node_ids):
        raise LAMMPSContractError('fixed IDs must be unique known nodes with at least one mobile bead')
    forces = {} if forces_N is None else {k: _vector(v, 'force_N') for k, v in forces_N.items()}
    velocities = {} if velocities_m_s is None else {k: _vector(v, 'velocity_m_s') for k, v in velocities_m_s.items()}
    if not (set(forces) | set(velocities)) <= set(node_ids):
        raise LAMMPSContractError('force/velocity mapping contains unknown node IDs')
    if set(forces) & set(fixed) or any(any(velocities.get(n, (0, 0, 0))) for n in fixed):
        raise LAMMPSContractError('fixed nodes cannot carry applied force or nonzero velocity')
    if len(forces) > 20:
        raise LAMMPSContractError('first qualification allows at most 20 loaded nodes (LAMMPS group limit)')
    points, masses = {}, {}
    for nid in node_ids:
        node = graph.nodes[nid]
        if node.node_id != nid:
            raise LAMMPSContractError('node record ID disagrees with graph key')
        points[nid] = tuple(_number(x*scale, 'converted node position') for x in _vector(node.position, 'node position'))
        if any(not box[a][0] < points[nid][a] < box[a][1] for a in range(3)):
            raise LAMMPSContractError('every bead must start strictly inside the explicit box')
        masses[nid] = _number(masses_kg[nid], 'mass_kg', True)
    # Only mass is type-dependent in this exact, pair-free bond model.
    # Group normalized binary64 values exactly; metadata/loads never become keys.
    # Future per-type properties require a new full key schema before admission.
    mass_keys = sorted({m.hex() for m in masses.values()}, key=float.fromhex)
    type_by_mass = {key:i+1 for i,key in enumerate(mass_keys)}
    atom_type = {n:type_by_mass[masses[n].hex()] for n in node_ids}
    type_mass = {i+1:float.fromhex(key) for i,key in enumerate(mass_keys)}
    type_table_lower_bound = 16*(len(type_mass)+1)**2
    seen, coefficients = set(), {}
    for eid in edge_ids:
        edge = graph.edges[eid]
        pair = tuple(sorted((edge.node_i, edge.node_j)))
        if edge.edge_id != eid or not set(pair) <= set(node_ids) or pair[0] == pair[1]:
            raise LAMMPSContractError('invalid edge identity/endpoints')
        if pair in seen:
            raise LAMMPSContractError('parallel edges need an explicit model contract; never aggregated')
        seen.add(pair)
        if edge.internal_points is not None and len(edge.internal_points):
            raise LAMMPSContractError('curved/discretized edges are unsupported; no silent flattening')
        if points[pair[0]] == points[pair[1]]:
            raise LAMMPSContractError('coincident bonded bead positions are unsupported')
        if not isinstance(bonds[eid], HarmonicBond):
            raise LAMMPSContractError('each edge requires explicit HarmonicBond coefficients')
        coefficients[eid] = bonds[eid].checked()
        separation = math.dist(points[pair[0]], points[pair[1]])
        if separation > reach or coefficients[eid][1] > reach:
            raise LAMMPSContractError('initial/rest bond length exceeds declared maximum reach')
        _number(coefficients[eid][0]/2, 'LAMMPS harmonic K', True)
    graph_json = _json(graph.to_dict())
    contract = dict(model='explicit-synthetic-or-coarse-grained-beads', chemistry=False,
                    material_mapping='graph material retained; no inferred force-field parameters',
                    length_unit=length_unit, length_to_m=scale, units='si',
                    maximum_bond_length_m=reach, pair_policy='none',
                    execution_scope='one-rank qualification; all retained integration steps checked; no continuous-time or MPI deformation guarantee',
                    masses_kg=sorted(masses.items()), bonds=sorted(coefficients.items()),
                    atom_type_key_schema='mass_kg.binary64.v1',
                    atom_to_type_id=[(atom[n],atom_type[n]) for n in node_ids],
                    type_to_mass_kg=sorted(type_mass.items()),
                    type_pair_cutoff_payload_lower_bound_bytes=type_table_lower_bound,
                    boundary=['f','f','f'], fixed_node_ids=fixed,
                    forces_N=sorted(forces.items()), velocities_m_s=sorted(velocities.items()),
                    timestep_s=dt, steps=steps, sample_every=every, box_m=box,
                    required_packages=['MOLECULE'],
                    required_styles={'atom':['bond'],'bond':['harmonic'],'fix':['nve']},
                    energy='total unnormalized J; load work not included in etotal',
                    stress='global pressure tensor Pa, compression-positive; not fiber/Cauchy stress',
                    restart='not supported; cancellation cannot resume')
    lines = ['FiberNet explicit bead model (SI)', '', f'{len(node_ids)} atoms',
             f'{len(edge_ids)} bonds', '', f'{len(type_mass)} atom types', f'{len(edge_ids)} bond types', '']
    lines += [f'{lo:.17g} {hi:.17g} {axis}lo {axis}hi' for (lo,hi),axis in zip(box,'xyz')]
    lines += ['', 'Masses', ''] + [f'{t} {mass:.17g}' for t,mass in sorted(type_mass.items())]
    lines += ['', 'Atoms # bond', '']
    lines += [f'{atom[n]} 1 {atom_type[n]} ' + ' '.join(f'{x:.17g}' for x in points[n]) for n in node_ids]
    lines += ['', 'Bonds', '']
    lines += [f'{i} {i} {atom[graph.edges[e].node_i]} {atom[graph.edges[e].node_j]}' for i,e in enumerate(edge_ids,1)]
    lines += ['', 'Velocities', '']
    lines += [f'{atom[n]} ' + ' '.join(f'{x:.17g}' for x in velocities.get(n,(0,0,0))) for n in node_ids]
    script = ['units si', 'dimension 3', 'boundary f f f', 'atom_style bond',
              'atom_modify map array', 'bond_style harmonic', 'read_data model.data',
              'pair_style none', 'special_bonds lj/coul 1 1 1']
    script += [f'bond_coeff {i} {coefficients[e][0]/2:.17g} {coefficients[e][1]:.17g}' for i,e in enumerate(edge_ids,1)]
    # A nonzero communication cutoff avoids lost bonds with pair-free models.
    cutoff = reach
    script += [f'comm_modify cutoff {cutoff:.17g}', 'neighbor 0.0 nsq',
               'neigh_modify every 1 delay 0 check no']
    if fixed:
        script += ['group held id ' + ' '.join(str(atom[n]) for n in fixed),
                   'group mobile subtract all held']
    else:
        script += ['group mobile union all']
    # Held beads are excluded from integration, preserving their native forces.
    for i,n in enumerate(sorted(forces)):
        script += [f'group loaded{i} id {atom[n]}',
                   f'fix load{i} loaded{i} addforce ' + ' '.join(f'{x:.17g}' for x in forces[n])]
    script += ['fix integrate mobile nve', f'timestep {dt:.17g}', f'thermo {every}',
               'thermo_style custom step atoms pe ke etotal pxx pyy pzz pxy pxz pyz',
               'thermo_modify norm no flush yes format float %.17g lost error',
               f'dump trajectory all custom {every} trajectory.dump id type x y z vx vy vz fx fy fz',
               'dump_modify trajectory sort id units yes time yes format float %.17g', f'run {steps}']
    data_text = '\n'.join(lines)+'\n'
    marker = 'FIBERNET_LAMMPS_COMPLETE_' + _digest(_json([graph_json,_json(contract),data_text]))
    script += [f'print "{marker}"']
    return PreparedRun(graph_json, _json(contract), '\n'.join(lines)+'\n',
                       '\n'.join(script)+'\n', node_ids, edge_ids, dt, steps, every)


def validate_prepared(prepared: PreparedRun):
    """Regenerate the compiler protocol; a public dataclass is not authority."""
    if not isinstance(prepared, PreparedRun):
        raise LAMMPSContractError('expected PreparedRun')
    try:
        c = json.loads(prepared.contract_json)
        graph = StructureGraph.from_dict(json.loads(prepared.graph_json))
        regenerated = prepare_harmonic_beads(graph, length_unit=c['length_unit'],
            maximum_bond_length_m=c['maximum_bond_length_m'],
            masses_kg=dict(c['masses_kg']),
            bonds={k:HarmonicBond(*v) for k,v in c['bonds']},
            run=RunSpec(c['timestep_s'],c['steps'],c['sample_every'],tuple(tuple(r) for r in c['box_m'])),
            fixed_node_ids=c['fixed_node_ids'], forces_N=dict(c['forces_N']),
            velocities_m_s=dict(c['velocities_m_s']))
    except Exception as exc:
        raise LAMMPSContractError('prepared contract cannot be regenerated') from exc
    if regenerated != prepared:
        raise LAMMPSContractError('prepared files/identity differ from deterministic compiler output')
    return prepared


@dataclass(frozen=True)
class TrajectoryFrame:
    step: int
    time_s: float
    node_ids: Tuple[int, ...]
    positions_m: tuple
    velocities_m_s: tuple
    forces_N: tuple


@dataclass(frozen=True)
class ThermoFrame:
    step: int
    potential_J: float
    kinetic_J: float
    total_J: float
    pressure_tensor_Pa: tuple  # xx, yy, zz, xy, xz, yz; compression-positive


@dataclass(frozen=True)
class LAMMPSRunResult:
    identity: str
    trajectory: Tuple[TrajectoryFrame, ...]
    thermo: Tuple[ThermoFrame, ...]
    workdir: str


def parse_outputs(prepared: PreparedRun, trajectory_text: str, log_text: str,
                  *, max_output_bytes=8*1024*1024) -> LAMMPSRunResult:
    """Strict parser for this compiler's exact dump/thermo protocol only."""
    validate_prepared(prepared)
    if len(trajectory_text.encode()) + len(log_text.encode()) > max_output_bytes:
        raise LAMMPSContractError('output byte budget exceeded')
    if log_text.splitlines().count(prepared.completion_marker) != 1:
        raise LAMMPSContractError('native completion marker missing')
    expected_types = dict(json.loads(prepared.contract_json)['atom_to_type_id'])
    lines = iter(trajectory_text.splitlines())
    frames = []
    expected = list(range(0, prepared.steps+1, prepared.sample_every))
    def take():
        try: return next(lines)
        except StopIteration as exc: raise LAMMPSContractError('truncated trajectory') from exc
    try:
        if take() != 'ITEM: UNITS' or take() != 'si':
            raise LAMMPSContractError('missing or incorrect native unit label')
        for step in expected:
            if take() != 'ITEM: TIME':
                raise LAMMPSContractError('native physical time missing')
            actual_time = float(take())
            if not math.isfinite(actual_time) or not math.isclose(actual_time,step*prepared.timestep_s,rel_tol=1e-12,abs_tol=0.0):
                raise LAMMPSContractError('native physical time differs from step/timestep')
            if take() != 'ITEM: TIMESTEP' or int(take()) != step:
                raise LAMMPSContractError('missing, extra or out-of-order trajectory step')
            if take() != 'ITEM: NUMBER OF ATOMS' or int(take()) != len(prepared.node_ids):
                raise LAMMPSContractError('atom count changed')
            if take() != 'ITEM: BOX BOUNDS ff ff ff':
                raise LAMMPSContractError('unexpected trajectory box type')
            box = json.loads(prepared.contract_json)['box_m']
            for row in box:
                actual = tuple(float(x) for x in take().split())
                if len(actual) != 2 or any(not math.isfinite(x) for x in actual) or any(
                    not math.isclose(a,b,rel_tol=1e-12,abs_tol=0.0) for a,b in zip(actual,row)):
                    raise LAMMPSContractError('trajectory box differs from contract')
            if take() != 'ITEM: ATOMS id type x y z vx vy vz fx fy fz':
                raise LAMMPSContractError('unexpected dump fields')
            rows = {}
            for _ in prepared.node_ids:
                parts = take().split()
                if len(parts) != 11:
                    raise LAMMPSContractError('malformed atom row')
                aid, atype = int(parts[0]), int(parts[1])
                if aid in rows or aid not in range(1,len(prepared.node_ids)+1) or atype != expected_types[aid]:
                    raise LAMMPSContractError('duplicate, unknown or mismatched atom ID/type')
                values = tuple(float(x) for x in parts[2:])
                if not all(math.isfinite(x) for x in values):
                    raise LAMMPSContractError('nonfinite atom output')
                rows[aid] = values
            ordered = [rows[i] for i in range(1,len(rows)+1)]
            frames.append(TrajectoryFrame(step, step*prepared.timestep_s, prepared.node_ids,
                          tuple(r[:3] for r in ordered), tuple(r[3:6] for r in ordered), tuple(r[6:] for r in ordered)))
        if any(line.strip() for line in lines):
            raise LAMMPSContractError('unexpected extra trajectory data')
        source = json.loads(prepared.graph_json)
        contract = json.loads(prepared.contract_json)
        by_id = {nid:i for i,nid in enumerate(prepared.node_ids)}
        for frame in frames:
            for point in frame.positions_m:
                if any(not contract['box_m'][a][0] < point[a] < contract['box_m'][a][1] for a in range(3)):
                    raise LAMMPSContractError('trajectory site left the declared fixed-box interior')
            for edge in source['edges']:
                r = math.dist(frame.positions_m[by_id[edge['node_i']]], frame.positions_m[by_id[edge['node_j']]])
                if not 0 < r <= contract['maximum_bond_length_m']:
                    raise LAMMPSContractError('trajectory bond left declared maximum-reach domain')
        initial_velocity = dict(contract['velocities_m_s'])
        for i,node in enumerate(source['nodes']):
            initial = tuple(x*contract['length_to_m'] for x in node['position'])
            velocity = initial_velocity.get(node['id'],(0,0,0))
            if any(not math.isclose(a,b,rel_tol=1e-13,abs_tol=0) for a,b in zip(initial,frames[0].positions_m[i])) or any(
                not math.isclose(a,b,rel_tol=1e-13,abs_tol=0) for a,b in zip(velocity,frames[0].velocities_m_s[i])):
                raise LAMMPSContractError('initial trajectory does not match prepared graph/state')
        header = 'Step Atoms PotEng KinEng TotEng Pxx Pyy Pzz Pxy Pxz Pyz'.split()
        thermo, reading = [], False
        for line in log_text.splitlines():
            parts = line.split()
            if parts == header:
                if reading or thermo: raise LAMMPSContractError('multiple thermo runs')
                reading = True
                continue
            if reading:
                if parts and parts[0] == 'Loop':
                    reading = False
                    continue
                if len(parts) != 11:
                    raise LAMMPSContractError('unexpected content inside thermo output')
                values = tuple(float(x) for x in parts)
                if not all(math.isfinite(x) for x in values) or values[0] != int(values[0]) or values[1] != len(prepared.node_ids):
                    raise LAMMPSContractError('invalid thermo values/count')
                if values[3] < 0 or not math.isclose(values[4],values[2]+values[3],rel_tol=1e-12,abs_tol=0.0):
                    raise LAMMPSContractError('inconsistent native energy values')
                thermo.append(ThermoFrame(int(values[0]), *values[2:5], values[5:]))
        if reading or [r.step for r in thermo] != expected:
            raise LAMMPSContractError('missing/inconsistent thermo steps')
    except (ValueError, OverflowError) as exc:
        if isinstance(exc, LAMMPSContractError): raise
        raise LAMMPSContractError('malformed numeric output') from exc
    return LAMMPSRunResult(prepared.identity, tuple(frames), tuple(thermo), '')


class TaskState(str, Enum):
    CREATED = 'created'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    TIMED_OUT = 'timed_out'
    RESOURCE_LIMIT = 'resource_limit'


@dataclass(frozen=True)
class ResourceLimits:
    cpu_seconds: int = 5
    wall_seconds: float = 10.0
    memory_bytes: int = 1024*1024*1024
    output_bytes: int = 8*1024*1024
    file_size_bytes: int = 8*1024*1024  # native regular files; distinct from saved artifacts

    def checked(self):
        _integer(self.cpu_seconds, 'cpu_seconds', 1, 60)
        _number(self.wall_seconds, 'wall_seconds', True)
        if self.wall_seconds > 120: raise LAMMPSContractError('wall_seconds exceeds bounded pilot cap')
        _integer(self.memory_bytes, 'memory_bytes', 64*1024*1024, 2*1024*1024*1024)
        _integer(self.output_bytes, 'output_bytes', 1024, 16*1024*1024)
        _integer(self.file_size_bytes, 'file_size_bytes', 1024, 16*1024*1024)
        return self


def _check_serial_static_elf(path):
    """Admit only the bounded ELF64 x86_64 static executable layout.

    This is an admission filter, not a general ELF loader or a capability probe.
    In particular, dynamic tables and interpreters are never admitted.
    """
    with open(path, 'rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        header = stream.read(64)
        if len(header) != 64:
            raise LAMMPSContractError('truncated serial ELF header')
        fields = struct.unpack('<16sHHIQQQIHHHHHH', header)
        ident, kind, machine, version, entry, phoff, shoff, flags, ehsize, phsize, phnum, shsize, shnum, shstr = fields
        if (ident != b'\x7fELF\x02\x01\x01' + b'\0'*9 or
                (kind, machine, version, flags, ehsize) != (2, 62, 1, 0, 64)):
            raise LAMMPSContractError('unsupported serial ELF identity/header')
        if not (phoff >= 64 and phsize == 56 and 1 <= phnum <= 64 and
                phoff + phsize*phnum <= size):
            raise LAMMPSContractError('invalid or unbounded serial ELF program table')
        # Section contents are not interpreted. Reject extended numbering and
        # malformed table extents; dynamic execution is rejected by segments.
        if shnum == 0:
            if shoff != 0 or shstr != 0 or shsize not in (0, 64):
                raise LAMMPSContractError('unsupported serial ELF section numbering')
        elif not (1 <= shnum <= 4096 and shsize == 64 and shoff >= 64 and
                  shoff + shsize*shnum <= size and shstr < shnum):
            raise LAMMPSContractError('invalid or unbounded serial ELF section table')
        stream.seek(phoff)
        table = stream.read(phsize*phnum)
        if len(table) != phsize*phnum:
            raise LAMMPSContractError('truncated serial ELF program table')
        executable_entry = False
        for index in range(phnum):
            segment, permissions, offset, address, _, filesz, memsz, alignment = struct.unpack_from('<IIQQQQQQ', table, index*56)
            if segment not in (1, 7, 0x6474e551, 0x6474e552):
                raise LAMMPSContractError('unsupported or dynamic serial ELF segment')
            if (offset > size or filesz > size-offset or filesz > memsz or
                    address + memsz > 2**64 or permissions & ~7 or
                    (alignment not in (0, 1) and alignment & (alignment-1))):
                raise LAMMPSContractError('malformed serial ELF segment')
            if segment == 0x6474e551 and permissions & 1:
                raise LAMMPSContractError('executable serial ELF stack is not admitted')
            if segment == 1:
                if alignment > 1 and address % alignment != offset % alignment:
                    raise LAMMPSContractError('misaligned serial ELF load segment')
                if permissions & 1 and address <= entry < address+filesz:
                    executable_entry = True
        if not executable_entry:
            raise LAMMPSContractError('serial ELF entry is not file-backed executable code')


def _check_serial_static_runtime(runtime):
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise LAMMPSContractError('native execution requires the qualified Linux x86_64 runtime layout')
    paths = (runtime.python_executable, runtime.lammps_executable)
    if any(not isinstance(p, str) or not p or not Path(p).is_absolute() or
           str(Path(p).resolve()) != p or not Path(p).is_file() or
           not os.access(p, os.X_OK) for p in paths):
        raise LAMMPSContractError('serial runtime requires existing canonical absolute executable files')
    if runtime.package_dir != '' or runtime.environment != ():
        raise LAMMPSContractError('serial static runtime requires no package directory or environment overrides')
    _integer(runtime.expected_version, 'expected_version', 20000101, 21000101)
    if (not isinstance(runtime.expected_banner, str) or
            not runtime.expected_banner.startswith('LAMMPS (') or '\n' in runtime.expected_banner):
        raise LAMMPSContractError('exact preflight-verified native log banner required')
    pins = {}
    if not isinstance(runtime.file_sha256, tuple):
        raise LAMMPSContractError('serial runtime identity pins must be explicit pairs')
    for pin in runtime.file_sha256:
        if not isinstance(pin, tuple) or len(pin) != 2:
            raise LAMMPSContractError('invalid serial runtime identity pin')
        path, expected = pin
        if (not isinstance(path, str) or not Path(path).is_absolute() or
                str(Path(path).resolve()) != path or not Path(path).is_file() or
                path in pins or not isinstance(expected, str) or len(expected) != 64 or
                any(c not in '0123456789abcdef' for c in expected)):
            raise LAMMPSContractError('invalid or duplicate serial runtime identity pin')
        pins[path] = expected
    required = {str(Path(p).resolve()) for p in paths}
    if not required <= set(pins):
        raise LAMMPSContractError('serial native/Python identity pins incomplete')
    for path, expected in pins.items():
        with open(path, 'rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
        if digest.hexdigest() != expected:
            raise LAMMPSContractError('runtime identity mismatch: ' + path)
    _check_serial_static_elf(runtime.lammps_executable)
    return runtime


@dataclass(frozen=True)
class RuntimeSpec:
    mode: str  # cli or cli_serial_static; linked Python-engine execution is not admitted
    python_executable: str  # runs the isolated stdlib resource-limit worker
    lammps_executable: Optional[str] = None
    package_dir: str = ''
    expected_version: int = 0
    expected_banner: str = ''
    file_sha256: Tuple[Tuple[str, str], ...] = ()
    environment: Tuple[Tuple[str, str], ...] = ()

    def checked(self):
        if self.mode == 'cli_serial_static':
            return _check_serial_static_runtime(self)
        if self.mode != 'cli':
            raise LAMMPSContractError('only CLI execution is supported; linked Python-engine mode is not admitted')
        if sys.platform != 'linux' or platform.machine() != 'x86_64':
            raise LAMMPSContractError('native execution requires the qualified Linux x86_64 runtime layout')
        paths = [self.python_executable, self.lammps_executable]
        if any(not p or not Path(p).is_absolute() or not Path(p).is_file() for p in paths):
            raise LAMMPSContractError('runtime executables must be explicit existing absolute files')
        if len(dict(self.environment)) != len(self.environment) or any(k not in ('LD_LIBRARY_PATH','UCX_TLS') for k,v in self.environment):
            raise LAMMPSContractError('only explicit existing-runtime library/MPI environment is supported')
        package = Path(self.package_dir)
        if not package.is_absolute() or not package.is_dir():
            raise LAMMPSContractError('explicit installed LAMMPS package directory required')
        _integer(self.expected_version, 'expected_version', 20000101, 21000101)
        if not self.expected_banner.startswith('LAMMPS (') or '\n' in self.expected_banner:
            raise LAMMPSContractError('exact preflight-verified native log banner required')
        pins = dict(self.file_sha256)
        required = {str(Path(self.python_executable).resolve()), str(package/'liblammps.so'), str(package/'liblammps.so.0')}
        required.update(str(p) for p in (package.parent/'lammps.libs').glob('*') if p.is_file())
        if Path(self.lammps_executable).resolve() != (package/'lmp').resolve():
            raise LAMMPSContractError('CLI must use native package lmp, never bin/lmp wrapper')
        required.add(str(Path(self.lammps_executable).resolve()))
        environment = dict(self.environment)
        library_dir = environment.get('LD_LIBRARY_PATH')
        if not library_dir or ':' in library_dir:
            raise LAMMPSContractError('single explicit existing MPI library directory required')
        required.add(str(Path(library_dir)/'libmpi.so.12'))
        if not required <= set(pins) or len(pins) != len(self.file_sha256):
            raise LAMMPSContractError('runtime binary/library/Python source identity pins incomplete')
        for path, expected in pins.items():
            if not Path(path).is_absolute() or len(expected) != 64:
                raise LAMMPSContractError('invalid runtime identity pin')
            with open(path, 'rb') as stream:
                h = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
            if h.hexdigest() != expected:
                raise LAMMPSContractError('runtime identity mismatch: ' + path)
        return self


class LAMMPSTask:
    """One-shot POSIX child lifecycle. No native work happens on construction.

    Watchdog enforces wall and aggregate output limits without polling by user.
    CPU/address-space/per-file limits are applied in the child before native
    loading. Output aggregate limit may overshoot by one watchdog interval;
    per-file kernel limits remain hard. Cancel/timeout are terminal, never success.
    """
    def __init__(self, prepared: PreparedRun, runtime: RuntimeSpec,
                 workdir, limits=ResourceLimits()):
        self.prepared, self.limits = validate_prepared(prepared), limits.checked()
        lower_bound = json.loads(self.prepared.contract_json)['type_pair_cutoff_payload_lower_bound_bytes']
        if lower_bound > self.limits.memory_bytes:
            raise LAMMPSContractError('type-pair cutoff tables alone exceed declared memory budget')
        self.runtime = runtime.checked()
        self.workdir = Path(workdir).absolute()
        self.state, self.error, self.result = TaskState.CREATED, None, None
        self._process, self._thread, self._stream = None, None, None
        self._lock, self._done = threading.RLock(), threading.Event()

    def start(self):
        with self._lock:
            if self.state != TaskState.CREATED:
                raise LAMMPSContractError('task can only start once')
            if os.name != 'posix':
                raise LAMMPSContractError('bounded execution currently requires POSIX resource limits')
            validate_prepared(self.prepared)
            self.runtime.checked()
            self.workdir.mkdir(parents=False, exist_ok=False)
            try:
                for name, data in [('model.data',self.prepared.data_text),('in.model',self.prepared.input_text),
                                   ('graph.json',self.prepared.graph_json),('manifest.json',_json(dict(self.prepared.manifest(),runtime=asdict(self.runtime),limits=asdict(self.limits))))]:
                    (self.workdir/name).write_text(data, encoding='utf-8')
                scratch = self.workdir/'scratch'
                scratch.mkdir(exist_ok=False)
                env = {'PATH':'/usr/bin:/bin', 'LANG':'C', 'TMPDIR':str(scratch)}
                env.update(dict(self.runtime.environment))
                env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                worker = Path(__file__).with_name('_lammps_worker.py').resolve()
                if self._artifact_bytes() > self.limits.output_bytes:
                    raise LAMMPSContractError('input artifacts exceed total byte budget')
                config = dict(mode=self.runtime.mode, executable=self.runtime.lammps_executable,
                              package_dir=self.runtime.package_dir, expected_version=self.runtime.expected_version,
                              file_sha256=self.runtime.file_sha256,
                              cpu_seconds=self.limits.cpu_seconds, memory_bytes=self.limits.memory_bytes,
                              file_size_bytes=self.limits.file_size_bytes)
                self._stream = (self.workdir/'process.log').open('xb')
                isolation_flags = ['-I','-S']
                self._process = subprocess.Popen([self.runtime.python_executable,*isolation_flags,str(worker),_json(config)],
                    cwd=self.workdir, env=env, stdin=subprocess.DEVNULL, stdout=self._stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                self.state = TaskState.RUNNING
                self._started = time.monotonic()
                self._thread = threading.Thread(target=self._watch, daemon=True)
                self._thread.start()
            except BaseException as exc:
                message = f'{type(exc).__name__}: {exc}'
                try:
                    if self._process is not None:
                        # A stored handle is owned even before RUNNING was assigned.
                        self.state = TaskState.RUNNING
                        self._stop(TaskState.FAILED, message)
                    else:
                        self.state, self.error = TaskState.FAILED, message
                    if self._stream is not None: self._stream.close()
                except BaseException as cleanup_exc:
                    self.error = message + f'; cleanup incomplete: {type(cleanup_exc).__name__}: {cleanup_exc}'
                    # RUNNING retains retriable ownership until group cleanup and wait finish.
                    if self.state != TaskState.RUNNING: self._done.set()
                    raise exc from cleanup_exc
                self._done.set()
                raise
        return self

    def _stop(self, state, message):
        with self._lock:
            if self.state != TaskState.RUNNING: return
            try: os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: self._process.wait(timeout=0.25)
            except subprocess.TimeoutExpired: pass
            # Kill the entire owned group even if its leader already exited.
            try: os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            self._process.wait()
            self.state, self.error = state, message

    def cancel(self):
        with self._lock:
            if self.state == TaskState.CREATED and self._process is None:
                self.state, self.error = TaskState.CANCELLED, 'cancelled before start'
                self._done.set()
                return
            if self.state == TaskState.CREATED:
                self.state = TaskState.RUNNING
            self._stop(TaskState.CANCELLED, 'cancelled by caller')
            if self._thread is None or not self._thread.is_alive():
                # An interrupted start may own a child without an active watchdog.
                try:
                    if self._stream is not None: self._stream.close()
                finally:
                    if self.state != TaskState.RUNNING: self._done.set()

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            if self._thread is not None and self._thread.is_alive():
                raise TimeoutError('task still running; its independent watchdog remains active')
            raise TimeoutError('task still pending with no active watchdog; inspect state and retry cleanup if needed')
        return self.state

    def _artifact_bytes(self):
        total = 0
        for path in self.workdir.rglob('*'):
            if path.is_symlink():
                raise LAMMPSContractError('artifact symlinks are unsupported')
            try:
                if path.is_file(): total += path.stat().st_size
            except FileNotFoundError: pass
        return total

    def _watch(self):
        try:
            while self._process.poll() is None:
                if time.monotonic()-self._started > self.limits.wall_seconds:
                    self._stop(TaskState.TIMED_OUT, 'wall-clock limit exceeded')
                    break
                size = self._artifact_bytes()
                if size > self.limits.output_bytes:
                    self._stop(TaskState.RESOURCE_LIMIT, 'aggregate artifact limit exceeded')
                    break
                time.sleep(0.025)
            self._process.wait()
            self._stream.close()
            with self._lock:
                if self.state != TaskState.RUNNING: return
                if self._artifact_bytes() > self.limits.output_bytes:
                    self.state, self.error = TaskState.RESOURCE_LIMIT, 'final aggregate artifact limit exceeded'
                    return
                if self._process.returncode:
                    self.state, self.error = TaskState.FAILED, f'native child exit {self._process.returncode}; see process.log'
                    return
                texts = []
                for name in ('trajectory.dump','log.lammps'):
                    path = self.workdir/name
                    if path.stat().st_size > self.limits.output_bytes:
                        raise LAMMPSContractError('output exceeds byte limit')
                    texts.append(path.read_text(encoding='utf-8'))
                if not texts[1].splitlines() or texts[1].splitlines()[0] != self.runtime.expected_banner:
                    raise LAMMPSContractError('native log version/banner differs from runtime pin')
                parsed = parse_outputs(self.prepared, *texts, max_output_bytes=self.limits.output_bytes)
                self.result = LAMMPSRunResult(parsed.identity, parsed.trajectory, parsed.thermo, str(self.workdir))
                self.state = TaskState.SUCCEEDED
        except Exception as exc:
            self._stop(TaskState.FAILED, f'{type(exc).__name__}: {exc}')
            with self._lock:
                if self.state == TaskState.RUNNING:
                    self.state, self.error = TaskState.FAILED, f'{type(exc).__name__}: {exc}'
        finally:
            # All terminal paths clean the owned group, including fast leader exit.
            if self._process is not None:
                try: os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                self._process.wait()
            if self._stream is not None: self._stream.close()
            self._done.set()
