"""Bounded offline saved-run reader; no engine, runtime admission, or restart.

A matching caller-supplied manifest digest is a byte-identity anchor,
not provenance, authentication, permission, or evidence that an engine ran.
Inputs must be quiescent regular files in a caller-selected local directory.
The cooperative deadline is not a hard CPU/RSS or network-filesystem timeout.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from fibernet.integrations.lammps_backend import (
    LAMMPSContractError, LAMMPSRunResult, PreparedRun,
    parse_outputs, validate_prepared,
)


_FILES = ('manifest.json', 'graph.json', 'model.data', 'in.model',
         'trajectory.dump', 'log.lammps')
_MIB = 1024 * 1024


class LAMMPSSavedReadError(LAMMPSContractError):
    """Saved bytes cannot be accepted by this saved-run reader."""


class LAMMPSCompilerCompatibilityError(LAMMPSSavedReadError):
    """Bytes may be modified or from another compiler; no migration attempted."""


@dataclass(frozen=True)
class LAMMPSSavedReadLimits:
    total_bytes: int = 20 * _MIB
    prepared_bytes: int = 2 * _MIB
    output_bytes: int = 16 * _MIB
    json_bytes: int = _MIB
    json_depth: int = 32
    json_items: int = 100000
    max_nodes: int = 256
    max_edges: int = 1024
    max_steps: int = 2048
    deadline_seconds: float = 10.0

    def checked(self):
        ceilings = dict(total_bytes=20*_MIB, prepared_bytes=2*_MIB,
                        output_bytes=16*_MIB, json_bytes=_MIB, json_depth=32,
                        json_items=100000, max_nodes=256, max_edges=1024,
                        max_steps=2048)
        for key, ceiling in ceilings.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise LAMMPSSavedReadError('invalid reader limit: ' + key)
        if (type(self.deadline_seconds) not in (int, float)
                or not math.isfinite(self.deadline_seconds)
                or not 0 < self.deadline_seconds <= 60):
            raise LAMMPSSavedReadError('invalid cooperative deadline')
        return self


@dataclass(frozen=True)
class SavedLAMMPSRead:
    prepared: PreparedRun
    result: LAMMPSRunResult
    receipt_json: str
    unverified_metadata_json: str

    @property
    def receipt(self):
        """A detached copy; the recorded receipt cannot be mutated by callers."""
        return json.loads(self.receipt_json)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _hex_digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _integer(value, name, maximum=2**31-1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise LAMMPSSavedReadError(name + ' must be a bounded integer, not boolean')
    return value


def _strict_json(text, limits, label):
    # Bound containers/tokens before the recursive JSON decoder allocates them.
    depth = tokens = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
            tokens += 1
            if depth > limits.json_depth:
                raise LAMMPSSavedReadError(label + ' JSON depth exceeded')
        elif char in ']}':
            depth -= 1
        elif char in ',:':
            tokens += 1
        if tokens > limits.json_items:
            raise LAMMPSSavedReadError(label + ' JSON item budget exceeded')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise LAMMPSSavedReadError(label + ' duplicate JSON key')
            result[key] = value
        return result

    def integer(token):
        if len(token) > 20:
            raise LAMMPSSavedReadError(label + ' JSON integer too large')
        value = int(token)
        if not -(2**63) <= value < 2**63:
            raise LAMMPSSavedReadError(label + ' JSON integer too large')
        return value

    def real(token):
        value = float(token)
        if not math.isfinite(value):
            raise LAMMPSSavedReadError(label + ' nonfinite JSON number')
        return value

    def constant(_):
        raise LAMMPSSavedReadError(label + ' nonfinite JSON number')

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_int=integer,
                           parse_float=real, parse_constant=constant)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, LAMMPSSavedReadError):
            raise
        raise LAMMPSSavedReadError(label + ' invalid JSON') from exc
    if type(value) is not dict:
        raise LAMMPSSavedReadError(label + ' must be a JSON object')
    return value


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _restore(manifest, texts, limits):
    graph = _strict_json(texts['graph.json'], limits, 'graph')
    contract = manifest.get('contract')
    if type(contract) is not dict:
        raise LAMMPSSavedReadError('manifest contract must be an object')
    nodes, edges = graph.get('nodes'), graph.get('edges')
    if (type(nodes) is not list or type(edges) is not list
            or not 2 <= len(nodes) <= limits.max_nodes
            or len(edges) > limits.max_edges):
        raise LAMMPSSavedReadError('graph count budget exceeded or malformed graph')
    def ids(rows, name):
        result = []
        for row in rows:
            if type(row) is not dict:
                raise LAMMPSSavedReadError(name + ' must contain records')
            result.append(_integer(row.get('id'), name + ' ID'))
        if len(set(result)) != len(result):
            raise LAMMPSSavedReadError('duplicate ' + name + ' ID')
        return tuple(sorted(result))
    node_ids, edge_ids = ids(nodes, 'node'), ids(edges, 'edge')
    for edge in edges:
        for key in ('node_i', 'node_j'):
            if _integer(edge.get(key), key) not in node_ids:
                raise LAMMPSSavedReadError('unknown endpoint ID')
    steps = _integer(contract.get('steps'), 'steps', limits.max_steps)
    every = _integer(contract.get('sample_every'), 'sample_every', limits.max_steps)
    if every == 0:
        raise LAMMPSSavedReadError('sample_every must be positive')
    dt = contract.get('timestep_s')
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
        raise LAMMPSSavedReadError('timestep_s must be finite and positive')
    profile_record = contract.get('profile')
    if profile_record is not None and (type(profile_record) is not dict
            or type(profile_record.get('name')) is not str):
        raise LAMMPSSavedReadError('profile must be a named record or absent')
    profile = profile_record['name'] if profile_record is not None else None
    prepared = PreparedRun(texts['graph.json'], _canonical(contract),
                           texts['model.data'], texts['in.model'],
                           node_ids, edge_ids, dt, steps, every, profile)
    # Require the full manifest emitted by an owned CLI task. Runtime paths,
    # hash pins, environment and limits are retained as unverified JSON only.
    if not {'runtime', 'limits'} <= set(manifest):
        raise LAMMPSSavedReadError('saved task manifest requires runtime and limits records')
    if type(manifest['runtime']) is not dict or type(manifest['limits']) is not dict:
        raise LAMMPSSavedReadError('saved runtime and limits must be objects')
    try:
        expected = prepared.manifest()
        actual = {k: v for k, v in manifest.items() if k not in ('runtime', 'limits')}
        # Canonical text comparison is type-sensitive: True != 1, 1 != 1.0.
        if _canonical(actual) != _canonical(expected):
            raise LAMMPSSavedReadError('manifest schema, identity, digest, or ID map mismatch')
        validate_prepared(prepared)
    except LAMMPSSavedReadError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise LAMMPSCompilerCompatibilityError(
            'saved preparation is incompatible with this compiler; bytes may be '
            'modified or from another revision/runtime; no migration attempted') from exc
    return prepared


def read_saved_lammps_run(root, *, expected_manifest_sha256, expected_banner=None,
                   limits=LAMMPSSavedReadLimits()):
    """Validate saved bytes/protocol without runtime admission or engine work.

    Only six named files are opened. Other entries (including scratch/logs) are
    ignored, not recursively inspected or certified. File and directory symlinks
    are refused by this route; existing public APIs are unchanged.
    Expected manifest digest and optional banner come from the caller, not the
    data. Even a matching external digest does not authenticate the caller's
    source or prove physics. Compiler drift is a hard compatibility failure.
    """
    if type(limits) is not LAMMPSSavedReadLimits:
        raise LAMMPSSavedReadError('limits must be an inert LAMMPSSavedReadLimits record')
    limits.checked()
    if not _hex_digest(expected_manifest_sha256):
        raise LAMMPSSavedReadError('caller must supply an explicit lowercase manifest SHA256')
    if expected_banner is not None and (type(expected_banner) is not str
            or not expected_banner or expected_banner.splitlines() != [expected_banner]):
        raise LAMMPSSavedReadError('expected_banner must be a nonempty single line')
    if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW'):
        raise LAMMPSSavedReadError('saved-run descriptor reader requires POSIX O_NOFOLLOW')
    if type(root) not in (str, Path) and not isinstance(root, Path):
        raise LAMMPSSavedReadError('root must be a path string or pathlib Path')
    path = Path(root).absolute()
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started > limits.deadline_seconds:
            raise LAMMPSSavedReadError('cooperative read deadline exceeded')
    directory_fd = None
    try:
        if path.resolve(strict=True) != path:
            raise LAMMPSSavedReadError('root symlink or noncanonical path refused')
        directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        before, contents = {}, {}
        total = prepared_size = output_size = 0
        for name in _FILES:
            deadline()
            output = name in ('trajectory.dump', 'log.lammps')
            budget = min(limits.total_bytes-total,
                         limits.output_bytes-output_size if output else limits.prepared_bytes-prepared_size)
            if name.endswith('.json'):
                budget = min(budget, limits.json_bytes)
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory_fd)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise LAMMPSSavedReadError(name + ' must be a regular file')
                if info.st_size > budget:
                    raise LAMMPSSavedReadError(name + ' byte budget exceeded')
                chunks, count = [], 0
                while True:
                    deadline()
                    chunk = os.read(fd, min(65536, budget-count+1))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    count += len(chunk)
                    if count > budget:
                        raise LAMMPSSavedReadError(name + ' byte budget exceeded while reading')
                if _signature(info) != _signature(os.fstat(fd)) or count != info.st_size:
                    raise LAMMPSSavedReadError(name + ' changed while reading')
                before[name] = _signature(info)
                contents[name] = b''.join(chunks)
                total += count
                if output:
                    output_size += count
                else:
                    prepared_size += count
            finally:
                os.close(fd)
        if _sha(contents['manifest.json']) != expected_manifest_sha256:
            raise LAMMPSSavedReadError('caller manifest digest mismatch')
        try:
            texts = {name: data.decode('utf-8') for name, data in contents.items()}
        except UnicodeDecodeError as exc:
            raise LAMMPSSavedReadError('saved files must be UTF-8') from exc
        manifest = _strict_json(texts['manifest.json'], limits, 'manifest')
        deadline()
        prepared = _restore(manifest, texts, limits)
        deadline()
        if expected_banner is not None and texts['log.lammps'].splitlines()[:1] != [expected_banner]:
            raise LAMMPSSavedReadError('saved log banner differs from caller expectation')
        # Reuse the public parser on captured immutable bytes, not reopened paths.
        result = parse_outputs(prepared, texts['trajectory.dump'], texts['log.lammps'],
                               max_output_bytes=limits.output_bytes)
        deadline()
        for name in _FILES:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _signature(current) != before[name]:
                raise LAMMPSSavedReadError(name + ' changed during validation')
        receipt = dict(schema='fibernet.lammps.saved-read/1',
            status='saved_bytes_and_protocol_validated',
            expected_manifest_sha256=expected_manifest_sha256,
            caller_anchor_authenticated=False, file_sha256={n: _sha(contents[n]) for n in _FILES},
            bytes_read=total, prepared_identity=prepared.identity,
            node_ids=list(prepared.node_ids), edge_ids=list(prepared.edge_ids),
            trajectory_frames=len(result.trajectory), thermo_frames=len(result.thermo),
            graph_length_unit=manifest['contract']['length_unit'],
            native_state_units='SI', compiler_compatibility='exact_current_regeneration',
            compiler_revision_in_saved_manifest='not_recorded',
            new_execution=False, runtime_checked=False, native_execution_verified=False,
            physics_verified=False, historical_task_status='not_recorded_in_input_manifest',
            extra_files='ignored; not validated', filesystem_assumption='quiescent local regular files')
        return SavedLAMMPSRead(prepared, result, _canonical(receipt),
                              _canonical({k: manifest[k] for k in ('runtime', 'limits')}))
    except OSError as exc:
        raise LAMMPSSavedReadError('cannot read the required saved regular files') from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
