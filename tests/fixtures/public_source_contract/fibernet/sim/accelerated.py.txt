"""
Taichi-accelerated simulations for fiber networks.

Two backends:
TaichiEngine is the primary simulation backend (mass-spring dynamics).
2. **TaichiEngine** — Mass-spring dynamics (unit-mass kick/drift integration, Taichi parallel forces)

Both accept StructureGraph directly via high-level test methods.
"""

from __future__ import annotations

import time
import sys
import os
import stat
import tempfile
import threading
import warnings
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union

import numpy as np

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False

from fibernet.core.structure_graph import StructureGraph
from ._taichi_viscoelastic import material_options as _material_options


def _finite_real_array(value, name, shape=None):
    """Detach numeric input without silently accepting masks or cast underflow."""
    def unmasked(item):
        if np.ma.isMaskedArray(item) or item is np.ma.masked:
            raise ValueError(f"{name} must not be masked")
        if isinstance(item, (list, tuple)):
            for child in item:
                unmasked(child)
        elif isinstance(item, np.ndarray) and item.dtype.kind == 'O':
            for child in item.flat:
                unmasked(child)
    unmasked(value)
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or (shape is not None and raw.shape != shape):
        raise ValueError(f"{name} must be a real numeric array with shape {shape}")
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        out = np.array(raw, dtype=np.float64, copy=True)
    if (not np.isfinite(raw).all() or not np.isfinite(out).all() or
            np.any((raw != 0) & (out == 0))):
        raise ValueError(f"{name} must remain finite and nonzero after float64 conversion")
    return out


def _strict_state_ids(value, name='node_ids', *, nonnegative=False):
    from numbers import Integral
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an ordered integer list")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in value):
        raise ValueError(f"{name} must contain strict integer IDs")
    out = [int(x) for x in value]
    if nonnegative and any(x < 0 for x in out):
        raise ValueError(f'{name} must be nonnegative')
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must be unique")
    return out


def _state_units(value):
    if value is None:
        return None
    keys = {'length', 'time', 'mass', 'frame'}
    if (not isinstance(value, dict) or set(value) != keys or
            any(not isinstance(v, str) or not v.strip() for v in value.values())):
        raise ValueError('state_units must declare nonempty length/time/mass/frame strings')
    return dict(value)


def _native_axial_event_dataclass(cls):
    """Keep this internal frozen event slotted on supported Python 3.9 too.

    The event has only ordinary instance fields and directly inherits object.
    This is deliberately not a general dataclass compatibility utility.
    """
    if sys.version_info >= (3, 10):
        return dataclass(frozen=True, slots=True)(cls)
    if cls.__bases__ != (object,) or '__slots__' in vars(cls):
        raise TypeError('native axial event requires a plain unslotted class')
    decorated = dataclass(frozen=True)(cls)
    names = tuple(decorated.__dataclass_fields__)
    namespace = dict(vars(decorated))
    namespace['__slots__'] = names
    for name in names:
        namespace.pop(name, None)
    namespace.pop('__dict__', None)
    namespace.pop('__weakref__', None)
    event_type = type(decorated)(decorated.__name__, decorated.__bases__, namespace)
    event_type.__qualname__ = decorated.__qualname__

    # Frozen slotted dataclasses on 3.10+ expose equivalent state hooks. These
    # support normal copy/pickle reconstruction, not a security boundary.
    def getstate(self):
        return [getattr(self, name) for name in names]

    def setstate(self, state):
        for name, value in zip(names, state):
            object.__setattr__(self, name, value)

    event_type.__getstate__ = getstate
    event_type.__setstate__ = setstate
    return event_type


@_native_axial_event_dataclass
class NativeAxialObserverEvent:
    """Detached completed-state observation; not a writable integration view.

    Snapshot arrays are backed by newly owned immutable bytes. A consumer cannot
    re-enable their write flag. Units are sorted key/value pairs, or None when
    the caller has not declared them. No event is retained by the engine.
    """
    schema: str
    step: int
    time: float
    node_ids: tuple
    units: Optional[tuple]
    positions: np.ndarray = field(repr=False, compare=False)
    velocities: np.ndarray = field(repr=False, compare=False)


def _emit_axial_observation(observer, step, dt, node_ids, units, positions, velocities, *, material_state=None):
    # Own immutable bytes, not aliases to native fields, export buffers or events.
    shape = (len(node_ids), positions.shape[1])
    def snapshot(value, name):
        a = _finite_real_array(value, name, shape)
        return np.frombuffer(a.tobytes(order='C'), dtype=np.float64).reshape(shape)
    event = NativeAxialObserverEvent(
        schema='native_axial_observer/1', step=int(step), time=float(step * dt),
        node_ids=node_ids, units=units,
        positions=snapshot(positions, 'observer positions'),
        velocities=snapshot(velocities, 'observer velocities'))
    if material_state is not None:
        from ._taichi_viscoelastic import material_observation
        event = material_observation(event, material_state)
    # Propagate the original callback exception: no shortened successful result.
    observer(event)


def _validated_fixed_rows(value, n_nodes):
    from numbers import Integral
    if value is None:
        return []
    if np.ma.isMaskedArray(value) or not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError('fixed_nodes must be an unmasked row-index sequence')
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError('fixed_nodes must be one-dimensional')
    rows = list(value)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or
           not 0 <= int(x) < n_nodes for x in rows):
        raise ValueError('fixed_nodes must contain strict integer row indices in [0, N)')
    return [int(x) for x in rows]


def _resolve_nodal_masses(node_ids, masses, by_id):
    from collections.abc import Mapping
    if masses is not None and by_id is not None:
        raise ValueError('nodal_masses and nodal_masses_by_id are mutually exclusive')
    if by_id is not None:
        ids = _strict_state_ids(list(node_ids))
        if not isinstance(by_id, Mapping):
            raise ValueError('nodal_masses_by_id must be a complete ID mapping')
        keys = _strict_state_ids(list(by_id.keys()), 'mass mapping keys')
        if set(keys) != set(ids):
            raise ValueError('mass mapping must cover exactly the graph node IDs')
        masses = [by_id[node] for node in ids]
    if masses is None:
        return np.ones(len(node_ids), dtype=np.float64)
    out = _finite_real_array(masses, 'nodal_masses', (len(node_ids),))
    if np.any(out <= 0):
        raise ValueError('nodal_masses must be strictly positive')
    return out


def _checked_kinetic_energy(masses, velocities):
    """Float64 result without premature overflow/underflow in m*v*v.

    Scale binary exponents before summation. Reject a nonzero energy that is
    unrepresentable as finite, nonzero float64 rather than reporting fake zero.
    """
    import math
    velocities = np.asarray(velocities)
    if not np.isfinite(velocities).all():
        raise ValueError('kinetic energy requires finite velocity')
    terms = []
    for mass, row in zip(masses, velocities):
        mf, me = math.frexp(float(mass))
        for value in row:
            if value != 0:
                vf, ve = math.frexp(abs(float(value)))
                terms.append((.5 * mf * vf * vf, me + 2 * ve))
    if not terms:
        return 0.
    exponent = max(e for _, e in terms)
    scaled = math.fsum(math.ldexp(m, e-exponent) for m, e in terms)
    try:
        energy = math.ldexp(scaled, exponent)
    except OverflowError as exc:
        raise ValueError('kinetic energy is outside float64 range') from exc
    if not math.isfinite(energy) or energy == 0:
        raise ValueError('nonzero kinetic energy is outside float64 range')
    return energy


def _resolve_initial_velocities(node_ids, by_id, dim, fixed_rows, units):
    """Sparse stable-ID velocities in declared length/time units; no conversion."""
    from collections.abc import Mapping
    ids = _strict_state_ids(list(node_ids))
    if not isinstance(by_id, Mapping):
        raise ValueError('initial_velocities_by_id must be a stable-ID mapping')
    keys = _strict_state_ids(list(by_id.keys()), 'initial velocity IDs')
    if not set(keys) <= set(ids):
        raise ValueError('initial velocity mapping contains unknown node IDs')
    if units is None:
        raise ValueError('initial velocities require explicit state_units')
    out = np.zeros((len(ids), dim), dtype=np.float64)
    rows = {node: row for row, node in enumerate(ids)}
    def reject_bools(value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('initial velocity must not contain booleans')
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                reject_bools(item)
    for node in keys:
        vector = _finite_real_array(by_id[node], 'initial velocity', (dim,))
        reject_bools(by_id[node])
        out[rows[node]] = vector
    if any(np.any(out[row] != 0) for row in fixed_rows):
        raise ValueError('fixed nodes cannot have nonzero initial velocity')
    return out

def _canonical_rest_lengths(positions, elements):
    """Canonical geometric rest in graph coordinate units, without a provider."""
    import math
    return np.array([math.dist(positions[i], positions[j]) for i, j in elements], dtype=np.float64)

def _resolve_rest_lengths(edge_ids, positions, elements, by_id):
    from collections.abc import Mapping
    ids = _strict_state_ids(list(edge_ids), 'edge_ids')
    if not isinstance(by_id, Mapping):
        raise ValueError('rest_lengths_by_id must be a complete stable-edge-ID mapping')
    keys = _strict_state_ids(list(by_id), 'rest mapping keys')
    if set(keys) != set(ids):
        raise ValueError('rest mapping must cover exactly the graph edge IDs')
    values = [_finite_real_array(by_id[e], 'rest length', ()) for e in ids]
    rest = np.array(values, dtype=np.float64)
    if np.any(rest <= 0) or not np.array_equal(rest, _canonical_rest_lengths(positions, elements)):
        raise ValueError('rest lengths must exactly equal canonical geometric rest; prestress is unsupported')
    return rest

def _validate_mass_inputs(value, n_nodes, node_ids):
    if (not isinstance(value, dict) or set(value) != {'schema', 'node_ids', 'masses', 'units'} or
            value['schema'] != 'native_axial_inputs/1'):
        raise ValueError('unsupported native_axial_inputs record')
    ids = _strict_state_ids(value['node_ids'])
    if len(ids) != n_nodes or (node_ids is not None and ids != _strict_state_ids(list(node_ids))):
        raise ValueError('mass input IDs disagree with result rows')
    masses = _resolve_nodal_masses(ids, value['masses'], None)
    return dict(schema='native_axial_inputs/1', node_ids=ids,
                masses=masses.tolist(), units=_state_units(value['units']))


def _validate_dynamics_state(value, n_nodes=None, node_ids=None):
    """Strict v1 JSON record; summaries never fabricate sampled state."""
    if not isinstance(value, dict) or value.get('schema') != 'native_axial_state/1':
        raise ValueError('unsupported dynamics_state schema')
    rep = value.get('representation')
    if rep not in ('sampled', 'summary'):
        raise ValueError('unsupported dynamics_state representation')
    common = {'schema', 'representation', 'availability', 'units', 'dt', 'final',
              'work_ledger', 'cutoff_check'}
    arrays = {'node_ids', 'masses', 'steps', 'times', 'positions', 'velocities',
              'kinetic_energy', 'potential_energy', 'total_energy', 'near_cutoff'}
    if set(value) != common | (arrays if rep == 'sampled' else set()):
        raise ValueError('dynamics_state has missing or unknown fields')
    if value['availability'] != ('available' if rep == 'sampled' else 'summary_only'):
        raise ValueError('inconsistent dynamics_state availability')
    if value['work_ledger'] != {'status': 'unavailable', 'value': None, 'reason': 'not_collected'}:
        raise ValueError('work ledger must be unavailable/not_collected')
    if value['cutoff_check'] != 'sampled_only':
        raise ValueError('cutoff check must explicitly be sampled_only')
    units = _state_units(value['units'])
    dt = _finite_real_array(value['dt'], 'state dt', ())
    if dt <= 0:
        raise ValueError('state dt must be positive')
    final = value['final']
    scalars = {'step', 'time', 'kinetic_energy', 'potential_energy', 'total_energy', 'near_cutoff'}
    if not isinstance(final, dict) or set(final) != scalars | ({'positions', 'velocities'} if rep == 'sampled' else set()):
        raise ValueError('invalid final state fields')
    from numbers import Integral
    if isinstance(final['step'], (bool, np.bool_)) or not isinstance(final['step'], Integral) or final['step'] < 0:
        raise ValueError('final step must be a nonnegative integer')
    for name in ('time', 'kinetic_energy', 'potential_energy', 'total_energy'):
        _finite_real_array(final[name], 'final ' + name, ())
    if (final['time'] != final['step'] * float(dt) or
            final['kinetic_energy'] < 0 or final['potential_energy'] < 0 or
            not np.isclose(final['total_energy'], final['kinetic_energy'] + final['potential_energy'], rtol=1e-12, atol=0)):
        raise ValueError('inconsistent final time or energy')
    if type(final['near_cutoff']) is not bool:
        raise ValueError('near_cutoff must be boolean')
    if rep == 'sampled':
        ids = _strict_state_ids(value['node_ids'])
        if (n_nodes is not None and len(ids) != n_nodes) or (node_ids is not None and ids != _strict_state_ids(list(node_ids))):
            raise ValueError('state node ID order disagrees with result')
        n = len(ids)
        mass = _resolve_nodal_masses(ids, value['masses'], None)
        steps = _strict_state_ids(value['steps'], 'steps', nonnegative=True)
        if not steps or steps[0] != 0 or steps != sorted(steps) or steps[-1] != final['step']:
            raise ValueError('state steps must increase from zero through final step')
        count = len(steps)
        times = _finite_real_array(value['times'], 'times', (count,))
        if not np.array_equal(times, np.asarray(steps, dtype=np.float64) * dt):
            raise ValueError('state time axis does not match steps*dt')
        pos = _finite_real_array(value['positions'], 'positions', (count, n, 3))
        vel = _finite_real_array(value['velocities'], 'velocities', pos.shape)
        kin = _finite_real_array(value['kinetic_energy'], 'kinetic_energy', (count,))
        pot = _finite_real_array(value['potential_energy'], 'potential_energy', (count,))
        total = _finite_real_array(value['total_energy'], 'total_energy', (count,))
        if (np.any(kin < 0) or np.any(pot < 0) or
                not np.allclose(kin, np.array([_checked_kinetic_energy(mass, frame) for frame in vel]), rtol=1e-12, atol=0) or
                not np.allclose(total, kin + pot, rtol=1e-12, atol=0)):
            raise ValueError('inconsistent sampled energies')
        flags = value['near_cutoff']
        if not isinstance(flags, list) or len(flags) != count or any(type(x) is not bool for x in flags):
            raise ValueError('near_cutoff must align with sampled states')
        for name in ('positions', 'velocities', 'kinetic_energy', 'potential_energy', 'total_energy', 'near_cutoff'):
            if not np.array_equal(np.asarray(final[name]), np.asarray(value[name][-1])):
                raise ValueError('final state does not match last sample')
    # All accepted fields are now known JSON-native data; detach on export/load.
    return json.loads(json.dumps(value, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()))


def _validate_result_state(result):
    if isinstance(result.dynamics_state, dict) and result.dynamics_state.get("schema") in ("lammps_harmonic_state/1", "lammps_cg_state/1"):
        from ._lammps_result import validate_result
        return validate_result(result)
    record = _validate_dynamics_state(result.dynamics_state, result.n_nodes,
                                      result.metadata.get('node_ids'))
    if not np.isclose(result.energy, record['final']['potential_energy'], rtol=1e-12, atol=0):
        raise ValueError('state final potential disagrees with result energy')
    if 'dt' in result.metadata and result.metadata['dt'] != record['dt']:
        raise ValueError('state dt disagrees with result metadata')
    if 'num_steps' in result.metadata and result.metadata['num_steps'] != record['final']['step']:
        raise ValueError('state final step disagrees with result metadata')
    if record['representation'] == 'sampled' and result.history:
        if [row.get('step') for row in result.history] != record['steps']:
            raise ValueError('state steps disagree with result history')
    inputs = result.metadata.get('native_axial_inputs')
    if inputs is not None:
        inputs = _validate_mass_inputs(inputs, result.n_nodes, result.metadata.get('node_ids'))
        if record['units'] != inputs['units']:
            raise ValueError('state units disagree with mass provenance')
        if record['representation'] == 'sampled' and record['masses'] != inputs['masses']:
            raise ValueError('state masses disagree with mass provenance')
    if record['representation'] == 'sampled':
        if result.deformed_positions is not None and not np.array_equal(result.deformed_positions, record['final']['positions']):
            raise ValueError('state final positions disagree with result')
        if len(result.positions_trajectory) and not np.array_equal(result.positions_trajectory, record['positions']):
            raise ValueError('state positions disagree with result trajectory')
    return record


def _dynamics_state_summary(value):
    if isinstance(value, dict) and value.get("schema") in ("lammps_harmonic_state/1", "lammps_cg_state/1"):
        from ._lammps_result import state_summary
        return state_summary(value)
    record = _validate_dynamics_state(value)
    if record['representation'] == 'summary':
        return record
    return {**{k: record[k] for k in ('schema', 'units', 'dt', 'work_ledger', 'cutoff_check')},
            'representation': 'summary', 'availability': 'summary_only',
            'final': {k: v for k, v in record['final'].items() if k not in ('positions', 'velocities')}}


@dataclass
class SimResult:
    """Unified result container for spring and beam-frame simulations.

    Simple mode: basic displacements + energy
    Detailed mode: per-edge forces, max values, trajectory data

    metadata['node_ids'] identifies displacement, deformed-position and trajectory
    rows; metadata['edge_ids'] identifies edge-force and stretch rows. Producers
    preserve solver row order. Old results may omit these maps, so consumers must
    use their documented historical ordering rather than infer identity.
    """
    # Basic fields
    displacements: np.ndarray = None
    energy: float = 0.0
    time_seconds: float = 0.0
    mode: str = ""
    deformed_positions: np.ndarray = None
    positions_trajectory: List[np.ndarray] = field(default_factory=list)

    # Detailed fields (populated by compute_detailed or detailed=True)
    edge_forces: np.ndarray = None       # per-edge axial force magnitude
    edge_stretches: np.ndarray = None    # per-edge stretch ratio (L/L0)
    max_force: float = 0.0              # maximum edge force
    max_stretch: float = 0.0            # maximum edge stretch ratio
    mean_stretch: float = 0.0           # mean edge stretch ratio
    std_stretch: float = 0.0            # std of edge stretch ratios
    max_displacement: float = 0.0       # maximum node displacement magnitude
    n_nodes: int = 0
    n_edges: int = 0

    # History
    history: List[Dict] = field(default_factory=list)

    # Metadata
    metadata: Dict = field(default_factory=dict)
    dynamics_state: Optional[Dict] = None  # opt-in; old JSON absence stays absent
    algorithmic_energy_ledger: Optional[Dict] = None  # separate opt-in v1 record

    def compute_detailed(self, graph, stiffness: Union[float, np.ndarray] = None):
        """Compute detailed metrics from deformed positions.

        Parameters
        ----------
        graph : StructureGraph
            Original (undeformed) graph.
        stiffness : float or (number of edges,) array, optional
            Coefficients in F = k * strain and E = 0.5 * k * L0 * strain**2.
        """
        if 'native_axial_viscoelastic' in self.metadata:
            from ._taichi_viscoelastic import populate_result
            return populate_result(self, self.metadata['native_axial_viscoelastic'], graph, stiffness)
        if self.metadata.get('backend') == 'lammps':
            raise ValueError('LAMMPS results use explicit equilibrium lengths; geometric-rest recomputation is unsupported')

        from fibernet.sim.accelerated import _graph_to_arrays, _element_data

        pos_orig, elements, _, _ = _graph_to_arrays(graph)
        lengths, _ = _element_data(pos_orig, elements)

        self.n_nodes = len(pos_orig)
        self.n_edges = len(elements)

        if self.deformed_positions is not None:
            pos_def = self.deformed_positions
            # Per-edge stretch (vectorized)
            final_lengths = np.linalg.norm(pos_def[elements[:, 1]] - pos_def[elements[:, 0]], axis=1)
            self.edge_stretches = final_lengths / lengths
            self.max_stretch = float(np.max(self.edge_stretches))
            self.mean_stretch = float(np.mean(self.edge_stretches))
            self.std_stretch = float(np.std(self.edge_stretches))

            # Per-edge forces (F = k * (L - L0) / L0)
            if stiffness is not None:
                spring_coefficients = _normalize_spring_stiffness(
                    stiffness, len(elements)
                )
                strain = final_lengths / lengths - 1.0
                self.edge_forces = spring_coefficients * strain
                self.max_force = float(np.max(np.abs(self.edge_forces)))
                self.energy = float(
                    0.5 * np.sum(spring_coefficients * strain ** 2 * lengths)
                )

        # Max displacement
        if self.displacements is not None:
            self.max_displacement = float(np.max(np.linalg.norm(self.displacements, axis=1)))

        return self

    def to_dict(self, detailed: bool = False) -> Dict:
        """Convert to dictionary for CSV/JSON export.

        Parameters
        ----------
        detailed : bool
            If True, include per-edge arrays (large).
        """
        if 'native_axial_viscoelastic' in self.metadata:
            from ._taichi_viscoelastic import validate_result as validate_material_result
            validate_material_result(self)
        profile = None
        if self.metadata.get('backend') == 'lammps':
            from fibernet.integrations import _harmonic_profile as hp
            profile = hp.result_guard(self.metadata, self.n_nodes, self.n_edges,
                                      self.dynamics_state, vars(self))
        d = {
            "mode": self.mode,
            "energy": self.energy,
            "time_seconds": self.time_seconds,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "max_force": self.max_force,
            "max_stretch": self.max_stretch,
            "mean_stretch": self.mean_stretch,
            "std_stretch": self.std_stretch,
            "max_displacement": self.max_displacement,
        }
        # Metadata shares the historical flat JSON namespace. Never let it
        # replace result fields (or smuggle detailed arrays into a summary).
        d.update({key: value for key, value in self.metadata.items()
                  if key not in self.__dataclass_fields__})
        if 'native_axial_viscoelastic' in d:
            import copy
            d['native_axial_viscoelastic'] = copy.deepcopy(d['native_axial_viscoelastic'])
        if detailed:
            if self.edge_forces is not None:
                d["edge_forces"] = self.edge_forces.tolist()
            if self.edge_stretches is not None:
                d["edge_stretches"] = self.edge_stretches.tolist()
            if self.displacements is not None:
                d["displacements"] = self.displacements.tolist()
            if self.deformed_positions is not None:
                d["deformed_positions"] = self.deformed_positions.tolist()
            d["history"] = self.history
            if self.positions_trajectory:
                d["positions_trajectory"] = [frame.tolist() for frame in self.positions_trajectory]
        if 'native_axial_inputs' in d:
            d['native_axial_inputs'] = _validate_mass_inputs(d['native_axial_inputs'], self.n_nodes, self.metadata.get('node_ids'))
        if self.dynamics_state is not None:
            record = _validate_result_state(self)
            d['dynamics_state'] = record if detailed else _dynamics_state_summary(record)
        if self.algorithmic_energy_ledger is not None:
            from ._energy_accounting import validate_ledger, ledger_summary
            if self.dynamics_state is None:
                raise ValueError('algorithmic_energy_ledger requires dynamics_state')
            ledger = validate_ledger(self.algorithmic_energy_ledger, record,
                self.metadata.get('node_ids'), self.metadata.get('native_axial_inputs'), self.n_nodes)
            d['algorithmic_energy_ledger'] = ledger if detailed else ledger_summary(ledger)
        if profile:
            hp.json_bytes(d, profile['export_bytes'], indent=2)
        return d

    def save(self, path: str, detailed: bool = False):
        """Atomically save JSON via an exclusively owned sibling temporary.

        Existing ordinary file permissions are retained; new files are private
        (0600). Symlinks, nonregular targets and special mode bits are rejected.
        This is replacement atomicity, not crash durability or writer locking.
        """
        data = self.to_dict(detailed=detailed)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            info = destination.lstat()
        except FileNotFoundError:
            mode = 0o600
        else:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("result destination must be a regular file, not a symlink or nonregular target")
            mode = stat.S_IMODE(info.st_mode)
            if mode & 0o7000:
                raise ValueError("result destination has unsupported special mode bits")
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f'Cannot serialize {type(value).__name__}')
        fd, name = tempfile.mkstemp(prefix="." + destination.name + ".result-",
                                    dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = None  # The stream now owns the descriptor, including on error.
                if self.algorithmic_energy_ledger is not None:
                    from ._energy_accounting import dump_binary64_json
                    dump_binary64_json(data, f, default=convert)
                else:
                    json.dump(data, f, indent=2, default=convert)
            os.chmod(temporary, mode)
            temporary.replace(destination)
        finally:
            # Cleanup must not mask a serialization/replace error or report a
            # committed save as failed, even when warnings are treated as errors.
            for cleanup in ((lambda: os.close(fd)) if fd is not None else None,
                            lambda: temporary.unlink(missing_ok=True)):
                if cleanup is None:
                    continue
                try:
                    cleanup()
                except Exception as exc:
                    try:
                        warnings.warn(f"Result cleanup failed for owned temporary {temporary}: {exc!r}; "
                                      "the save outcome is unchanged", RuntimeWarning, stacklevel=2)
                    except Exception:
                        pass

    @staticmethod
    def load(path: str, *, profile=None) -> "SimResult":
        from fibernet.integrations import _harmonic_profile as hp
        from fibernet.integrations import _cg_network_profile as network
        requested = network.select(profile) if profile in network.NAMES else hp.select(profile)
        with open(path, 'rb' if requested else 'r') as f:
            if requested and os.fstat(f.fileno()).st_size > requested['export_bytes']:
                raise ValueError('harmonic profile file exceeds predecode byte bound')
            if requested:
                payload = f.read(requested['export_bytes']+1)
                if len(payload)>requested['export_bytes']:
                    raise ValueError('harmonic profile file exceeds predecode byte bound')
                data = json.loads(payload)
                del payload
            else:
                data = json.load(f)
            if 'algorithmic_energy_ledger' in data:
                if requested:
                    raise ValueError('harmonic profile does not support algorithmic_energy_ledger')
                from ._energy_accounting import parse_binary64
                del data  # Release the first parse before allocating the ledger-aware tree.
                f.seek(0)
                data = json.load(f, parse_float=parse_binary64)
        if data.get('backend') == 'lammps':
            selected = hp.result_guard(data,data.get('n_nodes'),data.get('n_edges'),data.get('dynamics_state'),data)
            if selected != requested:
                raise ValueError('profile-bearing result requires explicit matching load profile')
            if selected: hp.json_bytes(data,selected['export_bytes'],indent=2)
        elif requested:
            raise ValueError('explicit harmonic profile requires a LAMMPS result')
        # Older FEM summary files stored [axial, shear, moment] resultants
        # under the canonical one-dimensional edge_forces name. Recover the
        # axial forces while retaining the original resultants separately.
        if data.get("mode") == "fem_beam_frame" and data.get("edge_forces") is not None:
            legacy_forces = np.asarray(data["edge_forces"])
            if legacy_forces.ndim == 2 and legacy_forces.shape[1] == 3:
                data.setdefault("beam_resultants", data["edge_forces"])
                data["edge_forces"] = legacy_forces[:, 0].tolist()
        r = SimResult(
            mode=data.get("mode", ""),
            energy=data.get("energy", 0),
            time_seconds=data.get("time_seconds", 0),
            n_nodes=data.get("n_nodes", 0),
            n_edges=data.get("n_edges", 0),
            max_force=data.get("max_force", 0),
            max_stretch=data.get("max_stretch", 0),
            mean_stretch=data.get("mean_stretch", 0),
            std_stretch=data.get("std_stretch", 0),
            max_displacement=data.get("max_displacement", 0),
            history=data.get("history", []),
            metadata={k: v for k, v in data.items()
                      if k not in {"mode", "energy", "time_seconds", "n_nodes", "n_edges",
                                   "max_force", "max_stretch", "mean_stretch", "std_stretch",
                                   "max_displacement", "history",
                                   "edge_forces", "edge_stretches",
                                   "displacements", "deformed_positions",
                                   "positions_trajectory", "dynamics_state", "algorithmic_energy_ledger"}},
        )
        for key in ("edge_forces", "edge_stretches", "displacements", "deformed_positions"):
            if data.get(key) is not None:
                setattr(r, key, np.array(data[key]))
        if data.get("positions_trajectory"):
            r.positions_trajectory = [np.array(frame) for frame in data["positions_trajectory"]]
        if 'native_axial_inputs' in r.metadata:
            r.metadata['native_axial_inputs'] = _validate_mass_inputs(r.metadata['native_axial_inputs'], r.n_nodes, r.metadata.get('node_ids'))
        if 'dynamics_state' in data:
            r.dynamics_state = data['dynamics_state']
            r.dynamics_state = _validate_result_state(r)
        if 'algorithmic_energy_ledger' in data:
            from ._energy_accounting import validate_ledger
            r.algorithmic_energy_ledger = validate_ledger(data['algorithmic_energy_ledger'],
                r.dynamics_state, r.metadata.get('node_ids'), r.metadata.get('native_axial_inputs'), r.n_nodes)
        if 'native_axial_viscoelastic' in r.metadata:
            from ._taichi_viscoelastic import validate_result as validate_material_result, populate_result
            validate_material_result(r)
            populate_result(r, r.metadata['native_axial_viscoelastic'])
        return r


# Runtime ownership is process-wide because Taichi fields and kernels are global.
# Strong references avoid recycled Python/native thread IDs masquerading as owners.
_taichi_runtime = None
_taichi_program = None
_taichi_owner = None
_taichi_lifecycle_lock = threading.RLock()


def _taichi_runtime_state():
    """Narrow private-API adapter, source-verified against Taichi 1.7.4."""
    if not HAS_TAICHI:
        raise ImportError("Taichi required: pip install taichi==1.7.4")
    if tuple(getattr(ti, "__version__", ())) != (1, 7, 4):
        raise RuntimeError(
            "FiberNet runtime lifecycle support is verified for Taichi 1.7.4; "
            "install taichi==1.7.4 before using TaichiEngine."
        )
    try:
        runtime = ti.lang.impl.get_runtime()
        program = runtime.prog
    except AttributeError as exc:
        raise RuntimeError("Unsupported Taichi runtime API; expected Taichi 1.7.4") from exc
    return runtime, program


def _check_taichi_owner():
    if _taichi_owner is not None and threading.current_thread() is not _taichi_owner:
        raise RuntimeError(
            "TaichiEngine must run on its runtime-initializing thread "
            "(which may be a worker thread); cross-thread engine/cache use is unsupported."
        )


def _discard_taichi_caches():
    # Never inspect old fields: even querying their shape after reset can segfault.
    TaichiEngine._field_cache.clear()
    TaichiEngine._compute_forces_cache.clear()


def _ensure_taichi(arch: str = "cpu", num_threads: int = 4):
    global _taichi_runtime, _taichi_program, _taichi_owner
    if arch not in ("cpu", "gpu"):
        raise ValueError("arch must be 'cpu' or 'gpu'")
    with _taichi_lifecycle_lock:
        _check_taichi_owner()
        runtime, program = _taichi_runtime_state()
        if runtime is not _taichi_runtime or program is not _taichi_program:
            _discard_taichi_caches()
            if program is not None:
                raise RuntimeError(
                    "Taichi was initialized outside FiberNet; runtime thread ownership "
                    "cannot be verified. On the intended runtime-owning thread, call "
                    "ti.reset() and then construct/use TaichiEngine."
                )
        if program is None:
            if os.environ.get("TI_ARCH") is not None:
                raise RuntimeError("Remove TI_ARCH before TaichiEngine initialization; use arch= explicitly")
            ti.init(arch=ti.cpu if arch == "cpu" else ti.gpu,
                    cpu_max_num_threads=num_threads, enable_fallback=False)
            runtime, program = _taichi_runtime_state()
            if program is None:
                raise RuntimeError("Taichi initialization did not create a runtime")
            _taichi_runtime, _taichi_program = runtime, program
            _taichi_owner = threading.current_thread()
        actual_arch = ti.cfg.arch
        compatible = actual_arch == ti.cpu if arch == "cpu" else actual_arch in ti.gpu
        if not compatible:
            raise RuntimeError(
                f"Requested Taichi architecture {arch!r} conflicts with active "
                f"{actual_arch}; use the existing backend or reset on its owning thread."
            )


def _graph_to_arrays(graph: StructureGraph):
    """Extract positions, elements, node_id mapping from StructureGraph."""
    node_ids = list(graph.nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    pos = np.array([graph.nodes[nid].position for nid in node_ids])
    elements = np.array([[nid_to_idx[e.node_i], nid_to_idx[e.node_j]] for e in graph.edges.values()])
    return pos, elements, node_ids, nid_to_idx


def _get_boundary_indices(pos: np.ndarray, tol: float = None, pct: float = 0.10) -> Dict[str, List[int]]:
    """Find boundary node indices for each side.
    
    Uses percentile-based detection for robust boundary identification
    across different structure types (crystal lattices, TPMS, etc.).
    
    Parameters
    ----------
    pos : (N, D) array of node positions
    tol : float, optional
        Fixed tolerance. If None, uses percentile-based detection.
    pct : float
        Percentile for boundary detection (0.10 = 10% of nodes at each end).
    """
    result = {}
    if tol is not None:
        # Legacy fixed tolerance mode
        bb_min, bb_max = pos.min(0), pos.max(0)
        result["left"] = list(np.where(pos[:, 0] < bb_min[0] + tol)[0])
        result["right"] = list(np.where(pos[:, 0] > bb_max[0] - tol)[0])
        result["bottom"] = list(np.where(pos[:, 1] < bb_min[1] + tol)[0])
        result["top"] = list(np.where(pos[:, 1] > bb_max[1] + tol)[0])
        if pos.shape[1] >= 3:
            result["back"] = list(np.where(pos[:, 2] < bb_min[2] + tol)[0])
            result["front"] = list(np.where(pos[:, 2] > bb_max[2] - tol)[0])
    else:
        # Percentile-based detection (robust for TPMS/complex structures)
        x_sorted = np.argsort(pos[:, 0])
        y_sorted = np.argsort(pos[:, 1])
        n = len(pos)
        n_bnd = max(int(n * pct), 1)
        result["left"] = list(x_sorted[:n_bnd])
        result["right"] = list(x_sorted[-n_bnd:])
        n_bnd_y = max(int(n * pct), 1)
        result["bottom"] = list(y_sorted[:n_bnd_y])
        result["top"] = list(y_sorted[-n_bnd_y:])
        if pos.shape[1] >= 3:
            z_sorted = np.argsort(pos[:, 2])
            n_bnd_z = max(int(n * pct), 1)
            result["back"] = list(z_sorted[:n_bnd_z])
            result["front"] = list(z_sorted[-n_bnd_z:])
    return result


def _element_data(pos, elements):
    """Compute element lengths, directions, areas."""
    diff = pos[elements[:, 1]] - pos[elements[:, 0]]
    lengths = np.linalg.norm(diff, axis=1)
    lengths = np.maximum(lengths, 1e-12)
    directions = diff / lengths[:, None]
    return lengths, directions



# ======================================================================
# ======================================================================


def _normalize_spring_stiffness(stiffness, num_edges):
    """Return one finite, nonnegative spring coefficient per edge.

    Convention: F = k * strain; energy = 0.5 * k * L0 * strain**2.
    k therefore has force units, not force/length units.
    """
    values = np.asarray(stiffness, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(num_edges, float(values), dtype=np.float64)
    elif values.shape != (num_edges,):
        raise ValueError("stiffness must be scalar or have shape (num_edges,)")
    else:
        values = values.copy()
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("stiffness must be finite and nonnegative")
    return values


def _prepare_displacement_schedule(schedule, num_nodes, dim):
    """Normalize schedules expressed in integration-step coordinates.

    Empty schedules impose no constraint. Input order is immaterial.
    Duplicate timestamps are ambiguous and rejected. A single waypoint
    represents a constant displacement. Values outside the waypoint
    range are held at the nearest endpoint.
    """
    prepared = {}
    if schedule is None:
        return prepared
    for node, waypoints in schedule.items():
        if (
            isinstance(node, (bool, np.bool_))
            or not isinstance(node, (int, np.integer))
            or not 0 <= node < num_nodes
        ):
            raise ValueError("Schedule node must be a valid node-array index")
        rows = []
        for waypoint in waypoints:
            if len(waypoint) != 2:
                raise ValueError("Each waypoint must be (step, displacement)")
            step, displacement = waypoint
            step = float(step)
            vector = np.asarray(displacement, dtype=np.float64)
            if not np.isfinite(step) or step < 0:
                raise ValueError("Schedule steps must be finite and nonnegative")
            if vector.shape != (dim,) or not np.all(np.isfinite(vector)):
                raise ValueError(
                    f"Schedule displacements must be finite vectors of length {dim}"
                )
            rows.append((step, vector.copy()))
        if not rows:
            continue
        rows.sort(key=lambda row: row[0])
        if any(a[0] == b[0] for a, b in zip(rows, rows[1:])):
            raise ValueError("Schedule steps must be unique")
        prepared[int(node)] = rows
    return prepared


class TaichiEngine:
    """Mass-spring dynamics with Taichi parallel force computation.

    Axial spring model: F = k * (L - L0) / L0 * direction
    Unit-mass velocity kick, exponential drag, then position drift and clamping.

    Runtime lifetime: sequential engines share one backend on the thread that
    first initializes FiberNet's runtime (including a worker thread). Other
    threads are rejected before field access. External reset between calls is
    supported on the owner; external init requires an explicit reset before
    reuse. External reset/init during a solve is unsupported. Taichi 1.7.4 is
    required by the version-tested private runtime-generation adapter.
    """

    # Cache Taichi fields and kernels to avoid SNode exhaustion
    # Key: (dim, num_nodes, num_edges)
    _field_cache = {}
    _compute_forces_cache = {}

    @classmethod
    def clear_field_cache(cls):
        """Drop cached field/kernel references on the runtime-owning thread.

        This does not destroy native Taichi SNodes. Subsequent simulations
        allocate new cache entries; this is not a native-memory reclamation API.
        """
        with _taichi_lifecycle_lock:
            _check_taichi_owner()
            _discard_taichi_caches()

    def __init__(self, arch: str = "cpu", num_threads: int = 4):
        _ensure_taichi(arch, num_threads)
        self.arch = arch
        self.num_threads = num_threads
        if arch == "cpu" and ti.cfg.cpu_max_num_threads != num_threads:
            warnings.warn(
                "TaichiEngine shares the active runtime's CPU thread count; "
                "num_threads only configures the first initialization.",
                RuntimeWarning, stacklevel=2,
            )

    def compute_forces(
        self,
        positions: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Compute spring forces in parallel (fields cached to avoid SNode leak)."""
        _ensure_taichi(self.arch, self.num_threads)
        num_nodes = positions.shape[0]
        num_edges = edges.shape[0]
        dim = positions.shape[1]

        cache_key = (dim, num_nodes, num_edges)
        if cache_key not in TaichiEngine._compute_forces_cache:
            pos = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            forces = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            edge_arr = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
            L0 = ti.field(dtype=ti.f64, shape=num_edges)
            k_field = ti.field(dtype=ti.f64, shape=num_edges)
            f_temp = ti.Vector.field(dim, dtype=ti.f64, shape=num_edges)

            @ti.kernel
            def _cf_compute():
                for e in range(num_edges):
                    i = edge_arr[e][0]
                    j = edge_arr[e][1]
                    diff = pos[j] - pos[i]
                    L = diff.norm()
                    if L > 1e-12:
                        strain_val = (L - L0[e]) / L0[e]
                        force_mag = k_field[e] * strain_val
                        f_temp[e] = force_mag * (diff / L)
                    else:
                        for d in ti.static(range(dim)):
                            f_temp[e][d] = 0.0

            @ti.kernel
            def _cf_zero():
                for n in range(num_nodes):
                    for d in ti.static(range(dim)):
                        forces[n][d] = 0.0

            @ti.kernel
            def _cf_accumulate():
                for e in range(num_edges):
                    i = edge_arr[e][0]
                    j = edge_arr[e][1]
                    for d in ti.static(range(dim)):
                        forces[i][d] += f_temp[e][d]
                        forces[j][d] -= f_temp[e][d]

            TaichiEngine._compute_forces_cache[cache_key] = {
                'pos': pos, 'forces': forces, 'edge_arr': edge_arr,
                'L0': L0, 'k_field': k_field, 'f_temp': f_temp,
                'compute': _cf_compute, 'zero': _cf_zero, 'accumulate': _cf_accumulate,
            }

        cached = TaichiEngine._compute_forces_cache[cache_key]
        pos = cached['pos']
        forces = cached['forces']
        edge_arr = cached['edge_arr']
        L0 = cached['L0']
        k_field = cached['k_field']

        pos.from_numpy(positions.astype(np.float64))
        edge_arr.from_numpy(edges.astype(np.int32))
        L0.from_numpy(rest_lengths.astype(np.float64))
        k_field.from_numpy(stiffness.astype(np.float64))

        cached['zero']()
        cached['compute']()
        cached['accumulate']()
        return forces.to_numpy()

    @staticmethod
    def dynamic_material_capabilities():
        from ._taichi_viscoelastic import source_capabilities
        return source_capabilities()

    @_material_options
    def dynamics(
        self,
        graph: StructureGraph,
        fixed_nodes: List[int] = None,
        displacement_schedule: Dict[int, List[Tuple[float, np.ndarray]]] = None,
        external_force: np.ndarray = None,
        stiffness: float = 1e5,
        damping: float = 0.3,
        dt: float = 1e-4,
        num_steps: int = 5000,
        save_interval: int = 500,
        spring_k: float = None,
        dashpot: float = 10.0,
        drag: float = 1.0,
        max_nodes: int = 20000,
        *,
        nodal_masses=None,
        nodal_masses_by_id=None,
        record_state: bool = False,
        state_units=None,
        record_energy_ledger: bool = False,
        state_observer=None,
        initial_velocities_by_id=None,
        rest_lengths_by_id=None,
        material=None,
        reference_areas_by_id=None,
        max_abs_strain=None,
    ) -> SimResult:
        """Run mass-spring dynamics with dashpot damping and air drag.

        Existing unit-mass/nondimensional kick-then-drift scheme:
        - Elastic force matches compute_forces for distance > 1e-12; zero below
        - Velocity kick uses implicit unit nodal mass (no mass inference)
        - Dashpot forces use one common old-step velocity snapshot
        - Dashpot damping: F_damp = -damp * (vi-vj).dot(dir) * dir * rest_len
        - Air drag: v *= exp(-drag * dt)
        - Constraints by position clamping + velocity zeroing

        Parameters
        ----------
        graph : StructureGraph
        fixed_nodes : list of node-array indices to fix
            Indices follow graph.nodes insertion order, not graph node IDs.
        displacement_schedule : dict mapping node-array index to waypoints
            Waypoints are (step, displacement) pairs, linearly interpolated and
            held outside their range. Keys follow graph.nodes insertion order.
            Displacements have pos_orig.shape[1] components (XYZ for StructureGraph).
        external_force : (N, 3) array
        stiffness : float
            Global force-like axial coefficient k in F = k * strain, overridden
            by spring_k. With consistent SI units k is in N (like EA), not N/m.
            Neither k nor node inertia is inferred from edge Material.
        damping : legacy metadata parameter; does not control the velocity update
        dt : time step
        num_steps : total integration steps
        save_interval : save trajectory every N steps
        spring_k : scalar or (number of edges,) stiffness array (overrides stiffness)
            Finite, nonnegative force-like coefficients in F = k * strain.
            Element energy is 0.5 * k * strain**2 * rest_length. A conventional
            force/extension spring constant is k / rest_length; no automatic
            conversion from Young's modulus or graph radius is performed.
        dashpot : active dashpot damping coefficient (existing softened direction)
        drag : air drag coefficient
        state_observer : optional synchronous callable, keyword-only
            Receives NativeAxialObserverEvent at t=0 and after every completed
            step including fixed projection. Owned immutable float64 snapshots
            carry stable IDs and declared units. Exceptions propagate and stop
            the call. Ledger and nonempty schedules are initially unsupported.
            None adds no snapshot exports and leaves sampling/results unchanged.
        """
        material_mode = material is not None
        if material_mode:
            if spring_k is not None:
                raise ValueError('material conflicts with spring_k')
            if record_energy_ledger:
                raise ValueError('dynamic material has its own energy record; elastic record_energy_ledger is unsupported')
            if dashpot != 0:
                raise ValueError('dynamic material requires explicit dashpot=0; numerical damping is not material viscosity')
            if displacement_schedule:
                raise ValueError('dynamic material supports fixed supports, not displacement_schedule')
            if nodal_masses_by_id is None or nodal_masses is not None or rest_lengths_by_id is None:
                raise ValueError('dynamic material requires explicit nodal_masses_by_id and rest_lengths_by_id')
        elif any(value is not None for value in (reference_areas_by_id, max_abs_strain,
                                                 initial_velocities_by_id, rest_lengths_by_id)):
            raise ValueError('material-specific areas, strain bound, initial velocities and rest lengths require material')
        if state_observer is not None:
            if not callable(state_observer):
                raise TypeError('state_observer must be callable or None')
            if record_energy_ledger:
                raise ValueError('state_observer does not support record_energy_ledger')
            if displacement_schedule:
                raise ValueError('state_observer does not support displacement_schedule')
        _check_taichi_owner()
        if type(record_energy_ledger) is not bool:
            raise ValueError('record_energy_ledger must be boolean')
        if record_energy_ledger and record_state is not True:
            raise ValueError('record_energy_ledger requires record_state=True')
        for name, value in (
            ("num_steps", num_steps),
            ("save_interval", save_interval),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < (0 if record_energy_ledger and name == 'num_steps' else 1)
            ):
                raise ValueError(f"{name} must be a positive integer (zero steps only with energy ledger)")

        # Memory guard: warn by default, hard block if FIBERNET_STRICT_MEMORY=1
        if graph.num_nodes > max_nodes:
            import os as _os
            msg = (
                f"Structure has {graph.num_nodes} nodes, exceeding max_nodes={max_nodes}. "
                f"Simulation may be slow or run out of memory. "
                f"Use a smaller grid or set max_nodes higher to suppress this warning."
            )
            if _os.environ.get("FIBERNET_STRICT_MEMORY", "0") == "1":
                raise ValueError(msg)
            import warnings as _warnings
            _warnings.warn(msg, ResourceWarning, stacklevel=2)

        t0 = time.time()
        pos_orig, elements, node_ids, _ = _graph_to_arrays(graph)
        dim = pos_orig.shape[1]
        num_nodes = len(node_ids)
        num_edges = len(elements)

        explicit_mass = nodal_masses is not None or nodal_masses_by_id is not None
        masses = _resolve_nodal_masses(node_ids, nodal_masses, nodal_masses_by_id)
        if type(record_state) is not bool:
            raise ValueError('record_state must be boolean')
        units = _state_units(state_units)
        if state_observer is not None:
            observer_node_ids = tuple(_strict_state_ids(node_ids))
            observer_units = None if units is None else tuple(sorted(units.items()))
        if explicit_mass or record_state or units is not None:
            fixed_nodes = _validated_fixed_rows(fixed_nodes, num_nodes)
            _strict_state_ids(list(node_ids))
            for name, value in (('dt', dt), ('dashpot', dashpot), ('drag', drag)):
                scalar = _finite_real_array(value, name, ())
                if (name == 'dt' and scalar <= 0) or (name != 'dt' and scalar < 0):
                    raise ValueError(f'{name} is outside the supported mass/state range')
            if external_force is not None:
                external_force = _finite_real_array(external_force, 'external_force', (num_nodes, dim))

        schedules = _prepare_displacement_schedule(
            displacement_schedule, num_nodes, dim
        )

        initial_velocity = None
        if initial_velocities_by_id is not None:
            initial_velocity = _resolve_initial_velocities(
                node_ids, initial_velocities_by_id, dim, fixed_nodes, units)
        lengths, _ = _element_data(pos_orig, elements)
        rest_lengths_np = (_resolve_rest_lengths(list(graph.edges), pos_orig, elements, rest_lengths_by_id)
                           if material_mode else lengths.copy())

        stiff_np = _normalize_spring_stiffness(
            spring_k if spring_k is not None else stiffness, num_edges
        )

        material_history = None
        if material_mode:
            from ._taichi_viscoelastic import DynamicAxialMaterial
            material_history = DynamicAxialMaterial(graph, node_ids, elements,
                rest_lengths_np, reference_areas_by_id, material, max_abs_strain,
                units, masses, dt, num_steps, save_interval, pos_orig,
                np.zeros_like(pos_orig) if initial_velocity is None else initial_velocity)
            stiff_np = np.zeros(num_edges, dtype=np.float64)
        _ensure_taichi(self.arch, self.num_threads)
        # Taichi fields and kernel (cached to avoid SNode exhaustion and kernel recompilation)
        cache_key = ('material-state-1' if material_mode else 'energy-ledger-1' if record_energy_ledger else 'mass-state-1', dim, num_nodes, num_edges)
        if (cache_key not in TaichiEngine._field_cache or
                not {'ti_mass', 'ti_mass_mode', 'ti_vel_old'}.issubset(TaichiEngine._field_cache[cache_key])):
            # Allocate fields once; reject reuse of a pre-mass/state layout.
            ti_pos = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_vel = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_vel_old = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_mass = ti.field(dtype=ti.f64, shape=num_nodes)
            ti_mass_mode = ti.field(dtype=ti.i32, shape=())
            ti_edges = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
            ti_L0 = ti.field(dtype=ti.f64, shape=num_edges)
            ti_k = ti.field(dtype=ti.f64, shape=num_edges)
            ti_fixed = ti.field(dtype=ti.i32, shape=num_nodes)
            ti_ext = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_pos0 = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_sched = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_sched_mask = ti.field(dtype=ti.i32, shape=num_nodes)
            ti_dt = ti.field(dtype=ti.f64, shape=())
            ti_dashpot = ti.field(dtype=ti.f64, shape=())
            ti_drag = ti.field(dtype=ti.f64, shape=())
            ti_ramp = ti.field(dtype=ti.f64, shape=())

            # Bind inactive closure cells before Taichi inspects ti.static branches.
            ti_material_force = material_v_kick = material_v_drag = material_x_drift = None
            if material_mode:
                ti_material_force = ti.field(dtype=ti.f64, shape=num_edges)
                material_v_kick = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
                material_v_drag = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
                material_x_drift = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            if record_energy_ledger:
                ledger_v_edge = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
                ledger_v_external = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
                ledger_v_drag = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
                ledger_x_drift = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
                ledger_x_fixed = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
                ledger_v_fixed = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)

            # Compile kernel once (closure captures fields + constants)
            @ti.kernel
            def _cached_substep():
                _dt = ti_dt[None]
                _dash = ti_dashpot[None]
                _drag = ti_drag[None]
                _ramp = ti_ramp[None]

                # Complete the old-step snapshot before any edge velocity kick.
                for i in range(num_nodes):
                    ti_vel_old[i] = ti_vel[i]

                for e in range(num_edges):
                    ia = ti_edges[e][0]
                    ib = ti_edges[e][1]
                    d = ti_pos[ia] - ti_pos[ib]
                    dist = d.norm()
                    # Same elastic law/zero-distance branch as compute_forces.
                    f_spring = ti.Vector.zero(ti.f64, dim)
                    if dist > 1e-12:
                        strain_val = (dist - ti_L0[e]) / ti_L0[e]
                        if ti.static(material_mode):
                            f_spring = -ti_material_force[e] * (d / dist)
                        else:
                            f_spring = -ti_k[e] * strain_val * (d / dist)
                    # Preserve the softened dashpot law; evaluate all members
                    # against the same pre-kick velocity state.
                    dir_ = d / (dist + 1e-6)
                    rel_v = ti_vel_old[ia] - ti_vel_old[ib]
                    f_damp = -_dash * rel_v.dot(dir_) * dir_ * ti_L0[e]
                    f_total = f_spring + f_damp
                    for k in ti.static(range(dim)):
                        if ti_mass_mode[None]:
                            ti.atomic_add(ti_vel[ia][k], f_total[k] * _dt / ti_mass[ia])
                            ti.atomic_add(ti_vel[ib][k], -f_total[k] * _dt / ti_mass[ib])
                        else:
                            ti.atomic_add(ti_vel[ia][k], f_total[k] * _dt)
                            ti.atomic_add(ti_vel[ib][k], -f_total[k] * _dt)

                for i in range(num_nodes):
                    for k in ti.static(range(dim)):
                        if ti_mass_mode[None]:
                            ti_vel[i][k] += ti_ext[i][k] * _dt / ti_mass[i]
                        else:
                            ti_vel[i][k] += ti_ext[i][k] * _dt
                    if ti.static(material_mode):
                        material_v_kick[i] = ti_vel[i]
                    ti_vel[i] *= ti.exp(-_drag * _dt)
                    if ti.static(material_mode):
                        material_v_drag[i] = ti_vel[i]
                    for k in ti.static(range(dim)):
                        ti_pos[i][k] += ti_vel[i][k] * _dt
                    if ti.static(material_mode):
                        material_x_drift[i] = ti_pos[i]

                for i in range(num_nodes):
                    if ti_fixed[i] == 1:
                        for k in ti.static(range(dim)):
                            ti_pos[i][k] = ti_pos0[i][k]
                            ti_vel[i][k] = 0.0

                for i in range(num_nodes):
                    if ti_sched_mask[i] == 1:
                        for k in ti.static(range(dim)):
                            ti_pos[i][k] = ti_pos0[i][k] + ti_sched[i][k] * _ramp
                            ti_vel[i][k] = 0.0

            if record_energy_ledger:
                # Compile kernel once (closure captures fields + constants)
                @ti.kernel
                def _ledger_substep():
                    _dt = ti_dt[None]
                    _dash = ti_dashpot[None]
                    _drag = ti_drag[None]
                    _ramp = ti_ramp[None]

                    # Complete the old-step snapshot before any edge velocity kick.
                    for i in range(num_nodes):
                        ti_vel_old[i] = ti_vel[i]

                    for e in range(num_edges):
                        ia = ti_edges[e][0]
                        ib = ti_edges[e][1]
                        d = ti_pos[ia] - ti_pos[ib]
                        dist = d.norm()
                        # Same elastic law/zero-distance branch as compute_forces.
                        f_spring = ti.Vector.zero(ti.f64, dim)
                        if dist > 1e-12:
                            strain_val = (dist - ti_L0[e]) / ti_L0[e]
                            f_spring = -ti_k[e] * strain_val * (d / dist)
                        # Preserve the softened dashpot law; evaluate all members
                        # against the same pre-kick velocity state.
                        dir_ = d / (dist + 1e-6)
                        rel_v = ti_vel_old[ia] - ti_vel_old[ib]
                        f_damp = -_dash * rel_v.dot(dir_) * dir_ * ti_L0[e]
                        f_total = f_spring + f_damp
                        for k in ti.static(range(dim)):
                            if ti_mass_mode[None]:
                                ti.atomic_add(ti_vel[ia][k], f_total[k] * _dt / ti_mass[ia])
                                ti.atomic_add(ti_vel[ib][k], -f_total[k] * _dt / ti_mass[ib])
                            else:
                                ti.atomic_add(ti_vel[ia][k], f_total[k] * _dt)
                                ti.atomic_add(ti_vel[ib][k], -f_total[k] * _dt)

                    # This completed loop only snapshots the existing combined kick.
                    for i in range(num_nodes):
                        ledger_v_edge[i] = ti_vel[i]

                    for i in range(num_nodes):
                        for k in ti.static(range(dim)):
                            if ti_mass_mode[None]:
                                ti_vel[i][k] += ti_ext[i][k] * _dt / ti_mass[i]
                            else:
                                ti_vel[i][k] += ti_ext[i][k] * _dt
                        ledger_v_external[i] = ti_vel[i]
                        ti_vel[i] *= ti.exp(-_drag * _dt)
                        ledger_v_drag[i] = ti_vel[i]
                        for k in ti.static(range(dim)):
                            ti_pos[i][k] += ti_vel[i][k] * _dt
                        ledger_x_drift[i] = ti_pos[i]

                    for i in range(num_nodes):
                        if ti_fixed[i] == 1:
                            for k in ti.static(range(dim)):
                                ti_pos[i][k] = ti_pos0[i][k]
                                ti_vel[i][k] = 0.0

                    for i in range(num_nodes):
                        ledger_x_fixed[i] = ti_pos[i]
                        ledger_v_fixed[i] = ti_vel[i]

                    for i in range(num_nodes):
                        if ti_sched_mask[i] == 1:
                            for k in ti.static(range(dim)):
                                ti_pos[i][k] = ti_pos0[i][k] + ti_sched[i][k] * _ramp
                                ti_vel[i][k] = 0.0
                _cached_substep = _ledger_substep

            TaichiEngine._field_cache[cache_key] = {
                'ti_pos': ti_pos, 'ti_vel': ti_vel, 'ti_vel_old': ti_vel_old,
                'ti_mass': ti_mass, 'ti_mass_mode': ti_mass_mode,
                'ti_edges': ti_edges,
                'ti_L0': ti_L0, 'ti_k': ti_k, 'ti_fixed': ti_fixed,
                'ti_ext': ti_ext, 'ti_pos0': ti_pos0, 'ti_sched': ti_sched,
                'ti_sched_mask': ti_sched_mask, 'ti_dt': ti_dt,
                'ti_dashpot': ti_dashpot, 'ti_drag': ti_drag, 'ti_ramp': ti_ramp,
                'substep': _cached_substep,
            }
            if material_mode:
                TaichiEngine._field_cache[cache_key].update(
                    ti_material_force=ti_material_force, material_v_kick=material_v_kick,
                    material_v_drag=material_v_drag, material_x_drift=material_x_drift)
            if record_energy_ledger:
                TaichiEngine._field_cache[cache_key].update(
                    ledger_v_edge=ledger_v_edge, ledger_v_external=ledger_v_external,
                    ledger_v_drag=ledger_v_drag, ledger_x_drift=ledger_x_drift,
                    ledger_x_fixed=ledger_x_fixed, ledger_v_fixed=ledger_v_fixed)

        # Retrieve cached fields and kernel
        cached = TaichiEngine._field_cache[cache_key]
        ti_pos = cached['ti_pos']
        ti_vel = cached['ti_vel']
        ti_edges = cached['ti_edges']
        ti_L0 = cached['ti_L0']
        ti_k = cached['ti_k']
        ti_fixed = cached['ti_fixed']
        ti_ext = cached['ti_ext']
        ti_pos0 = cached['ti_pos0']
        ti_sched = cached['ti_sched']
        ti_sched_mask = cached['ti_sched_mask']
        ti_dt = cached['ti_dt']
        ti_dashpot = cached['ti_dashpot']
        ti_drag = cached['ti_drag']
        ti_ramp = cached['ti_ramp']
        substep = cached['substep']

        # Reset masses and mode on EVERY call, including explicit -> unit mass.
        cached['ti_mass'].from_numpy(masses)
        cached['ti_mass_mode'][None] = int(explicit_mass)
        # Reset field data (always, even on cache hit)
        ti_pos0.from_numpy(pos_orig.astype(np.float64))
        ti_pos.from_numpy(pos_orig.astype(np.float64))
        ti_vel.from_numpy(np.zeros_like(pos_orig) if initial_velocity is None else initial_velocity)
        ti_edges.from_numpy(elements.astype(np.int32))
        ti_L0.from_numpy(rest_lengths_np.astype(np.float64))
        ti_k.from_numpy(stiff_np.astype(np.float64))
        if material_mode:
            cached['ti_material_force'].from_numpy(material_history.area * material_history.stress)

        fixed_arr = np.zeros(num_nodes, dtype=np.int32)
        for ni in (fixed_nodes or []):
            fixed_arr[ni] = 1
        ti_fixed.from_numpy(fixed_arr)

        ext_f = external_force if external_force is not None else np.zeros((num_nodes, dim))
        ti_ext.from_numpy(ext_f.astype(np.float64)[:, :dim])

        # Per-node, piecewise-linear schedules. Reset on every call,
        # including calls reusing fields from a previous simulation.
        schedule_targets = np.zeros((num_nodes, dim), dtype=np.float64)
        sched_mask = np.zeros(num_nodes, dtype=np.int32)
        for ni in schedules:
            sched_mask[ni] = 1
        ti_sched.from_numpy(schedule_targets)
        ti_sched_mask.from_numpy(sched_mask)

        # The kernel consumes absolute scheduled displacements.
        ti_ramp[None] = 1.0

        # Run loop
        ti_dt[None] = dt
        ti_dashpot[None] = dashpot
        ti_drag[None] = drag

        trajectory = [pos_orig.copy()]
        state_samples = []
        ledger_rows = []
        if record_energy_ledger:
            from ._energy_accounting import ExactLedger
            def ledger_potential(positions):
                if not np.isfinite(positions).all():
                    raise ValueError('ledger positions must be finite')
                lengths_now = np.linalg.norm(positions[elements[:, 1]] - positions[elements[:, 0]], axis=1)
                potential = float(.5 * np.sum(stiff_np * (lengths_now / rest_lengths_np - 1)**2 * rest_lengths_np))
                if not np.isfinite(potential) or (potential == 0 and np.any((stiff_np > 0) & (lengths_now / rest_lengths_np - 1 != 0))):
                    raise ValueError('ledger potential is outside finite nonzero float64 range')
                return potential, bool(np.any(lengths_now <= 1e-12))
            ledger_x = pos_orig.copy()
            ledger_v = np.zeros_like(pos_orig)
            initial_u, ledger_cutoff = ledger_potential(ledger_x)
            ledger_endpoint = dict(kinetic_energy=0., potential_energy=initial_u, total_energy=initial_u)
            ledger = ExactLedger(ledger_endpoint, ext_f)
            ledger.near_cutoff_any = ledger_cutoff
        def sample_state(step, positions):
            if record_energy_ledger:
                state_samples.append(dict(step=int(step), time=float(step * dt),
                    positions=positions.tolist(), velocities=ledger_v.tolist(),
                    **ledger_endpoint, near_cutoff=ledger_cutoff))
                ledger_rows.append(ledger.sample(step, step * dt, ledger_endpoint, dashpot))
                return
            velocity = ti_vel.to_numpy()
            lengths_now = np.linalg.norm(positions[elements[:, 1]] - positions[elements[:, 0]], axis=1)
            kinetic = _checked_kinetic_energy(masses, velocity)
            potential = (float(np.sum(material_history.stored)) if material_mode else
                float(.5 * np.sum(stiff_np * (lengths_now / rest_lengths_np - 1)**2 * rest_lengths_np)))
            if potential == 0 and np.any((stiff_np > 0) & (lengths_now / rest_lengths_np - 1 != 0)):
                raise ValueError('nonzero sampled potential is outside float64 range')
            state_samples.append(dict(step=int(step), time=float(step * dt),
                positions=positions.tolist(), velocities=velocity.tolist(),
                kinetic_energy=kinetic, potential_energy=potential,
                total_energy=kinetic + potential, near_cutoff=bool(np.any(lengths_now <= 1e-12))))
        if record_state:
            sample_state(0, pos_orig)
        max_stretch_history = []
        if state_observer is not None:
            _emit_axial_observation(state_observer, 0, dt, observer_node_ids,
                                    observer_units, ti_pos.to_numpy(), ti_vel.to_numpy(),
                                    material_state=material_history.snapshot() if material_mode else None)

        for step in range(num_steps):
            if schedules:
                for ni, schedule in schedules.items():
                    schedule_targets[ni] = self._interpolate_schedule(
                        schedule, step + 1, num_steps
                    )
                ti_sched.from_numpy(schedule_targets)
            substep()
            if material_mode:
                candidate = material_history.prepare(ti_pos.to_numpy(), ti_vel.to_numpy(),
                    cached['material_v_kick'].to_numpy(), cached['material_v_drag'].to_numpy(),
                    cached['material_x_drift'].to_numpy(), ext_f)
                cached['ti_material_force'].from_numpy(candidate['force'])
                material_history.commit(candidate)
            if state_observer is not None:
                _emit_axial_observation(state_observer, step + 1, dt, observer_node_ids,
                                        observer_units, ti_pos.to_numpy(), ti_vel.to_numpy(),
                                    material_state=material_history.snapshot() if material_mode else None)
            if record_energy_ledger:
                # Read only actual completed stages. No force or velocity is reconstructed.
                v_edge = cached['ledger_v_edge'].to_numpy()
                v_external = cached['ledger_v_external'].to_numpy()
                v_drag = cached['ledger_v_drag'].to_numpy()
                x_drift = cached['ledger_x_drift'].to_numpy()
                kb = _checked_kinetic_energy(masses, v_edge)
                kc = _checked_kinetic_energy(masses, v_external)
                kd = _checked_kinetic_energy(masses, v_drag)
                ue, near_e = ledger_potential(x_drift)
                if fixed_nodes:
                    x_fixed = cached['ledger_x_fixed'].to_numpy()
                    v_fixed = cached['ledger_v_fixed'].to_numpy()
                    kf = _checked_kinetic_energy(masses, v_fixed)
                    uf, near_f = ledger_potential(x_fixed)
                else:
                    x_fixed, v_fixed, kf, uf, near_f = x_drift, v_drag, kd, ue, near_e
                if schedules:
                    x_final = ti_pos.to_numpy()
                    v_final = ti_vel.to_numpy()
                    kg = _checked_kinetic_energy(masses, v_final)
                    ug, near_g = ledger_potential(x_final)
                else:
                    x_final, v_final, kg, ug, near_g = x_fixed, v_fixed, kf, uf, near_f
                ledger.advance(step + 1, (step + 1) * dt,
                    (ledger_endpoint['kinetic_energy'], kb, kc, kd, kf, kg),
                    (ledger_endpoint['potential_energy'], ue, uf, ug),
                    ledger_x, x_drift, near_e or near_f or near_g)
                ledger_x, ledger_v, ledger_cutoff = x_final, v_final, near_g
                ledger_endpoint = ledger.previous

            if (step + 1) % save_interval == 0 or (record_state and step + 1 == num_steps):
                cur_pos = ti_pos.to_numpy()
                trajectory.append(cur_pos.copy())
                if material_mode:
                    material_history.sample()
                if record_state:
                    sample_state(step + 1, cur_pos)
                # Vectorized edge length computation
                new_len = np.linalg.norm(cur_pos[elements[:, 1]] - cur_pos[elements[:, 0]], axis=1)
                max_stretch_history.append(float(np.max(new_len / rest_lengths_np)))

        pos_final = ti_pos.to_numpy()
        displacements = pos_final - pos_orig
        if dim < 3:
            displacements = np.hstack([displacements, np.zeros((num_nodes, 3 - dim))])

        result = SimResult(
            displacements=displacements,
            time_seconds=time.time() - t0,
            mode="dynamics",
            deformed_positions=pos_final,
            positions_trajectory=trajectory,
            history=([{"step": 0, "max_stretch": float(np.max(np.linalg.norm(pos_orig[elements[:, 1]] - pos_orig[elements[:, 0]], axis=1) / rest_lengths_np))}] if record_state else []) +
                    [{"step": (state_samples[i+1]['step'] if record_state else (i+1)*save_interval), "max_stretch": ms}
                     for i, ms in enumerate(max_stretch_history)],
            metadata={"node_ids": list(node_ids), "edge_ids": list(graph.edges),
                      "stiffness": float(stiffness), "damping": float(damping),
                      "dt": float(dt), "num_steps": int(num_steps)},
        )
        if material_mode:
            result.metadata.pop('stiffness', None)
            result.metadata['native_axial_viscoelastic'] = material_history.result(
                graph, fixed_nodes, ext_f, drag)
            result.compute_detailed(graph)
        else:
            result.compute_detailed(graph, stiffness=stiff_np)
        if explicit_mass or units is not None:
            result.metadata['native_axial_inputs'] = {
                'schema': 'native_axial_inputs/1', 'node_ids': list(node_ids),
                'masses': masses.tolist(), 'units': units}
        if material_mode and record_state:
            result.metadata['native_axial_viscoelastic']['sampled_native_state'] = state_samples
        if record_state and not material_mode:
            state = dict(schema='native_axial_state/1', representation='sampled',
                availability='available', units=units, dt=float(dt),
                node_ids=list(node_ids), masses=masses.tolist(),
                steps=[v['step'] for v in state_samples], times=[v['time'] for v in state_samples],
                final=state_samples[-1], cutoff_check='sampled_only',
                work_ledger={'status': 'unavailable', 'value': None, 'reason': 'not_collected'})
            for name in ('positions', 'velocities', 'kinetic_energy', 'potential_energy', 'total_energy', 'near_cutoff'):
                state[name] = [v[name] for v in state_samples]
            result.dynamics_state = _validate_dynamics_state(state, num_nodes, list(node_ids))
            if record_energy_ledger:
                result.algorithmic_energy_ledger = ledger.result(ledger_rows, result.dynamics_state,
                    dashpot, drag, 'explicit' if explicit_mass else 'unit',
                    result.metadata.get('native_axial_inputs'))
        return result
    def stretch_test(
        self,
        graph: StructureGraph,
        target_stretch: float = 2.0,
        stiffness: float = 1e5,
        damping: float = 0.3,
        num_steps: int = None,
        save_interval: int = 1000,
        ramp_fraction: float = 0.2,
        auto_steps: bool = True,
        max_nodes: int = 20000,
    ) -> SimResult:
        """Displacement-controlled uniaxial stretch with automatic step calculation.

        Two-phase simulation:
        1. Ramp phase: gradually move right boundary to target_stretch
        2. Relax phase: hold displacement, let waves propagate and settle

        Parameters
        ----------
        target_stretch : float
            Target stretch ratio (e.g., 2.0 = double the length)
        stiffness : float
            Force-like axial coefficient k in F = k * strain. With a consistent
            physical unit system this has force units (N in SI), not N/m.
            Energy is 0.5 * k * strain**2 * rest_length. Edge Material properties
            are not used to determine this coefficient.
        damping : float
            Velocity damping coefficient (0-1)
        num_steps : int, optional
            Total simulation steps. If None and auto_steps=True, calculated from graph diameter
        save_interval : int
            Save trajectory every N steps
        ramp_fraction : float
            Fraction of total steps for ramp phase (default 0.2 = 20%)
        auto_steps : bool
            Auto-calculate steps based on graph diameter if num_steps is None

        Returns
        -------
        SimResult
            Result with deformed_positions and trajectory
        """
        _check_taichi_owner()
        if not np.isfinite(ramp_fraction) or not 0 <= ramp_fraction <= 1:
            raise ValueError("ramp_fraction must be finite and between 0 and 1")
        if num_steps is not None and (
            isinstance(num_steps, (bool, np.bool_))
            or not isinstance(num_steps, (int, np.integer))
            or num_steps <= 0
        ):
            raise ValueError("num_steps must be a positive integer")

        import networkx as nx

        pos, elements, _, _ = _graph_to_arrays(graph)
        bnd = _get_boundary_indices(pos)
        L_x = pos[:, 0].max() - pos[:, 0].min()
        target_disp = L_x * (target_stretch - 1)

        # Auto-calculate steps based on graph diameter
        if num_steps is None and auto_steps:
            # Build graph to find diameter
            G = nx.Graph()
            for e in elements:
                G.add_edge(int(e[0]), int(e[1]))
            
            # Find max hops from right to left (sample for speed)
            right_set = set(bnd['right'])
            left_set = set(bnd['left'])
            max_hops = 0
            for rid in list(right_set)[:5]:
                for lid in list(left_set)[:5]:
                    try:
                        hops = nx.shortest_path_length(G, rid, lid)
                        max_hops = max(max_hops, hops)
                    except:
                        pass
            
            # 3D structures need more steps than 2D
            is_3d = pos.shape[1] == 3 and pos[:, 2].max() > 1e-6
            
            # Formula: base + hops * per_hop, capped reasonably
            base_steps = 3000 if is_3d else 2000
            per_hop = 300 if is_3d else 150
            num_steps = base_steps + max_hops * per_hop
            num_steps = min(num_steps, 20000)
        elif num_steps is None:
            num_steps = 5000

        ramp_steps = int(num_steps * ramp_fraction)
        schedule = {}
        target = np.array([target_disp, 0.0, 0.0])
        for ni in bnd["right"]:
            if ramp_steps == 0:
                schedule[ni] = [(0, target.copy())]
            else:
                schedule[ni] = [
                    (0, np.zeros(3)),
                    (ramp_steps, target.copy()),
                ]

        fixed = bnd["left"] + (bnd.get("bottom", [])[:1] if bnd.get("bottom") else [])

        return self.dynamics(
            graph,
            fixed_nodes=fixed,
            displacement_schedule=schedule,
            stiffness=stiffness,
            damping=damping,
            dt=1e-5,
            num_steps=num_steps,
            save_interval=save_interval,
            max_nodes=max_nodes,
        )

    @staticmethod
    def _interpolate_schedule(schedule, step, total_steps):
        """Linearly interpolate displacement from schedule."""
        if not schedule:
            return np.zeros(3)
        schedule = sorted(schedule, key=lambda x: x[0])
        if step <= schedule[0][0]:
            return np.array(schedule[0][1])
        if step >= schedule[-1][0]:
            return np.array(schedule[-1][1])
        for i in range(len(schedule) - 1):
            s0, d0 = schedule[i]
            s1, d1 = schedule[i + 1]
            if s0 <= step <= s1:
                t = (step - s0) / (s1 - s0)
                return np.array(d0) * (1 - t) + np.array(d1) * t
        return np.array(schedule[-1][1])


