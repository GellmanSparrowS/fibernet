"""Experimental Exudyn/contact preparation contracts. No provider or execution.

StructureGraph remains authoritative. Immutable canonical source/request/map
bytes are provenance and compilation records, not a second graph. Source-only
support does not establish native readiness, scientific validity or restart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real

PROFILE = 'straight-shared-chain/1'
SCHEMA = 'fibernet.exudyn_contact_prepared/1'
COMPILER = 'fibernet.exudyn_shared_chain_source/1'
_MAX_BYTES = 4 * 1024 * 1024


class ExudynContractError(ValueError):
    def __init__(self, code, message, source_id=None):
        self.code = code
        self.source_id = source_id
        super().__init__(message)


def _fail(code, message, source_id=None):
    raise ExudynContractError(code, message, source_id)


def _real(value, label, *, positive=False, nonnegative=False):
    # NumPy bool_ is not Real; built-in bool must be excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail('nonfinite', label + ' must be a finite real number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ExudynContractError('nonfinite', label + ' cannot be represented') from exc
    if not math.isfinite(result):
        _fail('nonfinite', label + ' must be finite')
    if (positive and result <= 0) or (nonnegative and result < 0):
        _fail('invalid_request', label + ' is outside its admitted domain')
    return result


def _id(value, label='physical ID'):
    if isinstance(value, bool) or not isinstance(value, Integral):
        _fail('invalid_id', label + ' must be an integer, not a boolean')
    return int(value)


def _tuple(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        _fail('invalid_request', label + ' must have ' + str(size) + ' components')
    return tuple(_real(x, label) for x in value)


def _canonical(value):
    """Bounded JSON-only, string-keyed provenance; never pickle objects."""
    active = set()
    def check(v, depth=0):
        if depth > 64:
            _fail('invalid_request', 'JSON provenance exceeds depth 64')
        if v is None or type(v) in (str, bool, int):
            return
        if type(v) is float:
            if not math.isfinite(v):
                _fail('nonfinite', 'nonfinite JSON provenance')
            return
        if type(v) in (list, tuple, dict):
            if id(v) in active:
                _fail('invalid_request', 'cyclic JSON provenance')
            if len(v) > 100000:
                _fail('invalid_request', 'JSON container exceeds bounded preparation scope')
            active.add(id(v))
            try:
                if type(v) is dict:
                    if any(type(k) is not str for k in v):
                        _fail('invalid_request', 'JSON provenance keys must be strings')
                    items = v.values()
                else:
                    items = v
                for item in items:
                    check(item, depth + 1)
            finally:
                active.remove(id(v))
            return
        _fail('invalid_request', 'provenance must contain only JSON builtin values')
    check(value)
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('ascii')
    except (ValueError, OverflowError, RecursionError) as exc:
        raise ExudynContractError('invalid_request', 'invalid JSON provenance') from exc
    if len(raw) > _MAX_BYTES:
        _fail('invalid_request', 'canonical payload exceeds 4 MiB')
    return raw


def _decode(raw):
    if type(raw) is not bytes or len(raw) > _MAX_BYTES:
        _fail('prepared_integrity', 'prepared payload must be bounded immutable bytes')
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ExudynContractError('prepared_integrity', 'invalid prepared JSON') from exc
    if _canonical(value) != raw:
        _fail('prepared_integrity', 'prepared bytes are not canonical finite JSON')
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SphereSpec:
    radius_m: float
    mass_kg: float
    center_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, 'radius_m', _real(self.radius_m, 'sphere radius', positive=True))
        object.__setattr__(self, 'mass_kg', _real(self.mass_kg, 'sphere mass', positive=True))
        object.__setattr__(self, 'center_m', _tuple(self.center_m, 3, 'sphere center'))
        object.__setattr__(self, 'velocity_m_s', _tuple(self.velocity_m_s, 3, 'sphere velocity'))


@dataclass(frozen=True)
class ChordContactSpec:
    k_N_per_m2: float
    w_N: float
    range_m: float

    def __post_init__(self):
        for name in ('k_N_per_m2', 'range_m'):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        object.__setattr__(self, 'w_N', _real(self.w_N, 'w_N', nonnegative=True))
        try:
            derived = (6 * self.w_N / self.range_m**2,
                       1.5 * self.w_N / self.range_m,
                       self.k_N_per_m2 * 1e-4,
                       self.k_N_per_m2 * 1e-8 / 2 + self.w_N)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ExudynContractError('nonfinite', 'contact coefficient intermediate overflow') from exc
        if not all(math.isfinite(x) for x in derived):
            _fail('nonfinite', 'contact coefficient intermediate overflow')


@dataclass(frozen=True)
class StepPhase:
    start_s: float
    end_s: float
    steps: int

    def __post_init__(self):
        start = _real(self.start_s, 'phase start')
        end = _real(self.end_s, 'phase end')
        steps = _id(self.steps, 'phase steps')
        if start < 0 or end <= start or not 1 <= steps <= 1000000:
            _fail('invalid_schedule', 'phase requires end > start and 1..1000000 steps')
        if not math.isfinite(end - start) or (end - start) / steps <= 0:
            _fail('invalid_schedule', 'phase duration/timestep is not finite and positive')
        object.__setattr__(self, 'start_s', start)
        object.__setattr__(self, 'end_s', end)
        object.__setattr__(self, 'steps', steps)

    @property
    def dt_s(self):
        return (self.end_s - self.start_s) / self.steps


@dataclass(frozen=True)
class ExudynContactRequest:
    profile: str
    length_unit: str
    clamped_node_ids: tuple[int, ...]
    shear_correction: tuple[float, float]
    sphere: SphereSpec
    contact: ChordContactSpec
    phases: tuple[StepPhase, ...]
    geometry_energy_budget_J: float
    gravity_m_s2: tuple[float, float, float]
    external_forces_N: tuple
    initial_fiber_state: str

    def __post_init__(self):
        if type(self.profile) is not str or self.profile != PROFILE:
            _fail('unsupported_profile', 'only ' + PROFILE + ' preparation is implemented')
        if type(self.length_unit) is not str or self.length_unit != 'm':
            _fail('invalid_units', 'this preparation profile requires explicit length_unit="m"')
        if not isinstance(self.clamped_node_ids, (tuple, list)):
            _fail('invalid_boundary', 'clamps must be a sequence of physical IDs')
        clamps = tuple(_id(n) for n in self.clamped_node_ids)
        if len(clamps) != 2 or len(set(clamps)) != 2:
            _fail('invalid_boundary', 'exactly two distinct endpoint clamps are required')
        object.__setattr__(self, 'clamped_node_ids', tuple(sorted(clamps)))
        shear = _tuple(self.shear_correction, 2, 'shear correction')
        if min(shear) <= 0:
            _fail('unsupported_material', 'shear correction factors must be positive')
        object.__setattr__(self, 'shear_correction', shear)
        if type(self.sphere) is not SphereSpec or type(self.contact) is not ChordContactSpec:
            _fail('invalid_request', 'typed sphere and contact records are required')
        if not isinstance(self.phases, (tuple, list)) or not 1 <= len(self.phases) <= 16:
            _fail('invalid_schedule', 'one to sixteen explicit phases are required')
        phases = tuple(self.phases)
        if any(type(p) is not StepPhase for p in phases):
            _fail('invalid_schedule', 'each phase must be a StepPhase')
        if phases[0].start_s != 0 or any(a.end_s != b.start_s for a, b in zip(phases, phases[1:])):
            _fail('invalid_schedule', 'phases start at zero and share exactly equal declared boundaries')
        if sum(p.steps for p in phases) > 1000000:
            _fail('invalid_schedule', 'total step count exceeds preparation cap')
        object.__setattr__(self, 'phases', phases)
        object.__setattr__(self, 'geometry_energy_budget_J',
                           _real(self.geometry_energy_budget_J, 'geometry budget', nonnegative=True))
        gravity = _tuple(self.gravity_m_s2, 3, 'gravity')
        if any(gravity):
            _fail('unsupported_load', 'nonzero gravity is outside this preparation profile')
        object.__setattr__(self, 'gravity_m_s2', gravity)
        if not isinstance(self.external_forces_N, (list, tuple)) or len(self.external_forces_N):
            _fail('unsupported_load', 'external forces must be explicitly empty')
        object.__setattr__(self, 'external_forces_N', ())
        if type(self.initial_fiber_state) is not str or self.initial_fiber_state != 'reference_rest':
            _fail('unsupported_profile', 'initial fiber state must be reference_rest')

    def to_dict(self):
        return json.loads(_canonical(asdict(self)))


def _request_from_record(record):
    if type(record) is not dict or set(record) != set(ExudynContactRequest.__dataclass_fields__):
        _fail('prepared_integrity', 'request fields mismatch')
    try:
        values = dict(record)
        values['sphere'] = SphereSpec(**values['sphere'])
        values['contact'] = ChordContactSpec(**values['contact'])
        values['phases'] = tuple(StepPhase(**p) for p in values['phases'])
        return ExudynContactRequest(**values)
    except (TypeError, KeyError) as exc:
        raise ExudynContractError('prepared_integrity', 'malformed typed request record') from exc


def _identity(source, request, mapping):
    return _sha(_canonical([SCHEMA, COMPILER, _sha(source), _sha(request), _sha(mapping)]))


@dataclass(frozen=True)
class PreparedExudynContact:
    source_record_bytes: bytes
    request_bytes: bytes
    mapping_bytes: bytes
    source_sha256: str
    request_sha256: str
    mapping_sha256: str
    identity: str
    schema: str = SCHEMA
    compiler: str = COMPILER

    def __post_init__(self):
        validate_prepared(self)

    def source_record(self):
        return _decode(self.source_record_bytes)

    def request(self):
        return _decode(self.request_bytes)

    def mapping(self):
        return _decode(self.mapping_bytes)

    def manifest(self):
        validate_prepared(self)
        mapping = self.mapping()
        return {'schema': self.schema, 'compiler': self.compiler, 'identity': self.identity,
                'source_sha256': self.source_sha256, 'request_sha256': self.request_sha256,
                'mapping_sha256': self.mapping_sha256, 'profile': PROFILE,
                'node_ids': mapping['node_ids'], 'edge_ids': mapping['edge_ids'],
                'counts': mapping['counts'],
                'source_graph_json_roundtrip_supported': mapping['source_graph_json_roundtrip_supported'],
                'qualification': 'source_supported_not_natively_qualified',
                'execution_available': False, 'restart_supported': False}


def validate_prepared(prepared):
    if type(prepared) is not PreparedExudynContact or prepared.schema != SCHEMA or prepared.compiler != COMPILER:
        _fail('prepared_integrity', 'unsupported prepared type/schema/compiler')
    source, request, mapping = (_decode(b) for b in
                              (prepared.source_record_bytes, prepared.request_bytes, prepared.mapping_bytes))
    expected = (_sha(prepared.source_record_bytes), _sha(prepared.request_bytes),
                _sha(prepared.mapping_bytes), _identity(prepared.source_record_bytes,
                                                       prepared.request_bytes, prepared.mapping_bytes))
    actual = (prepared.source_sha256, prepared.request_sha256, prepared.mapping_sha256, prepared.identity)
    if actual != expected:
        _fail('prepared_integrity', 'prepared identity/checksum mismatch')
    from ._exudyn_mapping import _compile_snapshot
    rebuilt = _compile_snapshot(source, _request_from_record(request))
    if _canonical(rebuilt) != prepared.mapping_bytes:
        _fail('prepared_integrity', 'prepared mapping disagrees with source/request')
    return True


def prepare_exudyn_contact(graph, request):
    """Pure preparation only. No native execution, file output or task creation."""
    from ._exudyn_mapping import _snapshot_graph, _compile_snapshot
    if type(request) is not ExudynContactRequest:
        raise TypeError('request must be ExudynContactRequest')
    # Revalidate a detached request as defense against deliberate frozen-object bypass.
    request_bytes = _canonical(request.to_dict())
    checked_request = _request_from_record(_decode(request_bytes))
    source_bytes = _canonical(_snapshot_graph(graph))
    mapping_bytes = _canonical(_compile_snapshot(_decode(source_bytes), checked_request))
    return PreparedExudynContact(source_bytes, request_bytes, mapping_bytes,
                                 _sha(source_bytes), _sha(request_bytes), _sha(mapping_bytes),
                                 _identity(source_bytes, request_bytes, mapping_bytes))


def source_capabilities():
    """Descriptive source support, never provider discovery or qualification."""
    return {'backend': 'exudyn', 'preparation_available': True, 'execution_available': False,
            'runtime_readiness': 'not_probed', 'restart_supported': False,
            'profiles': [PROFILE], 'qualification': 'source_supported_not_natively_qualified',
            'units': {'graph_length': 'm', 'material': 'SI'},
            'unsupported': ['native execution', 'restart', 'expanded/welded web execution',
                            'refinement', 'curved or branched graphs', 'nonzero loads',
                            'easy.simulate backend dispatch']}
