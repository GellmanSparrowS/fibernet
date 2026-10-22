"""Provider-free export/parser controls; no new scientific trajectories."""
import copy
import inspect
import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from fibernet.integrations import lammps_backend as backend
from fibernet.integrations import _harmonic_profile as profiles
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult
from test_lammps_backend_contract import fixture
from test_lammps_fixed_state import typed_raw, RUNTIME, edit_dump
from test_lammps_full_history_profile import small


@pytest.fixture(autouse=True)
def forbid_native_launch():
    with patch('subprocess.Popen', side_effect=AssertionError('native launch forbidden')):
        yield


def result():
    p = small(4)
    return adapter.to_sim_result(p, typed_raw(p), RUNTIME, 0., True)


class FallbackResult(SimResult):
    pass


def fallback(r):
    out = FallbackResult()
    out.__dict__.update(copy.deepcopy(vars(r)))
    return out


def encoded(r, detailed=True):
    return json.dumps(r.to_dict(detailed), indent=2).encode('utf-8')


def test_plain_full_history_uses_one_guard_and_preserves_exact_bytes():
    r = result()
    assert adapter._plain_full_history_export(r, True)
    original = profiles.result_guard
    with patch.object(profiles, 'result_guard', wraps=original) as guard:
        fast = encoded(r)
        assert guard.call_count == 1
    with patch.object(profiles, 'result_guard', wraps=original) as guard:
        slow = encoded(fallback(r))
        assert guard.call_count == 2
    assert fast == slow
    assert encoded(r, False) == encoded(fallback(r), False)
    assert not adapter._plain_full_history_export(r, False)


def test_every_export_revalidates_caller_mutation_and_detaches_state():
    r = result()
    d = r.to_dict(True)
    d['dynamics_state']['positions'][1][0][0] = 99.
    assert r.dynamics_state['positions'][1][0][0] != 99.
    r.to_dict(True)
    r.dynamics_state['positions'][2][0][0] = float('nan')
    with pytest.raises(ValueError):
        r.to_dict(True)


@pytest.mark.parametrize('field', ['positions', 'velocities', 'nodal_forces',
                                  'pressure_tensor', 'potential_energy',
                                  'kinetic_energy', 'total_energy', 'times'])
@pytest.mark.parametrize('bad', [True, float('nan'), float('inf')])
def test_middle_sample_bad_scalars_are_not_skipped(field, bad):
    r = result()
    rows = r.dynamics_state[field]
    if field in ('positions', 'velocities', 'nodal_forces'):
        rows[2][0][0] = bad
    elif field == 'pressure_tensor':
        rows[2][0] = bad
    else:
        rows[2] = bad
    with pytest.raises(ValueError):
        r.to_dict(True)


@pytest.mark.parametrize('change', [
    lambda r: r.dynamics_state['steps'].__setitem__(3, 4),
    lambda r: r.dynamics_state['positions'].pop(3),
    lambda r: r.dynamics_state.__setitem__('unexpected', 1),
    lambda r: r.dynamics_state['final'].__setitem__('time', 1.),
    lambda r: r.metadata['source_identity'].__setitem__('graph_sha256', 'x'*64),
    lambda r: r.positions_trajectory[2].__setitem__((0, 0), 1.),
])
def test_late_schema_schedule_provenance_and_result_mutation_rejected(change):
    r = result()
    r.to_dict(True)
    change(r)
    with pytest.raises(ValueError):
        r.to_dict(True)


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize('change', [
    lambda r: setattr(r, 'metadata', DictSubclass(r.metadata)),
    lambda r: setattr(r, 'dynamics_state', DictSubclass(r.dynamics_state)),
    lambda r: r.dynamics_state.__setitem__('positions', ListSubclass(r.dynamics_state['positions'])),
    lambda r: setattr(r, 'history', ListSubclass(r.history)),
    lambda r: setattr(r, 'positions_trajectory', ListSubclass(r.positions_trajectory)),
    lambda r: setattr(r, 'edge_forces', r.edge_forces.view(ArraySubclass)),
    lambda r: setattr(r, 'energy', np.float64(r.energy)),
])
def test_accepted_nonplain_values_retain_original_path(change):
    r = result()
    change(r)
    assert not adapter._plain_full_history_export(r, True)
    with patch.object(profiles, 'result_guard', wraps=profiles.result_guard) as guard:
        assert encoded(r) == encoded(fallback(r))
        assert guard.call_count == 4


def test_array_tolist_mutation_cannot_gain_fast_path():
    r = result()
    class Mutator(np.ndarray):
        def tolist(self):
            r.dynamics_state['velocities'][2][0][0] = float('nan')
            return super().tolist()
    r.edge_forces = r.edge_forces.view(Mutator)
    assert not adapter._plain_full_history_export(r, True)
    with pytest.raises(ValueError):
        r.to_dict(True)


@pytest.mark.parametrize('kind', ['object', 'masked', 'bool', 'underflow'])
def test_unusual_arrays_never_gain_a_coercion_shortcut(kind):
    r = result()
    if kind == 'object':
        r.displacements = np.array(r.displacements, dtype=object)
    elif kind == 'masked':
        r.displacements = np.ma.array(r.displacements, mask=True)
    elif kind == 'bool':
        r.displacements = np.array(r.displacements, dtype=bool)
    else:
        if np.finfo(np.longdouble).tiny >= np.finfo(np.float64).tiny:
            pytest.skip('platform lacks a wider real dtype')
        r.displacements = np.array(r.displacements, dtype=np.longdouble)
        r.displacements[0, 0] = np.finfo(np.longdouble).tiny
    if kind != 'underflow':
        assert not adapter._plain_full_history_export(r, True)
    with pytest.raises(ValueError):
        r.to_dict(True)


def test_oversized_and_deep_metadata_still_rejected():
    r = result()
    r.metadata['extra'] = 'x'*(profiles.FULL_HISTORY_PROFILE['prepared_bytes']+1)
    with pytest.raises(ValueError, match='budget'):
        r.to_dict(True)
    r = result()
    nested = 0
    for _ in range(65):
        nested = [nested]
    r.metadata['extra'] = nested
    assert not adapter._plain_full_history_export(r, True)
    with pytest.raises(ValueError, match='nesting'):
        r.to_dict(True)


def test_exact_utf8_and_escaped_json_byte_bounds_remain():
    value = {'text': '汉字"\\\n', 'number': 1.25}
    for indent in (None, 2):
        expected = len(json.dumps(value, allow_nan=False, indent=indent).encode('utf-8'))
        assert profiles.json_bytes(value, expected, indent=indent) == expected
        with pytest.raises(ValueError, match='byte budget'):
            profiles.json_bytes(value, expected-1, indent=indent)
    text = '汉字a'
    assert profiles.string_bytes((text,), 7) == 7
    with pytest.raises(ValueError):
        profiles.string_bytes((text,), 6)


def test_parser_hoists_only_its_own_immutable_contract_decode():
    p = small(4)
    dump, log = fixture(p)
    calls = []
    original = json.loads
    def observe(value, *args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        if value is p.contract_json:
            calls.append(caller)
        return original(value, *args, **kwargs)
    with patch.object(json, 'loads', new=observe), \
            patch.object(backend, 'validate_prepared', wraps=backend.validate_prepared) as validate:
        raw = backend.parse_outputs(p, dump, log)
    assert calls.count('parse_outputs') == 1
    validate.assert_called_once_with(p)
    assert [f.step for f in raw.trajectory] == list(range(5))
    assert [f.step for f in raw.thermo] == list(range(5))


@pytest.mark.parametrize('step', [1, 3, 4])
def test_changed_middle_and_late_boxes_still_rejected(step):
    p = small(4)
    dump, log = fixture(p)
    rows = dump.splitlines()
    indices = [i for i, line in enumerate(rows) if line == 'ITEM: BOX BOUNDS ff ff ff']
    rows[indices[step]+1] = '-0.02 0.01'
    with pytest.raises(ValueError, match='box differs'):
        backend.parse_outputs(p, '\n'.join(rows)+'\n', log)


@pytest.mark.parametrize('step', [1, 3, 4])
@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_late_rows_rejected(step, bad):
    p = small(4)
    dump, log = fixture(p)
    with pytest.raises(ValueError, match='nonfinite'):
        backend.parse_outputs(p, edit_dump(p, dump, step, 40, 2, bad), log)


@pytest.mark.parametrize('change', ['truncated', 'extra', 'missing_marker', 'duplicate_marker',
                                  'bad_step', 'bad_type'])
def test_parser_protocol_failures_rejected(change):
    p = small(4)
    dump, log = fixture(p)
    if change == 'truncated':
        dump = '\n'.join(dump.splitlines()[:-1])+'\n'
    elif change == 'extra':
        dump += 'unexpected\n'
    elif change == 'missing_marker':
        log = log.replace(p.completion_marker, '')
    elif change == 'duplicate_marker':
        log += p.completion_marker+'\n'
    elif change == 'bad_step':
        dump = dump.replace('ITEM: TIMESTEP\n3\n', 'ITEM: TIMESTEP\n4\n')
    else:
        dump = dump.replace('2 2 0.001', '2 3 0.001')
    with pytest.raises(ValueError):
        backend.parse_outputs(p, dump, log)


def test_eligibility_is_bounded_before_authoritative_json_guard():
    r = result()
    shared = [0]
    for _ in range(60):
        shared = [shared, shared]
    r.metadata['extra'] = shared
    assert not adapter._plain_full_history_export(r, True)
    with pytest.raises(ValueError, match='budget'):
        r.to_dict(True)
    r = result()
    r.positions_trajectory = [r.positions_trajectory[0]]*2050
    assert not adapter._plain_full_history_export(r, True)
    with pytest.raises(ValueError, match='shape'):
        r.to_dict(True)


def test_custom_instance_dictionary_and_keys_cannot_gain_fast_path():
    r = result()
    class HookKey:
        def __hash__(self):
            return hash('max_force')
        def __eq__(self, other):
            raise AssertionError('eligibility must not invoke custom-key equality')
    value = vars(r).pop('max_force')
    vars(r)[HookKey()] = value
    assert not adapter._plain_full_history_export(r, True)
    r = result()
    r.__dict__ = DictSubclass(vars(r))
    assert not adapter._plain_full_history_export(r, True)


def test_custom_nested_mapping_keys_are_checked_before_any_lookup():
    class HookKey:
        def __init__(self, name):
            self.name = name
        def __hash__(self):
            return hash(self.name)
        def __eq__(self, other):
            raise AssertionError('eligibility must inspect key types before lookup')
    for owner, key in (('metadata', 'profile'), ('dynamics_state', 'schema')):
        r = result()
        mapping = getattr(r, owner)
        value = mapping.pop(key)
        mapping[HookKey(key)] = value
        assert not adapter._plain_full_history_export(r, True)


def test_extra_instance_fields_select_original_validation_path():
    r = result()
    r.extra_unrelated = object()
    assert not adapter._plain_full_history_export(r, True)
    with patch.object(profiles, 'result_guard', wraps=profiles.result_guard) as guard:
        r.to_dict(True)
        assert guard.call_count == 2
