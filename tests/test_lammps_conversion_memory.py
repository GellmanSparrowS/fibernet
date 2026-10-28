"""Conversion-only regressions; synthetic records, never native execution."""
import copy
from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest

from fibernet.integrations import _harmonic_profile as profiles
from fibernet.integrations import lammps_backend as backend
from fibernet.sim import _lammps_result as adapter
from test_lammps_fixed_state import CASES, prepared_case, typed_raw, RUNTIME
from test_lammps_full_history_profile import small
from test_lammps_backend_contract import prepared


KINDS = (*CASES, 'full-history', 'zero-step')


def case(kind):
    if kind == 'full-history':
        return small()
    if kind == 'zero-step':
        return prepared(run=backend.RunSpec(1e-6, 0, 1, ((-.01, .01),)*3))
    return prepared_case(kind)


@pytest.fixture(autouse=True)
def no_native():
    with patch('subprocess.Popen', side_effect=AssertionError('native forbidden')):
        yield


@pytest.mark.parametrize('kind', KINDS)
def test_drop_state_matches_recorded_result_exactly_and_leaves_inputs(kind, monkeypatch):
    p = case(kind)
    raw = typed_raw(p)
    before = copy.deepcopy((p, raw))
    recorded = adapter.to_sim_result(p, raw, RUNTIME, .125, True)
    expected = copy.deepcopy(recorded)
    expected.dynamics_state = None
    expected.positions_trajectory = []
    def no_detach(*args, **kwargs):
        raise AssertionError('discarded history must not be detached')
    monkeypatch.setattr(adapter, 'validate_state', no_detach)
    dropped = adapter.to_sim_result(p, raw, RUNTIME, .125, False)
    assert dropped.to_dict(True) == expected.to_dict(True)
    assert (p, raw) == before


@pytest.mark.parametrize('kind', KINDS)
def test_lazy_json_and_byte_budget_match_eager_state(kind, monkeypatch):
    p = case(kind)
    raw = typed_raw(p)
    state = adapter.to_sim_result(p, raw, RUNTIME, 0., True).dynamics_state
    # Only recorded results retain the new original-SI reference. The transient
    # no-state validation view deliberately remains reference-absent.
    state.pop('displacement_reference')
    eager_bytes = profiles.json_bytes(state, 1024**2)
    original = profiles.json_bytes
    observed = []
    def check(value, limit, **kwargs):
        if isinstance(value, dict) and value.get('representation') == 'sampled':
            assert isinstance(value['positions'], adapter._ConvertedFrames)
            assert ''.join(json.JSONEncoder(allow_nan=False).iterencode(value)) == json.dumps(state, allow_nan=False)
            assert original(value, 1024**2) == eager_bytes
            with pytest.raises(ValueError):
                original(value, eager_bytes-1)
            observed.append(True)
        return original(value, limit, **kwargs)
    monkeypatch.setattr(profiles, 'json_bytes', check)
    adapter.to_sim_result(p, raw, RUNTIME, 0., False)
    assert bool(observed) == ('profile' in state)


@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('attr,value', [
    ('positions_m', ((0., 0.), (.001, 0., 0.))),
    ('forces_N', ((float('nan'), 0., 0.), (0., 0., 0.))),
    ('forces_N', ((True, 0., 0.), (0., 0., 0.))),
    ('velocities_m_s', ((np.bool_(False), 0., 0.), (0., 0., 0.))),
    ('time_s', True), ('time_s', 123.), ('step', True),
])
def test_discarded_intermediate_frame_is_still_checked(record, attr, value):
    p = prepared_case('harmonic_sampled_v1')
    raw = typed_raw(p)
    frames = list(raw.trajectory)
    frames[1] = replace(frames[1], **{attr: value})
    with pytest.raises(ValueError):
        adapter.to_sim_result(p, replace(raw, trajectory=tuple(frames)), RUNTIME, 0., record)


@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('attr,value', [('potential_J', -1.), ('kinetic_J', -1.),
    ('total_J', 8.), ('total_J', True), ('total_J', float('inf')),
    ('pressure_tensor_Pa', (0.,)*5), ('pressure_tensor_Pa', (False,)*6), ('step', True)])
def test_discarded_intermediate_thermo_is_still_checked(record, attr, value):
    p = prepared_case('harmonic_sampled_v1')
    raw = typed_raw(p)
    thermo = list(raw.thermo)
    thermo[1] = replace(thermo[1], **{attr: value})
    with pytest.raises(ValueError):
        adapter.to_sim_result(p, replace(raw, thermo=tuple(thermo)), RUNTIME, 0., record)



@pytest.mark.parametrize('record', [False, True])
def test_discarded_balanced_negative_harmonic_potential_is_still_checked(record):
    p = prepared_case('harmonic_sampled_v1')
    raw = typed_raw(p)
    # Valid output retains steps 0 and 2, not the observed native step 1.
    intact = adapter.to_sim_result(p, raw, RUNTIME, 0., True)
    assert intact.dynamics_state['steps'] == [0, 2]
    thermo = list(raw.thermo)
    assert thermo[1].step == 1 and thermo[1].kinetic_J == 2.
    thermo[1] = replace(thermo[1], potential_J=-1., total_J=1.)
    with pytest.raises(ValueError, match='inconsistent native frame energy'):
        adapter.to_sim_result(p, replace(raw, thermo=tuple(thermo)), RUNTIME, 0., record)


def test_json_view_is_repeatable_and_only_converts_requested_frame():
    calls = []
    view = adapter._ConvertedFrames(range(3), lambda n: calls.append(n) or [n])
    assert len(view) == 3 and bool(view)
    assert view[-1] == [2] and calls == [2]
    calls.clear()
    assert ''.join(json.JSONEncoder().iterencode(view)) == '[[0], [1], [2]]'
    assert calls == [0, 1, 2]
    assert list(view) == [[0], [1], [2]]
    assert ''.join(json.JSONEncoder().iterencode(adapter._ConvertedFrames([], list))) == '[]'
