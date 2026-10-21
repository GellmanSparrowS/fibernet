"""Inert typed-result domain contracts, not engine or physical evidence."""
from dataclasses import replace
import json

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import lammps_backend as backend
from fibernet.sim import _lammps_result as adapter
from test_lammps_fixed_state import (
    HARMONIC, SCALES, RUNTIME, harmonic_case, typed_raw, texts, edit_dump,
)


PROFILE_IDS = ('ordinary', 'large', 'sampled')
# Each edit violates one invariant while preserving identities, schedules,
# thermo, held node 10, and the final frame. Step 1 is omitted by sampled output.
MUTATIONS = {
    'initial-pose': (0, 'positions_m', (.001001, 0., 0.),
                     'native initial/fixed state mismatch', 'initial trajectory'),
    'initial-velocity': (0, 'velocities_m_s', (1e-4, 0., 0.),
                         'native initial/fixed state mismatch', 'initial trajectory'),
    'box-boundary': (1, 'positions_m', (.01, 0., 0.),
                     'native site left declared box', 'fixed-box interior'),
    'bond-reach': (1, 'positions_m', (.009, .009, 0.),
                   'native bond left declared reach', 'maximum-reach domain'),
    'collapsed-bond': (1, 'positions_m', (0., 0., 0.),
                       'native bond left declared reach', 'maximum-reach domain'),
}


def replace_mobile(raw, step, attr, value):
    frames = list(raw.trajectory)
    index = next(i for i, frame in enumerate(frames) if frame.step == step)
    frame = frames[index]
    rows = list(getattr(frame, attr))
    rows[frame.node_ids.index(40)] = value
    frames[index] = replace(frame, **{attr: tuple(rows)})
    return replace(raw, trajectory=tuple(frames))


def replace_mobile_text(p, dump, step, attr, value):
    first = 2 if attr == 'positions_m' else 5
    for axis, component in enumerate(value):
        dump = edit_dump(p, dump, step, 40, first+axis, component)
    return dump


def declared_mobile_case(profile, unit='mm'):
    # Reuse the graph/ID fixture, reverse its source order, and explicitly
    # declare nonzero mobile velocity. No held-ID guard can supply validation.
    p = harmonic_case(profile, unit, fixed=())
    data = json.loads(p.graph_json)
    data['nodes'].reverse()
    graph = StructureGraph.from_dict(data)
    args = dict(length_unit=unit, maximum_bond_length_m=.01,
                masses_kg={10: 1e-9, 40: 2e-9},
                bonds={77: backend.HarmonicBond(2., .001)},
                run=backend.RunSpec(p.timestep_s, p.steps, 1, ((-.02, .02),)*3),
                velocities_m_s={40: (.01, 0., 0.)}, profile=profile)
    if profile == 'harmonic_sampled_v1':
        args['output_stride'] = 2
    return backend.prepare_harmonic_beads(graph, **args)


@pytest.mark.parametrize('profile', HARMONIC, ids=PROFILE_IDS)
@pytest.mark.parametrize('mutation', tuple(MUTATIONS))
@pytest.mark.parametrize('record_state', [False, True], ids=['drop-state', 'keep-state'])
def test_typed_mobile_state_and_domain_rejected(profile, mutation, record_state):
    p = harmonic_case(profile)
    raw = typed_raw(p)  # Independently constructed, never passed through parser.
    step, attr, value, converter_error, _ = MUTATIONS[mutation]
    bad = replace_mobile(raw, step, attr, value)
    assert bad.identity == p.identity
    assert bad.thermo == raw.thermo
    assert bad.trajectory[-1] == raw.trajectory[-1]
    assert [f.step for f in bad.trajectory] == [f.step for f in raw.trajectory]
    assert all(f.node_ids == p.node_ids for f in bad.trajectory)
    assert all(f.positions_m[0] == raw.trajectory[0].positions_m[0] and
               f.velocities_m_s[0] == (0., 0., 0.) for f in bad.trajectory)
    if step:
        assert bad.trajectory[0] == raw.trajectory[0]
    with pytest.raises(ValueError, match=converter_error):
        adapter.to_sim_result(p, bad, RUNTIME, 0., record_state)


@pytest.mark.parametrize('profile', HARMONIC, ids=PROFILE_IDS)
@pytest.mark.parametrize('mutation', tuple(MUTATIONS))
def test_parser_rejects_matching_mobile_domain_edits(profile, mutation):
    p = harmonic_case(profile)
    dump, log = texts(p)
    step, attr, value, _, parser_error = MUTATIONS[mutation]
    bad = replace_mobile_text(p, dump, step, attr, value)
    with pytest.raises(backend.LAMMPSContractError, match=parser_error):
        backend.parse_outputs(p, bad, log)


@pytest.mark.parametrize('profile', HARMONIC, ids=PROFILE_IDS)
@pytest.mark.parametrize('unit', tuple(SCALES))
@pytest.mark.parametrize('record_state', [False, True], ids=['drop-state', 'keep-state'])
def test_declared_mobile_state_units_and_inclusive_reach(profile, unit, record_state):
    p = declared_mobile_case(profile, unit)
    before = (p.graph_json, p.contract_json, p.data_text, p.input_text)
    assert json.loads(p.contract_json)['fixed_node_ids'] == []
    raw = typed_raw(p)
    dump, log = texts(p)
    # Exactly maximum reach, strictly inside the larger declared box. This
    # synthetic trajectory is a protocol positive, not an integrated solution.
    for step in range(1, p.steps+1):
        raw = replace_mobile(raw, step, 'positions_m', (.01, 0., 0.))
        dump = replace_mobile_text(p, dump, step, 'positions_m', (.01, 0., 0.))
    parsed = backend.parse_outputs(p, dump, log)
    for candidate in (raw, parsed):
        result = adapter.to_sim_result(p, candidate, RUNTIME, 0., record_state)
        assert result.metadata['node_ids'] == [10, 40]
        assert result.metadata['edge_ids'] == [77]
        np.testing.assert_allclose(result.deformed_positions*SCALES[unit],
                                   [[0., 0., 0.], [.01, 0., 0.]], rtol=1e-14, atol=0.)
        np.testing.assert_allclose(result.displacements*SCALES[unit],
                                   [[0., 0., 0.], [.009, 0., 0.]], rtol=1e-14, atol=0.)
        assert result.metadata['reactions'] is None
        if record_state:
            state = result.dynamics_state
            assert state['velocities'][0][1] == [.01, 0., 0.]
            stride = 2 if profile == 'harmonic_sampled_v1' else 1
            assert state['steps'] == list(range(0, p.steps+1, stride))
        else:
            assert result.dynamics_state is None
            assert result.positions_trajectory == []
    assert before == (p.graph_json, p.contract_json, p.data_text, p.input_text)


@pytest.mark.parametrize('profile', HARMONIC, ids=PROFILE_IDS)
def test_initial_relative_tolerance_and_zero_absolute_tolerance(profile):
    p = declared_mobile_case(profile)
    raw = typed_raw(p)
    dump, log = texts(p)
    for attr, value, inside_delta, outside_delta in (
            ('positions_m', .001, 1e-17, 1e-12),
            ('velocities_m_s', .01, 1e-16, 1e-10)):
        inside = (value+inside_delta, 0., 0.)
        assert inside[0] != value
        for record in (False, True):
            adapter.to_sim_result(p, replace_mobile(raw, 0, attr, inside), RUNTIME, 0., record)
        backend.parse_outputs(p, replace_mobile_text(p, dump, 0, attr, inside), log)
        for outside in ((value+outside_delta, 0., 0.), (value, 1e-100, 0.)):
            bad = replace_mobile(raw, 0, attr, outside)
            for record in (False, True):
                with pytest.raises(ValueError, match='native initial/fixed state mismatch'):
                    adapter.to_sim_result(p, bad, RUNTIME, 0., record)
            with pytest.raises(backend.LAMMPSContractError, match='initial trajectory'):
                backend.parse_outputs(p, replace_mobile_text(p, dump, 0, attr, outside), log)
