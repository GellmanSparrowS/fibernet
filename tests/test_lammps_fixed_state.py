"""Inert result-contract regressions; no native LAMMPS or physics evidence."""
from dataclasses import replace
import json

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import lammps_backend as backend
from fibernet.integrations._lammps_cg import MorsePair, prepare_cg
from fibernet.sim import _lammps_result as adapter
from test_lammps_backend_contract import graph, fixture as harmonic_texts
from test_lammps_cg import fixture as cg_fixture, RUNTIME
from test_lammps_cg_network_profile import texts as network_texts


HARMONIC = (None, 'harmonic_large_v1', 'harmonic_sampled_v1')
CASES = (*HARMONIC, 'cg', 'cg_fiber_only',
         'cg_network_sampled_v1', 'cg_network_mixed_sampled_v1')
SCALES = {'m': 1., 'mm': .001, 'um': 1e-6}


def harmonic_case(profile, unit='mm', offset_m=0., fixed=(10,)):
    data = graph().to_dict()
    for node in data['nodes']:
        node['position'] = [(x*.001 + (offset_m if axis == 0 else 0.))/SCALES[unit]
                            for axis, x in enumerate(node['position'])]
    g = StructureGraph.from_dict(data)
    steps = 32 if profile == 'harmonic_large_v1' else 2
    args = dict(length_unit=unit, maximum_bond_length_m=.01,
                masses_kg={10: 1e-9, 40: 2e-9},
                bonds={77: backend.HarmonicBond(2., .001)},
                run=backend.RunSpec(1e-6, steps, 1, ((-.01, .01),)*3),
                fixed_node_ids=fixed, profile=profile)
    if profile == 'harmonic_sampled_v1':
        args['output_stride'] = 2
    return backend.prepare_harmonic_beads(g, **args)


def prepared_case(kind):
    if kind in HARMONIC:
        return harmonic_case(kind)
    g, request = cg_fixture(steps=2)
    request = replace(request, fixed_node_ids=(0,))
    if kind == 'cg_fiber_only':
        request = replace(request, projectile_node_id=None, projectile_pairs={})
    elif kind.startswith('cg_network_'):
        request = replace(request, profile=kind, thermo_every=2,
                          pair_encoding='static-groups-zero-cutoff/1',
                          run=replace(request.run, steps=8, sample_every=4))
        if kind == 'cg_network_mixed_sampled_v1':
            request = replace(request, projectile_pairs={
                **request.projectile_pairs,
                1: MorsePair(1.5, 2., 2.5, 2.5, role='repulsive_site'),
            })
    return prepare_cg(g, request)


def typed_raw(p):
    """Build the public typed object independently; never call the parser."""
    contract = json.loads(p.contract_json)
    nodes = {n['id']: n for n in json.loads(p.graph_json)['nodes']}
    points = tuple(tuple(x*contract['length_to_m'] for x in nodes[nid]['position'])
                   for nid in p.node_ids)
    velocities = dict(contract['velocities_m_s'])
    velocity_rows = tuple(tuple(velocities.get(nid, (0., 0., 0.))) for nid in p.node_ids)
    forces = ((7., -8., 9.),)*len(p.node_ids)
    frames = tuple(backend.TrajectoryFrame(step, step*p.timestep_s, p.node_ids,
                                          points, velocity_rows, forces)
                   for step in range(0, p.steps+1, p.sample_every))
    thermo = tuple(backend.ThermoFrame(step, 1., 2., 3., (4., 5., 6., 7., 8., 9.))
                   for step in range(0, p.steps+1, contract.get('thermo_every', 1)))
    return backend.LAMMPSRunResult(p.identity, frames, thermo, '')


def texts(p):
    if p.profile in ('cg_network_sampled_v1', 'cg_network_mixed_sampled_v1'):
        return network_texts(p)
    return harmonic_texts(p)


def edit_dump(p, dump, step, nid, column, value):
    """Change exactly one observed field, preserving reversed native row order."""
    lines = dump.splitlines()
    atom_id = p.node_ids.index(nid)+1
    current_step = None
    for index, line in enumerate(lines):
        if line == 'ITEM: TIMESTEP':
            current_step = int(lines[index+1])
        if line == 'ITEM: ATOMS id type x y z vx vy vz fx fy fz' and current_step == step:
            for row in range(index+1, index+1+len(p.node_ids)):
                fields = lines[row].split()
                if int(fields[0]) == atom_id:
                    fields[column] = repr(value)
                    lines[row] = ' '.join(fields)
                    return '\n'.join(lines)+'\n'
    raise AssertionError('test fixture has no requested atom/step')


def edit_raw(raw, step, nid, attr, value):
    frames = list(raw.trajectory)
    index = next(i for i, frame in enumerate(frames) if frame.step == step)
    frame = frames[index]
    rows = list(getattr(frame, attr))
    row = frame.node_ids.index(nid)
    rows[row] = (value, *rows[row][1:])
    frames[index] = replace(frame, **{attr: tuple(rows)})
    return replace(raw, trajectory=tuple(frames))


@pytest.mark.parametrize('kind', CASES)
@pytest.mark.parametrize('column,value', [(2, 1e-6), (5, 1e-4)])
def test_parser_rejects_observed_held_motion(kind, column, value):
    p = prepared_case(kind)
    held = json.loads(p.contract_json)['fixed_node_ids'][0]
    dump, log = texts(p)
    # For harmonic_sampled_v1, step 1 would not survive output projection.
    bad = edit_dump(p, dump, p.sample_every, held, column, value)
    with pytest.raises(backend.LAMMPSContractError, match='fixed native state changed'):
        backend.parse_outputs(p, bad, log)


@pytest.mark.parametrize('kind', CASES)
@pytest.mark.parametrize('attr,value', [('positions_m', 1e-6), ('velocities_m_s', 1e-4)])
@pytest.mark.parametrize('record_state', [False, True])
def test_converter_independently_rejects_observed_held_motion(kind, attr, value, record_state):
    p = prepared_case(kind)
    held = json.loads(p.contract_json)['fixed_node_ids'][0]
    raw = typed_raw(p)
    bad = edit_raw(raw, p.sample_every, held, attr, value)
    assert bad.identity == p.identity
    assert bad.thermo == raw.thermo
    assert bad.trajectory[0] == raw.trajectory[0]
    with pytest.raises(ValueError, match='native initial/fixed state mismatch'):
        adapter.to_sim_result(p, bad, RUNTIME, 0., record_state)


@pytest.mark.parametrize('kind', CASES)
@pytest.mark.parametrize('record_state', [False, True])
def test_intact_held_state_and_nonzero_native_force_are_valid(kind, record_state):
    p = prepared_case(kind)
    held = json.loads(p.contract_json)['fixed_node_ids'][0]
    dump, log = texts(p)
    for step in range(0, p.steps+1, p.sample_every):
        dump = edit_dump(p, dump, step, held, 8, 7.)
    parsed = backend.parse_outputs(p, dump, log)
    row = p.node_ids.index(held)
    assert all(frame.forces_N[row][0] == 7. for frame in parsed.trajectory)
    for raw in (parsed, typed_raw(p)):
        result = adapter.to_sim_result(p, raw, RUNTIME, 0., record_state)
        assert result.metadata['node_ids'] == list(p.node_ids)
        assert result.metadata['edge_ids'] == list(p.edge_ids)
        assert result.metadata['reactions'] is None
        if record_state:
            assert result.dynamics_state['final']['nodal_forces'][row][0] == 7.
        else:
            assert result.dynamics_state is None


@pytest.mark.parametrize('profile', HARMONIC)
@pytest.mark.parametrize('unit', SCALES)
@pytest.mark.parametrize('held', [10, 40])
def test_harmonic_units_nonzero_origins_and_mobile_motion(profile, unit, held):
    p = harmonic_case(profile, unit, offset_m=.002, fixed=(held,))
    original_graph = p.graph_json
    dump, log = texts(p)
    raw = typed_raw(p)
    mobile = 40 if held == 10 else 10
    row = p.node_ids.index(mobile)
    mobile_x = raw.trajectory[0].positions_m[row][0]+1e-6
    for step in range(1, p.steps+1):
        dump = edit_dump(p, dump, step, mobile, 2, mobile_x)
        dump = edit_dump(p, dump, step, mobile, 5, 1e-4)
        raw = edit_raw(raw, step, mobile, 'positions_m', mobile_x)
        raw = edit_raw(raw, step, mobile, 'velocities_m_s', 1e-4)
    parsed = backend.parse_outputs(p, dump, log)
    for candidate in (parsed, raw):
        result = adapter.to_sim_result(p, candidate, RUNTIME, 0., True)
        np.testing.assert_allclose(result.deformed_positions*SCALES[unit],
                                   candidate.trajectory[-1].positions_m, rtol=1e-14, atol=0.)
        assert result.metadata['node_ids'] == [10, 40]
        assert result.metadata['edge_ids'] == [77]
    assert p.graph_json == original_graph


@pytest.mark.parametrize('profile', HARMONIC)
def test_graph_boundary_flags_do_not_imply_held_atoms(profile):
    p = harmonic_case(profile, fixed=())
    assert any(json.loads(p.graph_json)['nodes'][0]['boundary'])
    dump, log = texts(p)
    dump = edit_dump(p, dump, 1, 10, 2, 1e-6)
    backend.parse_outputs(p, dump, log)
    raw = edit_raw(typed_raw(p), 1, 10, 'positions_m', 1e-6)
    adapter.to_sim_result(p, raw, RUNTIME, 0., False)


@pytest.mark.parametrize('profile', HARMONIC)
def test_existing_parser_exact_and_converter_relative_policies_remain_distinct(profile):
    p = harmonic_case(profile, offset_m=.002, fixed=(40,))
    raw = typed_raw(p)
    x = raw.trajectory[0].positions_m[1][0]
    dump, log = texts(p)
    in_tolerance = x+1e-16
    assert in_tolerance != x
    with pytest.raises(backend.LAMMPSContractError, match='fixed native state changed'):
        backend.parse_outputs(p, edit_dump(p, dump, 1, 40, 2, in_tolerance), log)
    adapter.to_sim_result(p, edit_raw(raw, 1, 40, 'positions_m', in_tolerance), RUNTIME, 0., False)
    with pytest.raises(ValueError, match='native initial/fixed state mismatch'):
        adapter.to_sim_result(p, edit_raw(raw, 1, 40, 'positions_m', x+1e-12), RUNTIME, 0., False)


@pytest.mark.parametrize('profile', HARMONIC)
def test_converter_checks_held_frame_zero_against_prepared_graph(profile):
    p = harmonic_case(profile, offset_m=.002)
    raw = typed_raw(p)
    # Corrupt every observed held position equally: a frame-zero-only anchor
    # would miss this, but the conversion contract is bound to prepared geometry.
    for frame in raw.trajectory:
        raw = edit_raw(raw, frame.step, 10, 'positions_m', .002001)
    with pytest.raises(ValueError, match='native initial/fixed state mismatch'):
        adapter.to_sim_result(p, raw, RUNTIME, 0., True)
