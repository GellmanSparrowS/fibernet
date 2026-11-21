"""Stable-ID adapter for the existing row-keyed schedule, without Taichi."""
import subprocess
import sys

import numpy as np
import pytest

import fibernet
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.nodal_controls import prepare_taichi_schedule_by_id


def graph(reverse=False):
    g = StructureGraph(dimension=3)
    for x in range(4):
        g.add_node([x, 0, 0], merge=False)
    g.add_edge(0, 1, radius=.05)
    g.add_edge(1, 3, radius=.05)
    g.remove_node(2)
    payload = g.to_dict()
    change = dict(zip(g.nodes, [109, 7, 903]))
    for node in payload['nodes']:
        node['id'] = change[node['id']]
    for edge in payload['edges']:
        edge['node_i'] = change[edge['node_i']]
        edge['node_j'] = change[edge['node_j']]
    if reverse:
        payload['nodes'].reverse()
    payload['next_node_id'] = 904
    return StructureGraph.from_dict(payload)


@pytest.mark.parametrize('reverse', [False, True])
def test_schedule_uses_native_rows_and_copies(reverse):
    from fibernet.sim.accelerated import TaichiEngine, _graph_to_arrays, _prepare_displacement_schedule
    g = graph(reverse)
    before = g.to_dict()
    vector = np.array([.125, 0., -.25])
    prepared = prepare_taichi_schedule_by_id(g, {np.int64(903): [(2.5, vector), (0., [0,0,0])], 7: []})
    vector[:] = 99
    _, _, _, rows = _graph_to_arrays(g)
    got = prepared.taichi_kwargs(g)['displacement_schedule']
    expected = _prepare_displacement_schedule({rows[903]: [(0., [0,0,0]), (2.5,[.125,0,-.25])]},3,3)
    assert list(got) == list(expected) == [rows[903]]
    # Preserve the caller's ordering; only the native parser owns the timeline.
    assert [step for step, _ in got[rows[903]]] == [2.5, 0.]
    waypoints = _prepare_displacement_schedule(got, 3, 3)[rows[903]]
    for (_,actual),(_,wanted) in zip(waypoints,expected[rows[903]]):
        np.testing.assert_array_equal(actual,wanted)
    np.testing.assert_allclose(TaichiEngine._interpolate_schedule(waypoints, 1, 2), [.05,0,-.1],
                               atol=2e-17, rtol=2e-15)
    got[rows[903]][0][1][:] = 11
    got.clear()
    np.testing.assert_array_equal(prepared.taichi_kwargs(g)['displacement_schedule'][rows[903]][0][1],
                                  [.125,0,-.25])
    assert g.to_dict() == before


def test_real_wrapper_forwards_schedule_and_fixed_rows_unchanged(monkeypatch):
    from fibernet import easy
    g = graph()
    prepared = prepare_taichi_schedule_by_id(g, {109: [(1.5, (.25,0.,0.))]})
    kw = prepared.taichi_kwargs(g)
    fixed = [next(iter(kw['displacement_schedule']))]
    token = object()
    class Capture:
        def __init__(self, **runtime):
            assert runtime == {'arch':'cpu','num_threads':1}
        def dynamics(self, actual_graph, **controls):
            assert actual_graph is g
            assert controls['fixed_nodes'] is fixed
            assert controls['displacement_schedule'] is kw['displacement_schedule']
            return token
    monkeypatch.setitem(vars(easy), 'TaichiEngine', Capture)
    assert fibernet.simulate(g, mode='dynamics', backend='taichi',
                             num_threads=1, fixed_nodes=fixed, **kw) is token


@pytest.mark.parametrize('key', [0, -1, True, np.bool_(False), np.timedelta64(109, 'ns'), 109., '109', 999])
def test_bad_schedule_ids_are_not_fallback_rows(key):
    with pytest.raises(ValueError, match='schedule_by_id key'):
        prepare_taichi_schedule_by_id(graph(), {key: []})


@pytest.mark.parametrize('waypoints', [
    None, [(0,)], [(0,[0,0])], [(0,[0,0,0,0])], [(0,'000')], [('0',[0,0,0])],
    [(.5,[0,np.ma.masked,0])], [(.5,np.ma.array([0,1,0],mask=[0,1,0]))],
    [(.5,np.ma.array([0,1,0],mask=False))], [(True,[0,0,0])], [(.5,[True,1.,0])],
    [(np.nan,[0,0,0])], [(0,[0,np.inf,0])], [(2**53+1,[0,0,0])], [(0,[0,2**53+1,0])],
    [(1., {1., 2., 3.})], [{1., (2., 3., 4.)}], {(1., (2., 3., 4.))},
    [(1., {'x': 1., 'y': 2., 'z': 3.})], iter([(1., [2., 3., 4.])]),
    [(1., iter([2., 3., 4.]))], [(1., np.zeros((1, 3)))], np.empty((0, 3)),
    [(np.timedelta64(7, 'ns'), [0.,0.,0.])],
    [(1., [0., np.timedelta64(7, 'ns'), 0.])],
    [(1., np.array([1,2,3], dtype='timedelta64[ns]'))],
    [(np.datetime64('2026-01-01'), [0.,0.,0.])],
])
def test_rejects_invalid_waypoints(waypoints):
    with pytest.raises(ValueError):
        prepare_taichi_schedule_by_id(graph(), {109: waypoints})


@pytest.mark.parametrize('style', ['list', 'tuple', 'array'])
def test_ordered_container_forms_are_equivalent(style):
    if style == 'list':
        waypoints = [[.5, [2., -3., 4.]], [1.25, [1., 0., .5]]]
    elif style == 'tuple':
        waypoints = ((.5, (2., -3., 4.)), (1.25, (1., 0., .5)))
    else:
        waypoints = np.empty((2, 2), dtype=object)
        waypoints[:, 0] = [np.float32(.5), np.float64(1.25)]
        waypoints[0, 1] = np.array([2, -3, 4], dtype=np.int32)
        waypoints[1, 1] = np.array([1., 0., .5])
    g = graph()
    kw = prepare_taichi_schedule_by_id(g, {109: waypoints}).taichi_kwargs(g)
    steps, vectors = zip(*kw['displacement_schedule'][0])
    assert steps == (.5, 1.25)
    np.testing.assert_array_equal(vectors, [[2., -3., 4.], [1., 0., .5]])


@pytest.mark.parametrize('waypoints,error', [
    ([(-.5, [0,0,0])], 'nonnegative'),
    ([(1.5, [0,0,0]), (1.5, [.1,0,0])], 'unique'),
])
def test_native_parser_still_owns_timeline_validation(waypoints, error):
    from fibernet.sim.accelerated import _prepare_displacement_schedule
    g = graph()
    kw = prepare_taichi_schedule_by_id(g, {109: waypoints}).taichi_kwargs(g)
    with pytest.raises(ValueError, match=error):
        _prepare_displacement_schedule(kw['displacement_schedule'], len(g.nodes), 3)


def test_preparation_without_touching_optional_accelerated_module():
    code = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name in ('taichi', 'fibernet.sim.accelerated'):
            raise AssertionError(name)
sys.meta_path.insert(0, Block())
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.nodal_controls import prepare_taichi_schedule_by_id
g = StructureGraph(dimension=3)
nid = g.add_node([0,0,0])
assert prepare_taichi_schedule_by_id(g, {nid: [(1.5, [0,0,0])]}).taichi_kwargs(g)
"""
    subprocess.run([sys.executable, '-B', '-c', code], check=True, timeout=20)


def test_graph_edits_and_reordering_require_preparing_again():
    g = graph()
    p = prepare_taichi_schedule_by_id(g, {109: [(1., [0,0,0])]})
    with pytest.raises(ValueError, match='graph changed'):
        p.taichi_kwargs(graph(reverse=True))
    g.nodes[109].position[0] += 1e-8
    with pytest.raises(ValueError, match='graph changed'):
        p.taichi_kwargs(g)
