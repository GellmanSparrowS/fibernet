"""Synthetic runner/NPZ/UI contract tests; no numerical solve or user cache."""
import ast
import copy
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FiberScope'))
import fslab.simcache as sc
from fslab.structure import StructureFactory
from fibernet.core.structure_graph import StructureGraph


def graph():
    g = StructureGraph(dimension=2)
    for p in ((0, 0), (2, 1), (4, 0), (6, 2)):
        g.add_node(p, merge=False)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        g.add_edge(a, b)
    return g


class Factory:
    def __init__(self, g): self.g, self.builds = g, 0
    def build(self): self.builds += 1; return copy.deepcopy(self.g)
    def key(self): return 'same-config'


def result(g):
    xy = g.node_positions()[:, :2]
    edges = g.edge_array()
    return sc.StretchRun(xy[None].copy(), edges, np.ones(len(edges)),
        np.zeros((1, len(edges))), np.ones(1), np.zeros(1),
        np.array([0]), np.array([len(xy)-1]),
        {k: np.zeros(1) for k in ('axial', 'bend', 'contact', 'kinetic', 'work')},
        np.empty((0, 2), int), [np.empty((0, 2), int)], np.zeros(1),
        {'fixture': 'synthetic, not physics'})


@pytest.fixture
def engine(monkeypatch):
    calls = []
    class FakeEngine:
        def __init__(self, g, cfg): calls.append(g); self.g = g
        def run(self, progress_cb=None): return result(self.g)
    monkeypatch.setattr(sc, 'Engine2', FakeEngine)
    return calls


def tag(g): return sc.cache_tag(Factory(g), sc.RunConfig(), graph=g)


def test_required_graph_and_no_build():
    f = Factory(graph())
    with pytest.raises(TypeError): sc.cache_tag(f, sc.RunConfig())
    sc.cache_tag(f, sc.RunConfig(), graph=f.g)
    assert f.builds == 0


@pytest.mark.parametrize('change', ['xy', 'connectivity', 'order', 'direction', 'edge_id', 'node_id', 'weld'])
def test_consumed_changes_miss(change, tmp_path, engine):
    f = Factory(graph()); cfg = sc.RunConfig()
    first = sc.run_stretch(f, cfg, cache_dir=str(tmp_path))
    old = tmp_path / ('run_' + first.cache_key + '.npz'); data = old.read_bytes()
    g = f.g
    if change == 'xy': g.nodes[0].position[0] = np.nextafter(0., 1.)
    elif change == 'connectivity': g.edges[0].node_j = 2
    elif change == 'order': g._edges = dict(reversed(list(g.edges.items())))
    elif change == 'direction': g.edges[0].node_i, g.edges[0].node_j = 1, 0
    elif change == 'edge_id': g._edges[20] = g._edges.pop(0)
    elif change == 'node_id':
        g._nodes[20] = g._nodes.pop(3); g.edges[2].node_j = 20
    elif change == 'weld': g.metadata['weld_triplets'] = [[0, 1, 3]]
    second = sc.run_stretch(f, cfg, cache_dir=str(tmp_path))
    assert not first.cache_hit and not second.cache_hit
    assert first.cache_key != second.cache_key
    assert len(engine) == 2 and f.builds == 2 and old.read_bytes() == data


def test_hit_sentinel_runtime_fields_and_old_namespace(tmp_path, engine, monkeypatch):
    f = Factory(graph()); cfg = sc.RunConfig()
    raw = json.dumps({'v': sc.ENGINE_VERSION, 'h': sc.ENGINE_SRC_HASH,
        's': f.key(), 'c': asdict(cfg), 'mf': sc.MAX_FRAMES}, sort_keys=True, default=str)
    oldkey = hashlib.sha1(raw.encode()).hexdigest()[:16]
    old = tmp_path / ('run_' + oldkey + '.npz')
    sc._save_cache(str(old), result(f.g)); oldbytes = old.read_bytes()
    first = sc.run_stretch(f, cfg, cache_dir=str(tmp_path))
    assert len(engine) == 1 and first.cache_key.startswith('g1_')
    def sentinel(*a, **k): raise RuntimeError('before physics')
    monkeypatch.setattr(sc, 'Engine2', sentinel)
    cached = sc.run_stretch(f, cfg, cache_dir=str(tmp_path))
    assert cached.cache_hit and cached.cache_key == first.cache_key and f.builds == 2
    assert np.array_equal(cached.frames_xy, first.frames_xy)
    with np.load(tmp_path / ('run_' + first.cache_key + '.npz'), allow_pickle=True) as z:
        assert 'cache_hit' not in z and 'cache_key' not in z
    with pytest.raises(RuntimeError, match='before physics'):
        sc.run_stretch(f, cfg, cache_dir=str(tmp_path), use_cache=False)
    f.g.nodes[0].position[0] += .375
    with pytest.raises(RuntimeError, match='before physics'):
        sc.run_stretch(f, cfg, cache_dir=str(tmp_path))
    assert old.read_bytes() == oldbytes


def test_unreadable_is_miss(tmp_path, engine):
    f = Factory(graph()); key = tag(f.g)
    (tmp_path / ('run_' + key + '.npz')).write_bytes(b'not npz')
    run = sc.run_stretch(f, cache_dir=str(tmp_path))
    assert not run.cache_hit and run.cache_key == key and len(engine) == 1


def test_no_cache_does_not_require_factory_key(engine):
    f = SimpleNamespace(build=graph)
    run = sc.run_stretch(f, use_cache=False)
    assert not run.cache_hit and run.cache_key == '' and len(engine) == 1


def test_post_weld_graph_built_once(tmp_path, engine, monkeypatch):
    from fslab.welding import IntersectionWelder
    f = Factory(graph()); seen = []
    def weld(self, g):
        seen.append(g); g.nodes[0].position[0] += .5; return g
    monkeypatch.setattr(IntersectionWelder, 'apply', weld)
    cfg = sc.RunConfig(weld_intersections=True)
    run = sc.run_stretch(f, cfg, cache_dir=str(tmp_path))
    assert f.builds == 1 and len(seen) == 1
    assert run.cache_key == sc.cache_tag(f, cfg, graph=engine[0])
    assert run.frames_xy[0, 0, 0] == .5


def test_numeric_canonicalization_and_unused_fields():
    a = graph(); b = copy.deepcopy(a)
    for n in b.nodes.values(): n.position = n.position.astype('>f8'); n.position[2] = 99
    for e in b.edges.values(): e.radius = 123; e.internal_points = np.empty((0, 3))
    b._nodes = dict(reversed(list(b.nodes.items())))
    assert tag(a) == tag(b)
    b.metadata['weld_triplets'] = np.empty((0, 3), dtype='>i8')
    assert tag(a) == tag(b)


@pytest.mark.parametrize('change', ['nonplanar', 'internal_points'])
def test_unsupported_geometry_cannot_reuse_a_planar_cache(change, tmp_path, engine):
    factory = Factory(graph())
    initial = sc.run_stretch(factory, cache_dir=str(tmp_path))
    assert not initial.cache_hit and len(engine) == 1
    if change == 'nonplanar':
        factory.g.nodes[0].position[2] = 1.
    else:
        factory.g.edges[0].internal_points = np.array([[1., 2., 0.]])
    with pytest.raises(ValueError, match='XY plane|actual graph nodes'):
        sc.run_stretch(factory, cache_dir=str(tmp_path))
    assert len(engine) == 1


@pytest.mark.parametrize('bad', ['bool_id', 'float_id', 'overflow_id', 'float_endpoint',
    'missing_endpoint', 'nan', 'inf', 'masked_xy', 'masked_trip', 'float_trip', 'bool_trip',
    'missing_trip', 'bad_trip_shape', 'nested_masked_trip'])
def test_invalid_inputs_rejected(bad):
    g = graph()
    if bad == 'bool_id': g._nodes = {False: g.nodes[0], **{k:v for k,v in g.nodes.items() if k}}
    elif bad == 'float_id': g._edges[1.5] = g._edges.pop(0)
    elif bad == 'overflow_id': g._edges[2**64] = g._edges.pop(0)
    elif bad == 'float_endpoint': g.edges[0].node_j = 1.5
    elif bad == 'missing_endpoint': g.edges[0].node_j = 99
    elif bad in ('nan', 'inf'): g.nodes[0].position[0] = float(bad)
    elif bad == 'masked_xy': g.nodes[0].position = np.ma.array([0., 0., 0.], mask=[1,0,0])
    elif bad == 'nested_masked_trip': g.metadata['weld_triplets'] = [np.ma.array([0,1,2], mask=[0,1,0])]
    elif bad == 'masked_trip': g.metadata['weld_triplets'] = np.ma.array([[0,1,2]], mask=True)
    elif bad == 'float_trip': g.metadata['weld_triplets'] = [[0,1.,2]]
    elif bad == 'bool_trip': g.metadata['weld_triplets'] = [[False,1,2]]
    elif bad == 'missing_trip': g.metadata['weld_triplets'] = [[0,1,99]]
    elif bad == 'bad_trip_shape': g.metadata['weld_triplets'] = [[0,1]]
    with pytest.raises(ValueError): tag(g)


@pytest.mark.parametrize('status', [True, False, 'failure'])
def test_worker_uses_actual_status(status):
    # Execute the production worker method with signal stubs, no Qt installation
    # or module-level default-cache probing required.
    tree = ast.parse((ROOT/'FiberScope/studio/sim_tab.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimWorker')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    emitted = []; failed = []
    def runner(*args, **kwargs):
        if status == 'failure': raise RuntimeError('sentinel')
        return SimpleNamespace(cache_hit=status)
    namespace = dict(CACHE_DIR='unused', run_stretch=runner,
                     compute_percolation=lambda *a, **k: 'perc')
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<SimWorker.run>', 'exec'), namespace)
    worker = SimpleNamespace(factory=None, cfg=None, quantile=.5,
        finished_ok=SimpleNamespace(emit=lambda *a: emitted.append(a)),
        failed=SimpleNamespace(emit=failed.append), progress=SimpleNamespace(emit=lambda x: None))
    namespace['run'](worker)
    if status == 'failure': assert not emitted and failed == ['RuntimeError: sentinel']
    else: assert not failed and emitted[0][2] is status


def test_actual_legacy_generator_change_rejects_saved_xy(tmp_path, monkeypatch):
    pattern = sc.ensure_fibernet()
    f = StructureFactory(unit='square', grid_x=1, grid_y=1,
                         n_pts_per_side=0, topology='legacy')
    cfg = sc.RunConfig(weld_intersections=False)
    original = pattern.pattern_2d
    builds = []
    def counted(*a, **k):
        g = original(*a, **k); builds.append(g); return g
    monkeypatch.setattr(pattern, 'pattern_2d', counted)
    g = f.build(); key = sc.cache_tag(f, cfg, graph=g)
    path = tmp_path / ('run_' + key + '.npz')
    sc._save_cache(str(path), result(g)); original_bytes = path.read_bytes()
    def sentinel(*a, **k): raise RuntimeError('before physics')
    monkeypatch.setattr(sc, 'Engine2', sentinel)
    builds.clear()
    hit = sc.run_stretch(f, cfg, cache_dir=str(tmp_path))
    assert hit.cache_hit and hit.cache_key == key and len(builds) == 1
    def changed(*a, **k):
        g = counted(*a, **k)
        next(iter(g.nodes.values())).position[0] += .375
        return g
    monkeypatch.setattr(pattern, 'pattern_2d', changed)
    builds.clear()
    with pytest.raises(RuntimeError, match='before physics'):
        sc.run_stretch(f, cfg, cache_dir=str(tmp_path))
    assert len(builds) == 1 and path.read_bytes() == original_bytes
    assert sc.cache_tag(f, cfg, graph=builds[0]) != key


def test_write_failure_is_fresh_status(tmp_path, engine, monkeypatch):
    def fail(*a): raise OSError('write unavailable')
    monkeypatch.setattr(sc, 'atomic_replace', fail)
    run = sc.run_stretch(Factory(graph()), cache_dir=str(tmp_path))
    assert not run.cache_hit and run.cache_key.startswith('g1_')
    assert not list(tmp_path.glob('*.npz'))
