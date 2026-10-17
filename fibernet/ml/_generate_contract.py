"""Private exact-run contract for FiberNetDataset.generate, not a cache framework."""
import hashlib
import importlib.metadata
import json
import math
import platform
from numbers import Real, Integral
from pathlib import Path
import numpy as np

SCHEMA = 'fibernet.generate-checkpoint/1'
SAMPLING = 'cartesian-radius/2'
SOURCE_FILES = ('ml/dataset.py', 'ml/_generate_contract.py', 'gen/pattern.py',
                'core/tiling.py', 'core/transforms.py', 'core/structure_graph.py',
                'core/material.py', 'sim/accelerated.py')
BUILTINS = ('square', 'triangle', 'hexagon', 'honeycomb', 'kagome', 'reentrant',
            'chiral', 'star', 'cross', 'missing_rib', 'diamond', 'voronoi')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def real(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be finite numeric')
    out = float(value)
    if not math.isfinite(out) or (positive and out <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return out


def sequence(value, default, name):
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'{name} must be a list or tuple')
    return list(value)


def make_plan(*, units, grid_range, radius_range, n_internal_range, box,
              target_stretch, stiffness, damping, num_steps, max_samples,
              seed, checkpoint_every):
    units = sequence(units, ['honeycomb','square','triangle'], 'units')
    if any(not isinstance(u,str) or not u.strip() for u in units):
        raise ValueError('units must contain nonempty strings')
    grids = sequence(grid_range, [(3,3),(5,5)], 'grid_range')
    normalized_grids = []
    for g in grids:
        if not isinstance(g,(list,tuple)) or len(g)!=2:
            raise ValueError('grid_range entries must be pairs')
        normalized_grids.append([integer(v,'grid',1) for v in g])
    radii = [real(v,'radius',True) for v in sequence(radius_range,[.05,.1,.2],'radius_range')]
    internal = [integer(v,'n_internal') for v in sequence(n_internal_range,[0,2,4],'n_internal_range')]
    if not isinstance(box,(list,tuple)) or len(box)!=2:
        raise ValueError('box must be a pair')
    box = [real(v,'box',True) for v in box]
    seed = integer(seed,'seed')
    if seed > 2**32-1:
        raise ValueError('seed must fit RandomState unsigned 32-bit seed')
    cap = None if max_samples is None else integer(max_samples,'max_samples',1)
    integer(checkpoint_every,'checkpoint_every',1)
    solver = dict(target_stretch=real(target_stretch,'target_stretch'),
                  stiffness=real(stiffness,'stiffness'), damping=real(damping,'damping'),
                  num_steps=integer(num_steps,'num_steps',1))
    if solver['stiffness'] < 0:
        raise ValueError('stiffness must be nonnegative')
    solver.update(save_interval=solver['num_steps'],auto_steps=False,
                  ramp_fraction=.2,max_nodes=20000,arch='cpu',num_threads=4)
    declaration = dict(units=units,grid_range=normalized_grids,radius_range=radii,
                       n_internal_range=internal,box=box,seed=seed,max_samples=cap)
    plan = [dict(sampling_version=SAMPLING,unit=u,grid=g,n_internal=n,
                 radius=r,box=box,seed=seed)
            for u in units for g in normalized_grids for n in internal for r in radii]
    ids = [sample_id(p) for p in plan]
    if len(ids)!=len(set(ids)):
        raise ValueError('duplicate or colliding sample specifications')
    if cap is not None:
        np.random.RandomState(seed).shuffle(plan)
        plan = plan[:cap]
    return declaration, solver, plan


def sample_id(params):
    return 'v2-' + digest(params)


def implementation_recipe(pattern, units):
    factories = {}
    for unit in units:
        name = unit.lower()
        actual = pattern._UNIT_FACTORIES.get(name)
        expected = getattr(pattern, '_unit_'+name, None) if name in BUILTINS else None
        if actual is None or actual is not expected:
            raise ValueError('checkpoint generation requires a known built-in factory; '
                             'custom registered factories require save_dir=None')
        factories[unit] = actual.__module__ + '.' + actual.__qualname__
    root = Path(__file__).resolve().parents[1]
    sources = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    versions = {'python':platform.python_version()}
    for name in ('numpy','scipy','networkx','taichi'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    return dict(source_sha256=sources,versions=versions,factories=factories)


def binding(declaration, solver, plan, implementation):
    return dict(sampling_version=SAMPLING,declaration=declaration,ordered_plan=plan,
                solver=solver,implementation=implementation,
                material_policy='built-in default Material; no Material-derived spring stiffness',
                feature_label_policy='dataset.py extraction and result-attribute mapping; source-bound')


def validate_rows(rows, plan):
    if not isinstance(rows,list):
        raise ValueError('checkpoint samples must be a list')
    expected = {sample_id(p):p for p in plan}
    seen = set()
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError('checkpoint sample must be an object')
        sid = row.get('id'); params = row.get('params')
        if not isinstance(sid,str) or sid in seen or sid not in expected or params != expected[sid]:
            raise ValueError('checkpoint sample identity/plan mismatch or duplicate')
        if sid != sample_id(params) or row.get('unit') != params['unit'] or row.get('grid') != params['grid']:
            raise ValueError('checkpoint sample fields contradict its specification')
        for key in ('features','labels'):
            values=row.get(key)
            if not isinstance(values,dict) or any(not isinstance(k,str) for k in values):
                raise ValueError('checkpoint feature/label mappings must have string keys')
            for v in values.values():
                real(v,'checkpoint '+key)
        if not {'max_force','max_stretch','mean_stretch','std_stretch'} <= row['labels'].keys():
            raise ValueError('checkpoint required labels missing')
        if not isinstance(row.get('metadata', {}), dict):
            raise ValueError('checkpoint metadata must be an object')
        canonical(row)
        seen.add(sid)
    if rows and any(set(r['features'])!=set(rows[0]['features']) or
                    set(r['labels'])!=set(rows[0]['labels']) for r in rows):
        raise ValueError('checkpoint feature/label columns differ between samples')
    return seen


def read_checkpoint(path, requested, plan):
    def unique_keys(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate JSON key in generate checkpoint')
            out[key] = value
        return out
    with path.open() as f:
        saved=json.load(f, object_pairs_hook=unique_keys)
    if not isinstance(saved,dict) or saved.get('checkpoint_schema') != SCHEMA:
        raise ValueError('legacy or unknown generate checkpoint; use a new save_dir (no automatic migration)')
    if canonical(saved.get('run_binding')) != canonical(requested) or saved.get('run_binding_sha256') != digest(requested):
        raise ValueError('generate run binding mismatch; use a new save_dir')
    rows=saved.get('samples')
    if not isinstance(rows,list):
        raise ValueError('checkpoint samples must be a list')
    count = integer(saved.get('n_samples'), 'checkpoint n_samples')
    if saved.get('samples_digest') != digest(rows) or count != len(rows):
        raise ValueError('checkpoint sample payload/count mismatch')
    validate_rows(rows,plan)
    return rows
