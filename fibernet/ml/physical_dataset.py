"""Callback-driven physical datasets with atomic, resumable checkpoints.

The core owns selection and checkpoint semantics; graph/label adapters are
supplied by a desktop APP or by a pure-library workflow.
"""
import json
import os
import tempfile
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from .physical_learning import FEATURE_SCHEMA, select_candidate
from .physical_surrogate import light_features
from fibernet.gen.manufacturing import PlanarManufacturingConfig, manufacturable_graph
from fibernet.sim.reduced_beam import ReducedBeamConfig, ReducedBeamSolver


SIM_KW = dict(target_stretch=1.8, num_steps=1500, save_interval=500,
              n_increments=24)


class DatasetStream:
    def __init__(self, config, path=None, model=None, pool_size=24,
                 factory_builder=None, feature_fn=None, sample_fn=None,
                 spec_fn=None):
        self.config, self.path, self.model = config, path, model
        if not all(callable(fn) for fn in
                   (factory_builder, feature_fn, sample_fn, spec_fn)):
            raise TypeError('factory_builder, feature_fn, sample_fn and spec_fn must be callable')
        self.factory_builder = factory_builder
        self.feature_fn = feature_fn
        self.sample_fn = sample_fn
        self.spec_fn = spec_fn
        self.pool_size = max(1, min(int(pool_size), 128))
        self.X, self.Y, self.specs = [], [], []
        self.next_seed = int(config['seed0'])
        self.failures = []
        if config['mode'] not in ('generate', 'physics', 'surrogate'):
            raise ValueError('unknown generation mode')
        if config['mode'] == 'surrogate' and model is None:
            raise ValueError('train a model before requesting predictions')
        # Predictions depend on model state: do not silently resume another model.
        if config['mode'] == 'surrogate' and path and os.path.exists(path):
            raise ValueError('use a new prediction dataset path for each model')
        if path and os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['meta']))
                if meta.get('config') != config:
                    raise ValueError('dataset configuration/schema mismatch; use a new cache path')
                self.X, self.Y = list(z['X']), list(z['Y'])
                self.specs = json.loads(str(z['specs']))
                self.next_seed = int(meta['next_seed'])
                self.failures = meta.get('failures', [])
                if len(self.X) != len(self.Y) or len(self.X) != len(self.specs):
                    raise ValueError('incomplete dataset checkpoint')

    def _save(self):
        if not self.path:
            return
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='dataset_', suffix='.npz', dir=folder)
        os.close(fd)
        try:
            np.savez_compressed(tmp, X=self.arrays()[0], Y=self.arrays()[1],
                                specs=json.dumps(self.specs),
                                meta=json.dumps(dict(config=self.config, next_seed=self.next_seed,
                                                     failures=self.failures[-100:])))
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def arrays(self):
        return np.asarray(self.X, float).reshape(-1, 14), np.asarray(self.Y, float).reshape(-1, 3)

    def generate(self, n, progress_cb=None, stop_cb=None):
        n = int(n)
        if not 1 <= n <= 2000:
            raise ValueError('dataset size must be 1..2000')
        cfg = self.config
        attempts = 0
        while len(self.X) < n:
            if stop_cb and stop_cb():
                break
            candidates, features = [], []
            count = self.pool_size if cfg['acquisition'] != 'random' else 1
            for _ in range(count):
                seed = self.next_seed
                self.next_seed += 1
                rng = np.random.default_rng(seed)
                factory = self.factory_builder(seed, cfg, rng)
                try:
                    graph = factory.build()
                    x = self.feature_fn(graph.node_positions(), graph.edge_array())
                    if not np.isfinite(x).all():
                        raise ValueError('non-finite features')
                    candidates.append(factory)
                    features.append(x)
                except (ValueError, MemoryError) as exc:
                    self.failures.append(dict(seed=seed, error=str(exc)))
                if stop_cb and stop_cb():
                    break
            if not candidates:
                self._save()
                raise ValueError('no valid candidate structures')
            if stop_cb and stop_cb():
                self._save()
                break
            index = select_candidate(features, *self.arrays(), cfg['acquisition'], np.random.default_rng(self.next_seed))
            factory, x = candidates[index], features[index]
            try:
                if cfg['mode'] == 'physics':
                    x, y = self.sample_fn(factory)
                elif cfg['mode'] == 'surrogate':
                    y = self.model.predict(x)
                else:
                    y = np.full(3, np.nan)
                self.X.append(x)
                self.Y.append(y)
                self.specs.append(self.spec_fn(factory))
            except (ValueError, FloatingPointError) as exc:
                self.failures.append(dict(seed=factory.seed, error=str(exc)))
            self._save()
            attempts += 1
            if attempts > n * 3:
                raise ValueError('too many rejected structures; checkpoint retained')
            if progress_cb:
                progress_cb(len(self.X), n)
        return self.arrays()


@dataclass
class _PlanarCandidate:
    unit: str
    grid_x: int
    grid_y: int
    n_pts_per_side: int
    seed: int
    perturbation: float
    line_displacements: list
    max_nodes: int
    max_edges: int

    def build(self):
        graph = manufacturable_graph(PlanarManufacturingConfig(
            unit=self.unit, grid_x=self.grid_x, grid_y=self.grid_y,
            n_pts_per_side=self.n_pts_per_side, seed=self.seed,
            perturbation=self.perturbation,
            line_displacements=self.line_displacements))
        if graph.num_nodes > self.max_nodes or graph.num_edges > self.max_edges:
            raise MemoryError('physical dataset graph exceeds node/edge budget')
        return graph


class PlanarPhysicalDataset(DatasetStream):
    """Generate bounded, resumable reduced-solver labels without FiberScope."""

    def __init__(self, unit='hexagon', amp=0.2, pert=0.1, seed0=0,
                 path=None, mode='physics', acquisition='random', model=None,
                 grid=3, pool_size=24, max_nodes=3000, max_edges=5000):
        grid = int(grid)
        max_nodes, max_edges = int(max_nodes), int(max_edges)
        if (grid < 1 or grid > 8 or max_nodes < 1 or max_edges < 1 or
                not np.isfinite(amp) or not 0 <= amp <= 0.5 or
                not np.isfinite(pert) or not 0 <= pert <= 0.5):
            raise ValueError('invalid planar dataset geometry or budget')
        sources = (Path(__file__),
                   Path(__file__).resolve().parents[1] / 'sim' / 'reduced_beam.py',
                   Path(__file__).resolve().parents[1] / 'gen' / 'manufacturing.py')
        digest = hashlib.sha256()
        for source in sources:
            digest.update(source.read_bytes())
        config = dict(schema=FEATURE_SCHEMA, topology='topnet26',
                      unit=unit, amp=float(amp), pert=float(pert),
                      seed0=int(seed0), mode=mode, acquisition=acquisition,
                      simulation=SIM_KW, engine_source=digest.hexdigest()[:16],
                      grid=grid, max_nodes=max_nodes, max_edges=max_edges)
        super().__init__(config, path, model, pool_size,
                         factory_builder=self._factory,
                         feature_fn=light_features,
                         sample_fn=self._sample,
                         spec_fn=asdict)

    @staticmethod
    def _factory(seed, cfg, rng):
        return _PlanarCandidate(
            unit=cfg['unit'], grid_x=cfg['grid'], grid_y=cfg['grid'],
            n_pts_per_side=5, seed=seed,
            line_displacements=rng.uniform(-cfg['amp'], cfg['amp'],
                                           (5, 2)).tolist(),
            perturbation=cfg['pert'] * rng.uniform(),
            max_nodes=cfg['max_nodes'], max_edges=cfg['max_edges'])

    @staticmethod
    def _sample(factory):
        graph = factory.build()
        x = light_features(graph.node_positions(), graph.edge_array())
        run = ReducedBeamSolver(graph, ReducedBeamConfig(**SIM_KW)).run()
        force = np.asarray(run.force_curve, float)
        stretch = np.asarray(run.strain_levels, float)
        low, high = float(stretch.min()), float(stretch.max())
        mask = ((stretch >= low + 0.05 * (high - low)) &
                (stretch <= low + 0.40 * (high - low)))
        if int(mask.sum()) < 3:
            mask = np.ones(len(stretch), bool)
            mask[0] = False
        stiffness = float(np.polyfit(stretch[mask], force[mask], 1)[0])
        work = float((0.5 * (force[1:] + force[:-1]) *
                      np.diff(stretch)).sum())
        return x, np.array([float(force.max()), stiffness, work],
                           dtype=np.float64)
