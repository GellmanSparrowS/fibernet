"""Experimental, uncalibrated linear-static brittle frame fracture.

Explicit SI material parameters and grip IDs are required. This API does not
replace ``simulate`` or the reduced contact solver. Install ``fibernet[fracture]``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Optional, Sequence

import numpy as np

from . import _engine, _sparse

__all__ = ['FractureConfig', 'FractureResult', 'FractureSolver']
SCHEMA = 'fibernet-linear-fracture-result-1'
CHECKPOINT_SCHEMA = 'fibernet-linear-fracture-checkpoint-1'


def _digest(value):
    return hashlib.sha256(_sparse.canonical(value)).hexdigest()


def _sources():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('__init__.py', '_beam.py', '_sparse.py', '_engine.py')}


def _literal_ids(values, name):
    values = list(values)
    if not values or any(type(x) is not int or x < 0 for x in values) or len(set(values)) != len(values):
        raise ValueError(name + ' must contain unique nonnegative integer IDs')
    return values


@dataclass(frozen=True)
class FractureConfig:
    """Uniform circular welded beams. All material inputs use SI units.

    ``maximum_displacement_m`` is an imposed translation, not a stretch ratio.
    ``event_tol`` groups near-tied failure candidates; it is not uncertainty in
    measured strength. Resource checks are cooperative, not an OS hard limit.
    """
    young_modulus_Pa: float
    poisson_ratio: float
    normal_strength_Pa: float
    shear_strength_Pa: float
    axis: int = 1
    maximum_displacement_m: float = 0.001
    reference_displacement_m: float = 1e-6
    event_tol: float = 1e-10
    max_epochs: int = 128
    max_avalanche_solves: int = 128
    max_nodes: int = 1000
    max_edges: int = 2000
    memory_budget_mb: int = 768
    reserve_memory_mb: int = 512
    timeout_seconds: int = 300
    output_quota_bytes: int = 64 * 1024**2

    def __post_init__(self):
        for key in ('young_modulus_Pa', 'normal_strength_Pa', 'shear_strength_Pa',
                    'maximum_displacement_m', 'reference_displacement_m'):
            _sparse.number(getattr(self, key), key, positive=True)
        nu = _sparse.number(self.poisson_ratio, 'poisson_ratio')
        if not -1 < nu < 0.5:
            raise ValueError('poisson_ratio must be in (-1, 0.5)')
        if type(self.axis) is not int or self.axis not in (0, 1, 2):
            raise ValueError('axis must be 0, 1 or 2')
        if type(self.event_tol) is not float or self.event_tol not in (1e-10, 1e-8):
            raise ValueError('event_tol must be 1e-10 or 1e-8 (lightweight-near-tie-v1)')
        bounds = dict(max_epochs=(1,128), max_avalanche_solves=(2,128),
                      max_nodes=(2,6000), max_edges=(1,7000),
                      memory_budget_mb=(128,8192), reserve_memory_mb=(128,1048576),
                      timeout_seconds=(1,3600), output_quota_bytes=(1024,256*1024**2))
        for key, (low, high) in bounds.items():
            v = getattr(self, key)
            if type(v) is not int or not low <= v <= high:
                raise ValueError('%s must be an integer in [%s, %s]' % (key,low,high))
        if self.reference_displacement_m > self.maximum_displacement_m:
            raise ValueError('reference displacement cannot exceed maximum displacement')


class _Guard:
    def __init__(self, config):
        try:
            import psutil
        except ImportError as exc:
            raise ImportError('Install fibernet[fracture] for resource-guard dependencies') from exc
        self.psutil = psutil
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.config = config
        self.start = time.monotonic()

    def check(self, additional_bytes=0):
        c = self.config
        if time.monotonic() - self.start > c.timeout_seconds:
            raise TimeoutError('Fracture cooperative time budget exceeded')
        if (self.process.memory_info().rss + additional_bytes > self.baseline+c.memory_budget_mb*1024**2
                or self.psutil.virtual_memory().available-additional_bytes < c.reserve_memory_mb*1024**2):
            raise MemoryError('Fracture memory budget/reserve exceeded')


@dataclass
class FractureResult:
    """Explicit original IDs, equilibrium event history and scientific scope.

    Free fragment positions/rotations are JSON null. A lost grip-to-grip path
    describes graph connectivity only. No fracture toughness is inferred.
    """
    data: dict

    @property
    def events(self):
        return self.data['events']

    @property
    def observations(self):
        return self.data['observations']

    @property
    def termination_reason(self):
        return self.data['termination_reason']

    def save(self, path):
        """Create a new JSON file; never silently overwrite an existing result."""
        raw = _sparse.canonical(self.data)
        if len(raw) > self.data['config']['output_quota_bytes']:
            raise MemoryError('Result exceeds output quota')
        with Path(path).open('xb') as f:
            f.write(raw)


class FractureSolver:
    """Adapt a StructureGraph without deleting, merging or mutating its edges.

    Only straight, uniform-radius, homogeneous isotropic, welded beams are
    supported. Parallel edges, curved points, hinges and mixed materials fail
    explicitly. Original (possibly sparse) node/edge IDs are retained.
    Config E/nu must match every edge material; strengths are explicit config
    inputs and do not implicitly inherit Material.tensile_strength.
    """
    def __init__(self, graph, config: FractureConfig, *, fixed_nodes: Sequence[int],
                 loaded_nodes: Sequence[int], length_unit: str,
                 initially_broken_edges: Sequence[int] = ()):
        from fibernet.core.structure_graph import StructureGraph
        if not isinstance(graph, StructureGraph) or not isinstance(config, FractureConfig):
            raise TypeError('StructureGraph and FractureConfig required')
        if length_unit not in ('m', 'mm', 'um'):
            raise ValueError('Explicit length_unit must be m, mm or um')
        scale = {'m':1., 'mm':1e-3, 'um':1e-6}[length_unit]
        fixed = _literal_ids(fixed_nodes, 'fixed_nodes')
        loaded = _literal_ids(loaded_nodes, 'loaded_nodes')
        if set(fixed) & set(loaded):
            raise ValueError('Grip node sets must be disjoint')
        order = sorted(graph.nodes)
        if not 2 <= len(order) <= config.max_nodes or not 1 <= len(graph.edges) <= config.max_edges:
            raise ValueError('Graph exceeds declared node/edge budget')
        if any(type(i) is not int or i < 0 for i in order) or not set(fixed+loaded) <= set(order):
            raise ValueError('Invalid node/grip IDs')
        positions = {}
        for i in order:
            n = graph.nodes[i]
            p = np.asarray(n.position, dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all() or any(n.metadata.get(key,'welded') != 'welded' for key in ('joint_type','junction_type')):
                raise ValueError('Finite XYZ positions and welded junctions required')
            positions[i] = (p*scale).tolist()
        edges, seen, radii = [], set(), []
        for sid, e in sorted(graph.edges.items()):
            if type(sid) is not int or sid < 0 or type(e.node_i) is not int or type(e.node_j) is not int or e.node_i not in positions or e.node_j not in positions or e.node_i == e.node_j:
                raise ValueError('Invalid edge ID/endpoints')
            pair = tuple(sorted((e.node_i,e.node_j)))
            if pair in seen:
                raise ValueError('Parallel edges are unsupported; no silent deduplication')
            seen.add(pair)
            a, b = np.asarray(positions[e.node_i]), np.asarray(positions[e.node_j])
            if float(np.linalg.norm(b-a)) <= 0:
                raise ValueError('Zero-length beam')
            if e.internal_points is not None and len(e.internal_points):
                pts = np.asarray(e.internal_points,dtype=float)*scale
                v = b-a
                if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
                    raise ValueError('Invalid internal points')
                t = (pts-a)@v/(v@v)
                if np.any(t<=0) or np.any(t>=1) or np.any(np.diff(t)<=0) or not np.allclose(pts,a+t[:,None]*v,rtol=0,atol=1e-12*max(1.,np.linalg.norm(v))):
                    raise ValueError('Curved/nonordered internal points unsupported; no silent straightening')
            if isinstance(e.radius, (bool, np.bool_)):
                raise ValueError('Numeric radius required, not boolean')
            radius = _sparse.number(float(e.radius)*scale, 'radius',positive=True)
            radii.append(radius)
            if any(e.metadata.get(key, 'welded') != 'welded' for key in ('junction_type', 'joint_type')):
                raise ValueError('Only welded beam joints are supported')
            m = e.material
            if (not math.isclose(float(m.youngs_modulus),config.young_modulus_Pa,rel_tol=1e-12)
                    or not math.isclose(float(m.poissons_ratio),config.poisson_ratio,rel_tol=1e-12,abs_tol=1e-15)
                    or not math.isclose(float(m.shear_modulus),config.young_modulus_Pa/(2*(1+config.poisson_ratio)),rel_tol=1e-12)
                    or m.stiffness_tensor is not None):
                raise ValueError('Every edge must match configured homogeneous isotropic E and nu')
            edges.append((sid,e.node_i,e.node_j))
        if not all(math.isclose(r,radii[0],rel_tol=1e-12,abs_tol=0) for r in radii):
            raise ValueError('Heterogeneous radii unsupported')
        broken = list(initially_broken_edges)
        if any(type(x) is not int or x < 0 for x in broken) or len(set(broken)) != len(broken) or not set(broken) <= set(graph.edges):
            raise ValueError('Invalid initially_broken_edges')
        mask = [sid not in broken for sid,_,_ in edges]
        span = max(p[config.axis] for p in positions.values())-min(p[config.axis] for p in positions.values())
        if span <= 0:
            raise ValueError('Positive loading-axis span required')
        self.geometry = dict(order=order,positions=positions,original_edges=edges,
                             initial_mask=mask,fixed=fixed,loaded=loaded,axis=config.axis)
        topo = _engine.topology(self.geometry, mask)
        if not topo['path_present'] or any(v==0 for v in list(topo['fixed_incident_rods'].values())+list(topo['loaded_incident_rods'].values())):
            raise ValueError('Initial finite material must connect grips and touch every selected grip node')
        self.config = config
        self.parameters = dict(diameter_m=2*radii[0], young_modulus_Pa=config.young_modulus_Pa,
            poisson_ratio=config.poisson_ratio,sigma_Pa=config.normal_strength_Pa,
            tau_Pa=config.shear_strength_Pa, shear_coefficient=6*(1+config.poisson_ratio)/(7+6*config.poisson_ratio),
            characteristic_length_m=span, reference_displacement_m=config.reference_displacement_m,
            maximum_displacement_m=config.maximum_displacement_m,max_avalanche_solves=config.max_avalanche_solves,
            numerics=dict(schema='lightweight-near-tie-v1',event_tol=config.event_tol))
        self.binding = dict(config_sha256=_digest(asdict(config)),geometry_sha256=_digest(self.geometry),
                            sources=_sources(), parameters_sha256=_digest(self.parameters))

    def _checkpoint(self, path, payload):
        envelope = dict(schema=CHECKPOINT_SCHEMA,payload=payload,payload_sha256=_digest(payload))
        raw = _sparse.canonical(envelope)
        if len(raw) > self.config.output_quota_bytes:
            raise MemoryError('Checkpoint exceeds output quota')
        path = Path(path)
        if path.is_symlink():
            raise ValueError('Checkpoint symlinks are unsupported')
        fd, name = tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(path.parent))
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            os.replace(name,path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def run(self, *, checkpoint: Optional[str] = None, resume: bool = False,
            stop_after_epochs: Optional[int] = None) -> FractureResult:
        """Run serially, checkpointing only complete equilibrated epochs.

        Resume verifies config, geometry, source and payload hashes and solves
        the saved equilibrium again. A checkpoint is an integrity aid, not an
        authenticated record against a malicious editor. Epoch-budget exhaustion
        is incomplete, never successful fracture. No finite-contact dynamics run.
        """
        try:
            from threadpoolctl import threadpool_limits
        except ImportError as exc:
            raise ImportError('Install fibernet[fracture] for threadpool control') from exc
        if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
            raise ValueError('stop_after_epochs must be a positive integer')
        if resume and checkpoint is None:
            raise ValueError('resume requires checkpoint')
        if checkpoint and not resume and Path(checkpoint).exists():
            raise FileExistsError('Existing checkpoint: explicitly resume or choose a new path')
        current = dict(config_sha256=_digest(asdict(self.config)),geometry_sha256=_digest(self.geometry),
                       sources=_sources(), parameters_sha256=_digest(self.parameters))
        if current != self.binding:
            raise ValueError('Solver source/config/geometry changed after construction')
        guard = _Guard(self.config)
        guard.check()
        with threadpool_limits(limits=1):
            return self._run(checkpoint,resume,stop_after_epochs,guard)

    def _run(self, checkpoint, resume, stop_after_epochs, guard):
        p, g = self.parameters, self.geometry
        events, observations, epoch = [], [], 0
        def solve(*args):
            guard.check()
            return _engine.equilibrium(*args)
        def observe(phase,state,pending,cascade):
            observations.append(dict(phase=phase,epoch=epoch,cascade=cascade,
                displacement_m=state['displacement_m'],loaded_reaction_force_N=state['loaded_reaction_force_N'],
                strain_energy_J=state['strain_energy_J'],half_constraint_work_J=state['half_constraint_work_J'],
                mask=list(state['mask']),pending_failed_ids=[e['id'] for e in pending],
                pending_failure_members=deepcopy(pending),
                near_tie_added_ids=[e['id'] for e in pending if e['criterion']<1.],
                numerical_grouping_tolerance=p['numerics']['event_tol'],
                path_present=state['topology']['path_present'],solver=deepcopy(state['solver'])))
            if len(_sparse.canonical(observations)) > self.config.output_quota_bytes//2:
                raise MemoryError('Observation history budget exceeded')
        if resume:
            path=Path(checkpoint)
            if path.is_symlink() or path.stat().st_size > self.config.output_quota_bytes:
                raise ValueError('Bounded ordinary checkpoint required')
            envelope=json.loads(path.read_text(encoding='utf-8'))
            saved=envelope['payload']
            if envelope.get('schema')!=CHECKPOINT_SCHEMA or envelope.get('payload_sha256')!=_digest(saved) or saved['binding']!=self.binding:
                raise ValueError('Checkpoint source/config/geometry/payload mismatch')
            epoch=saved['epoch']; events=saved['events']; observations=saved['observations']
            if type(epoch) is not int or not 0 <= epoch <= self.config.max_epochs:
                raise ValueError('Invalid checkpoint epoch')
            state=solve(p,g,saved['state']['mask'],saved['state']['displacement_m'],guard)
            if _digest(state)!=_digest(saved['state']):
                raise ValueError('Checkpoint equilibrium replay mismatch')
            removed={sid for (sid,_,_),initial,active in zip(g['original_edges'],g['initial_mask'],state['mask']) if initial and not active}
            if len(events)!=len(removed) or {e['segment_id'] for e in events}!=removed:
                raise ValueError('Checkpoint event/mask mismatch')
            for e in events:
                q=_engine.stress_envelope(e['local_end_resultants'],p)['criterion']
                if q < 1-p['numerics']['event_tol'] or q!=e['criterion']:
                    raise ValueError('Checkpoint failure criterion mismatch')
        else:
            state=solve(p,g,g['initial_mask'],0.,guard); observe('initial',state,[],None)
        reason='epoch_bound'
        start_epoch=epoch
        while epoch < self.config.max_epochs and state['topology']['path_present']:
            if state['displacement_m']>=p['maximum_displacement_m']:
                reason='loading_bound'; break
            epoch+=1
            state,new,reason=_engine.advance(p,g,state['mask'],state['displacement_m'],guard,solve,observe)
            events.extend(new)
            if checkpoint:
                self._checkpoint(checkpoint,dict(binding=self.binding,epoch=epoch,state=state,events=events,observations=observations))
            if not state['topology']['path_present']:
                reason='material_path_loss'; break
            if not new:
                break
            if stop_after_epochs is not None and epoch-start_epoch>=stop_after_epochs:
                reason='paused'; break
        if not state['topology']['path_present']:
            reason='material_path_loss'
        elif epoch>=self.config.max_epochs and state['displacement_m']<p['maximum_displacement_m']:
            reason='epoch_bound'
        if reason == 'loading_bound_or_roundoff_step' and state['displacement_m'] < p['maximum_displacement_m']:
            reason = 'numerical_stall'
        data=dict(schema=SCHEMA,binding=deepcopy(self.binding),config=asdict(self.config),
            numerics=p['numerics'],epoch=epoch,termination_reason=reason,
            state=state,events=events,observations=observations,
            scope=dict(model='linear-static-Timoshenko-stress-removal',
                material_path_loss_is_spatial_separation=False,contact_verified=False,
                finite_kinematics_verified=False,material_calibrated=False,
                free_fragment_pose='null',energy_scope='elastic strain energy and half constraint work; not Gc'),
            complete=reason not in ('paused','epoch_bound','numerical_stall'))
        if len(_sparse.canonical(data))>self.config.output_quota_bytes:
            raise MemoryError('Result exceeds output quota')
        return FractureResult(data)
