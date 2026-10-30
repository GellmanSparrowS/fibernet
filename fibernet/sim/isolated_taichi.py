"""Opt-in isolated CPU worker; existing simulation dispatch remains unchanged."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
import numpy as np
from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.core.material import Material
from ._isolated_taichi_io import read_progress

MAX_REQUEST = 8 * 1024**2
MAX_RESULT = 64 * 1024**2

class IsolatedTaichiError(RuntimeError):
    """Worker failure, cancellation or bounded protocol failure."""

@dataclass(frozen=True)
class WorkerRuntime:
    python_executable: str
    provider_site: str
    native_sha256: str

    def checked(self):
        python = Path(self.python_executable).resolve(strict=True)
        site = Path(self.provider_site).resolve(strict=True)
        native = site / 'taichi/_lib/core/taichi_python.cpython-312-x86_64-linux-gnu.so'
        if sys.platform != 'linux' or not python.is_file() or not site.is_dir():
            raise ValueError('prototype requires explicit Linux CPython 3.12 CPU runtime')
        with native.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != self.native_sha256:
            raise ValueError('native provider digest mismatch')
        return dict(python_executable=str(python), provider_site=str(site), native_sha256=digest)

def _json_value(x):
    """Pre-serialization finite JSON budget; cycles fail without recursion leaks."""
    active=set();count=0;strings=0
    def visit(v, depth):
        nonlocal count,strings
        count+=1
        if depth>16 or count>200000:raise ValueError('JSON depth/item budget exceeded')
        if v is None or type(v) in (bool,int):return
        if type(v) is str:
            if len(v)>4096:raise ValueError('JSON string budget exceeded')
            strings+=len(v.encode('utf-8'))
            if strings>1024**2:raise ValueError('JSON string budget exceeded')
            return
        if type(v) is float and math.isfinite(v):return
        if type(v) not in (list,dict):raise ValueError('finite built-in JSON values required')
        if id(v) in active:raise ValueError('cyclic metadata is unsupported')
        active.add(id(v))
        try:
            if type(v) is dict:
                for k,value in v.items():
                    if type(k) is not str:raise ValueError('JSON keys must be strings')
                    visit(k,depth+1);visit(value,depth+1)
            else:
                for value in v:visit(value,depth+1)
        finally:active.remove(id(v))
    visit(x,0)

def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def _read_json(path, limit):
    try:
        with Path(path).open('rb') as stream:raw=stream.read(limit+1)
        if len(raw)>limit:raise ValueError('JSON byte bound exceeded')
        value=json.loads(raw)
        if type(value) is not dict:raise ValueError('JSON object required')
        return value
    except (OSError,ValueError,TypeError,RecursionError) as exc:
        raise IsolatedTaichiError('bounded status/protocol decode failed') from exc

def _positive_int(x, name, cap):
    if type(x) is not int or not 1 <= x <= cap:
        raise ValueError(f'{name} must be an integer in [1, {cap}]')
    return x

def _real(x, name, positive=False):
    if type(x) not in (int, float) or not math.isfinite(x) or (x <= 0 if positive else x < 0):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} real')
    return x

def prepare_request(graph, *, fixed_nodes, external_force, spring_k, dt,
                    num_steps, save_interval, dashpot=10.0, drag=1.0):
    """Narrow unit-mass axial CPU domain; preserves native row order exactly.

    fixed_nodes and force rows use graph insertion order, not sorted IDs.
    No callbacks, material constitutive model, state restart, curves or schedules.
    """
    if type(graph) is not StructureGraph:
        raise ValueError('an exact StructureGraph is required')
    _json_value(graph._metadata)
    node_ids = list(graph.nodes); edge_ids = list(graph.edges)
    if not 2 <= len(node_ids) <= 2048 or not 1 <= len(edge_ids) <= 8192:
        raise ValueError('prototype graph bounds exceeded')
    if any(type(i) is not int or i < 0 for i in node_ids + edge_ids):
        raise ValueError('prototype requires nonnegative built-in integer stable IDs')
    if graph.dimension not in (2, 3): raise ValueError('dimension must be 2 or 3')
    for i, node in graph.nodes.items():
        if type(node) is not SNode or type(node.node_id) is not int or node.node_id != i:
            raise ValueError('unsupported node type or record/key identity mismatch')
        if type(node.position) is not np.ndarray or node.position.shape!=(3,) or node.position.dtype != np.dtype('float64') or not np.isfinite(node.position).all():
            raise ValueError('node position must be a finite ordinary float64 ndarray of shape (3,)')
        _json_value(node.metadata)
        if type(node.boundary) not in (tuple,list) or len(node.boundary)!=6 or any(type(b) is not bool for b in node.boundary):
            raise ValueError('boundary flags must be six built-in booleans')
    for i, edge in graph.edges.items():
        if type(edge) is not SEdge or type(edge.edge_id) is not int or edge.edge_id != i or type(edge.material) is not Material:
            raise ValueError('unsupported edge/material type or identity mismatch')
        if set(vars(edge.material))!=set(Material.__dataclass_fields__):
            raise ValueError('material has undeclared fields')
        _json_value(vars(edge.material))
        for name,value in vars(edge.material).items():
            if name not in ('name','extra','stiffness_tensor') and value is not None and type(value) is not float:
                raise ValueError('material scalar fields must be ordinary binary64 floats')
        if type(edge.node_i) is not int or type(edge.node_j) is not int or type(edge.segments) is not int:
            raise ValueError('edge endpoints/segments must be built-in integers')
        _real(edge.radius,'edge radius',True)
        if edge.internal_points is not None and (type(edge.internal_points) is not np.ndarray or edge.internal_points.shape!=(0,3) or edge.internal_points.dtype!=np.dtype('float64')):
            raise ValueError('internal curve points are outside the prototype domain')
        if edge.node_i not in graph.nodes or edge.node_j not in graph.nodes:
            raise ValueError('edge endpoint missing')
        a=np.asarray(graph.nodes[edge.node_i].position); b=np.asarray(graph.nodes[edge.node_j].position)
        if not np.isfinite(a).all() or not np.isfinite(b).all() or np.array_equal(a,b):
            raise ValueError('edge reference coordinates must be finite and distinct')
    payload=graph.to_dict()
    # to_dict sorts records: put them back into solver insertion order BEFORE decode.
    nodes={r['id']:r for r in payload['nodes']}; edges={r['id']:r for r in payload['edges']}
    payload['nodes']=[nodes[i] for i in node_ids]; payload['edges']=[edges[i] for i in edge_ids]
    _json_value(payload)
    restored=StructureGraph.from_dict(payload)
    if list(restored.nodes)!=node_ids or list(restored.edges)!=edge_ids:
        raise ValueError('graph row identity round-trip failed')
    canonical=restored.to_dict()
    canonical_nodes={r['id']:r for r in canonical['nodes']}
    canonical_edges={r['id']:r for r in canonical['edges']}
    canonical['nodes']=[canonical_nodes[i] for i in node_ids]
    canonical['edges']=[canonical_edges[i] for i in edge_ids]
    _json_value(canonical)
    # Strict fixed point, not approximate equality: preserve float spellings,
    # signed zeros, IDs, material values, counters and all metadata exactly.
    # Unsupported canonicalization is rejected in the caller before launch.
    if json.dumps(canonical,allow_nan=False,separators=(',',':')) != json.dumps(payload,allow_nan=False,separators=(',',':')):
        raise ValueError('graph serialization is not an exact fixed point; normalize explicitly before isolation')
    if type(fixed_nodes) is not list or any(type(i) is not int or not 0 <= i < len(node_ids) for i in fixed_nodes) or len(set(fixed_nodes))!=len(fixed_nodes):
        raise ValueError('fixed_nodes must be unique insertion-order row indices')
    _json_value(external_force)
    if type(external_force) is not list or len(external_force)!=len(node_ids) or any(type(row) is not list or len(row)!=3 for row in external_force):
        raise ValueError('external_force must be built-in JSON rows (nodes, 3)')
    for row in external_force:
        for x in row:
            if type(x) not in (int,float) or not math.isfinite(x) or float(x)!=x: raise ValueError('force must be exactly representable finite binary64 data')
    steps=_positive_int(num_steps,'num_steps',10000); interval=_positive_int(save_interval,'save_interval',10000)
    if (steps//interval+2)*len(node_ids)*3*32 > MAX_RESULT//2:
        raise ValueError('trajectory exceeds conservative JSON output forecast')
    controls=dict(fixed_nodes=list(fixed_nodes),external_force=external_force,
                  spring_k=_real(spring_k,'spring_k',True),dt=_real(dt,'dt',True),
                  num_steps=steps,save_interval=interval,dashpot=_real(dashpot,'dashpot'),drag=_real(drag,'drag'))
    request=dict(schema='isolated-taichi-cpu/0',graph=payload,node_ids=node_ids,edge_ids=edge_ids,controls=controls)
    raw=json.dumps(request,allow_nan=False,separators=(',',':')).encode()
    if len(raw)>MAX_REQUEST:raise ValueError('request byte bound exceeded')
    if json.dumps(json.loads(raw),allow_nan=False,separators=(',',':')).encode()!=raw:
        raise ValueError('request JSON is not an exact fixed point')
    return raw

class IsolatedTaichiTask:
    """Owned broker lease; cancel closes the pipe, preserving all evidence.

    Use as a context manager. Broker enforces wall/output limits without poll().
    Parent exit closes the lease; broker then terminates and reaps its worker.
    No global signal handler or native runtime operation is installed in caller.
    """
    def __init__(self, raw, runtime, wall_seconds):
        self.workdir=Path(tempfile.mkdtemp(prefix='fibernet-isolated-taichi-'))
        self._broker=None;self._lease=None;self._log=None;self._pipe=None
        self._request_sha256=hashlib.sha256(raw).hexdigest()
        try:
            (self.workdir/'request.json').write_bytes(raw)
            config=dict(runtime=runtime,wall_seconds=wall_seconds,request_sha256=self._request_sha256)
            (self.workdir/'config.json').write_text(json.dumps(config))
            for name in ('home','tmp','cache'): (self.workdir/name).mkdir()
            env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',HOME=str(self.workdir/'home'),TMPDIR=str(self.workdir/'tmp'),XDG_CACHE_HOME=str(self.workdir/'cache'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TI_SKIP_VERSION_CHECK='ON',TI_OFFLINE_CACHE='0')
            self._log=(self.workdir/'broker.log').open('xb')
            self._pipe=os.pipe()
            self._lease=self._pipe[1] # retained before a child can exist
            try:
                self._broker=subprocess.Popen([runtime['python_executable'],'-I','-B',str(Path(__file__).with_name('_isolated_taichi_broker.py')),str(self.workdir)],stdin=self._pipe[0],stdout=self._log,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True)
            finally:
                os.close(self._pipe[0])
        except BaseException:
            self.cancel();raise
    @property
    def status(self):
        if self._broker is None:return 'not-started'
        if self._broker.poll() is None:return 'running'
        f=self.workdir/'status.json'
        if not f.is_file():return 'error'
        state=_read_json(f,65536).get('status')
        if state not in ('completed','cancelled','timeout','output-limit','log-limit','worker-error','broker-error','cleanup-error','error'):
            raise IsolatedTaichiError('invalid task status')
        return state
    @property
    def progress(self):
        """Detached historical milestones, or None before the first receipt.

        A receipt is not a current live phase, completion, or cancellation proof.
        Reading it does not import or initialize the native provider.
        """
        try:
            return read_progress(self.workdir/'progress.json',self._request_sha256)
        except FileNotFoundError:
            return None
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as exc:
            raise IsolatedTaichiError('bounded progress decode failed') from exc
    def cancel(self):
        if self._lease is not None:
            os.close(self._lease);self._lease=None
        if self._broker is not None:
            try:self._broker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                raise IsolatedTaichiError('broker did not acknowledge cancellation; evidence retained')
        if self._log is not None:self._log.close();self._log=None
    def result(self, timeout=None):
        try:
            self._broker.wait(timeout=timeout)
            if self.status!='completed':raise IsolatedTaichiError(f'{self.status}: {self.workdir}')
            path=self.workdir/'result.json'
            if path.stat().st_size>MAX_RESULT:raise IsolatedTaichiError('result byte bound exceeded')
            receipt=_read_json(self.workdir/'status.json',65536)
            if _sha(path)!=receipt.get('result_sha256'):
                raise IsolatedTaichiError('result integrity mismatch')
            # Public class identity and its native validators remain unchanged.
            # This can import Taichi in parent, but never initializes/resets it.
            from fibernet.sim.accelerated import SimResult
            try:result=SimResult.load(path)
            except (OSError,ValueError,TypeError,KeyError,RecursionError) as exc:
                raise IsolatedTaichiError('native result decoding failed') from exc
            req=_read_json(self.workdir/'request.json',MAX_REQUEST)
            if result.metadata.get('node_ids')!=req['node_ids'] or result.metadata.get('edge_ids')!=req['edge_ids']:
                raise IsolatedTaichiError('returned row identity mismatch')
            return result
        except (OSError,ValueError,TypeError,KeyError,RecursionError) as exc:
            self.cancel();raise IsolatedTaichiError('bounded result/protocol decoding failed') from exc
        except BaseException:
            self.cancel();raise
        finally:
            if self._broker.poll() is not None:self.cancel()
    def __enter__(self):return self
    def __exit__(self,*exc):self.cancel()

def start_isolated_taichi(graph, *, runtime, wall_seconds=60, **controls):
    """Start an isolated CPU task without changing fn.simulate or engine defaults."""
    raw=prepare_request(graph,**controls) # Reject unsupported kwargs before launch.
    if type(runtime) is not WorkerRuntime:raise ValueError('WorkerRuntime required')
    config=runtime.checked();_positive_int(wall_seconds,'wall_seconds',120)
    return IsolatedTaichiTask(raw,config,wall_seconds)
