# Opt-in isolated Taichi CPU backend

`fibernet.sim.isolated_taichi.start_isolated_taichi` launches the existing FiberNet axial dynamics API in a fresh worker and returns an owned task. `task.result()` loads the existing public `SimResult`; `task.cancel()` closes the broker lease. A context manager cancels/reaps outstanding work on exit. Work directories and logs remain available for diagnosis. Existing `simulate` routing/defaults remain unchanged.

## Initial supported domain

Linux CPython 3.12 with waitid/WNOWAIT, explicitly identified Taichi 1.7.4 CPU runtime/native SHA256, one CPU/thread. An exact `StructureGraph`, 2D/3D positions stored as three coordinates, 2–2048 nodes and 1–8192 edges, nonnegative integer stable IDs, straight noncoincident edges, finite built-in JSON metadata, ordinary scalar `Material` records. Parallel physical edges remain distinct. Material records are preserved for graph interoperability; they do not determine spring coefficients or masses in this unit-mass model. Positions must be ordinary finite float64 arrays, including three coordinates for planar graphs. Material numeric scalars must be ordinary floats; optional None values remain allowed only when the graph roundtrip preserves them exactly. The standard StructureGraph and default Material constructors are the intended input path. Integer position arrays or material scalars must be normalized explicitly before calling; the bridge never silently changes them.

The caller supplies `fixed_nodes`, a three-component `external_force` row per node, positive `spring_k`, positive `dt`, `num_steps` and `save_interval`; optional nonnegative `dashpot` and `drag`. `spring_k` is force-like (EA), not EA/L. Mass is the native default unit mass; this interface does not infer physical units or expose a calibrated material model. Steps are bounded to 10,000. Request JSON is capped at 8 MiB, result at 64 MiB, trajectory forecast is checked before launch, and the worker has a 2 GiB address-space limit, one CPU and bounded CPU/wall/file/output limits.

Unsupported controls are rejected before launch: observer/callback, custom Python objects or graph subclasses, signed IDs, curves, displacement schedules, restart state, material constitutive models, custom mass/velocity/rest-length maps, energy ledger, GPU selection. No callback is silently discarded. These are admission restrictions of this initial interface, not changes to the native solver capabilities.

The selected executable is not explicitly version-probed before launch. Callers must supply the qualified CPython 3.12 interpreter; the native-library filename and digest checks are not a Python-version preflight. Other interpreter versions are outside this profile.

## Identity and serialization

`StructureGraph.to_dict()` sorts records. The bridge restores each list to the original dictionary insertion order before existing `StructureGraph.from_dict()` reconstruction; both sides check exact node and edge ID order. Fixed-node indices and force rows therefore retain their original meaning. Result metadata IDs must match those same ordered IDs. No new graph type or physics implementation is introduced.

The worker calls `fn.simulate(..., mode='dynamics', backend='taichi', arch='cpu', num_threads=1)`, then `SimResult.save(detailed=True)`. The caller uses `SimResult.load()` and its existing validation. This export is not a restart checkpoint. Timing fields describe worker simulation, not total startup/IPC latency.

## Ownership and arithmetic boundary

Only the worker constructs/initializes Taichi. The caller never initializes/resets it or changes FP mode. Loading `SimResult` may import Taichi through its existing module; import-only arithmetic neutrality is evidence for the qualified test environment, not a universal provider-neutral-import promise.

A stdlib broker watches an inherited dedicated pipe lease. Do not fork the caller while a task is active: forked descendants can retain the lease writer and delay EOF. Parent cancellation or normal process termination (without inherited writers) closes that lease; the broker kills/reaps the native worker's group and preserves status. The native worker never inherits the lease descriptor. The broker also enforces wall/output limits without caller polling. No library-global signal handlers are changed. Broker destruction by uncatchable SIGKILL, host failure, process-group escape and blocked OS teardown are outside this guarantee. Aggregate output checks can overshoot within a polling interval; address-space and file limits are per process/file, not aggregate process-tree limits.

## Bounds

Metadata is checked before serialization for cycles, depth (16), item count (200,000), individual strings (4,096 characters), and aggregate UTF-8 string bytes (1 MiB). Request/result byte limits are not parent-process RSS guarantees. Graph serialization must be an exact fixed point, including material records, counters, metadata, IDs and row ordering. Non-preserving material reconstruction is rejected before launch.

The transferred object is graph schema 3.1 plus its JSON metadata, not arbitrary Python instance attributes. Extra attributes on graph/node/edge objects are outside that contract; Material attributes must exactly match its declared dataclass fields. Record IDs, not only dictionary keys, must be built-in integers.

## Python usage

Install the existing `accel` extra in the worker interpreter. This initial profile uses the Taichi 1.7.4 CPython 3.12 Linux x86-64 wheel; other platforms and native payloads require separate qualification. Configure `WorkerRuntime` with that interpreter, its package directory, and the trusted native-library SHA256 from the runtime provenance. The caller must also be able to import Taichi for the existing public `SimResult` loader; importing the provider does not request initialization.

```python
import sys
from pathlib import Path
import taichi
from fibernet import StructureGraph
from fibernet.sim.isolated_taichi import WorkerRuntime, start_isolated_taichi

runtime = WorkerRuntime(
    python_executable=sys.executable,
    provider_site=str(Path(taichi.__file__).resolve().parent.parent),
    native_sha256="66fd0497dcba13009c3a24418fcc39dbd5af5949ae81f901f98381ed7acfdd50",
)
# Synthetic dimensionless example; rows follow insertion order [9, 2, 7].
graph = StructureGraph.from_dict({
    "version": "3.1", "dimension": 2,
    "nodes": [{"id": 9, "position": [0., 0., 0.]},
              {"id": 2, "position": [1., 0., 0.]},
              {"id": 7, "position": [1., 1., 0.]}],
    "edges": [{"id": 18, "node_i": 9, "node_j": 2},
              {"id": 4, "node_i": 2, "node_j": 7},
              {"id": 11, "node_i": 2, "node_j": 9}],
})
with start_isolated_taichi(
    graph, runtime=runtime, wall_seconds=60,
    fixed_nodes=[0],
    external_force=[[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.]],
    spring_k=2., dt=.001, num_steps=10, save_interval=5,
) as task:
    result = task.result(timeout=70)
    result.save("result.json", detailed=True)
    print(result.metadata["node_ids"], task.workdir)
```

The SHA256 identifies the native `.so` payload, not the wheel file; do not replace it with the digest of an unqualified runtime simply to bypass a mismatch. The wheel includes the four Python backend modules. This usage document is a repository/source-distribution document, not a wheel package resource. Importing `fibernet.sim.isolated_taichi` and preparing supported requests remain usable without the optional provider; execution requires the explicit qualified runtime.

## Progress snapshots

`task.progress` reads a bounded, detached JSON snapshot without loading the native provider. It is `None` until the worker publishes its first milestone. Schema `isolated-taichi-progress/1` binds each receipt to the request SHA256 and request schema, retaining two ordered events with sequence numbers and worker monotonic nanosecond timestamps:

- `provider_validated`: actual provider origin/version and uninitialized runtime checked, before simulation dispatch.
- `result_ready`: simulation returned and ordered result IDs were checked, before result export.

These are historical milestones, not live phase locks. The worker can finish between a progress read and `cancel()`; that case does not establish cancellation during native execution. `result_ready` does not mean export succeeded. Use `task.status` and `task.result()` for terminal outcome. Timestamps give worker-local ordering, not wall-clock dates or parent/worker clock synchronization. Both events remain in the final receipt; reading or modifying the returned dictionary cannot modify the task's retained file. No polling sleep, barrier, or callback is added to the solver.

Progress is a control-I/O contract: a receipt-write failure stops the worker with an error, including when simulation has returned but export has not begun. This is reported as a worker/control-I/O failure, not a physical-model failure. The progress reader admits only a bounded regular file, rejects symlinks and pipes without blocking on a pipe open, and reports malformed or truncated data as `IsolatedTaichiError`. This is control-file validation, not a hostile-process sandbox.

## Qualification scope

The reference profile is Linux CPython 3.12.14, NumPy 2.3.5 and Taichi 1.7.4 on one CPU thread with f32 and fast math. Fifty-eight non-native contract, lease/lifecycle and progress cases passed. Installed-package qualification covered offline wheel construction and installation, importing the entry point where the provider was absent, an 81-node/216-edge/100-step result compared with an archived inline reference, induced result-export I/O failure, and cancellation after observing `provider_validated` while task status was running. The reference computation was not rerun for that installed comparison.

The observed caller floating-point snapshots remained unchanged, caller runtime ownership remained unset, and native workers were reaped in those cases. The cancellation receipt establishes a historical milestone, not a known kernel stage. This evidence does not establish convergence, scalability, universal provider-import neutrality, or a whole-process-family memory/performance benefit.
