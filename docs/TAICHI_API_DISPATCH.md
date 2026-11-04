# Calling the optional Taichi spring backend

The current library entry point accepts a `StructureGraph` and returns the actual
`SimResult` from the existing engine:

```python
result = fibernet.simulate(
    graph, mode="dynamics", backend="taichi",  # "spring" is the same engine
    arch="cpu", num_threads=1,
    fixed_nodes=fixed_array_rows,
    spring_k=axial_coefficients,
    nodal_masses_by_id=masses_by_stable_id,
    state_units=declared_units,
    external_force=constant_nodal_forces,
    dt=dt, num_steps=steps, save_interval=sample_interval,
    dashpot=0.0, drag=0.0, record_state=True,
)
```

The variables above must describe a prepared, validated problem. This example
does not assign units, masses, coefficients or a stable timestep automatically.
`backend="spring"` remains the default. `backend="taichi"` is an explicit alias,
not a new solver, integrator or GPU default. Both stretch and dynamics modes use
the same existing methods. `arch` is `"cpu"` or `"gpu"`; `num_threads` is a
positive integer (default 4), with booleans rejected. These options configure the
engine and are not forwarded as physics-method arguments. Unknown backend/mode
or invalid runtime-option values fail before this wrapper constructs the engine.
This does not make imports of the entire library lazy.

Taichi is optional and the supported version remains 1.7.4 (the `accel` extra).
Missing dependency, unsupported version, unavailable requested architecture and
runtime ownership errors propagate; there is no automatic install or alternate
solver fallback. The first initialization owns the runtime thread and CPU thread
setting. A later engine cannot silently replace an active runtime's settings.
Requesting `"gpu"` does not establish GPU availability or validated GPU accuracy.
See [the lifecycle contract](architecture/taichi-lifecycle-audit.md).

## Physical inputs and identities

The native spring coefficient is force-like: F = k strain and
U = 0.5 k L0 strain². For SI axial rods k is EA in newtons, not EA/L in N/m.
The engine does not infer k or nodal inertia from stored Material/radius values.
Use explicit coefficients and nodal masses, and declare consistent length,
time, mass and coordinate frame. Unit strings describe the inputs; they do not
convert them. Active damping controls are `dashpot` and `drag`; the historical
`damping` parameter remains metadata-only.

Native node rows follow graph node insertion order. `fixed_nodes`, force rows
and displacement-schedule keys use that row order; `nodal_masses_by_id` uses
stable IDs. Node and edge IDs in the returned result identify the arrays.
The wrapper preserves the graph, its parallel members and the result unchanged.
It does not convert a FiberNetwork or silently reindex a graph.

Dynamics validates these caller controls even without explicit masses, state
recording or declared units: `fixed_nodes` must contain nonnegative integer rows
below the node count (not booleans); `dt` must be a finite positive real scalar;
`dashpot` and `drag` must be finite nonnegative real scalars; and a supplied
`external_force` must be a finite real array of shape `(N, 3)`. Invalid controls
raise `ValueError` before dynamics field allocation. Negative row indices no
longer select nodes from the end, and boolean rows no longer act as NumPy masks.
This is not a stability check. The public wrapper constructs the engine first,
so these checks do not promise rejection before Taichi runtime initialization.
For stable-ID supports and loads, use
`prepare_nodal_controls(...).taichi_kwargs(graph)` rather than passing IDs as rows.

The underlying model remains straight endpoint axial dynamics with its existing
kick/drift, damping and projection conventions. This alias adds no bending,
contact, prestress, initial velocity, new boundary/load model or material law.
See [native mass/state semantics](NATIVE_AXIAL_MASS_STATE.md).

## Read actual result quantities

- With `record_state=True`, actual final native velocities are in
  `result.dynamics_state["final"]["velocities"]`; sampled velocities and times
  remain aligned in that state record
- Without state recording, the result does not provide those velocity arrays
- `result.edge_forces` contains signed axial forces, one per edge; it is not a
  nodal force/reaction vector and is not velocity
- For this spring backend, `result.energy` is axial potential energy. Recorded
  kinetic and total mechanical energy live in the native state record
- `time_seconds` is measured solve wall time, not physical simulation time
- A summary save retains summary availability; it cannot reconstruct omitted
  velocity or trajectory data on reload

The wrapper does not fabricate missing fields, reinterpret quantities or convert
SimResult into a legacy dictionary. Direct engine calls remain supported.

## Legacy FiberNetwork function

`fibernet.api.simulate_dynamics` is a separate historical API. Its default
`backend="taichi"` now fails explicitly with `NotImplementedError` before
building a spring system. It previously called an absent method and expected
nonexistent result attributes. The default is retained to avoid silently
substituting a different calculation.

That function's crosslink rows, along-fiber connections and mixed millimetre /
EA-over-length conventions do not have a validated conversion to the current
StructureGraph engine. Endpoint-only `StructureGraph.from_fiber_network` is
approximate and is not equivalent. Prepare the full graph and consistent
physical inputs explicitly before using the current API.

Explicit `backend="numpy"` keeps the existing legacy builder, recurrence and
return dictionary unchanged. This dispatch repair does not correct or certify
that legacy path's physical units. Unknown legacy backend names now raise
`ValueError` instead of silently selecting NumPy. No legacy/native numerical
parity is claimed.

## Verification boundary

The interface includes explicit masses, sampled state and optional observer and
algorithmic-ledger diagnostics. Source and mock checks alone do not establish
native numerical parity, stable timesteps or performance for a particular
installation or network. Verify the supported Python/runtime combination and
use a bounded representative problem before a larger calculation. Source-bound
checkpoints intentionally reject incompatible source identities; use a fresh
output directory or retain the matching source rather than relabeling results.

## Floating-point environment and process-isolation limitation

The current CPU native validation is qualified only for a bounded isolated
process. It does not establish numerical neutrality inside a long-lived caller
process. In a Python 3.12.14 / NumPy 2.3.5 / Taichi 1.7.4 CPU run, Python
and NumPy half-min-normal arithmetic produced a nonzero subnormal value before
`fibernet.simulate(...)` and zero afterward, despite passing network parity.
The exact floating-point control register was not read, so this is an observed
arithmetic side effect, not a claim about a particular FTZ/DAZ register bit.

If surrounding numerical code depends on its floating-point semantics, run
the Taichi backend in a separately owned worker process and exchange explicit
input/result data. The wrapper itself remains an in-process callable; it does
not automatically provide isolation. No caller-state restoration is promised.
Do not assume `taichi.reset()` or an unchanged `numpy.geterr()` policy restores
or proves the caller's arithmetic behavior. `numpy.geterr()` reports NumPy's
error-handling policy, not a hardware denormal-mode measurement. See the
[NumPy error-policy documentation](https://numpy.org/doc/2.3/reference/generated/numpy.geterr.html).

This qualification does not cover arbitrary process reuse, other threads, GPU
execution, other hardware/runtime versions or all floating-point modes. No
process-global floating-point reset is provided as a workaround.
