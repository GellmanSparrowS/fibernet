# Reduced-model configuration admission

`ReducedBeamConfig` and `Engine2Config` are the same mutable dataclass.
`ReducedBeamSolver` is the NumPy Engine2 reduced axial-spring model, with
optional degree-two bending surrogates and node contact. These controls do
not configure the FEM, Taichi, or LAMMPS solvers.

## Accepted inputs

All real-valued controls must be finite real scalars. Python integers and
floats and NumPy integer, float16, float32, and float64 scalars are accepted;
booleans, strings, complex values, and arrays are not numeric configuration
scalars. Extended-precision NumPy floats are rejected because the solver's
force accumulator uses float64; they are not silently downcast.
Integer values in real-valued controls must be finite when converted to
float64, but need not fit NumPy int64 (for example, `stiffness=10**20`).
After all fields pass, admitted numeric controls are stored as Python
`float` or `int`, and switches as Python `bool`. The same normalization
applies at solver admission and run entry after a config has been mutated.
This keeps low-precision NumPy scalars from downcasting coefficient products
and keeps result metadata JSON-serializable for desktop caching. A float16
or float32 input retains its already-rounded value; normalization does not
recover precision lost when the input scalar was created.

- `target_stretch`, `r_contact`, and `dt` must be positive.
- `stiffness`, `k_bend_frac`, `k_contact`, and `drag` must be nonnegative.
  Zero coefficients remain available for disabling a contribution.
- `ramp_fraction` is in `[0, 1]`; zero applies the full displacement on the
  first step, and one ramps over the whole run.
- `grip_pct` is nonnegative and measures each grip band's fraction of the
  sample's initial X width. Zero selects the extreme columns. Values at or
  above `0.5` retain the legacy overlap fallback: nodes left of the sample
  midpoint join the left grip and all remaining nodes join the right grip.
- `num_steps`, `save_interval`, and `contact_rebuild` must be positive
  Python or NumPy integers. `n_increments` must be a nonnegative integer;
  zero selects ramp dynamics. Integral-valued floats are not step counts.
- The same numeric requirements apply when a contribution or loading mode
  is disabled, so switching it on does not expose an invalid latent value.
- `use_bending` and `use_contact` must be Python or NumPy booleans, rather
  than truthy numbers, strings, or arrays.

`target_stretch` is a stretch ratio, not engineering strain: `1.2` requests
20% extension, `1.0` requests no extension, and `0.8` requests compression.
For example:

```python
from fibernet import StructureGraph
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver

graph = StructureGraph(dimension=2)
graph.add_node([0.0, 0.0])
graph.add_node([1.0, 0.0])
graph.add_edge(0, 1)
config = ReducedBeamConfig(
    target_stretch=1.2, stiffness=100.0,
    num_steps=2, save_interval=1, ramp_fraction=1.0,
    use_bending=False, use_contact=False,
)
result = ReducedBeamSolver(graph, config).run()
# Fully gripped unit spring: final reaction = 100 * (1.2 - 1) = 20.
```

## Admission and compatibility

Invalid controls raise `ValueError` naming the field at config construction.
Because the dataclass remains mutable, solver construction validates it
again before reading geometry, and `run()` revalidates it before evaluating
forces or invoking the progress callback. Configure geometry-dependent
controls such as `grip_pct` before constructing a solver; construct a new
solver if those controls change. Validation is not a live reconfiguration
mechanism, and mutating the config during a run is unsupported.

The desktop `RunConfig.engine_cfg()` validates raw step counts before
automatic step scaling. `run_stretch()` constructs that validated config
before creating a cache directory or looking up a cached result. An invalid
config cannot return a historical cached trajectory.

Previously, `target_stretch=float("nan")` with contact disabled could return
nonfinite coordinates and forces without an exception. Nonfinite values,
malformed scalar types, and the out-of-range values listed above are now
rejected instead of reaching incidental arithmetic errors or silent
coercion. Callers should correct the input rather than replace NaN with an
arbitrary stretch. Public names, constructor fields, defaults, result schema,
and numerical formulas are unchanged. Ordinary Python-float inputs retain
their numerical behavior; NumPy scalar controls now use Python float64
coefficient arithmetic instead of dtype-dependent scalar promotion.

Finite, in-range controls do not prove numerical stability, convergence,
or calibrated physical behavior. Large finite values can still overflow;
choose timestep and loading for the graph and check convergence separately.
This change does not add a universal result-finiteness guarantee.

The FiberScope source hash changes so old caches and source-bound research
checkpoints are not silently reused. The engine version remains `e2v15`
because this change is input admission, not a numerical formula change.
The historical golden reference bytes, four cases, 40 arrays and comparison
tolerances remain fixed; only the audited current source identities are
updated for their read-only replay.
