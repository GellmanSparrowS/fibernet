# Plan storage for pair-free CG models

`forecast_cg_fiber_only()` estimates trajectory, export and memory sizes from a
prepared coarse-grained model. It reports each size alongside the caller's
limits, making it easy to compare schedules and identify which output would
exceed a budget before running a simulation.

Prepare the graph with `prepare_cg()`, supply the intended runtime configuration
and an explicit export budget, then inspect the returned comparisons:

```python
from fibernet import StructureGraph
from fibernet.integrations import (
    CGRequest, HarmonicBond, ResourceLimits, RunSpec, RuntimeSpec,
    forecast_cg_fiber_only, prepare_cg,
)

graph = StructureGraph(dimension=3)
left = graph.add_node([0.0, 0.0, 0.0], merge=False)
right = graph.add_node([1e-6, 0.0, 0.0], merge=False)
edge = graph.add_edge(left, right, radius=1e-8, segments=1)
request = CGRequest(
    length_unit="m", maximum_bond_length_m=3e-6,
    masses_kg={left: 1e-15, right: 1e-15},
    bonds={edge: HarmonicBond(2e-6, 1e-6)}, angles={},
    projectile_node_id=None, projectile_pairs={},
    run=RunSpec(1e-7, 100, 1, ((-5e-6, 5e-6),) * 3),
)
prepared = prepare_cg(graph, request)

# Planning placeholders: these files need not exist.
runtime = RuntimeSpec(
    mode="cli", python_executable="python", lammps_executable="lmp",
    expected_version=20250722, expected_banner="LAMMPS (22 Jul 2025)",
)
limits = ResourceLimits(memory_bytes=512 * 1024**2)
forecast = forecast_cg_fiber_only(
    prepared, runtime, limits, export_budget_bytes=16 * 1024**2,
)
assert forecast["native_frames"] == 101
print(forecast["budget_comparison"])
print(forecast["fits_declared_budgets"])
```

Use the intended `RuntimeSpec` strings when exact setup-byte accounting matters:
the planner counts their serialized manifest and metadata representation.

## Supported inputs

The planner accepts the existing unprofiled, pair-free CG contract:

- 2–64 nodes, 0–128 bonds, 0–128 angles and 0–1000 integration steps.
- `projectile_node_id=None`, empty projectile pairs, and unset `profile` and
  `pair_encoding`.
- Every-step output (`sample_every=1`). A zero-step model has one frame; the
  maximum schedule has 1001 frames.
- Explicit graph units and SI model coefficients supplied through `CGRequest`.
  Preparation retains the authoritative `StructureGraph`, stable IDs and
  existing model validation.
- `RuntimeSpec` mode `cli` or `cli_serial_static`, text configuration fields,
  an integer version date from 20000101 through 21000101, and unique text keys
  in environment/hash pairs.
- A positive integer `export_budget_bytes` up to `2**63 - 1` and a boolean
  `record_state`.

`ResourceLimits` keeps its existing ranges: 1–60 CPU seconds, a positive wall
time up to 120 seconds, 64 MiB–2 GiB memory, and 1024 bytes–16 MiB for each of
aggregate output and individual native files. Input counting has independent
bounds: 1 MiB for the combined prepared strings, 1 MiB for runtime configuration,
and 4 MiB for counted manifest/setup/metadata text.

A valid small output budget still returns a forecast with a false comparison.
Malformed contracts, configuration or limits, and oversized structural inputs
raise `ValueError` (including `LAMMPSContractError`).

## Read the forecast

- Counts include nodes, edges, angles, frames, type-pair entries and generated
  pair coefficient/exclusion commands. `source_identity` binds the result to
  the prepared input and graph.
- `prepared_bytes`, `prepared_artifact_bytes` and `metadata_base_bytes` count
  the supplied serialized content. Runtime strings and declared limits affect
  the exact setup count.
- Trajectory, thermo, detailed-export and parsing/copy-memory fields are
  conservative estimates. Future metadata includes a reserve for the unknown
  final-force scalar. The type-pair payload field is a lower bound for that
  native data structure, not a complete native-memory estimate.
- `budget_comparison` checks aggregate native output, individual trajectory
  and thermo files, detailed export and the modeled memory envelope.
  `fits_declared_budgets` is true when all five comparisons fit.
- `record_state=False` retains the same transient-memory estimate because
  current conversion constructs the full state before selecting saved output.

## Scope

Forecasting is an in-memory planning operation: it validates prepared records
and counts bytes without importing LAMMPS, accessing provider paths or launching
a task. `runtime_readiness` remains `not_checked` and `execution_admission`
remains false. A fitting report describes budget fit, not permission or runtime
readiness to execute.

The memory comparison models parent-side parsing and export copies, while
`ResourceLimits.memory_bytes` governs native child address space during an
actual run. CPU cost and measured peak RSS remain unknown. The explicit export
budget is a planning input rather than an additional enforced file limit.
Existing preparation caps, generated native inputs and harmonic/network
forecast results are preserved.
