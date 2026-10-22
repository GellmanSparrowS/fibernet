# Explicit one-way electrical-to-thermal heating

`fibernet.run_joule_heating` accepts a `StructureGraph`, an existing
`LinearTransportResult`, explicit endpoint heat fractions, and the usual
transient thermal inputs. It returns `JouleHeatingRun(heat_load, thermal)`.
It runs the existing thermal solver once, without rerunning electrical
transport or feeding temperature back into conductance.

```python
electrical = fn.solve_linear_transport(graph, electrical_S, voltage_V,
                                       length_unit="m")
heating = fn.run_joule_heating(
    graph, electrical, capacity_J_per_K, thermal_W_per_K,
    initial_temperature_K, times_s,
    endpoint_weights=endpoint_weights, length_unit="m",
    external_source_W=external_source_W,
    fixed_temperature_K=fixed_temperature_K,
)
node_heat_W = heating.heat_load.total_source_W
temperatures_K = heating.thermal.temperature_K
```

For load assembly without running heat transport, use
`fn.assemble_joule_heating(graph, electrical, endpoint_weights=..., length_unit=...)`.
Its source-map properties return copies; row arrays and ledger records are
detached. `to_dict()` gives a strict-JSON-ready copy.

## Explicit inputs and assumptions

- Electrical conductance is S; thermal conductance is W/K; capacity is J/K.
  They are separate quantities. Geometry and mechanical material fields do
  not calibrate or infer any of them
- `endpoint_weights[edge_id]` contains both endpoint stable IDs and fractions
  in [0, 1]. Cover every edge, including inactive and parallel edges. The sum
  must be within 8 binary64 epsilon of one; no normalization is performed
- Fractions are a caller-declared heat-lumping convention. Explicit .5/.5
  means equal-endpoint lumping, not a universally calibrated distribution
- Constant steady electrical dissipation is assumed to become local heat
  throughout the thermal interval. Electrical storage and relaxation are
  neglected. There is no temperature-dependent resistance or mechanical
  coupling
- Geometry, IDs, oriented connectivity, supplied electrical conductance,
  voltage boundaries and length declaration are checked against the
  electrical result's existing input fingerprint. Equivalent graph record
  ordering is allowed. The fingerprint is an input consistency binding, not
  proof that an arbitrary caller-created result was produced by a solver

## Heat ledger and boundaries

Each edge's nonnegative electrical power is allocated once. Per-edge,
node, electrical-component and global ledgers report allocation/summation
residuals and finite-precision bounds. Allowed weight-sum defects remain
visible; the ledger does not claim mathematically exact conservation.

Electrical voltage constraints do not prescribe thermal reservoir nodes.
Keep Joule and external source terms on fixed-temperature nodes: the thermal
solver's reservoir injection is outward heat flow minus that node's total
source. A reservoir can therefore remove generated heat immediately.

For each backward-Euler interval:

Δstored heat = Δt × (Joule power + external thermal sources + reservoir injection)

Electrical boundary supply and Joule dissipation describe the same energy
transfer. Do not add boundary electrical power, or fixed-node Joule
allocations, again as an extra thermal source. The adapter cannot prevent a
caller manually double-adding source maps later.

The adapter's components are electrical conducting components. Thermal
conducting components can differ: zero electrical conductance need not imply
zero thermal conductance, and conversely. Thermal capacities, sources,
boundaries, time-grid validation and failure handling remain the existing
thermal solver's contract.

## Joined example

Run `python -m examples.one_way_joule_heating`. It uses one two-node graph with
reversed parallel and inactive edges. Electrical powers are 4, 12 and 0 W;
declared fractions assign 4 and 12 W to the nodes. With capacities 2 and 3 J/K
and an isolated thermal network, stored heat at 1 s is 16 J. Temperatures use
backward Euler. These synthetic values demonstrate the workflow, not an
experimentally calibrated material or general time-accuracy guarantee.

## Save and reopen thermal fields without a solve

Use the existing result methods, not a generic dataclass/NumPy conversion:

```python
import json
from pathlib import Path

path = Path("joule_heating.json")
path.write_text(json.dumps({
    "electrical": electrical.to_dict(),
    "heat_load": heating.heat_load.to_dict(),
    "thermal": heating.thermal.to_dict(),
}, allow_nan=False), encoding="utf-8")

saved = json.loads(path.read_text(encoding="utf-8"))
thermal = saved["thermal"]
final_K_by_id = dict(zip(thermal["node_ids"], thermal["temperature_K"][-1]))
```

`TransientThermalResult.to_dict()` returns detached lists and scalars. It
preserves the existing field names, ID ordering, oriented edge endpoints,
`input_sha256`, `completed` and `interval_diagnostics_available`. The recorded
`provenance_json` is decoded to `provenance`, as for the Joule load ledger;
its units, input/source metadata and validation limits are retained unchanged.
A source commit is retained if supplied in model metadata, not inferred by the
exporter. Reopening this JSON does not reconstruct a solver or run one.

JSON `null` means an unavailable diagnostic, **not zero or a failed balance**:

- At t=0 there is no preceding interval, so all four interval diagnostic
  arrays have null entries and `interval_diagnostics_available[0]` is false.
- At fixed-temperature node IDs, `free_node_residual_J` and
  `node_acceptance_bound_J` remain null even on later intervals. These are not
  free-node equations; reservoir injection and component energy diagnostics
  remain available.
- Physical fields, including the initial zero stored-heat change, stay
  numerical. Unexpected NaN/infinity in an available field raises
  `ValueError`; the exporter does not silently sanitize numerical failures.

The original read-only NumPy arrays and NaN sentinels are unchanged. Use the
availability mask and `fixed_node_ids`, rather than treating null as zero,
when consuming diagnostics. `completed=True` means the requested thermal
steps were accepted, not a proof of time convergence, calibrated physics, or
mechanical/two-way coupling.

On `ThermalSolveError`, `error.partial_result` is either `None` (no initial
frame accepted) or an accepted prefix with `completed=False`. Its `to_dict()`
works even when only t=0 was accepted. Preserve `error.attempted_step`,
`error.attempted_time_s` and the error message alongside that export if the
failure context is needed. Rejected trial states are never added to the
prefix. The JSON export is for inspection/analysis, not a restart checkpoint.
