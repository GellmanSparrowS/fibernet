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
