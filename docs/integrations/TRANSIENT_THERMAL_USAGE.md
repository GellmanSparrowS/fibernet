# Explicit transient thermal network (isolated module)

Import `solve_transient_thermal` from `fibernet.sim.transient_thermal`. There is deliberately no root export or replacement of the legacy `simulate_thermal` API.

Supply a `StructureGraph`, complete stable-node-ID maps for `capacity_J_per_K` and `initial_temperature_K`, a complete stable-edge-ID map for `conductance_W_per_K`, and requested `times_s`. Keyword `source_W` is either omitted (all zero) or a complete node map; `fixed_temperature_K` is an ID map or iterable of pairs. Declare geometric `length_unit`, which is provenance only. Node rows are sorted stable IDs; edge rows retain insertion order. Parallel edges remain distinct.

Units are K, J/K, W/K, s and W. Inputs are explicit constant physical parameters, not inferred from material or geometry. Every node needs positive capacity, including fixed nodes. Zero heat conductance is allowed. Sources are nonnegative in this minimal model. Fixed temperatures equal initial values exactly. Disconnected/unanchored components are legal; their initial-value evolution is defined without artificial grounding. The time grid starts at zero and is uniform within the frozen floating-point tolerance; actual interval values are used, not silently rounded.

The method is backward Euler, solving increments and checking the fields reconstructed from returned absolute temperatures. A lost nonzero increment, nonrepresentable heat product, nonfinite result, excessive temperature-rounding allowance, local residual or component energy imbalance is an explicit failure. The local algebraic term is 1e-10 times a node-local energy scale, plus a representation allowance capped at 1e-6 of that scale. There is no universal dimensional absolute floor. See `thermal_frozen_contract.json` for exact formulas.

## Result interpretation

- `temperature_K`: time × node, including the initial frame
- `edge_heat_rate_W`: time × edge, positive from stored node_i to node_j; heat transfer rate, not heat-flux density
- `reservoir_injection_W`: positive external heat input at fixed nodes; zero at free nodes
- `node_stored_heat_change_J`: C(T−T_initial), with the initial reference explicit
- `free_node_residual_J`: backward-Euler interval energy residual; fixed-node entries are unavailable
- `component_energy_residual_J`: component storage change minus source/reservoir heat during that interval
- `node_acceptance_bound_J`, `component_acceptance_bound_J`: actual accepted numerical bounds
- `interval_diagnostics_available`: false at t=0, true thereafter. Unavailable numerical entries are NaN in NumPy arrays, **not valid JSON**. For any strict JSON export, explicitly replace unavailable values with null and preserve the availability mask; no automatic JSON method is currently exposed
- `completed`: all requested intervals passed the discrete numerical checks. It does not certify time accuracy for arbitrary networks
- `provenance_json`: explicit method, units, SciPy version and `time_accuracy_verified=false`

Boundary reactions are legitimate heat inputs, not errors. The t=0 edge field and instantaneous boundary reaction are available, while interval residuals are not. For this model g(ΔT)^2 has units W K; it is not edge heat-generation power and is not returned as such.

## Failure contract

Invalid input raises `ValueError` before simulation. Numerical failures raise `ThermalSolveError`. Its `partial_result` is a detached, read-only-array prefix containing only accepted frames, with `completed=false`; `attempted_step` and `attempted_time_s` identify the rejected attempt. If initial heat rates cannot be represented, `partial_result=None` and attempted_step=0. No failed trial is appended. Input graph and maps are never mutated.

A conservative representability refusal does not mean the physical system is invalid. It means the requested absolute-temperature output cannot support the frozen numerical checks at that scale. Do not remove that guard merely to return a result.

## Validation scope

The saved two-node, two-parallel-edge reservoir fixture has an independent continuous exponential reference and a separate discrete recurrence. Step refinement is checked against the continuous reference; exact discrete energy accounting alone is insufficient. Dedicated tests exercise floating components, isolated heating, boundary signs, sparse/parallel IDs, invalid inputs, bias loss, underflow and deliberately wrong solves. There are no claims for thermo-mechanics, phase change, radiation, convection, temperature-dependent properties, changing topology or effective conductivity.
