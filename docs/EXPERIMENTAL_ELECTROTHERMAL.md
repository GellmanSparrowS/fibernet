# Experimental two-way lumped electrothermal model

`fibernet.sim.electrothermal` provides an explicit, experimental module-level
API for voltage-driven, fixed-geometry electrothermal feedback. Import
`TemperatureConductanceLaw`, `solve_electrothermal`, and
`ElectrothermalSolveError` from that module. They are not exported from
`fibernet` or `fibernet.sim`.

This model is two-way: temperature changes electrical conductance; a new
steady electrical solution changes Joule heating; that heating changes the
next temperature. Unlike the one-way Joule adapter, it does not retain a
single electrical solution for the entire thermal trajectory.

The scope is deliberately narrow:

- Geometry and topology remain fixed.
- Electrical transport is quasistatic and resistive, driven by prescribed
  voltages. There is no current-driven boundary API or electrical storage.
- Thermal behavior uses positive nodal lumped heat capacities and constant,
  nonnegative edge thermal conductances, with optional constant nonnegative
  external sources and constant fixed-temperature reservoirs.
- There is no mechanics, deformation feedback, radiation, phase change, or
  inference of electrical or thermal properties from material or geometry
  fields. This is not a generic multiphysics coupling interface.

## Explicit constitutive and lumping inputs

For each edge `e`, the caller supplies

```text
T_edge[e] = sum(endpoint_temperature_weights[e][n] * T[n])
G[e] = G0_S[e] / (1 + alpha_per_K[e] *
                        (T_edge[e] - reference_temperature_K[e]))
```

`G0_S`, `alpha_per_K`, and `reference_temperature_K` are exact stable-edge-ID
maps, including every inactive and parallel edge. Values must be finite;
`G0_S >= 0`, `alpha_per_K >= 0`, and reference temperatures are nonnegative
Kelvin. The single global `temperature_domain_K=(lower_K, upper_K)` is a
finite nonnegative closed interval. Each reference temperature must lie in
it, and every edge denominator must be finite and strictly positive
throughout it. Nonnegative `alpha_per_K` means active conductance is
nonincreasing with edge temperature.

`endpoint_temperature_weights[edge_id]` must name both actual endpoint
node IDs, use finite nonnegative weights, and sum to one within eight
binary64 machine epsilons. Weights are not normalized. Every node temperature
and interpolated edge temperature is checked against the declared domain at
initialization and at every evaluated trial and candidate state. An edge
with `G0_S == 0` remains electrically inactive, but its parameters,
denominator, endpoint weights, and temperatures still undergo these checks.
No domain extrapolation or silent clipping is performed.

The distinct argument `endpoint_heat_weights[edge_id]` allocates each
edge's electrical dissipation to its endpoint heat sources. It must likewise
cover every edge and both endpoints; fractions are in `[0, 1]` and must sum
to one within eight binary64 machine epsilons. There is no normalization.
Temperature interpolation weights and heat-allocation weights are separate
caller-declared modeling choices. Neither determines the other. Equal
weights in the example below are a synthetic symmetric convention, not a
material calibration.

Supply capacities in J/K, thermal conductances in W/K, temperatures in K,
electrical conductances in S, voltages in V, times in s, and source powers
in W. Capacity, thermal-conductance, initial-temperature, and optional
external-source maps require exact node or edge coverage as appropriate.
Every capacity must be strictly positive. `length_unit` declares the graph
coordinate unit (`"m"`, `"cm"`, `"mm"`, `"um"`, or `"nm"`); it does not infer
or convert supplied conductances or capacities.

## Time stepping and same-state acceptance

The time grid must be uniform, strictly increasing, and start at zero.
Each interval uses backward Euler and at most `max_iterations` Picard
iterations, independently of other intervals. The default is 25 and the
allowed range is the integer interval `[1, 25]`.

For each interval, the previous accepted temperature is held fixed as the
backward-Euler initial state. Starting with that temperature as the trial:

1. Evaluate the conductance law at the trial temperature.
2. Solve steady voltage-driven electrical transport and assemble the Joule
   load using the declared heat weights.
3. Solve one constant-source backward-Euler thermal interval to obtain a
   candidate temperature.
4. Reevaluate the conductance law at that candidate, solve electrical
   transport again, and reassemble the Joule load. The values saved in an
   accepted frame are these final same-state values, not the lagged values
   used to generate the candidate.
5. Accept only if the candidate-to-trial temperature change is small at
   every node and the thermal equations and thermal-component energy
   balances pass using the recomputed final load.

The temperature-increment criterion is

```text
abs(candidate[n] - trial[n])
    <= 8 * ulp(candidate[n])
       + 1e-12 * abs(candidate[n] - previous_accepted[n])
```

Electrical transport and Joule allocation also retain their own
finite-precision residual and provenance checks. Thermal residual bounds
use `1e-10` times the local heat scale plus an explicit finite-precision
allowance, with that allowance itself limited to `1e-6` times the local
scale. There is no dimensional absolute residual floor.

A small temperature change alone is insufficient. A finite thermal
residual rejection can trigger another Picard iteration; nonfinite or
unrepresentable arithmetic and precision-budget failures do not become
accepted states. Exhausting the iteration limit rejects the interval.
There is no automatic damping, time-grid refinement, or fallback to a
one-way solution.

Acceptance establishes finite-tolerance joint consistency of this
discrete model. It does not establish an exact nonlinear solution,
uniqueness, convergence for arbitrary inputs, experimental validity, or
time-discretization accuracy.

## Energy accounting, output, and failures

Electrical constraints and thermal reservoirs are independent. Keep
allocated Joule heat and external sources on fixed-temperature nodes:
their reservoir injection is outward thermal heat flow minus total local
source power. Positive reservoir injection adds heat to the graph. Do not
add electrical boundary supply a second time as a thermal source.

An edge's thermal heat rate is positive from its `node_i` to its `node_j`.
Thermal components are determined by positive thermal conductances, not
by electrical connectivity. The electrical component ledgers reside in
each frame's `heat_load_json`.

`ElectrothermalResult` and its `ElectrothermalFrame` records are frozen,
tuple-backed snapshots. They contain no writable result arrays; nested
electrical and heat-load snapshots are serialized JSON strings. Decode
those strings with `json.loads` when a detached mutable dictionary is
needed. Node-indexed tuples follow `result.node_ids` (sorted stable IDs);
edge-indexed tuples follow `result.edge_ids` (graph edge insertion order).
Component-indexed tuples follow the labels in
`result.thermal_component_labels`.

For each accepted interval, output includes instantaneous fields, stored
heat, free-node and thermal-component residuals and their acceptance
bounds, the accepted iteration count, and the electrical and heat-load
snapshots at the saved temperature. `node_stored_heat_change_J` is storage
relative to the initial temperature; `interval_stored_heat_J` is storage
relative to the preceding accepted frame.
`cumulative_source_heat_J` and `cumulative_reservoir_heat_J` sum each
accepted interval's duration times its final right-endpoint power. They
are not final power multiplied by elapsed time.

The initial frame has `interval_diagnostics_available=False`, zero
accumulated heat, and empty interval-residual/bound tuples. Later frames
have interval diagnostics; free-node residual/bound entries at
fixed-temperature nodes are NaN because their free-node equation does
not apply.

Input errors can raise `TypeError` or `ValueError`. Failures before an
initial state is accepted do not provide a completed trajectory. An
interval failure after the initial frame raises
`ElectrothermalSolveError`, whose `partial_result` has `completed=False`
and contains only the accepted frame prefix. `attempted_step` and
`attempted_time_s` identify the rejected interval. The iteration history
may describe rejected attempts, but their trial states are never added
to `frames`. Callers must not treat that prefix as a completed run.

The implementation enforces API limits: at most 4,096 nodes, 16,384
edges, 2,048 time intervals, 2,000,000 retained scalar entries, 32 MiB of
serialized electrical/heat-load snapshots, and 16 KiB of JSON model
metadata. A conservative history-size check may reject a request before
stepping; retained-history checks can also stop a run with an accepted
prefix. These limits are not performance or scalability claims.

## Minimal two-node usage

```python
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.electrothermal import (
    TemperatureConductanceLaw,
    solve_electrothermal,
)

graph = StructureGraph()
left = graph.add_node([0.0, 0.0, 0.0])
right = graph.add_node([1.0, 0.0, 0.0])
edge = graph.add_edge(left, right)

law = TemperatureConductanceLaw(
    G0_S={edge: 1.0},
    alpha_per_K={edge: 0.01},
    reference_temperature_K={edge: 300.0},
    endpoint_temperature_weights={edge: {left: 0.5, right: 0.5}},
    temperature_domain_K=(290.0, 350.0),
)

result = solve_electrothermal(
    graph,
    law,
    prescribed_voltage_V={left: 0.0, right: 4.0},
    capacity_J_per_K={left: 2.0, right: 2.0},
    conductance_W_per_K={edge: 0.0},
    initial_temperature_K={left: 300.0, right: 300.0},
    times_s=[i / 10 for i in range(11)],
    endpoint_heat_weights={edge: {left: 0.5, right: 0.5}},
    length_unit="m",
)
assert result.completed
print(dict(zip(result.node_ids, result.frames[-1].temperature_K)))
```

Both thermal nodes are free and, with zero thermal edge conductance,
belong to distinct thermal components. Symmetric capacities and heat
weights make their temperatures equal even though they receive no
conductive heat from one another.

## Independent small benchmark

Run the explicitly invoked benchmark from the repository root:

```console
python -m benchmarks.electrothermal_two_node --output electrothermal_two_node.json
```

Omit `--output` to print the JSON report to standard output. Importing the
benchmark module does not import the solver or perform a solve. The
callable `run_benchmark()` returns the report as a dictionary.

The fixture is exactly the two-node model above, over 1 s, with time
steps 0.1, 0.05, and 0.025 s. It compares both saved node temperatures at
all saved times with independent 80-digit `Decimal` references. These
references use only the fixture's scalar equations, not the production
thermal or electrical helpers.

Let `theta = T - 300 K`, `a = 0.01 / K`, and `k = 4 K/s`. Equal endpoint
temperatures reduce the model to

```text
d(theta)/dt = k / (1 + a * theta)
theta(0) = 0
theta(t) = 2*k*t / (1 + sqrt(1 + 2*a*k*t))

b = 1 + a * theta_previous
backward_Euler_increment =
    2*k*dt / (b + sqrt(b*b + 4*a*k*dt))
```

The quadratic backward-Euler recurrence is a discrete reference, while
the continuous solution measures time-discretization error separately.
The frozen comparison criteria are a maximum discrete-reference
temperature error of `2e-10 K`, a finest-step final continuous-reference
error of `0.003 K`, and successive halving error ratios between `0.45`
and `0.55`. The report includes each criterion, errors, ratios, reference
and numerical trajectories, and an overall fixture-comparison result.
A failed comparison produces a nonzero command exit status.

These thresholds apply only to this symmetric fixture and its three
specified grids. They are not general accuracy tolerances, a convergence
guarantee for other networks, or a material validation.
