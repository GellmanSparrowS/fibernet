# Saved temperature → prescribed thermoelastic snapshot

Continue a one-way electrical → thermal workflow without rerunning either
upstream solver. Read a saved `TransientThermalResult.to_dict()` object (or a
bundle with a `thermal` member), bind one exact temperature frame to a caller-supplied
`StructureGraph`, then explicitly request a **new** mechanical snapshot.

```python
from fibernet import StructureGraph
from fibernet.sim.saved_temperature import (
    read_saved_thermal_json, prepare_saved_temperature,
)

graph = StructureGraph.load_json("graph.json")
saved = read_saved_thermal_json("electrical_thermal.json")
frame = prepare_saved_temperature(
    graph, saved, frame_index=20, length_unit="m",
)
result = frame.solve(
    graph, profile="network", E_Pa=2e9,
    expansion_alpha_per_K={eid: 1e-5 for eid in graph.edges},
    stress_free_temperature_K=300.,
    constraints={left_node_id: (0., 0., 0.)},
)
```

Choose supports for your structure, not from the example's variable name.
Alternatively, `frame.mechanical_kwargs(graph)` returns only `length_unit`
and a detached stable-node-ID `temperature_K` map for direct use with the
existing prescribed-temperature mechanics owner. Preparation runs no solver.

Save upstream fields once using `json.dumps(..., allow_nan=False)`:

```python
bundle = {
    "electrical": electrical.to_dict(),
    "heat_load": heating.heat_load.to_dict(),
    "thermal": heating.thermal.to_dict(),
}
```

Keep the graph, explicit input parameters, source identity and original failure
records with the bundle. A thermal failure's accepted prefix is not a completed
history. Selection rejects it by default; `allow_partial=True` permits an
explicit accepted-prefix use and preserves `source_completed=False`.

## Transfer contract

- Geometry coordinates and radii use the explicit `m`, `mm` or `um` declaration;
  saved thermal provenance must use the same declaration. Temperature is K,
  E is Pa and expansion coefficients are 1/K. No automatic conversion occurs.
- Node IDs and physical-edge IDs must cover the entire graph exactly. Parallel
  physical members remain distinct. Edge endpoints, orientation and original
  node coordinates must match; record ordering may differ.
- Select a nonnegative, existing frame index. No interpolation, projection,
  cropping, coordinate matching or nearest-time selection occurs.
- The preparation owns detached fields. Before solving, the full graph
  fingerprint is checked again, including radii, materials and internal points.
  Changed geometry requires fresh preparation, not a stale cached field map.
  Thermal exports contain no radii or internal-point fields: those values are
  frozen from the caller-supplied mechanical graph at preparation, NOT verified
  as the original upstream geometry. Preserve and verify the original graph
  and parameter files separately when upstream lineage matters.
- The file reader rejects duplicate JSON keys, NaN/Infinity, overflowing numbers
  and files over 64 MiB. Unavailable interval diagnostics may remain JSON null.
- Output includes the selected time, original thermal input hash, canonical
  thermal-content fingerprint, source completion status and new mechanics data.
  Hashes establish consistency, not authenticity or independent physics proof.
  A bundle's electrical member is retained by the reader but is not independently
  validated or certified as the causal source by this temperature adapter.

## Mechanical model and limits

`profile="elementary"` delegates to the existing two-node axial benchmark
profile. `profile="network"` delegates to the existing experimental 2D planar
welded Euler–Bernoulli network profile: by default 2–192 nodes and 1–256 physical edges,
straight/collinear members, sufficient supports, explicit common E and
per-edge alpha. The network uses endpoint-mean temperature for each member's
constant axial eigenstrain. It does not resolve through-section temperature
gradients or provide 3D mechanics.

Existing admission checks remain unchanged: endpoint thermal strains ≤1e-5,
mean total axial strain and extreme-fiber strain ≤1e-4, rotations/slopes ≤1e-3
rad. Admission is not buckling, material, joint, mesh or time-step validation.
Large saved fields can be reopened and bound without being admissible to the
default mechanics profile; the adapter never raises limits implicitly or cuts out a model.
For an explicit caller-chosen network size contract, pass `ThermoelasticSizeLimits`
as `size_limits` to `frame.solve(...)`. All non-size model checks remain unchanged.
See [THERMOELASTIC_NETWORK.md](THERMOELASTIC_NETWORK.md) for usage and the distinction
between size admission and factorization/resource guarantees.

Displacement, axial/bending stress, reactions and applicability diagnostics are
new mechanical outputs, not recovered upstream fields. There is no inertia,
deformation-dependent resistance, feedback into heat, or reciprocal three-field
coupling. Check the thermal heat ledger and the mechanical eigenstrain energy
separately; adding them does not establish total coupled energy conservation.

A 35-node/82-edge synthetic sequential example has exercised a saved electrical
checkpoint, a 20-interval thermal history and newly solved initial/final
mechanical snapshots against analytical rods and an independent frame-energy
reference. This does not qualify arbitrary networks or spatial/time convergence.

## Command-line example

`examples/saved_thermal_mechanics.py --graph graph.json --thermal fields.json
--mechanics mechanics.json --frame 20` inspects a selected saved frame without
solving. Add `--solve --output new_snapshot.json` only when a new mechanics solve
is intended. Existing outputs are never replaced. The mechanics JSON supplies
`profile`, `length_unit`, `E_Pa`, `expansion_alpha_per_K`,
`stress_free_temperature_K`, `constraints`, and optional `loads`. ID-keyed
objects use canonical decimal string keys. No coefficient or support is inferred.
This CLI retains default profile limits; explicit size contracts are available
through the Python library API, not additional CLI or MCP options.
