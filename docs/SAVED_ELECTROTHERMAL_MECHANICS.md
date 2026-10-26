# Reuse saved electrothermal fields for explicit mechanics

Save an existing electrothermal trajectory once, reopen it, and select its exact
nodal temperatures for a new mechanical snapshot. The adapter reuses
`StructureGraph` and the public thermoelastic API; it does not introduce another
solver, graph representation or root-level API.

Electrical and thermal states are retained together. The upstream solver's
fixed-geometry temperature-dependent conductance affects electrical Joule heating,
which drives backward-Euler thermal steps. Mechanics is a subsequent, one-way
prescribed-temperature calculation. Mechanical displacement and stress never feed
back into electrical conductance, geometry or temperature.

## Python workflow

Export a native `ElectrothermalResult` from an already completed calculation:

```python
import json
from pathlib import Path
from fibernet.sim.saved_electrothermal import export_saved_electrothermal

saved = export_saved_electrothermal(graph, native_result, length_unit="m")
# Exclusive creation avoids replacing an earlier saved result.
with Path("electrothermal.json").open("x", encoding="utf-8") as stream:
    json.dump(saved, stream, allow_nan=False, sort_keys=True, indent=2)
    stream.write("\n")
```

Reopen the saved file and explicitly choose an existing frame:

```python
from fibernet.sim.saved_electrothermal import (
    read_saved_electrothermal_json,
    prepare_saved_electrothermal_temperature,
)

saved = read_saved_electrothermal_json("electrothermal.json")
selection = prepare_saved_electrothermal_temperature(
    graph, saved, frame_index=selected_frame, length_unit="m",
)
temperature_inputs = selection.mechanical_kwargs(graph)
```

Reading, validation, selection and `mechanical_kwargs` perform no electrical,
thermal or mechanical solve. Selection neither interpolates nor restarts a
trajectory. `selection.to_dict()` returns a detached record of the selected
frame, source identities, completion state and diagnostics availability.

A mechanical calculation occurs only with an explicit call:

```python
result = selection.solve(
    graph, profile="network",
    E_Pa=youngs_modulus,
    expansion_alpha_per_K=expansion_by_edge_id,
    stress_free_temperature_K=reference_temperature,
    constraints=prescribed_components_by_node_id,
    loads=loads_by_node_id,
)
```

Supply these material, support and load values for the intended mechanical model;
electrodes are not mechanical supports. Scalar `E_Pa` and explicit edge expansion
coefficients govern mechanics. Retained graph materials do not silently activate
heterogeneous or anisotropic material laws. The result includes `temperature_source`
and the new `mechanical` result, with `feedback_to_electrical_or_thermal=False`.
The public thermoelastic facade is loaded lazily at the explicit solve boundary.
Its errors and applicability limits propagate unchanged.

The `network` profile is a static, small-deformation, welded planar
Euler–Bernoulli frame. The narrower `elementary` profile is also explicit.
See [the mechanical method](THERMOELASTIC_NETWORK.md) and
[the elementary method](EXPERIMENTAL_THERMOELASTIC.md). No mechanics is exposed as
an additional MCP tool by this adapter. The older saved-temperature API and root
exports are unchanged.

## Saved-byte replay

From the repository root, validate the included nonuniform planar saved fixture:

```sh
python -m benchmarks.saved_electrothermal_replay --fixture planar_network.json --frames 0 5 10 --length-unit m
```

The bundled fixtures use deterministic gzip/base64 storage. The explicit
`--fixture` route verifies fixed encoded/decoded size limits, a single complete
gzip member and the original decoded-byte SHA256, then passes temporary JSON
through the same strict native reader. An ordinary caller-supplied JSON path
continues to use that reader directly; compressed inputs are not added to the
native API. Container hashes and decoded envelope hashes are distinct.

The replay decodes the envelope, verifies graph/source consistency, and compares the
transferred nodal temperatures exactly in binary64 by stable ID. It prints the
saved-envelope file SHA256 separately from the canonical saved-content hash.
It calls no solver, reproduces no numerical trajectory and measures no solver
performance. Its checks complement, rather than replace, the numerical evidence
summarized below.

## Identity and input conditions

The schema is `fibernet.saved_electrothermal/1`. Native result fields are retained
under `source`; JSON-text native provenance/electrical/heat-load fields become
strict JSON objects. Every accepted frame retains its actual `G(T)`, not only
reference conductance. Stable node and physical-edge IDs, oriented endpoints and
parallel physical edges are preserved. Graph insertion order is immaterial.

The full canonical graph fingerprint covers geometry, radii, internal points,
materials and metadata. Electrical provenance establishes IDs, oriented endpoints,
coordinates and units. Dimension, radii, internal points, materials and metadata
are caller-supplied fields: captured and mutation-checked, not independently
established by the electrical solve. The caller must supply the original upstream
graph. Changing any captured graph field invalidates a prepared selection.

Distinct hashes have distinct meanings:

- Native `input_sha256` is a reported source identifier. The original complete
  input payload is not retained, so the adapter does not reconstruct or
  authenticate that identifier.
- Each nested electrical input fingerprint is verified using the existing
  electrical binding owner. Heat-load lineage and conductance-law consistency are
  checked without numerical solves.
- `content_sha256` is a versioned canonical saved-content consistency hash,
  excluding itself. It is not an authenticity signature.
- A saved-envelope file SHA256 identifies the actual encoded envelope bytes.
  It does not identify the separate raw native-result dataclass dump, and it
  differs from the canonical-content hash when formatting differs.

Mechanics requires `dimension=2`, nodes in one XY plane, positive finite Kelvin
temperatures and an explicit matching `m`, `mm` or `um` unit. There is no implicit
projection, coordinate rewriting or silent unit conversion.

## Strictness, partial results and bounds

The reader refuses files over 64 MiB, duplicate keys, nonfinite JSON numbers and
malformed outer objects. Preparation rejects missing/duplicate IDs, noncanonical
ID-map keys, wrong shapes and booleans used as numbers. Accepted time grids start
at zero and are uniform within the upstream owner's binary64 roundoff allowance.

`frame_index` is a required nonnegative integer naming an existing accepted frame.
Partial accepted prefixes reject by default; `allow_partial=True` permits only
available frames and preserves incomplete status. When exporting an accepted
prefix after an upstream exception, supply explicit `failure` context with
`reason`, `attempted_step` and `attempted_time_s`; it is not inferred. A terminal
successful trial in iteration history without a retained frame is not selectable.

At time zero, interval diagnostics are unavailable and their arrays remain empty.
Permitted fixed-temperature-node NaN sentinels become JSON null only in those
diagnostic positions. Available diagnostics stay finite; residuals must lie
within their recorded nonnegative bounds. Unavailable values never become zeros.
Thermal component labels remain separate from electrical component ledgers.

Saved-state shape limits are at most 4096 nodes, 16384 physical edges, 2049 frames
and 51200 iteration-history entries. The retained-history scalar estimate
`frames * (80 * nodes + 80 * edges)` and the nested-value traversal are each capped
at 2,000,000. These admission limits are not measured large-network performance
or numerical-validity guarantees. The default network mechanical profile remains
2–192 nodes and 1–256 physical edges. Explicit `ThermoelasticSizeLimits` may be
passed only for `profile="network"`; such limits do not bound factorization fill,
peak memory or execution time, and do not extend physical validity.

## Measured numerical scope

Two synthetic datasets exercise the unchanged native solver and adapter bytes.
Their exact saved envelope bytes are encoded under `tests/fixtures/saved_electrothermal_v1/`.
[VALIDATION.json](../tests/fixtures/saved_electrothermal_v1/VALIDATION.json) identifies
the exact envelope hashes, implementation hash and selected measured errors.
These are comparisons with independently constructed discrete references, not
comparisons with calibrated experimental data.

The axial-chain case has 3 nodes, 2 physical edges and 11 accepted frames over
0–1 s with 0.1 s backward-Euler steps. Four mechanical snapshots use initial/final
temperatures with free-expansion and axially blocked-end conditions. An
80-digit Decimal recurrence gives maximum temperature error below 9.04e−14 K;
axial stress error is below 7.49e−14 Pa. Uniform temperature and zero bending
make this an axial-transfer check, not a planar-network test.

The planar case has 25 nodes, 56 physical edges and 11 accepted frames over the
same time interval. It uses a 4×4 triangular-cell graph, fixed geometry,
temperature-dependent conductance, nonuniform temperatures, thermal conduction,
and welded-frame mechanical snapshots at 0, 0.5 and 1 s. Discrete coupled
references and mechanical references were constructed independently. At the
final frame, 48 edges dissipate at least 1e−6 W, all 56 carry thermal flux
of magnitude at least 1e−5 W, and 52 have bending moments of magnitude at least
1e−10 N m. Temperature spread reaches 0.3271 K; transverse displacement reaches
2.379e−7 m and rotation 3.720e−7 rad at that frame.

Maximum absolute comparison errors were 1.435e−13 K in temperature,
1.527e−15 V in voltage, 1.686e−15 A in current, 4.455e−20 m in displacement,
1.201e−19 rad in rotation and 1.013e−11 Pa in stress. Primitive-field balance
checks covered nodal current, electrical power, nodal/component/history heat,
force, moment and work. The largest whole-history heat-balance residual was
2.132e−12 J; global force and moment residuals were below 7.89e−18 N and
3.67e−18 N m. No upstream electrical or thermal solve occurred after persistence
while those mechanical snapshots were made.

These results establish only the tested synthetic fixed-geometry discrete
trajectory and selected one-way linear planar snapshots. They do not establish
time-step accuracy, mesh convergence, buckling/stability, material calibration,
general ordered/disordered-network validity, large-network performance, dynamic
mechanical history or reciprocal three-field coupling.
