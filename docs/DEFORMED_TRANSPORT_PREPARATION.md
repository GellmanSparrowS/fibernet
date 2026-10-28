# Prescribed-translation transport preparation

`fibernet.sim.deformed_transport.prepare_deformed_transport` is an additive,
module-level preparation API. It returns an existing `StructureGraph`, a detached
edge-ID conductance dictionary, and a detached finite-JSON binding report. It
never executes electrical, thermal, mechanical, reference, or generator code.
There are no new root exports, graph types, solver classes, result loaders, or
reusable solver plans.

## Call and model

```python
from fibernet.sim.deformed_transport import prepare_deformed_transport

derived_graph, conductance_S, binding = prepare_deformed_transport(
    graph,
    translations_m_by_id={nid: [ux_m, uy_m] for nid, ux_m, uy_m in prescribed_rows},
    reference_conductance_S=reference_G_by_edge_id,
    length_unit="m",
)
```

Translations are caller-prescribed SI Cartesian components along the source
graph's existing x/y axes. They are not automatically accepted mechanical
results. A row must contain exactly two components; neither xyz nor xy plus
rotation is accepted. A caller who extracts translations from another result is
responsible for that interpretation, identity, and unit conversion. No saved
mechanics artifact, `SimResult`, temperature frame, or equilibrium claim is
inferred or admitted.

The fixed model is:

- `x_new = x_reference + translation_m / length_scale`, preserving original z.
- Source and derived lengths are actual stored endpoint chords in metres.
- `G_new[e] = G_reference[e] * L_reference[e] / L_new[e]`.
- Equal actual chord lengths preserve the supplied conductance exactly.
- Effective area is fixed and reference `sigma*A = G_reference*L_reference` is
  frozen. Stored radius/material fields do not supply conductivity or area.

Coordinates and arithmetic use binary64. Nonrepresentable integer input,
nonfinite coordinates, collapsed chords, invalid frozen products, or invalid
conductance arithmetic are rejected, not clamped or silently repaired.
Translation does not recompute internal geometry, electrical electrodes, boundary
flags, box sizes, periodic cells, temperature-dependent resistivity, or mechanics.

An electrical solve remains a separate explicit operation, with separately
chosen electrical boundary conditions:

```python
from fibernet.sim.linear_transport import solve_linear_transport

# This is a separate physics step, not performed by preparation.
electrical = solve_linear_transport(
    derived_graph, conductance_S, explicit_dirichlet_voltage_by_node_id,
    length_unit="m", model_metadata=binding,
)
```

## Admitted domain

The first bounded slice intentionally accepts only exact ordinary
`StructureGraph`, `SNode`, `SEdge`, and `Material` records, rather than all custom
objects accepted elsewhere in the library. It requires:

- Dimension 2, nonempty nodes and edges, finite three-component stored positions,
  and one exactly constant z plane, including a nonzero plane.
- Complete ordinary Python dictionaries with stable nonnegative Python int64
  keys for translations and reference conductances. Bool, string, float, foreign,
  missing, and pair-list ID inputs are rejected. A dictionary cannot represent
  duplicate keys, and this API offers no separate JSON/pair-list loader.
- Two finite real non-bool translation components in a list, tuple, or ordinary
  numeric NumPy array; positive finite reference conductance for every physical
  edge, including each parallel member.
- Explicit graph `length_unit` of `m`, `mm`, or `um`; positive radii, tolerance,
  and segment counts; valid counters and endpoint identities; six Boolean
  boundary flags; no nonempty or malformed internal points. `None` and numeric
  shape `(0, 3)` are retained distinctly.
- Finite JSON-native graph/node/edge metadata and material data, valid ordinary
  material fields, and a bounded 6-by-6 material stiffness tensor when present.
- Valid exactly-once spatial membership and valid edge-lookup representatives.
  The source is never repaired as a side effect.

### Unit and frame declarations

The existing `fiber_network_profile._units` metadata scope is retained: root
geometry declarations and declarations directly inside any exact `geometry`
dictionary reached by its bounded recursive walk are recognized in graph/node/
edge metadata. Conflicting length declarations or conflicting/blank/nonstring
frame labels reject preparation. Plain nested load, stress, or calibration unit
and frame labels are not geometry declarations; a nested explicit `geometry`
dictionary is. `Material.extra` is preserved but not interpreted by this rule.

One consistent stored geometry frame label is retained, or the report says
`coordinate_frame: null` when none is declared. There is no extra frame argument,
no invented "cartesian" label, no frame transformation, and no frame authenticity
certification. The report always says `translation_frame: source_graph_xy_axes`.

## Admission bounds

Optional positive Python-int keyword limits can narrow, but cannot exceed:

| Keyword | Fixed ceiling |
| --- | ---: |
| `max_nodes` | 4096 |
| `max_edges` | 16384 |
| `max_metadata_bytes` | 4 MiB |
| `max_report_bytes` | 16 MiB |

The aggregate metadata/material budget counts every logical occurrence, including
material data shared across edges. It runs before full graph serialization,
copying, or hashing. Existing finite-JSON byte/depth/cycle validation is reused.
The spatial hash admits at most `4 * max_nodes` buckets, including empty buckets,
and exactly one member per node. A valid long-lived graph with too many empty
buckets can exceed this explicit payload limit. The source is not silently
rebuilt to get past it. These are payload-admission limits, not measured
peak-memory bounds or physical applicability claims.

## Ownership and identity

The source is unchanged. The derived graph is produced by the existing
`copy()` and `set_node_positions()` methods. All serialized fields except node
positions are preserved, including sparse IDs, insertion orders, oriented
endpoints, parallel edges, materials, metadata, radii, empty curves, segments,
reference box, boundary flags, and next-ID counters. Spatial lookup is consistent
with derived coordinates, though copying may discard old empty lookup buckets.

The report includes full source/derived graph hashes, separate graph-plus-order
envelope hashes, a preparation-input hash, stable sorted ID rows, oriented
endpoints, translations, both chord lengths and conductances, frozen products,
reference box, derived AABB, fixed policies, and admission declarations. Hashes
use complete schema-3.1 records, not the rounded/truncated `fingerprint()`.
They establish equality relative to retained records, not authenticity or physics.

The three outputs are detached from inputs and from each other's mutable data.
Their binding scope is preparation time only. Subsequent mutation invalidates
that recorded correspondence; there is no automatic check before an unrelated
solver call. Prepare again if inputs or output geometry change.

## Evidence and limits

Focused tests exercise analytical sparse/reordered-ID series/parallel controls,
units, translations, preservation/ownership, lookup identity, rejection behavior,
and provider tripwires. A compact retained 25-node/56-edge fixture compares the
actual public preparation to retained full graph/order hashes and an independent
same-host frozen-law arithmetic reference. Historical lengths, conductances, and
the prior electrical input identity are verified separately without changing
their bytes. Both arithmetic records satisfy input-derived rational accuracy
bounds; candidate values must additionally equal the host reference exactly.
The current-host electrical input digest is not claimed to be the historical
physical input digest across Python versions. It does not rerun the
three retained electrical solves or any thermal/mechanical solver.
The retained-output identity chain is pinned review → terminal hash → artifact
manifest → preparation/deformed-output bytes.

This slice does not qualify arbitrary mechanical results, curved conductors, 3D
geometry, reciprocal coupling, stability, material calibration, physical
accuracy, or production-scale performance. A later saved-mechanics reader or
broader domain needs its own explicit contract and independent review.
