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

## Prepare a 3D straight-member network

Use the explicit `prepare_deformed_transport_3d` entry point when the source
`StructureGraph` declares dimension 3. Supply three translations in metres per
stable node ID in the graph's x/y/z axes, even if one component is zero. The
planar function and its report remain unchanged.

```python
from fibernet.sim.deformed_transport import prepare_deformed_transport_3d
from fibernet.sim.linear_transport import solve_linear_transport

deformed_graph, conductance_S, binding = prepare_deformed_transport_3d(
    graph_3d,
    translations_m_by_id=translations_xyz_m_by_id,
    reference_conductance_S=reference_G_by_edge_id,
    length_unit="m",
)
electrical = solve_linear_transport(
    deformed_graph, conductance_S, electrode_voltages_by_id,
    length_unit="m", model_metadata=binding,
)
```

The XYZ report records the actual new three-dimensional chord lengths, the
preserved reference area–conductivity product, and the matching graph and ID
bindings. Nodes, rods, material data and electrode IDs are not rebuilt. The
preparation function does not read or authenticate mechanical results: extract
translations by matching the recorded stable node IDs and convert their units
to metres before the call. This path uses straight members and fixed effective
area; it does not model changing contacts, conductivity, or electrical feedback
onto the mechanical solution.

## Runnable sequential workflow

See [saved temperature through geometry export](THERMAL_TO_TRANSPORT_EXPORT.md)
for a complete example using existing APIs. It extracts SI translations by node
ID from saved-temperature mechanics, prepares conductance, solves the circuit,
and writes identity-preserving VTU and geometry-only STL.

## Run a complete 3D beam-network comparison

From a repository checkout with FiberNet installed, run:

```bash
python examples/mechanoelectric_network_3d.py --strain 0.005
```

The example stretches a 6 mm network of 91 nodes and 360 welded members
in the x direction using the linear 3D beam solver. Its interior nodes
are perturbed deterministically. Individual rods have one of two radii,
moduli and electrical conductivities. Each x, y and z measurement places
a separate 1 V difference across a pair of originally opposite faces.
Electrode membership follows the original stable node IDs during deformation.

The example maps the solver's deformed Cartesian positions
by stable node ID, then calls the existing resistor-network solver on a
separate, deformed graph. For each rod, the original conductance is
`G0 = sigma * pi * radius**2 / L0`. The two geometric assumptions give:

- Fixed cross-sectional area: `G = G0 * L0 / L`.
- Fixed member volume: `G = G0 * (L0 / L)**2`.

At 0.5% tensile strain, fixed area gives approximate resistance increases of
0.2652%, 0.0300% and 0.0325% on the x, y and z axes. The volume-preserving
rule gives 0.5306%, 0.0598% and 0.0649%. Thus both assumptions give a stronger
axial response in this example, while the assumed area evolution changes its
magnitude substantially.

The sample parameters are illustrative. This is one-way, small-deformation
beam mechanics with straight rods and fixed electrical topology, not a solid
model of the welded joints. The area laws are imposed hypotheses, not
inferred transverse strain; the electrical resistivity of each rod is fixed,
with no contact evolution, Joule heating, or electrical feedback on the
mechanics. The planar preparation API above is unchanged.

### Inspect signed rod contributions at fixed area

Run the same example with `--compare-affine` to add the
`fixed_area_attribution` section to its JSON output. The default output
is unchanged. The extra section compares the beam solution with a control
that moves each x coordinate in proportion to its distance from the fixed
grip, leaving y and z unchanged. Both states keep the same grip
displacements, cross-sections, conductivities, and electrode IDs.

For each measurement direction, normalize the two signed rod currents to
unit total terminal current. The contribution of a rod is its change in
resistance multiplied by the product of its two normalized currents,
with the same endpoint orientation in both solves. The sum is checked
against the direct terminal resistance difference and against bounds from
the two fixed current fields. The arrays follow the corresponding stable
`edge_ids` list; each comparison reports its positive and negative sums
separately.

For the example at 0.5% axial strain, the y-direction resistance increases
by about 0.0932% for the control and 0.0300% for the beam solution. In the
beam solution, contributions of about +2.094 and −0.995 Ω leave a net
+1.099 Ω. This cancellation describes this geometry and loading, not an
intrinsic piezoresistive law.

### Keep the network and change material scale or load control

```bash
python examples/mechanoelectric_material_controls_3d.py
```

On the same 360-member network, double every member's Young's modulus at
unchanged Poisson ratio and electrical conductivity. First prescribe the
same right-grip displacement with the left grip clamped. In this linear
beam model, the displacements and geometry-derived resistance do not
change under that uniform stiffness rescaling; reactions double.

Then remove the right displacement constraints and apply the three
components of their original mechanical reactions as force loads, with
the left grip still clamped. The original material reproduces the
prescribed-displacement solution. Doubling the moduli at the same force
halves those displacements. The reading along x changes by about 0.2652%
at the original moduli and 0.1326% at doubled moduli under this force
control.

The example uses zero numerical regularization for the uniform scaling
identity on this particular restrained network and checks the physical
free-equation residual. At fixed geometry, an independent electrical
control doubles all conductivities and halves all three electrical
resistances. These controls separate stiffness and conductivity changes;
they do not model strain-dependent intrinsic conductivity.

## Place electrical conductivity at fixed mechanical deformation

From the source checkout, run a matched two-conductivity comparison:

```bash
python examples/mechanoelectric_conductivity_placement_3d.py --phase 0
```

The same solved 91-node, 360-member geometry is used for all electrical cases in
each phase. The comparison keeps the amount of 1 and 2 S/m material fixed
within groups of equal undeformed rod vector (ignoring sign) and radius.
The two ranked placements use the change in normalized per-rod electrical
power share between the reference and deformed states. These rankings guide
the discrete comparison; they do not guarantee a global optimum. A third
placement based on stable member IDs provides a control.

In the JSON, check `unloaded_ohm` separately from `change_percent`. At 0.5%
prescribed extension, phase 0 gives about +0.020%, +0.034%, and +0.048% in
transverse resistance under the fixed-area rule. The ordering is unchanged
when the electrical law instead conserves each member's nominal volume.
`--phase 1`, `--phase 2`, and `--phase 3` repeat the comparison with three
related predetermined distortions. All cases deliberately keep the same
mechanics within a phase; no intrinsic piezoresistivity, contact switching,
or joint conductivity–stiffness material model is assumed.
