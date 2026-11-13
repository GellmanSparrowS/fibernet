# L3 saved-frame result projection

`fibernet.sim.l3_result` is a bounded, provider-free result family for externally
supplied saved-frame bytes. It consumes the existing authoritative
`StructureGraph` and its unchanged public `L3Binding`. It does not provide a
filesystem/archive reader, execute a provider, qualify a runtime, or validate
physical correctness. The preparation API remains separate.

## Entry points

Import directly from `fibernet.sim.l3_result`:

- `project_l3_result(graph, binding, *, frame_view_bytes, expected_view_sha256)`
- `load_l3_projection(graph, binding, *, projection_bytes, expected_projection_sha256)`
- `l3_render_inputs(graph, binding, projection)`
- `L3ResultProjection`, `L3RenderInputs`, `L3ProjectionError`

`L3ProjectionError` is an `L3InputError` subclass. The result is not a `SimResult`.
Calling its `compute_detailed` method raises an error rather than recomputing
spring stretch, force or energy. Existing `SimResult.load` rejects the dedicated
`fibernet.l3-result-projection/1` export schema.

The byte factory accepts `fibernet.l3-frame-view/1`. It also explicitly accepts
the existing `private.l3.saved-frame-view/1` byte profile with identical retained
field requirements. Supporting that schema name does not provide its producer,
raw archive reader, private worker, provider or source-data package.

The caller is responsible for supplying the original graph/binding and obtaining
the expected digest from a separately trusted receipt. Calculating a digest from
unknown bytes and immediately trusting it establishes no provenance. Neither a
manually constructed result object nor its own digest authenticates a source.
The loader and render-input factory revalidate the bounded record before use.

## Frame contract

Node and edge IDs are explicit, unique, nonnegative integers and must exactly
cover the bound physical graph. They identify array row order; sparse IDs,
permuted rows and parallel edges are not renumbered. Endpoint rows preserve
`node_i`/`node_j` orientation. Reference coordinates and the public graph/material
binding are rechecked, including the original coordinate/radius unit scales.

The exact nine retained fields are:

| Field | Shape | Unit / meaning |
| --- | --- | --- |
| displacements_m | N × 2 | planar x/y translation, m |
| positions_m | N × 2 | planar saved position, m |
| rotations_rad | N | rotation kept separately, rad |
| active | M | literal boolean segment activity |
| axial_strain | M | dimensionless axial strain, not stretch ratio |
| axial_force_N | M | saved signed axial force, N |
| end_moments_Nm | M × 2 | moments at node_i/node_j, N m |
| max_tensile_strain | M | dimensionless saved maximum tensile strain |
| element_energy_J | M | saved element energy, J |

`field_units`, `field_rows` and `field_components` explicitly label these axes.
`reference_positions_m`, `edge_node_ids`, `original_units`, active/failed ID maps,
frame number and observation channels are mandatory. Observation contains step,
phase, u_m, reaction_N, internal_energy_J, external_work_J,
equilibrium_jump_energy_J and energy_balance_residual_J. The bounded phase
identifier is retained, not interpreted as physical acceptance.

No force, energy, stress, material default or damage threshold is inferred.
Positions must agree with reference plus translation. Initially inactive edges
cannot reactivate; failed IDs and new-failure IDs must agree with the saved mask.
A single-frame record does not prove cross-frame chronology, force equilibrium,
stability or convergence. Native gradient is explicitly omitted; this is not a
lossless native archive or a restart checkpoint.

## Historical status and export boundary

The restricted profile preserves historical native `succeeded` together with
historical outer `SUCCEEDED` or `FAILED_WORKER`. These are source assertions,
not a new execution receipt. `new_execution` and `provider_license_verified`
must remain false: the projection makes neither claim. This does not decide
whether a provider is licensed; that requires separate evidence.

The projection exports an exact allowlist and source-view/binding/graph hashes.
It does not copy arbitrary source provenance, provider seal content, graph
metadata, source paths or unknown description fields. It retains numerical
results, stable IDs and phase labels, so it is not an anonymizer or permission
to publish a dataset. Display graph copies retain graph metadata and are also
not privacy-cleaned publication objects.

## Resource and read-only limits

The profile is limited to 8 MiB JSON bytes, 100,000 JSON values, depth 32,
4,096 nodes and 8,192 edges. Numeric values must be finite and integer values
must fit int64. Duplicate keys, malformed Unicode, cycles/aliases in graph
metadata, incorrect field types and unsupported shapes are rejected. Graph
provenance is bounded before public rebinding, and graph box/counter fields have
additional small-shape/integer checks.

These are format/size checks, not hard RSS/CPU limits or malicious-Python-object
isolation. JSON parsing allocates bounded input structures before member/depth
checking. The graph/binding are trusted, normally constructed in-process objects
and must not be mutated concurrently. The adapter contains no file opener,
archive extractor, process launcher, network client, pickle or provider import.

## Existing 2D consumers

`l3_render_inputs` returns detached graph copies, immutable byte-backed numerical
arrays, explicit row IDs and keyword arguments for `render_deformation` and
`render_graph`. The reference copy preserves the entire physical graph. The
deformed copy contains every node and only active edges, with atomic public
position updates maintaining its coordinate lookup. Force values are reordered
by stable edge ID into the active graph's actual insertion order.

Display translation/position is converted back to the graph's original length
unit; saved force, rotation, moment and energy values are not reinterpreted.
The display graphs themselves are mutable copies, not authoritative physical
models. Rotations are not a third displacement component and do not create a
beam-bending shape. Empty active graphs use uniform rather than empty custom
color arrays.

The existing renderer normalizes color independently per figure. This entry
point does not provide an absolute color scale, cross-frame quantitative color
comparison, generic `SimResult` alignment, 3D, trajectory, NPZ or VTK export.

## Synthetic tests and source scope

`tests/test_l3_result_projection.py` runs a separate synthetic-only contract
process and checks Python 3.9 grammar/provider-free imports. The child fixture
constructs its own small graph and field values. It blocks provider/task imports,
process/network operations and solver/postprocessing calls; actual 2D rendering
consumers are exercised with Agg and the no-fontconfig fallback. Its import and
audit configuration does not leak into the rest of the pytest session.

No provider implementation, private runtime, original archive, research fixture
or historical private path is included. This module uses public FiberNet APIs
under the repository's existing license; that license does not establish any
external provider's redistribution or execution rights.
