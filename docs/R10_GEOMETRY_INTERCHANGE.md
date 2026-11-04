# R10 source geometry: bounded in-library integration

This is the first integration boundary, **not the completed R10 solver merge**.
It uses the existing `StructureGraph`; there is no parallel graph package.
Import directly from `fibernet.io.r10_geometry`; existing facades and dependency
floors are unchanged. No optional mesher, compiler or solver is imported.

## API and units

`validate_r10_geometry(payload, *, length_unit)` returns detached finite JSON.
`import_r10_geometry(payload, *, edge_properties, length_unit)` returns a
`(StructureGraph, binding)` pair. `export_r10_geometry(graph, binding, *,
length_unit)` inverts an **unchanged import**, returning detached original R10
source JSON. All require explicit `length_unit="mm"`; no conversion is implied.

`edge_properties` maps each stable R10 edge ID to `radius` in mm, an explicit
FiberNet `Material` in its documented SI units (e.g. Young's modulus in Pa), and
positive integer `segments`. These are caller-supplied beam annotations, NOT
parameters inferred from or used by the fixed continuum material. The binding
retains their JSON records. Bare R10 JSON cannot encode those beam annotations:
retain the binding and reconstruct `Material.from_dict(record["material"])`
when reimporting. This is not a general lossless StructureGraph exporter.

```python
from fibernet.core.material import Material
from fibernet.io.r10_geometry import import_r10_geometry, export_r10_geometry

# `source` is an explicitly loaded R10 source JSON object. The caller chooses
# beam annotations; no automatic radius or elastic-material guess is made.
properties = {
    edge_id: {"radius": chosen_radius_mm,
              "material": chosen_material, "segments": chosen_segments}
    for edge_id in source["edge_ids"]
}
graph, binding = import_r10_geometry(
    source, edge_properties=properties, length_unit="mm")
original_source = export_r10_geometry(graph, binding, length_unit="mm")
```

Source schema is `FiberNet_original100_source_graph_material_v1`. Edges index
rows of `nodes_xy_mm`, NOT `node_ids`. Sparse IDs, source ordering, coincident
distinct nodes and parallel edges survive. Parent/source identity arrays and
unknown finite JSON fields are detached and retained. No outline is fabricated.
Node/edge IDs must be nonnegative integers below `2**63 - 1`, leaving space for
the next-ID counter; booleans and out-of-range IDs are rejected.
Parent/source identity entries are opaque JSON records with checked outer
edge-count alignment, not independently verified provenance identities. Grip
`native_node_ids`, if present, identify continuum mesh nodes, not StructureGraph
nodes; they are retained without being converted into graph boundary flags.

The supported associated profile is Y loading, 1 mm reference thickness, no
self-contact, C10=6 MPa, C01=3 MPa, D1=0.495 MPa^-1. Its thickness generalized
coordinate is not a third geometric coordinate. The adapter makes no claim of
plane stress, general 3D mechanics, or an equivalent linear elastic material.

Invalid schema, units, missing fields, nonfinite/boolean coordinates, malformed
identity/index arrays, zero-length edges, malformed rings/grips, incompatible
profile and missing beam annotations raise `ValueError`. Missing mandatory
keywords or a non-StructureGraph export argument raise `TypeError`. Export
rejects nonplanar/curved edges and any graph/source/annotation change. This
includes benign graph metadata edits: the first boundary is intentionally exact
and conservative. A changed graph needs explicit material-domain reconstruction
and reimport, not reused stale outlines or grip locations. Hashes are consistency
checks, not signatures or a security boundary against fabricated bindings.

## What validation does not establish

`geometry_only_not_solver_ready` means finite structural data and profile checks
only. Closed rings may still self-intersect or overlap; no polygon validity,
hole containment, material connectivity, source-to-outline registration, physical
grip membership, mesh quality or mechanics has been established. Caller-supplied
Material records are serialization-valid, not constitutively qualified.

## Next solver boundary (not implemented)

The eventual in-library optional backend should consume this StructureGraph and
its current associated geometry, then explicitly qualify fused-domain validity,
grips, source mapping, and nonoverlapping Q4/direct-T3/patch cell identities.
FE nodes/cells and field mappings are derived data, not another authoritative
Graph. Preserve original numerical kernel bytes with hashes and MIT notice where
possible. A lazy backend runner must own process-local thread/resource settings;
do not retain R10 driver's import-time environment mutation. Keep mesh preparation
(Shapely/Gmsh) separate from solver (NumPy/SciPy/Numba) and optional GUI/raster
preprocessing. No backend dependency is added by this batch.

Before a solver merge: run bounded synthetic constitutive, element, rigid-body,
Q4/T3, mixed-mesh and continuation parity tests; then obtain the missing original
states/verifier or regenerate an explicitly new comparison. Do not describe the
historical 168-state report as independently reproduced. Gmsh licensing and
optional-runtime compatibility require separate qualification; no bundled wheels
are redistributed here.

## Provenance and verification

Input release: FiberNet_Percolation_R10_20261031.zip, SHA-256
`e4d52b88ae84523a5f441f6e04f187a22c8a48030a729ee3ea68cba529998841`.
Package-reported upstream: `08679bdfa443ee91b4498f32a8a728baf49bf515`.
Integration base: `5a05c087629e6435aa1c3cc6bca6b5123bba8fce`.
Original package declares MIT, copyright 2026 FiberNet Contributors. This
adapter is newly authored from its data contract; no solver code is copied.
Original ZIP, local research paths, embedded images, raw receipts and benchmark
payloads remain private and unchanged. Only synthetic fixtures are included in
`tests/test_r10_geometry.py`; they contain no historical experimental evidence.
