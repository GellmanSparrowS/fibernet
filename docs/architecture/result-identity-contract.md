# Result identity and renderer boundary

This contract describes implemented row-identity metadata and rendering alignment.

## Historical failure and implemented repair

Before identity alignment, a 3D two-node bar loaded with reversed node records
was solved in insertion order but rendered against sorted IDs, drawing the fixed
node moving by .01. The implemented repair preserves backend ordering, publishes
row identities and aligns the consumer without globally sorting numerical inputs.

### Producer contract

`SimResult.metadata['node_ids']` is a list of original graph node IDs in the exact
row order of displacements, deformed_positions and every positions_trajectory
frame. `metadata['edge_ids']` names rows of edge_forces and edge_stretches.

- Spring dynamics already obtains node IDs from `_graph_to_arrays`; publish those
  IDs and graph edge insertion-order IDs
- FEM `to_sim_result(raw, graph)` obtains node order from the same helper. Edge
  IDs are graph edge IDs selected by raw `edge_list`, including legacy dedup
- With graph=None, the adapter cannot infer original IDs. Omit maps explicitly;
  never invent graph IDs from row numbers
- Preserve maps through summary/full JSON save/load. Current metadata flattening
  can carry these names without changing SimResult constructor compatibility
- No change to force signs, coordinates, solver ordering, BC indices, convergence,
  constitutive parameters or energy values

### Consumer contract

A shared private rendering helper returns an aligned copy, not a mutation of the
caller's result or graph. For 3D consumers, target node rows are sorted graph IDs;
edge rows follow graph edge insertion order. Validate IDs before constructing
figures: integer/nonnegative, no bools, unique, correct length and graph coverage.
Validate every populated array against its named axis. Do not silently truncate,
fall back to node 0, or nearest-match geometry after an ID mismatch.

Mapped results with full node/edge coverage are reorderable. If a result contains
only a subset of edge fields (e.g. legacy dedup), full-graph stress rendering must
raise a clear coverage error until an explicit partial-field policy exists. It
must not pad a physically absent result with zero. Node-only visualization should
not require full edge coverage when it does not consume edge fields.

For old results without maps, retain each renderer's documented historical row
order. Warn/document that unknown historical order cannot be reconstructed from
coordinates. Absence of IDs is backward compatibility, not identity verification.

### Explicit-array APIs

`render_trajectory(original, positions_trajectory, edge_stretches, ...)` accepts
optional keyword-only `node_ids` and `edge_ids` and resolves IDs to dense rows.
Without maps, coordinate rows use sorted node IDs and fields use graph edge
insertion order. Edited sparse graphs are supported through this mapping.

`render_deformation(original, deformed, displacement_data=...)` accepts optional
`node_ids`. Without that map, displacement coloring retains the historical
insertion-order assumption; it does not silently reinterpret legacy arrays in
sorted order.

`render_trajectory_3d` accepts `node_ids` for an explicitly supplied trajectory.
Passing `node_ids` without an explicit trajectory raises ValueError. Explicit
trajectory arrays do not silently inherit a SimResult map; callers must ensure
the documented row order or supply their own matching IDs.

## Bounded acceptance cases

1. Sparse/reversed node IDs, 2D FEM and 3D FEM: fixed node remains fixed in the
   actual graph passed to rendering helper; loaded result behaves identically
2. Permuted node map with valid arrays: all rendered coordinates and node colors
   align; source graph/result remain byte-for-byte unchanged
3. Distinct parallel fibers and reordered edge IDs: force colors attach to the
   right physical edge; differing radii make the fixture distinguishable
4. Duplicated, missing, foreign, boolean or wrong-length maps: clear ValueError,
   never plausible-looking output
5. Metadata-free historical contiguous results: existing figures still work
6. Explicit 2D sparse trajectory: no index error; correct segments and colors
7. FEM legacy dedup: node-only drawing works, full-graph missing edge field is
   reported rather than zero-filled
8. Spring producer IDs: test metadata without mocking the numerical solver as
   successful; execute kernels only if the optional validation environment is
   ready. Pure rendering tests may use hand-constructed SimResult fixtures clearly
   labeled synthetic

## Direct-array provenance

For a raw solve produced in a different row order, supply explicit input IDs as
explained in [raw FEM array identity](raw-array-identity-migration.md). A graph
alone cannot prove the provenance of arbitrary low-level arrays. Metadata-free
historical results retain documented ordering assumptions.

Known FEM diagnostic metadata follows the same entity permutations in the
private aligned rendering view. Metadata unrelated to entity axes is unchanged.
See `tests/test_result_identity_rendering.py` for regression coverage.

## Detailed geometric metrics

`result.compute_detailed(graph, stiffness=None)` aligns node rows by the
result's `node_ids` and selected physical edge rows by `edge_ids`. Node maps
cover every graph node; edge maps may select an ordered subset. Native signed
integer IDs are supported. Without maps, historical results retain graph
insertion order.

Every node record and selected edge record must carry the strict integer ID
used as its graph key. When deformed positions are supplied, selected edges
must have distinct reference endpoints. This exact-coincidence check adds no
minimum edge-length threshold and does not inspect unselected edges. Invalid
identity, shape, or nonfinite metric arithmetic raises `ValueError` before any
result field is changed.

With `stiffness=None`, geometric metrics are refreshed while force, energy and
coefficient provenance are retained. Explicit spring coefficients follow the
selected edge order. FEM solver channels and backend-specific result contracts
retain their own recomputation rules.
