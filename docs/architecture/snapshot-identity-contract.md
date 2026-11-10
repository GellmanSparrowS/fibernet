# Snapshot graph identity contract

`SnapshotFeatureExtractor.extract_graph` is the graph-to-array boundary for
snapshot structure, planar-pore and optional contact features.

## StructureGraph identity

For a `StructureGraph` (including subclasses), the adapter uses its `to_numpy()`
export. Position rows are copied in sorted node-ID order, and each edge endpoint
is mapped from its stored node ID to that dense row index. Edge rows retain graph
edge insertion order, endpoint direction and parallel-edge multiplicity. Neither
nodes nor edges are renumbered in the source graph. Sparse IDs such as 10, 30 and
70 therefore describe the same geometry as equivalent dense IDs 0, 1 and 2.

The type import is local to the adapter to avoid imposing a core dependency at
analysis module import time. The graph already owns the identity mapping; the
analysis layer does not duplicate that mapping or infer IDs from coordinates.

## Legacy and raw-array boundaries

Other graph-like objects continue to expose `node_positions()` and `edge_array()`.
Their edge endpoints must be zero-based row indices into their position array.
The adapter does not inspect a `nodes` mapping or call a generic `to_numpy()` to
guess a different identity contract. Sparse arbitrary endpoint IDs are rejected
by the existing array validation.

`extract(positions, edges)` remains a dense-row API. The identity adapter does
not perform general graph normalization, reordering or duplicate removal.
The separate open-chain straightness correction includes both boundary nodes
in chord / full path length and excludes closed walks. Individual edges and
paths at or below 1e-12 in caller units remain excluded; an empty distribution
has mean 0.0. Pore topology is calculated after translating and normalizing
coordinates by the larger XY span of active edge endpoints, then pore areas
are returned in caller units squared and perimeters in caller units. Nested
disconnected components contribute inner face boundaries: areas exclude holes,
perimeters include both sides of interior branches, and shared edges determine
pore adjacency. Approximate
tolerances remain;
structure/contact calculations, including the 1e-12 straightness cutoff,
are not normalized. See [pore units and scale](../SNAPSHOT_PORE_SCALING.md).

## Preserved behavior

- Parallel edges are independent structural observations, even with mixed radii
- Snapshot calculations use straight endpoint segments. Nonempty edge-owned
  `internal_points` are rejected rather than discarded. Explicitly discretizing
  paths into graph nodes and straight edges changes node/edge counts. Graph edge
  radii are not added to the analysis; optional finite-width contacts continue
  to use the explicit `ContactConfig`
- Three-dimensional coordinate arrays are projected to XY only after validating
  every source coordinate as finite and checking constant Z (tolerance 1e-9)
- Empty graphs and graphs with no active edges retain existing feature behavior
- Graph-size budgets are enforced before rectangle filtering
- Rectangle validation, midpoint selection, output key order, histogram order,
  and contact configuration retain their existing contracts. The pore pipeline
  additionally rejects unrepresentable normalized geometry, region area, area
  scale or denormalized pore measurements with `ValueError`
- Source graph dictionaries, node coordinates and edge records are not mutated

## Regression evidence

`tests/test_snapshot_graph_identity.py` covers dense/sparse IDs, reversed node
record order, non-sorted edge IDs, mixed-radius parallel edges, complete scalar
and histogram output equality, contact on/off, four region choices, generic
fallback, raw sparse-ID rejection, empty/isolated graphs, budget enforcement,
planarity, rectangle validation and input immutability.
