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

`extract(positions, edges)` remains a dense-row API. `compute_features` and all
numerical feature algorithms are unchanged. No general graph normalization,
reordering, duplicate removal or new validation policy is introduced.

## Preserved behavior

- Parallel edges are independent structural observations, even with mixed radii
- Snapshot calculations use straight endpoint segments; graph edge radii and
  internal points are not added to the analysis. Optional finite-width contacts
  continue to use the explicit `ContactConfig`
- Three-dimensional coordinate arrays are projected to XY only after the
  existing constant-Z planarity check (tolerance 1e-9)
- Empty graphs and isolated nodes retain existing feature behavior
- Graph-size budgets are enforced before rectangle filtering
- Rectangle validation, midpoint selection, output key order, histogram order,
  exceptions and contact configuration retain their existing contracts
- Source graph dictionaries, node coordinates and edge records are not mutated

## Regression evidence

`tests/test_snapshot_graph_identity.py` covers dense/sparse IDs, reversed node
record order, non-sorted edge IDs, mixed-radius parallel edges, complete scalar
and histogram output equality, contact on/off, four region choices, generic
fallback, raw sparse-ID rejection, empty/isolated graphs, budget enforcement,
planarity, rectangle validation and input immutability.
