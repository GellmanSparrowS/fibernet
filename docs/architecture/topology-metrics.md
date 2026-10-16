# Explicit physical topology, schema 1

```python
from fibernet.analysis import analyze_topology
result = analyze_topology(structure_graph)
assert result.schema == "fibernet.topology/1"
```

`TopologyMetricsV1` is a frozen typed result for a finite, undirected,
loop-free `StructureGraph`. It is not a feature vector or a mechanical
strength estimate. NetworkX supplies connectivity, bridges and articulation
algorithms; an explicitly constructed simple projection is used only internally.

## Scope and units

- `node_ids`: sorted original nonnegative integer IDs. All `*_by_node` tuples align with this
  tuple. `edge_ids` and `bridge_edge_ids` follow physical edge insertion order.
- `n_nodes`, `n_physical_edges`: counts including isolated nodes and every
  independent parallel edge. No fiber grouping beyond one `SEdge` is inferred.
- `n_unique_connections`: number of distinct unordered endpoint pairs.
- `degree_physical_by_node`: incident physical-edge count.
  `degree_unique_by_node`: number of distinct neighboring nodes.
- `n_components`, `n_isolates`: counts on all nodes. `component_by_node` labels
  each node with its component's current minimum ID; `component_sizes` contains
  sorted `(label, node_count)` pairs. Labels may change after edits and are not
  permanent component identities.
- `cycle_rank_physical`: `E - N + C`, the dimension of the undirected multigraph
  cycle space. Two parallel edges contribute one cycle. This is neither a
  count of all simple cycles nor a pore/face count.
- `bridge_edge_ids`: removing this single physical edge increases component
  count. A simple-projection bridge with multiplicity greater than one is
  excluded. `articulation_node_ids`: sorted IDs of nodes whose removal increases
  component count, including removal of their incident physical edges.
- `largest_component_fraction`: largest component's node count divided by all
  nodes. It is `None` for an empty graph, otherwise defined also for disconnected
  graphs. Empty graph counts and cycle rank are zero and tuples are empty.

Counts, degrees, cycle rank and fractions are dimensionless. No edge weights
are used. Radius, material, coordinate scale, internal curve points, endpoint
orientation and box size do not affect these metrics. Coordinates or materials
are not validated or certified by this topology-only function. Node and edge
record IDs must be nonnegative integers (not booleans or floats) and match their mapping keys; endpoints must exist. NumPy integer scalars are accepted when nonnegative
and returned IDs are normalized to Python integers; no additional integer
width limit is imposed. Self-loops are
rejected explicitly, including loops introduced through mutable edge records.

Only explicit adjacency is analyzed: geometric crossings do not create
junctions, coincident nodes are not merged, and periodic equivalence or image
shifts are not inferred from box sizes or boundary flags. A periodic quotient
or lifted graph needs a separately specified adapter before analysis.

## Legacy feature limits and migration

This API does not change `GraphFeatureExtractor`, `GraphFeatureExtractor3D`,
`SnapshotFeatureExtractor`, dataset features or saved-model inputs. In
particular it does not fix existing scientific limitations in legacy outputs:

- The two legacy graph extractors project `StructureGraph` to a simple graph,
  collapsing parallel edges. Their geometric lengths use endpoint chords,
  not internal-point contour lengths.
- The legacy 3D tortuosity path requests `length` weights that its own adapter
  does not populate; absent attributes default to hop counts. Dividing these
  by geometric distance is dimensionally invalid and scale-dependent.
- Legacy 3D k-core reporting can return zero because a missing local
  `mean_degree` value is caught as an exception. Its edge-betweenness Gini uses
  reversed ordering and can be negative. Disconnected graphs report zero
  `lambda_max` even when the Laplacian has positive eigenvalues.
- Legacy orientation entropy can depend on arbitrary undirected edge
  orientation. Spectrum and shortest-path routines do not have one consistent
  explicit weight policy. Zero fallbacks may represent unavailable results.

These outputs must not be interpreted as validated physical measurements.
Compatibility with old artifacts is not scientific validation. Existing
artifacts must retain their original feature definitions for reproducibility;
corrected numerical metrics require a new feature-schema version, named
columns with explicit scope/units/undefined policies, regeneration of training
features, retraining and validation, and schema checks at inference. Do not
silently substitute corrected values under an old artifact's feature names.
The topology result is not automatically integrated into that training or
artifact pipeline. Future distance, centrality, spectral and orientation
schemas must distinguish hop, chord and contour distances, edge multiplicity,
weight meanings and component selection explicitly.
