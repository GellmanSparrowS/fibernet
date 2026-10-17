# Explicit geometry and selected paths, schema 1

```python
from fibernet.analysis import analyze_geometry_paths
result = analyze_geometry_paths(
    graph, length_unit="mm", coordinate_frame="lab",
    node_pairs=[(10, 30), (30, 10)], distance_mode="polyline")
assert result.schema == "fibernet.geometry_paths/1"
```

`GeometryPathMetricsV1` is an immutable typed result for a finite undirected
`StructureGraph`, with original nonnegative integer IDs. It is not a mechanical
response, periodic quotient, or replacement feature vector for a trained model.
Caller-provided `length_unit` and `coordinate_frame` are required nonempty
strings. They are recorded without conversion, recognition or certification.

## Geometry and identity

`node_ids` is sorted. Edge-aligned tuples follow original insertion order in
`edge_ids`, retaining every parallel physical edge. Node/edge keys, record IDs
and endpoints must be nonnegative integers, excluding booleans and floats.
NumPy integer scalars are accepted and output IDs are Python integers.

- `chord_length_by_edge`: full XYZ Euclidean endpoint distance
- `polyline_length_by_edge`: sum of consecutive segment lengths along endpoint
  i, ordered internal points, endpoint j. This measures the discrete polyline,
  not an unknown continuous curve. Missing internal points means a straight edge
- `curve_straightness_by_edge`: chord divided by polyline length, dimensionless;
  `None` if polyline length is exactly zero. A positive-length curve with
  coincident endpoints has straightness zero
- `degenerate_edge_ids`: edges whose endpoint chord is exactly zero, including
  positive-length curves with coincident endpoints

No absolute epsilon is used to define degeneracy. Full XYZ positions must have
shape `(3,)`; internal points, if provided, must have shape `(M, 3)`. Masked,
nonfinite, nonnumeric or malformed arrays are rejected rather than filled or
projected. Every node and edge is checked, not only selected routes. Self-loops
are rejected; distinct nodes with coincident coordinates are allowed. The input
graph is not modified. Reversing an edge record requires reversing its internal
points to describe the same curve. The resulting path graph remains undirected.

## Selected paths and undefined values

`pair_node_ids` preserves caller order and duplicates. IDs are never interpreted
as dense row offsets. `distance_mode` must explicitly be `polyline` (default)
or `chord`. NetworkX Dijkstra uses those computed lengths, with every parallel
edge available; missing weights never fall back to hops. No radius, material,
metadata weight, boundary flag or box size changes costs or connectivity.

`pair_distance` and `pair_chord` use the declared coordinate length unit.
`pair_tortuosity` is selected weighted distance divided by endpoint chord,
dimensionless. There is no implicit all-pairs sampling or averaging. This version returns
pair distances/statuses and the complete edge inventory, not the selected
route's node sequence or physical-edge sequence.

Status precedence is:
1. `unreachable`: no route; distance and tortuosity are `None`, even for
   disconnected coincident nodes. Pair chord remains a geometric distance
2. `zero_chord`: route exists but endpoint chord is exactly zero; distance is
   retained and tortuosity is `None`. Self-pairs have distance zero and this status
3. `ok`: route exists and chord is nonzero; distance and ratio are defined

Lengths use stable `hypot` norms and `fsum` accumulation, avoiding square-based
underflow at scales such as 1e-200 and overflow at 1e200. Shortest-route selection
still uses ordinary finite-precision NetworkX Dijkstra arithmetic; exact
real-number ordering of nearly tied routes is not certified. The selected
route's edge costs are summed stably. Coordinate conversion rejects values
outside finite float64 range or nonzero values converted to zero; ordinary
float64 rounding is allowed, so not every loss of precision is rejected.
Nonrepresentable coordinate differences, lengths, sums or ratios raise
`OverflowError`; numerical
failure is never encoded as a valid zero distance or unreachable status.

## Bounded work and errors

Positive integer budgets exclude bool and float values; NumPy integers are
accepted. Defaults are `max_pairs=256`, `max_nodes=10000`, `max_edges=20000`,
`max_points=100000`, `max_path_work=1000000`. Points count nodes plus stored
internal points. Path work is conservatively charged as
`number_of_pairs * (number_of_nodes + number_of_physical_edges)`, counting repeated
pairs. This bounds traversal scale, not wall-clock time or exact Dijkstra heap
operations. Exceeding any budget raises `MemoryError` before any path query.
Invalid values/shapes/IDs raise `ValueError`; a non-StructureGraph input raises
`TypeError`. Budgets are configurable but callers own the cost of raising them.

## Compatibility

Legacy 94/60-dimensional graph features, snapshot metrics, GraphBatch and saved
model schemas are unchanged. This module has no dependency on the ML package.
Names distinguish chord and discrete polyline length explicitly. Using these
results in ML requires a separately versioned feature schema with defined
column ordering, units, undefined-value handling and training/inference checks;
never substitute corrected values into an old model's existing feature names.
