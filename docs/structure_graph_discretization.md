# Filling missing edge samples

`StructureGraph.discretize_edges(n_points_per_edge=8)` returns an independent
copy of the graph.

- An edge whose `internal_points` is `None` or empty receives the requested
  equally spaced samples on its endpoint chord.
- An edge with existing nonempty `internal_points` keeps every stored sample,
  in the same order. This includes collinear, duplicate and backtracking
  samples. Its point count need not equal the requested count.
- `n_points_per_edge=0` adds no samples to unsampled edges; it does not erase
  existing stored geometry.

For example, calling `graph.discretize_edges(0)` on a graph containing a sampled
curve retains that curve. Calling the default method on a mixed graph retains
the sampled curves and adds eight points to each previously unsampled edge.

Stored samples are absolute coordinates. Updating physical node positions does
not transport or interpolate existing internal points. Filling keeps those stored
coordinates; only previously unsampled edges use their current endpoints.

The result preserves physical node and edge IDs, parallel members, endpoint
orientation, positions, materials, metadata, boundary flags and `segments`.
Its arrays, materials and nested records are detached from the input.

This operation does not add physical graph nodes or edges, build a finite
element mesh, or infer an interior displacement field. Existing simulator
geometry policies still apply: nonempty edge-owned samples remain nonempty
geometry, including samples added to straight edges. Filling samples does not
establish curved-beam mechanical equivalence.

Use a deliberately authored representation when changing geometry. This method
is not a curve simplifier, fixed-count resampler, or curve-to-chord conversion.

