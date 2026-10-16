# Voronoi bridge ownership optimization

`pattern_2d(unit="voronoi", ...)` keeps the same generated coordinates, stable IDs,
member ordering, topology, seed handling and attributes. No new API is introduced.
The freshly tiled graph can avoid a second full graph copy when its material is
internally created and the actual factory called is the built-in Voronoi factory.
Nested graph, node and edge metadata are still independently deep-copied before
bridging. Tiling already owns node/internal-point arrays and separately copied
edge materials. Bridge traversal, ties, component order and randomness are unchanged.

Explicit `material` values, including subclasses/custom deepcopy behavior, and
any `unit_kwargs` material key (even `None`) retain the original full-copy path.
A factory registered through `register_unit("voronoi", custom_factory)` also
retains that path. The factory used is captured before invocation. Other patterns
and custom point inputs are unchanged. The existing case-sensitive bridge condition
`unit == "voronoi"` is intentionally preserved, even though factory lookup uses
lowercase. This change does not promise behavior under arbitrary monkeypatching
of internal tiling, transformation or material constructors.

The general private `_bridge_small_components` helper still returns a different,
deep-copied graph, including for empty and already-connected input. The owned helper
is internal only: callers must not use it for graphs whose ownership is unknown.
No nearest-neighbor, BFS queue, statistical-distribution or geometry algorithm changed.

## Evidence and limits

Three prior fixed small profiles in one CPU environment found full graph copying
inside bridging to be a hotspot. An isolated candidate had median wall times
0.02535/0.08765/0.21122 s against saved baseline 0.03335/0.11784/0.37688 s for
20/80/200 seed fixtures. These were separate batches, with timing noise; they are
small-scale signals, not a universal speedup guarantee. The original profile
and isolated candidate are preserved separately from this implementation stage.

Before editing production code, 13 additional small option-branch outputs were
frozen from the old source. Regression checks compare complete serialized graphs
plus insertion order: default/explicit seeds, mirrors, rotation, boundary modes,
internal/intermediate points, factory kwargs overrides, mixed-case unit spelling
and custom points. Separate tests cover registered factories, explicit Material
subclasses and deepcopy calls, mutable alias topology and mutation propagation,
and general-helper empty/connected/small/parallel graph semantics.
