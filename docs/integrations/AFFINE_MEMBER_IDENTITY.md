# Affine transforms preserve physical member identity

`fibernet.core.transforms.translate`, `rotate`, `mirror` (including x/y/z
shortcuts), and `scale` return independent StructureGraph geometry copies.
They preserve node and edge IDs, dictionary insertion order, allocation counters,
each member's endpoint orientation, parallel members, radius/material/segments,
and node/edge/graph metadata. Coordinates of nodes and internal curve points are
transformed. They do not weld, merge, subdivide or reassign IDs.

This fixes the former common helper's use of sorted add_node/add_edge rebuilding:
sparse IDs were renumbered, and reversed parallel members were deduplicated. In
the saved two-member case, one material and member disappeared. Applications
that accidentally relied on transform-driven renumbering/deduplication need to
request a separate explicit topology operation; no such operation is introduced
here. Dense, normally ordered nonparallel fixtures retain their former complete
serialized values. Sparse-ID/parallel results intentionally change.

The helper now uses existing StructureGraph.copy per-object deep-copy semantics.
Node positions, internal points, graph/node/edge metadata and each edge's Material
are detached. Shared input material instances are independently deep-copied per
edge, as in StructureGraph.copy; cross-object alias topology is not promised.
The box array is explicitly detached because StructureGraph.copy itself currently
shares that private array. The previous transform helper already had detached
box ownership through the public getter. StructureGraph.copy is not changed by
this fix. Custom objects retain their own deepcopy behavior.

The spatial hash is rebuilt from transformed node coordinates in original
insertion order, so subsequent merge=True lookup finds transformed stable IDs.
Existing _edge_set lookup and counters are retained; this does not change the
existing add_edge deduplication rule for future insertions or tiling/welding.
Coincident nodes created by a zero scale remain distinct graph records.

## Existing geometric conventions retained

- Translation and rotation keep boundary flags. Rotation is not an automatic
  recomputation of axis-aligned box-face classifications.
- Mirror swaps the selected axis's min/max flags and reflects node/curve points.
- Only a scalar Python int/float scale multiplies radii by abs(factor). A vector
  scale, even an equal-component vector, keeps radii unchanged. This historical
  policy is not a general cross-section mechanics model.
- Scale updates box components by its existing scale vector, including signs.
  Other transforms retain box_size values. Box updates are not derived bounding
  boxes. No revised negative/zero scale policy is introduced.
- Coordinates use the input graph's units; no conversion or physics is inferred.
- Empty internal-point arrays retain their shape. None remains None.
- Native to_dict serializes IDs in sorted order; that serialization order is
  different from the graph's in-memory insertion order. Both are checked with
  their respective conventions.

The focused regression uses synthetic 2D/3D graphs and independent coordinate
formulae, metadata mutations, transformed-position lookup, counter increments,
parallel orientations and existing radius/box policies. Separate saved original
and repaired records support the before/after evidence. There is no mechanical
simulation, fabrication qualification, tiling algorithm or persistence-schema
change in this slice.
