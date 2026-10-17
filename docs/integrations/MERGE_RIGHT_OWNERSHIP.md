# StructureGraph.merge: right-side ownership

`left.merge(right)` keeps its existing disjoint-append topology: the left graph is
copied, right nodes receive new IDs starting at the left allocation counter, and
right members are appended in input insertion order with parallel fibers allowed.
It does not weld coincident nodes. Existing left IDs, edge orientation, edge count,
lookup behavior, metadata selection and standard serialized values remain the same.

The appended right members' Material objects and internal-point arrays are now
copied, as are nested right node and edge metadata. Mutating these returned fields
no longer changes the right input; mutating those right input fields after merge
no longer changes the result. Right node coordinates already had independent
storage through add_node and retain that behavior.

Copies use existing per-object deepcopy semantics: shared input Material objects
are copied separately for each edge, matching StructureGraph.copy. Caller-defined
custom deepcopy methods retain responsibility for their own behavior; arbitrary
cross-object alias topology is not preserved or authenticated. No new material
catalog, material assignment policy or constitutive model is introduced.

## Deliberately retained limitations

- Left graph metadata is retained; right graph metadata is not merged or reconciled.
- Left StructureGraph.copy still shares its private box array with its source in
  the current implementation. This narrow repair does not change that separate
  behavior and does not promise independence of every graph field.
- No coordinate-unit/frame conflict checking or numeric conversion is added.
- No member equivalence policy, seam deduplication, tiling, welding, schema change
  or topology lineage API is introduced.
- StructureGraph is in the generation source recipe. This source change can cause
  conservative checkpoint mismatch; there is no bypass or automatic migration.

Tests cover both mutation directions for material extra/tensor, curve coordinates,
nested node/edge metadata and node positions; unchanged ID allocation/order,
reversed parallel members, empty curves/empty right graphs and custom deepcopy.
The independent saved pre/post fixture additionally compares complete ordinary
serialized graph values before any mutations. These are geometry/data-ownership
checks, not simulation validation.
