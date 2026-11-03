# StructureGraph edge endpoint IDs

`StructureGraph.add_edge(node_i, node_j)` accepts signed Python and NumPy integer
IDs of existing nodes, including negative stable IDs already present in a live
graph. NumPy integer scalars are stored as Python integers without renumbering.
Booleans, floating-point values and other non-integer inputs raise `ValueError`
before graph state changes.

Existing integer-ID behavior is retained: equal endpoint IDs return `-1`, unknown
unequal endpoints raise `ValueError`, and reversed duplicate edges reuse the
existing edge unless `allow_parallel=True`. Parallel fibers, stored endpoint
orientation, curve coordinates and stable IDs are preserved.

Stable IDs are not array-row indices: a negative ID does not index from the end
of an array. This live-graph compatibility contract does not broaden the existing
`from_dict`/JSON-reader schema, which requires nonnegative IDs, or downstream
adapters with their own nonnegative-ID admission rules. Nonnegative NumPy
integer endpoints remain JSON-roundtrippable as Python integers.
