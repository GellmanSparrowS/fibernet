# StructureGraph edge endpoint IDs

`StructureGraph.add_edge(node_i, node_j)` accepts nonnegative Python and NumPy
integer IDs. NumPy integer scalars are stored as Python integers so graph JSON
can be written and read consistently. Booleans, floating-point values and other
non-integer inputs raise `ValueError` before graph state changes.

Existing integer-ID behavior is retained: equal endpoint IDs return `-1`, unknown
unequal endpoints raise `ValueError`, and reversed duplicate edges reuse the
existing edge unless `allow_parallel=True`. Parallel fibers, stored endpoint
orientation, curve coordinates and stable IDs are preserved.
