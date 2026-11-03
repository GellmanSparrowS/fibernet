# Relative node movement

`StructureGraph.displace_node(node_id, displacement)` accepts two or three
numeric components. Nested array-like inputs are flattened, consistently with
absolute node-position setters. Two components mean `(dx, dy, 0)`, retaining the
node's current Z coordinate. Three components supply `(dx, dy, dz)`.

Scalars, one-component arrays, empty inputs and inputs with more than three
components raise `ValueError` before movement instead of broadcasting a scalar
across all axes. Invalid node IDs and unknown nodes retain their existing error
precedence. Nonfinite destinations, addition overflow and unsupported spatial
hash ranges remain rejected without updating positions or lookup state.

Existing numeric conversion rules are unchanged; this does not introduce a new
masked-array admission policy. Edge curves and material records are not moved by
this node-only operation.
