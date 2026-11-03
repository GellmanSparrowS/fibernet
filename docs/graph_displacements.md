# Relative node movement

`StructureGraph.displace_node(node_id, displacement)` accepts array-like inputs
with one, two or three numeric components. Nested inputs are flattened,
consistently with absolute node-position setters. A one-component input such as
`[d]` retains the historical broadcast `(d, d, d)`, including the stored Z
coordinate of a 2D graph. Two components mean `(dx, dy, 0)`, retaining the node's
current Z coordinate. Three components supply `(dx, dy, dz)`.

Scalars (including zero-dimensional arrays), empty inputs and inputs with more
than three components raise `ValueError` before movement. The singleton
broadcast applies only to relative displacement, not absolute position setters.
Invalid node-ID types and unknown nodes retain their existing error precedence.
Nonfinite destinations, addition overflow and unsupported spatial hash ranges
remain rejected without updating positions or lookup state.

Existing numeric conversion rules are unchanged; this does not introduce a new
masked-array admission policy. Edge curves and material records are not moved by
this node-only operation.
