# FiberNet 4.21.5

This patch prevents two ways that a valid-looking output can misrepresent
the input. It does not change the beam stiffness model, the electrical
solve for accepted inputs, package dependencies, or the desktop version.

## Keep explicit beam result IDs in input row order

When calling `BeamFrameFEM.to_sim_result` with optional `node_ids` and
`edge_ids`, pass both as ordered iterables aligned with the original solver
input rows, before deduplication. For arrays from `graph.to_numpy()`, start
with `sorted(graph.nodes)` and `list(graph.edges)`, then apply the same
permutations used to prepare the arrays. Do not sort the IDs after reordering
the arrays. Sets and frozensets now raise `ValueError`. This is not a geometry
binding: the caller still owns the row correspondence. Calls without explicit
maps retain their existing behavior. Re-export results if an old conversion
used unordered maps.

## Preserve whether an electrical edge is active

`solve_linear_transport` rejects an incoming nonzero `Fraction` or NumPy
floating value, such as a wider floating scalar, if it converts to zero in
binary64. Both signs are checked. Actual zero, including negative zero,
still disables the edge. Ordinary rounding such as `Fraction(1, 10)`
is unchanged. An input that was already converted to zero cannot be
recovered. Prescribed voltage rules and the MCP adapter wire format are
unchanged.

## Compare global and cross-cap responses separately

The [Tet4 guide](TET4_STATIC.md) includes a reproducible example with one fixed
cap and three distinct loaded caps on the same welded-cylinder geometry.
Stiffening only the junction reduces global compliance by a similar amount
across the two tested meshes. Some small individual cross-cap terms are
mesh-sensitive. This does not establish mesh convergence, local stress, or
strength. The source distribution includes the guide's inline Python recipe
and its helper; save the recipe as directed in the guide. The mesh workflow
uses optional Gmsh and Meshio.
