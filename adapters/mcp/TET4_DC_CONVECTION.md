# Steady DC heating and convection in bundled Tet4 solids

The optional Linux stdio profile exposes the native FiberNet one-way
steady DC → Joule heat → conduction/convection function without adding a
second solver. Temperature does not feed back into electrical conductivity.

In the operator manifest, keep the existing source tree and at least one
graph entry. Add the optional `"tet4_fixtures": ["simple", "welded"]`
member to select the bundled meshes. Omit the member to retain the
graph-only catalog and nine tools. On an opted-in session,
`fibernet_catalog` returns new session-specific mesh handles under
`tet4_meshes`; graph and mesh handles are not interchangeable. There are
no client-supplied files, paths, URLs, arbitrary meshes, or uploads.

Use `fibernet_solve_tet4_dc_convection` with the catalog's `mesh_id`.
Identify all nodes/cells/materials by canonical decimal strings
(including IDs above 2⁵³). For the bundled one-cell `simple` fixture:

| Control | Example |
| --- | --- |
| electrical conductivity (S/m), by material | `{"17": 0.5}` |
| voltages (V), by node | `{"9223372036854775808": 0, "9223372036854775819": 0.3}` |
| thermal conductivity (W/(m K)), by material | `{"17": 0.25}` |
| held temperatures (K), by node | both electrode nodes at `300` |
| selected exterior face, by node | `["9223372036854775808", "9223372036854775830", "9223372036854775841"]` |
| convection coefficient (W/(m² K)) / ambient (K) | `100` / `299.95` |

The exact argument names are in the tool's input schema. Electrical
and thermal material maps are separate. `faces_by_node_id` is a list
of triangles; no alternate thermal source is accepted. Setting the
coefficient to zero keeps existing face validation but disables film
exchange.

The reply gives separate Joule, convection and fixed-temperature powers
in watts, and a managed artifact handle. A positive signed boundary
value adds heat to the solid; a negative one removes it. Retrieve the
voltage, temperature, cell Joule power and nodal heat ledgers with the
existing chunk tool for that handle. The welded fixture is a synthetic
three-cylinder voxel union with 621 nodes and 2,064 cells, not a
mesh-converged CAD equivalence study. The worker and artifact stay
within the existing session resource and cancellation lifecycle.
