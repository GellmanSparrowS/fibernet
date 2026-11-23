# Steady DC heating and convection in registered Tet4 solids

The optional Linux stdio profile exposes the native FiberNet one-way
steady DC → Joule heat → conduction/convection function without adding a
second solver. Temperature does not feed back into electrical conductivity.

In the operator manifest, keep the existing source tree and at least one
graph entry. Add the optional `"tet4_fixtures": ["simple", "welded"]`
member to select the bundled meshes. Omit both Tet4 registration members
(`tet4_fixtures` and `tet4_sources`) to retain the graph-only catalog
and nine tools. On an opted-in session,
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

## Register an operator-prepared mesh

Only the trusted operator can prepare a source. First export a strict
FiberNet Tet4 VTU input bundle using the native exchange workflow; a
solver-result VTU or a generic VTU/ZIP is not an input. From the directory
housing your operator manifest, with Meshio available **offline**:

```sh
mkdir tet4-sources
python -m fibernet_mcp.tet4_operator my-native-input.zip my-network
```

The command writes `tet4-sources/my-network.tet4.json` without replacing
an existing file, and prints a JSON object with its `name` and `sha256`.
Include that object in the manifest's optional `tet4_sources` list; keep
the existing source-tree identity and graph entries. For example, the
new manifest member has this shape:

```json
{ "tet4_sources": [{"name": "my-network", "sha256": "<printed SHA-256>"}] }
```

Use a simple name of up to 64 ASCII characters, starting alphanumeric
and containing only letters, digits, underscores and hyphens. There is
no source-file name or path parameter in the manifest entry or client
tool. The server matches the exact pinned JSON bytes, preserves explicit
coordinate units, row order, stable decimal IDs and material tags, and
copies a bounded snapshot into the managed session. A new session
rechecks the digest and takes a new snapshot. Both profiles together can
register at most two Tet4 sources; graph-only defaults are unchanged.
The live service does not open native bundles or require Meshio.

For a reproducible fitted-geometry comparison, run from the repository
root once, with FiberNet, the adapter and Meshio installed:

```sh
python -m adapters.mcp.examples.prepare_fitted_tet4 operator-demo
```

It creates the fitted native bundle, a pinned JSON source in
`operator-demo/tet4-sources`, and `operator-demo/fitted-controls.json`.
Save your operator manifest in `operator-demo`, copy the printed
`tet4_sources` member into it, and keep the normal graph configuration.
Use `fibernet_catalog` to get the mesh handle for
`fitted-axial-junction`. In two calls, combine that same `mesh_id`
with each of the two exported `cases`. The recipe defines both
electrode sets, held temperatures, material property maps and selected
exterior face IDs. It changes only the junction electrical conductivities;
the values are synthetic and not derived from the elastic constants.
No row or cell tag changes between the two runs.
