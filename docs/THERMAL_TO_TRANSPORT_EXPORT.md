# From saved temperature to deformation, conductance and geometry

Run a complete sequential workflow with one command:

```bash
python -m pip install . meshio==5.3.5
python examples/thermal_to_transport_export.py --output-dir thermal_export
```

Run from a repository checkout with a new output directory. The example uses
installed FiberNet APIs and the optional meshio exporter. Its default geometry
unit is millimetres; `--length-unit m` runs the same physical sample in metres.
Importing the example creates no files and starts no solver.

## Follow the fields

The sample is a 3 × 3 square-cell network spanning 36 × 36 mm, with 16 welded
nodes, 24 straight members and a 0.05 mm member radius. Sparse node and edge IDs
are retained throughout. Source records are deliberately reversed; native JSON
reload sorts them, so subsequent field transfer explicitly uses stable IDs.

1. **Thermal history.** Solve four time points from 0 to 0.03 s. Each node has
   capacity 0.01 J/K and each member conductance 0.2 W/K. The left and right
   boundaries stay at 300 and 310 K. Other nodes start at 300 K.
2. **Saved temperature → mechanics.** Write the thermal result, reload it from
   disk and bind the final frame by node ID. Solve a planar linear beam network
   with `E=1 MPa`, thermal expansion `1e-6 /K`, stress-free temperature 300 K
   and the left edge fixed. Mechanical rows contain SI translations and a
   rotation: only `ux_m, uy_m` are passed to the next step.
3. **Deformation → electrical transport.** Use the existing prescribed-translation
   preparation with a reference conductance of 1 S per member. Its fixed-area
   length law is `G_new = G_reference * L_reference / L_new`. Apply 1 V at the
   original left electrode IDs and 0 V at the right, then solve the derived
   network. Conductance is explicit; it is not inferred from the stored radius
   or material record.
4. **Fields → geometry exports.** Write the derived network to VTU, preserving
   node IDs, physical edge IDs, endpoint incidence and the SI `translation_m`
   vector. Project the straight members through the existing explicit
   `FiberNetworkProfile`, save its source-edge-to-fiber mapping, and export
   capped cylinders to binary STL.

This is **sequential one-way transfer**. The electrical result does not update
temperature or mechanics: it is not bidirectional electro-thermo-mechanical
coupling. The thermal time grid selects a transient snapshot, not a steady-state
solution. The library's existing solver models and input contracts are unchanged.

## Inspect the output

| File | Content |
| --- | --- |
| `source.json`, `thermal.json` | Original stable-ID graph and thermal history |
| `mechanical.json` | Saved-temperature binding and complete mechanical result |
| `derived.json`, `transport_binding.json` | Deformed graph and explicit conductance preparation |
| `electrical.json` | Nodal voltages, member currents and power |
| `vtu/network.vtu`, `vtu/manifest.json` | Identity-preserving line geometry, translation field, units and mapping |
| `fiber_projection.json` | Source edge IDs mapped to fiber rows at conversion time |
| `network.stl`, `stl_geometry.json` | Cylinder geometry and its external unit declaration |
| `summary.json` | Key results and output file hashes |

To pair an electrical result with its preparation report, compare the
SHA-256 of `transport_binding.json` with `report_file_sha256` under
`electrical.json` → `provenance.model_metadata.deformed_transport_preparation`.
That record also includes the preparation schema and its input SHA-256.
These are handoff checks, not authentication. They do not change the
electrical solver's own input SHA-256 or state.

STL stores geometry only. Its coordinates use the selected input unit; unit and
identity information are retained in the adjacent JSON files. Cylinders remain
separate mesh pieces, so this export is a geometric handoff rather than a fused
print-ready solid. VTU is the export to use when retaining graph identities and
fields matters.

For the supplied parameters, the peak translation is approximately 0.211 µm,
electrical dissipation is 1.33333 W under 1 V, and the STL contains 1,152
triangles. The example rereads VTU to check IDs, incidence and displacement
fields, and checks source-edge-to-fiber correspondence before writing STL.
The accompanying integration test compares equivalent m/mm runs, including
saved thermal-field alignment and SI mechanical and electrical results.

For individual stages, see [saved-temperature mechanics](SAVED_THERMAL_MECHANICS.md)
and [prescribed-translation transport](DEFORMED_TRANSPORT_PREPARATION.md).
