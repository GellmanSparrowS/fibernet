# Legacy FEA export boundary

`fibernet.io.export_to_abaqus` and `export_to_ansys` write legacy templates for
independent, straight, circular fibers with uniform scalar E, nu, density and
radius. Ordered straight centerline samples are allowed. Curves, backtracking,
degenerate centerlines, declared crosslinks, anisotropic stiffness and differing
written beam properties reject before opening the output file. Coincident
endpoints are **not welded**. ANSYS uses the supplied radius even when material
is None; only missing material properties use the historical defaults.

Only two-node B31 and BEAM188 layouts are supported. Abaqus `element_size=1.0`
is an unused legacy placeholder, not a sizing instruction; other values reject.
`segments_per_fiber` controls straight-element count and must be a positive integer.

`export_to_gmsh_fea` applies the same geometry/incidence admission, but exports
Gmsh MSH 2.2 ASCII two-node line geometry only—not material or section properties. Different radii/materials
therefore do not cause rejection. These input checks do not certify commercial
solver acceptance, suitable boundary conditions, units,
or an equivalent mechanical model. Do not interpret an output file as validation.

`export_to_lammps_fea` now raises NotImplementedError without opening any file.
Its historical mass was a density-derived placeholder with unspecified units;
no automatic replacement or newly invented mass formula is supplied. This does
not change the separate `to_lammps` legacy bead exporter or the explicit native
StructureGraph LAMMPS backend. The latter requires caller-provided units,
masses, bond parameters and runtime/resource contracts and is not a migration.

To retain an existing FiberNetwork's centerlines and crosslinks, save native
JSON (`network.save_json(path)` or `fibernet.api.export(..., format='json')`).
This preserves source data; it does not create a solver-ready model. For a
solver mesh, explicitly specify connectivity, materials and units. In particular,
`StructureGraph.from_fiber_network` is an approximate endpoint projection, not a
lossless conversion or a substitute for unsupported FEA export.

Rejected inputs leave an existing destination untouched and do not create a
new destination. Successful writes retain legacy I/O behavior; this is not a
transactional file writer and does not protect against operating-system errors.

Coordinates, radii and emitted material scalars now use 17 significant digits
to retain float64 values in text, including small SI-scale values. This avoids
fixed-decimal zeroing/truncation; it does not provide unit conversion. Abaqus uses a fixed section orientation `(0, 0, 1)`;
fibers whose normalized XY direction is within 64 machine eps of zero reject
before file opening. Other orientations require an explicitly prepared model.

Periodic networks reject: these exporters do not encode periodic adjacency or
boundary conditions. Empty Abaqus/ANSYS beam networks also reject rather than
writing a default section and boundary condition without nodes. Empty Gmsh
input also rejects before opening the file.

The Gmsh writer uses the official [MSH 2.2 ASCII format](https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-2-_0028Legacy_0029):
node count followed by `id x y z`; element count followed by
`id 1 2 0 fiber_tag node1 node2`. Type 1 is a two-node line; the two tags are
no physical group (0) and a per-fiber elementary entity identifier. Sequential
export IDs remain distinct even at coincident coordinates. This replaces the
previous invalid mix of a 4.1 header with incompatible block records.
