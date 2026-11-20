# Tet4 geometry and material bundles

Use this optional interface to move a tetrahedral mesh and its two isotropic
elasticity constants per material between tools. Bundle I/O neither solves
the mesh nor attaches it to a `StructureGraph`. For supported small-strain
problems use the separate [Tet4 static solver](TET4_STATIC.md). This format
is distinct from the straight-line VTU adapter.

This interface was verified with the optional provider `meshio==5.3.5`.
Importing FiberNet or this module does not load Meshio; it is only needed
when reading or writing a bundle.

```python
from fibernet.io.tet4_meshio import read_tet4_vtu_bundle, write_tet4_vtu_bundle

path = write_tet4_vtu_bundle(
    "two_materials.zip",
    points=[(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 0., -1.)],
    tetrahedra=[(0, 1, 2, 3), (0, 2, 1, 4)],
    node_ids=[100, 29, 82, 111, 5],
    cell_ids=[901, 756],
    material_ids=[21, 2**63 + 1],
    materials={
        21: {"young_modulus_pa": 1e9, "poisson_ratio": 0.25},
        2**63 + 1: {"young_modulus_pa": 2.4e9, "poisson_ratio": 0.33},
    },
    length_unit="mm",
)
bundle = read_tet4_vtu_bundle(path)
assert bundle.cell_ids.tolist() == [901, 756]
assert bundle.material_ids.tolist() == [21, 2**63 + 1]
assert bundle.length_unit == "mm"
```

## Input and output

- `points` is nonempty `(N, 3)` finite binary64 geometry.
  Only exactly representable values are accepted when a conversion is needed.
  `length_unit` must be one of `m`, `cm`, `mm`, `um`, `nm`. The coordinates
  are never scaled or interpreted as SI based on the unit.
- `tetrahedra` is nonempty `(M, 4)`, using **zero-based point row indices**, not
  stable IDs. Both windings are retained. A mathematically zero volume at the
  supplied binary64 coordinates is rejected, including cells with repeated
  corners; there is no small-cell cutoff. This is not a mesh quality or
  topology validator: it does not check overlap, duplicates, conformity or
  whether the winding suits a given external solver.
- `node_ids` and `cell_ids` are unique within their respective columns.
  `material_ids` has one value per element row and may repeat. IDs are exact
  nonnegative integers in `0 .. 2**64-1`. Each stored column is `Int64` when
  its values fit and `UInt64` otherwise. Signed and unsigned integer inputs
  may be mixed if in range; floating-point IDs (even integral ones),
  booleans, negative and out-of-range integers are rejected. Supply the source
  Python sequence or a correctly typed ndarray: the writer cannot recover an
  ID rounded by an earlier external float cast.
- The material mapping must cover **exactly** the material IDs used by cells.
  It carries only `young_modulus_pa` (positive, in Pa) and `poisson_ratio`
  (dimensionless, strictly between `-1` and `0.5`) as finite binary64 values.
  These bounds describe this elastic metadata, not eligibility for a solver.
- Point, element, and corner ordering are preserved. The returned
  `Tet4Bundle` exposes the same named fields accepted by the writer.

## v1 interoperability boundary

The versioned ZIP envelope contains exactly two stored members:
`mesh.vtu` (up to 40 MiB) and UTF-8 `manifest.json` (up to 1 MiB). The VTU
contains exactly one piece and a single `tetra` block. Unzip it to open
`mesh.vtu` in tools that support ordinary VTK XML, such as Meshio. The
uncompressed binary VTU holds the geometry and all ID assignments. The JSON
is the only source of the coordinate unit and the material table, whose keys
are decimal strings so all uint64 values remain exact. The manifest also
carries the payload's SHA-256 for integrity checking, not authentication.

Both directions admit at most 250,000 points and 250,000 Tet4 cells. These
are mesh row limits plus the member byte limits above, not bounds on the
process's memory use. Parsing XML and creating arrays use additional memory.
Exact zero-volume checks work on batches of at most 4,096 tetrahedra. This is
a local interchange utility, not a parser for accepting untrusted uploads.

The FiberNet reader understands this specific writer's v1 files, not arbitrary
VTU files or bundles repacked with compression. It rejects extra ZIP members,
extra or missing metadata, non-tetra blocks, multiple pieces, unknown or
duplicate arrays, user fields, and VTU `FieldData` rather than dropping them.
The writer publishes a newly completed archive atomically on writable local
filesystems that support hard links (for example ext4, APFS, and NTFS).
Existing files, directories and symlinks are not replaced, including competing
writers. An unsupported filesystem fails; there is no overwrite fallback.

There are no loads, constraints, fields such as stress or displacement, or
continuum simulation APIs in this interface.
