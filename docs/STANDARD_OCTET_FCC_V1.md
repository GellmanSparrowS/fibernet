# `octet_fcc_v1`: ideal finite FCC bar geometry

**Scope:** finite FCC geometry. Mechanical behavior and solver accuracy require separate validation. Diamond is not implemented by this API.

```python
from fibernet.core.material import Material
from fibernet.gen.standard_lattice import octet_fcc_v1

graph = octet_fcc_v1(
    a=1e-6,                  # conventional cubic cell size, metres
    grid=(2, 2, 2),           # positive Python integers; tuple required
    radius=2e-8,              # uniform bar radius, metres
    material=Material(name="example", youngs_modulus=1e9),  # SI properties
)
# This finite geometry has 63 junctions and 240 physical bars.
```

Returns the existing **StructureGraph**, not a new graph type or FiberNetwork conversion. The example specifies geometry and material data; it does not run a simulation.

## Geometry

FCC lattice vertices fill the closed box `[0,nx*a] × [0,ny*a] × [0,nz*a]`. Every nearest-neighbor pair is joined once; shared-cell vertices and bars are deduplicated. Each straight bar has length `a/sqrt(2)`, requested radius and the same serialized Material values. Strict-interior junctions have12 neighbors; boundary junctions have fewer. There are no periodic image edges or imposed mechanical boundary conditions.

## Inputs and identities

All four arguments are required. Lengths must be finite, positive SI values; grid must be a three-integer tuple, without booleans or coercion. Anisotropic replication counts are allowed, but anisotropic cells, seeds, perturbations and subdivisions are not accepted. The wrapper checks numerical resolution and explicit resource/payload budgets before construction; budget errors are not physical limits of FCC lattices.

Material properties are preserved as detached native serialized values. Python object identity, arbitrary subclass behavior and nonserialized attributes are not preserved. Modulus/density are not rescaled with cell size.

Native integer node/edge IDs survive native serialization. Node metadata stores exact half-cell lattice labels; each edge has a canonical unordered physical-edge key. These keys identify corresponding bars across geometric scaling, not globally unique specimen hashes. Graph metadata records SI dimensions, termination, reviewed reference provenance and normalized-construction details.

The metadata field `source_blobs` contains the fixed Git blob SHA-1 identifiers of the four implementation files reviewed when this geometry definition was qualified: `material.py`, `pattern.py`, `tiling.py` and `structure_graph.py`. These values identify the reviewed reference sources; they are not computed from, or checked against, the modules loaded at runtime. They do not attest the current installation, later dependency edits, or the complete source of the generator wrapper. Use a separately verified package/source manifest when runtime source identity is required.

## Compatibility and limits

Existing `octet`, `bcc`, `fcc`, `kagome`, `diamond_3d` and named diamond functions are unchanged. The same function is available as `fibernet.octet_fcc_v1`, `fibernet.gen.octet_fcc_v1` and the explicit module import above. It is not an alias or new `pattern_3d` registry entry.

Validation covers finite geometry, adjacency, four scales from1m to1nm, material copying, identities and native lookup/roundtrip behavior in small cases. It does **not** establish beam slenderness, solid-overlap validity, junction mechanics, stiffness, strength, density, isotropy, homogenization or solver accuracy. Choose and independently validate the mechanical model and loading before physical interpretation.
