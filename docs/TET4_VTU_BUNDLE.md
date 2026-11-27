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
    node_ids=[100, 29, 82, 2**63 + 3, 5],
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

## Hand off a traction solve

With FiberNet 4.8.0 and Meshio installed, create the ZIP using the example
above in a fresh directory. As the sender, save the load recipe beside it:

```python
import json
from hashlib import sha256
from pathlib import Path

def bundle_sha256(path):
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

recipe = {
    "bundle_sha256": bundle_sha256(Path("two_materials.zip")),
    "faces": {"upper_slant": [["29", "82", str(2**63 + 3)]]},
    "traction_Pa": [0., 0., 5000.],
    "fixed_dofs": {node: ["x", "y", "z"] for node in ("100", "29", "82", "5")},
}
with Path("two_materials.loads.json").open("x", encoding="utf-8") as f:
    json.dump(recipe, f, indent=2)
```

Send **both** files. The ZIP stores geometry and materials. The companion
JSON stores the named faces, global traction in Pa, fixed Cartesian axes
and a checksum of the complete ZIP. Stable IDs are decimal strings so
tools that read JSON numbers as binary64 cannot round them.

As the recipient, put the two trusted files in the same directory and run
this block there. It reads both files without writing anything:

```python
import json
import re
from hashlib import sha256
from pathlib import Path
from fibernet.io.tet4_meshio import read_tet4_vtu_bundle
from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction

def bundle_sha256(path):
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def node_id(value):
    if type(value) is not str or re.fullmatch(r"0|[1-9][0-9]{0,19}", value) is None:
        raise ValueError("Node ID must be a canonical ASCII decimal string")
    number = int(value)
    if number > 2**64 - 1:
        raise ValueError("Node ID exceeds uint64")
    return number

bundle_path = Path("two_materials.zip")
recipe = json.loads(Path("two_materials.loads.json").read_text(encoding="utf-8"))
if bundle_sha256(bundle_path) != recipe["bundle_sha256"]:
    raise ValueError("Bundle ZIP does not match the load recipe checksum")
bundle = read_tet4_vtu_bundle(bundle_path)
faces = [tuple(map(node_id, face)) for face in recipe["faces"]["upper_slant"]]
loads = nodal_forces_from_traction(bundle, faces, traction_Pa=recipe["traction_Pa"])
holds = {node_id(node): axes for node, axes in recipe["fixed_dofs"].items()}
result = solve_static_tet4(bundle, forces_by_id=loads, fixed_dofs_by_id=holds)
tip_row = result.node_ids.tolist().index(2**63 + 3)
print("Tip displacement z (m):", result.displacement_m[tip_row, 2])
```

The expected tip displacement is about `7.22e-9 m`. This example uses
trusted local files that are not modified during replay. The checksum
pins the exact ZIP bytes, including the material manifest, not just the
mesh. Use a new recipe if the ZIP changes or is repacked. The checksum
detects a mismatched ZIP; it is not a signature and cannot authenticate
either file if someone changes both. This recipe is not an official
FiberNet format. Neither file is a
[solver result VTU](TET4_STATIC.md#view-a-static-result-in-paraview).

## Generate a tagged fused-cylinder CAD mesh

From the repository root or the extracted FiberNet source distribution:

```sh
python -m pip install 'fibernet>=4.21.0' 'gmsh==4.13.1' 'meshio==5.3.5'
python examples/tet4_fused_cylinder_cad.py --output-dir fused-cad-output
```

Gmsh is optional; its Python package may also require native libraries on
your platform (for example `libGLU.so.1` on Linux). The script runs
headlessly, uses the Gmsh/OpenCASCADE Boolean union of three perpendicular
cylinders (0.50 mm radius, 3.00 mm full length), and remeshes the entire
CAD model twice. The target sizes are 0.33 and 0.24 mm; this is not an
internal subdivision of one fixed exterior. The output directory must not
already exist; choose a parent directory you control. Only use a result
directory containing `COMPLETE`. The script creates this marker after
all six output files are in place. If publication is interrupted and
`COMPLETE` is absent, ignore any partial files and retry with a new
output directory.

Each of `coarse` and `fine` gives:

- `.msh`: ASCII MSH 4.1, one homogeneous volume named `solid`, six distinct
  exterior cap groups (`cap_neg_x`, `cap_pos_x`, and the same for `y` and
  `z`), and the rest of the exterior as `ambient_side`.
- `.zip`: native Tet4Bundle, readable with `read_tet4_vtu_bundle`. The
  coordinate unit is recorded as `mm`; the example material's Young's
  modulus is in pascals (1 GPa) and Poisson's ratio is dimensionless (0.25).
- `.json`: volume, mesh counts, structural budget preflight for an
  illustrative clamp on the negative x cap, bundle SHA-256, and each
  named exterior group's triangles as decimal-string native node IDs.
  Use the corresponding ZIP for those IDs; they are not Gmsh tags or
  shared across remeshes. Boundary names live in the MSH and JSON, not
  in the native ZIP.

The checks require positive tetrahedra, one material, an exhaustive and
disjoint named exterior, and genuinely different boundary triangulations.
Exact element counts can vary with platform or Gmsh build. The linear
tetrahedra approximate the curved surface; this is a mesh-interchange
example, not a convergence study or a physical solve. For other trusted
tagged meshes, use the general import recipe below.

### Optional conformal junction material

```sh
python examples/tet4_fused_cylinder_cad.py --junction-sphere --output-dir fused-cad-sphere
```

This opt-in route fragments the same fused solid with a centred, fully
embedded 0.30 mm sphere. The native bundle has two material regions named
`bulk` and `junction` in the MSH, linked to native decimal-string IDs
by the JSON's `material_id_by_name` map. Both have the same illustrative
elastic properties; adjust properties by region in a consuming workflow.
The coarse and fine targets are 0.33 and 0.26 mm, respectively, with
local targets of 0.165 and 0.13 mm around the sphere. Actual element
counts depend on the Gmsh build. The linear interface triangles facet the
analytic sphere rather than representing it exactly.

The two regions share tetrahedral faces that form a closed interface.
The six caps and `ambient_side` still partition only the exterior.
Do not use the internal material interface as a convection boundary.
Use generated files only after `COMPLETE` appears. The default route
remains homogeneous, and the homogeneous thermomechanics example does
not accept the two-region bundle. These are mesh and material-label
checks, not a solve or a convergence study.

### View the two material regions in ParaView

Open `mesh.vtu` extracted from the ZIP and select **Color By → material_id**
(cell data). This interchange mesh is in millimetres; the ZIP's
`manifest.json` records the unit and material properties. Use the matching
generator JSON (for example, `coarse.json`) to map IDs to `bulk` and
`junction` and recover the named exterior groups. If you open a
[thermoelastic result VTU](TET4_THERMOELASTIC.md#inspect-a-snapshot-in-paraview)
instead, its undeformed coordinates and displacement are in metres;
it preserves the IDs, but not the material table or named groups.

An exterior-only STL does not carry material IDs, the internal interface,
or units. FiberNet's graph STL export builds cylinders from a fiber
network rather than exporting this fused Tet4 solid. Reading the VTUs
with Meshio can verify the IDs and geometry; it is not a fused
multimaterial manufacturing export.

From a checkout containing the [shell check](../examples/tet4_material_shell_check.py),
run this on a trusted local bundle with Meshio installed:

```sh
python examples/tet4_material_shell_check.py fused-cad-sphere/coarse.zip
```

It compares each region's signed closed-shell volume (subtracting cavity
components) with its Tet4 volume. It does not check self-intersections
or slicer compatibility.

## Import named material and boundary groups from Gmsh

This recipe uses Meshio 5.3.5 to read a trusted, small **ASCII MSH 4.1**
export with linear tetrahedra and triangular boundary groups. In Gmsh,
name the two physical **volumes** `material_low` and `material_high`,
and the two physical **surfaces** `hold_face` and `pull_face`. Edit the
names, moduli, Poisson ratios, coordinate unit and traction for your
mesh. Export all the tetrahedra you want to solve; the recipe checks
coverage of the exported cells, not what was omitted at export.
Ancillary linear lines or vertices (for example, from physical curves or
points) are ignored only when their point rows already belong to the
tetrahedral mesh. Orphan points and unsupported element types are
rejected; points are not removed or renumbered.

A volume can also belong to an auxiliary group such as `whole_solid`.
Do not assign materials from the scalar `gmsh:physical` column: Meshio
can return that same auxiliary tag on both volumes. Instead select
the two **named** groups and require exactly one material for each
tetrahedron. Meshio returns row-based connectivity; this recipe assigns
new stable IDs for the bundle and does not preserve original Gmsh tags.

Run in a fresh directory with `two-materials.msh` and Meshio installed:

```python
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import meshio
import numpy as np
from fibernet.io.tet4_meshio import Tet4Bundle, write_tet4_vtu_bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission, solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction

source = Path("two-materials.msh")
with source.open("rb") as f:
    if f.readline(128).strip() != b"$MeshFormat" or f.readline(128).split() != [b"4.1", b"0", b"8"]:
        raise ValueError("Expected an ASCII Gmsh 4.1 mesh")
mesh = meshio.read(source)
if any(block.type not in {"tetra", "triangle", "line", "vertex"} for block in mesh.cells):
    raise ValueError("Export only linear tetrahedra, triangles, lines or vertices")
tets, triangles = (mesh.get_cells_type(kind) for kind in ("tetra", "triangle"))
npoints = len(mesh.points)
if (not len(tets) or tets.dtype.kind not in "iu"
        or not np.array_equal(np.unique(tets), np.arange(npoints))):
    raise ValueError("Every point row must belong to the tetrahedral mesh")
ancillary_widths = {"line": 2, "vertex": 1}
for block in mesh.cells:
    if block.type in ancillary_widths:
        rows = np.asarray(block.data)
        if (rows.ndim != 2 or rows.shape[1] != ancillary_widths[block.type]
                or rows.dtype.kind not in "iu" or np.any(rows < 0) or np.any(rows >= npoints)):
            raise ValueError("Line or vertex connectivity must use valid tetrahedral point rows")
groups = mesh.cell_sets_dict

def named_group(name, dimension, kind):
    if name not in mesh.field_data or int(mesh.field_data[name][1]) != dimension:
        raise ValueError(f"Missing or wrong-dimensional physical group: {name}")
    indices = groups.get(name, {}).get(kind)
    if indices is None or not len(indices):
        raise ValueError(f"No {kind} cells in physical group: {name}")
    return int(mesh.field_data[name][0]), indices

properties, covered = {}, np.zeros(len(tets), dtype=int)
material_ids = np.zeros(len(tets), dtype=np.uint64)
for name, young in (("material_low", 1e9), ("material_high", 2e9)):
    tag, rows = named_group(name, 3, "tetra")
    if tag in properties:
        raise ValueError("The two materials must use distinct physical volume tags")
    np.add.at(covered, rows, 1)
    material_ids[rows] = tag
    properties[tag] = {"young_modulus_pa": young, "poisson_ratio": 0.25}
if len(tets) == 0 or np.any(covered != 1):
    raise ValueError("Each exported tetrahedron needs exactly one named material")

counts = Counter(face for tet in tets for face in combinations(sorted(map(int, tet)), 3))
def exterior_faces(name):
    _, rows = named_group(name, 2, "triangle")
    faces = [tuple(sorted(map(int, f))) for f in triangles[rows]]
    if len(faces) != len(set(faces)) or any(counts[f] != 1 for f in faces):
        raise ValueError(f"{name} must contain distinct exterior triangles")
    return set(faces)

clamped, pulled = (exterior_faces(name) for name in ("hold_face", "pull_face"))
if clamped & pulled:
    raise ValueError("Clamp and traction groups must not share a triangle")
node_ids = np.arange(len(mesh.points), dtype=np.uint64)
cell_ids = np.arange(len(tets), dtype=np.uint64)
bundle = Tet4Bundle(mesh.points, tets, node_ids, cell_ids, material_ids, properties, "mm")
holds = {int(node_ids[i]): ("x", "y", "z") for face in clamped for i in face}
faces = [tuple(int(node_ids[i]) for i in face) for face in sorted(pulled)]
traction = (0., 0., 1000.)  # Pa, global +z, not pressure normal to the surface
loads = nodal_forces_from_traction(bundle, faces, traction_Pa=traction)
plan = inspect_static_tet4_admission(bundle, fixed_dofs_by_id=holds)
if not plan.within_factor_budget:
    raise ValueError("Tet4 mesh exceeds the structural factor budget")
print("Tetrahedra:", plan.tetrahedra, "factor entries:", plan.factor_entries)

path = write_tet4_vtu_bundle(
    "two_materials.zip", points=bundle.points, tetrahedra=bundle.tetrahedra,
    node_ids=node_ids, cell_ids=cell_ids, material_ids=material_ids,
    materials=properties, length_unit=bundle.length_unit,
)
digest = sha256()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)
recipe = {"bundle_sha256": digest.hexdigest(),
          "faces": {"upper_slant": [list(map(str, face)) for face in faces]},
          "traction_Pa": traction, "fixed_dofs": {str(i): axes for i, axes in holds.items()}}
with Path("two_materials.loads.json").open("x", encoding="utf-8") as f:
    json.dump(recipe, f, indent=2)
```

This checks the load faces and budget but does not solve. You can call
`solve_static_tet4(bundle, forces_by_id=loads, fixed_dofs_by_id=holds)`
in the same script. To send a repeatable input, use the preceding
[handoff recipe](#hand-off-a-traction-solve): send both files and run its
recipient block through the solve, leaving out the two lines that print
the earlier example's particular tip displacement. This conversion
does not transfer arbitrary Gmsh fields or turn physical group labels into
units or material constants.

## Run a tagged mesh above the banded solver limits

For a single larger bonded solid, use prescribed grip motion and the
optional iterative solver. Use a trusted local mesh file. The example
reads ASCII Gmsh 2.2 or 4.1. The default two-material mode uses
these named physical groups:

- Volumes: `material_low` and `material_high`, with exactly one group per cell.
- Surface triangles: `hold_face` and `pull_face`, with distinct exterior
  triangles and no shared grip nodes.

The names above are defaults. For an existing mesh, supply its real names
without editing the mesh file: `--material-groups "Soft phase" "Stiff phase"`
sets the order for the low and high modulus inputs, and
`--grip-groups "Fixed end" "Driven end"` selects the two exterior grips.

Install FiberNet 4.32.0 with the iterative back end and Meshio:

~~~sh
python -m pip install "fibernet[solid-iterative]==4.32.0" "meshio==5.3.5"
python -m fibernet.examples.tet4_tagged_grip_iterative --input my-mesh.msh --output grip-run \
  --length-unit mm --axis x --stroke-um 1.5 --material-low-gpa 1 --material-high-gpa 2
~~~

Choose the input unit, axis, stroke, and material values for your sample.
The held faces are fixed and the pull face is driven as a rigid grip
along the selected axis. Omit `--input` and `--length-unit` to build a
28,512-tetrahedron fixture. The output directory must be new.

Outputs are `tagged-solid.zip` (Tet4 geometry and material bundle),
`grips.json` (grip IDs, stroke in metres and bundle hash),
`summary.json` (force, energy and residual), and `nodal-fields.npz`
(stable `node_ids`, `displacement_m` and `reaction_N`). IDs are assigned
from the imported rows; they are not original Gmsh tag numbers. Change
grip settings through command-line options and rerun in a new output
directory rather than editing the recorded sidecar.
The example calls standalone `solve_static_tet4(..., solver="amg-cg")`.
The small-mesh traction, band inspection, and VTU result wrappers retain
their existing limits.

For a homogeneous solid, provide one physical volume name and its
modulus with `--material-group "Solid" --young-modulus-gpa 2` instead of
the two-material flags. In Gmsh, assign the volume group to the whole
solid you intend to export, even when it spans several geometric volumes.
The two exterior grip groups and coordinate units are still required.

To compare different materials on the same geometry, keep the same mesh
and pass `--materials-json materials.json` in place of the other material
flags. Use the exact Gmsh physical volume names. Each of the 1–64 named
regions has its own Young modulus in pascals and Poisson ratio:

~~~json
{
  "Soft phase": {"young_modulus_pa": 1000000000, "poisson_ratio": 0.24},
  "Junction phase": {"young_modulus_pa": 2000000000, "poisson_ratio": 0.32},
  "Stiff phase": {"young_modulus_pa": 5000000000, "poisson_ratio": 0.39}
}
~~~

~~~sh
python -m fibernet.examples.tet4_tagged_grip_iterative --input my-mesh.msh \
  --output grip-run --length-unit mm --materials-json materials.json --write-vtu
~~~

If the grip names differ from the defaults, also pass `--grip-groups`.
Rerun with another material file and a new output directory to preserve
both results. Each tetrahedral volume in Gmsh 4.1 must belong to exactly
one declared physical volume group; overlapping physical volume groups
are rejected rather than assigning whichever label comes first.

Add `--material-sensitivities` to rank which material changes matter most
for the signed grip reaction near the current settings. From the same
solve, `summary.json` reports `d_reaction_d_ln_young_modulus_N` and
`d_reaction_d_poisson_ratio_N` keyed by the Gmsh physical volume IDs.
For small changes, multiply those slopes by the change in log modulus
(or approximate fractional modulus change) and in Poisson ratio.
The geometry and grip displacement stay fixed. The report does not
predict a new displacement field; use a new solve for large changes.

To inspect the displacement directly, pass `--write-vtu` in the same run.
Open the resulting `deformation.vtu` in ParaView and select **Warp By Vector**
with `displacement_m`. Use scale factor 1 for the physical deformation;
the reference coordinates and displacement are both in metres. The file
also contains `reaction_N`, material IDs, and stable node/cell IDs. It is a
display result, separate from the unchanged interchange ZIP.

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
- Zero-volume cells are rejected for the represented coordinates. An
  outward-rounded determinant interval can establish that a cell is
  nonzero; if the interval cannot exclude zero or the coordinate scale is
  extreme, the exact binary64 determinant is used. This is not a mesh
  quality test and does not change the solver's conditioning limits.
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
