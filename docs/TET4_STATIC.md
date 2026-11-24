# Bounded linear Tet4 static solve

Solve small 3D problems with conforming four-node tetrahedra, one isotropic
linear elastic material per cell, nodal forces, and Cartesian displacement constraints.
This is separate from FiberNet's beam backends and does not change
`fibernet.simulate`, `StructureGraph`, or VTU bundle fields.

```python
from fibernet.sim.tet4 import solve_static_tet4

# bundle is a Tet4Bundle; use its stable IDs, not coordinate row indices.
result = solve_static_tet4(
    bundle,
    forces_by_id=nodal_forces,      # {stable_id: (Fx, Fy, Fz)}, in N
    fixed_dofs_by_id=held_axes,     # {stable_id: ("x", "z")}, for example
)
```

For prescribed nodal temperatures, use the separate experimental
[Tet4 thermoelastic snapshot](TET4_THERMOELASTIC.md).

Create a [Tet4Bundle](TET4_VTU_BUNDLE.md) in memory or read it with the
optional Meshio adapter. Solving itself does not import or need Meshio.
For a known constant Cartesian traction on selected exterior triangles,
see [conversion to Tet4 nodal forces](TET4_TRACTION.md). This is an
explicit helper, not a new load format for the solver.
See the runnable [two-material example](../examples/tet4_static_two_material.py)
for a fitted series cube with an analytical displacement and energy check.
From an installed or editable FiberNet environment:

```sh
python examples/tet4_static_two_material.py
```

For a larger reproducible junction, use the
[three-cylinder example](../examples/tet4_static_welded_junction.py) or
[download the exact 4.7.0 version](RELEASE_4_7_0.md#run-the-welded-junction-example).
Its quick and reference tiers are separate discretizations, not a
mesh-convergence claim. The current source example requires FiberNet 4.8.0
or newer. It passes the selected right-cap
triangles by stable node ID to the traction helper. The script
first divides 1 mN by the actual cap area to define a global +x traction,
not a surface-normal pressure. Both materials receive the same nodal loads.

Use `--preflight` with the current example to print one structural budget
check for the shared mesh and restraints to stderr before assembling
loads. A passing check still runs both material solves. It is opt-in
because the solver repeats its own validation, and stdout remains the JSON
report. The flag can be combined with `--tier reference` and
`--write-vtu-dir DIR`.

For a fitted mesh, see the [equal-volume stiffening example](TET4_FITTED_PLACEMENT.md)
to compare two material placements under x and y loading. It runs without
Gmsh or Meshio.

## Check the factor budget without solving

If you only need to know whether a mesh and restraint layout fit the current
banded solver's structural budget, inspect them before constructing loads:

```python
from fibernet.sim.tet4 import inspect_static_tet4_admission

report = inspect_static_tet4_admission(bundle, fixed_dofs_by_id=held_axes)
print(report.free_dofs, report.factor_entries, report.factor_entry_limit)
if not report.within_factor_budget:
    print("This mesh and restraint layout exceed the band budget.")
```

Pass the same stable-ID restraint and, if used, prescribed displacement
mappings you would pass to the solver. Only their held-axis union affects
the structural budget. The
function takes no loads and does not assemble or factor the stiffness
matrix. For example, a welded-junction discretization with 1,512 points
and 6,150 tetrahedra can still need 3,495,402 factor entries, exceeding the
3,000,000-entry budget despite being below both mesh-size limits.

An over-budget topology returns its estimated counts and
`within_factor_budget=False`. Invalid bundle fields, point/cell-count
violations, unsupported topology, or insufficient restraints still raise a
validation error. `factor_entries` is the conservative lower-plus-upper
structural envelope, while `padded_factor_entries` counts slots in the band
buffer. Neither is measured fill or memory use. Passing this check does not
validate loads, certify element conditioning or numerical material
stiffness, guarantee absence of overlap, or promise a successful solve.
The solver still validates inputs and checks for numerical failure.

## Contract

- Connectivity uses point row indices within the bundle. Forces and
  restraints use stable nonnegative node IDs through `2**64-1`. Forces
  must contain three finite real scalars exactly representable as binary64.
  Each restraint lists distinct Cartesian axes; unspecified axes are free.
- Coordinates are converted to m using the bundle's length unit. Its modulus
  is specified in Pa and Poisson ratio is dimensionless. The axes listed in
  `fixed_dofs_by_id` are held at zero displacement.
- The detached `Tet4StaticResult` holds `node_ids`, `displacement_m`,
  `reaction_N`, and `fixed_mask` in input node order. Reaction convention
  is support force on the mesh, `Ku-f` on held axes and zero on free axes.
  The result also holds `strain_energy_J`, `half_external_work_J`,
  `free_residual_inf_N`, `normalized_backward_error`, and
  `relative_energy_work_error`. A small normwise backward residual is not
  a per-component or forward-error guarantee.

## Solve several load cases on the same solid

`solve_static_tet4_load_cases` shares mesh validation, assembly, and the
checked factorization for 1 to 8 cases on one mesh with unchanged materials
and zero-displacement restraints. Supply an ordered list or tuple of the
same stable-node-ID force mappings used by `solve_static_tet4`. From the
source checkout or extracted source distribution:

```python
from examples.tet4_static_two_material import build_example
from fibernet.sim.tet4 import solve_static_tet4_load_cases

bundle, pull_N, held_axes = build_example()
push_N = {node_id: tuple(-f for f in load) for node_id, load in pull_N.items()}
pull, push = solve_static_tet4_load_cases(
    bundle, force_cases=[pull_N, push_N], fixed_dofs_by_id=held_axes,
)
print(pull.reaction_N.sum(axis=0), push.reaction_N.sum(axis=0))  # N
```

The tuple preserves case order, and each detached result keeps its own
reaction, energy, and residual checks. If any case fails, no partial result
is returned. The usual mesh, band-factor, and numerical limits still apply.
No shared factor is kept after the call. Use separate calls for different
material assignments and the single-case API for prescribed motion,
cell fields, or VTU export.

## Prescribe a measured displacement

For prescribed Cartesian motion, give the desired **signed displacement in
metres** from the reference geometry, even when the mesh coordinates are in
millimetres. A nonempty `prescribed_displacements_m_by_id` maps stable node
IDs to axis/value mappings. For example, to move every node of a selected
grip by +1.5 nm along global z while retaining other restraints:

```python
grip_motion = {node_id: {"z": 1.5e-9} for node_id in grip_node_ids}
result = solve_static_tet4(
    bundle,
    forces_by_id={},  # no separately applied nodal loads
    fixed_dofs_by_id=held_axes,
    prescribed_displacements_m_by_id=grip_motion,
)
```

Do not repeat a node-axis in both mappings. A listed zero displacement
still holds that axis, and other axes of the same node may remain free.
Each body still needs enough held axes to suppress rigid modes. A held
axis may also have an applied force: the reported reaction is then the
**net force on the mesh from the support or actuator**, not the applied
force or its opposite. Read `fixed_mask` as the union of held axes.

`half_external_work_J` is half the signed sum of applied force times nodal
displacement and net support/actuator force times prescribed displacement,
for proportional loading from zero in this static linear model. A moving
actuator can do negative work; a stationary support does no work. On a
held axis that also carries a load, the solver combines that applied force
and reaction before taking their work, then checks the combined total
against the integrated cell strain energy.
If intermediate arithmetic or cancellation prevents the check, the
solver raises rather than returning a zero or partial result.

The same optional keyword is accepted by `inspect_static_tet4_admission`
and `solve_static_tet4_to_vtu`. The diagnostic VTU fields and geometry/material
bundle format are unchanged; neither stores a load plan.

### Move a local exterior grip, not the entire cap

Mark the intended contact patch during meshing (for example as a named
Gmsh physical surface; see [named surface groups](TET4_VTU_BUNDLE.md#import-named-material-and-boundary-groups-from-gmsh)).
The patch must be a union of **whole exterior Tet4 triangles**. If its outline
cuts through existing triangles, mesh a boundary that conforms to that
outline first. Selecting nearby points, or clipping triangles just to
estimate area, does not create the specified grip. Compare the selected
footprint to the intended patch, not just its area.

Given the selected triangles as stable node ID triples from the same
`bundle`, check that each has one incident tetrahedron, measure the area,
and constrain the union of their vertices. This example assumes coordinates
in mm, a designed patch of 0.5 mm², and separate stationary `held_axes`:

```python
from collections import Counter
from itertools import combinations
from math import fsum
import numpy as np
from fibernet.sim.tet4 import solve_static_tet4

if bundle.length_unit != "mm":
    raise ValueError("This area check expects millimetre coordinates")
row = {int(node_id): i for i, node_id in enumerate(bundle.node_ids)}
owners = Counter(face for tet in bundle.tetrahedra
                 for face in combinations(sorted(map(int, tet)), 3))
faces = [tuple(sorted(row[node_id] for node_id in tri)) for tri in grip_faces_by_id]
if (not faces or len(set(faces)) != len(faces)
        or any(len(face) != 3 or owners[face] != 1 for face in faces)):
    raise ValueError("Grip must contain distinct whole exterior triangles")
a, b, c = np.asarray(bundle.points)[faces].transpose(1, 0, 2)
area_mm2 = fsum(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1))
if not np.isclose(area_mm2, 0.5, rtol=0, atol=1e-8):  # use your area and tolerance
    raise ValueError(f"Unexpected grip area: {area_mm2:g} mm²")
grip_ids = {int(bundle.node_ids[i]) for face in faces for i in face}
result = solve_static_tet4(
    bundle, forces_by_id={}, fixed_dofs_by_id=held_axes,
    prescribed_displacements_m_by_id={node_id: {"x": 1.5e-9} for node_id in grip_ids},
)
```

This moves the patch +1.5 nm along global x. Its y and z displacements
remain free unless separately restrained; nodes off the patch do not receive
this prescribed motion. For a known force per unit area instead, use the
[traction helper](TET4_TRACTION.md), not the displacement mapping.

## Limits

Inputs are limited to 1,800 points, 8,000 Tet4 cells, and a structural budget
of 3,000,000 lower-plus-upper entries for the banded factor. Admission
depends on topology and constraints, not just counts. Numerical shape
checks and a positive stiffness floor reject unsuitable or poorly
conditioned inputs. The floor does not add stiffness to the physical solve.
Subnormal nonzero forces or unrepresentable displacement and positive
energy are rejected rather than silently rounded to zero.

Each independently face-connected body needs enough restraints to suppress
its rigid modes. Duplicate cells, unused nodes, faces incident to more
than two cells, and bodies joined only at an edge or vertex are rejected.
The solver does not certify absence of overlap, independently coincident
mesh pieces, or general spatial conformity. It does not support
surface tractions as an input format, nodal projection
of stress, contact, nonlinear mechanics, or dynamics. Element-wise strain
and stress are available below. No performance or numerical guarantee is
made across mesh families or platforms.

## Optional cell strain and stress

Request fields during the solve with `return_cell_fields=True`. The default
is `False`; `result.cell_fields` is then `None` and no field arrays are returned.
For nonzero prescribed motion, the solver still integrates cell energy.

```python
result = solve_static_tet4(
    bundle, forces_by_id=nodal_forces, fixed_dofs_by_id=held_axes,
    return_cell_fields=True,
)
fields = result.cell_fields
# Integrate U = Σ V σ:ε / 2. The double contraction counts both shear entries.
import numpy as np
energy_J = 0.5 * np.einsum("c,cij,cij->", fields.volume_m3, fields.stress_Pa, fields.strain)
```

- `Tet4CellFields` contains detached arrays in bundle **cell** order,
  with NumPy write-protection enabled. It prevents accidental edits;
  it does not make an array immutable if a caller deliberately re-enables
  writing. Use exact uint64 `cell_ids` and `material_ids` to label or
  join rows within that solve and bundle, without converting IDs to
  floating point. Equal IDs across solves do not establish that the
  mesh or material definitions are the same.
- `strain` and `stress_Pa` each have shape `(M, 3, 3)`; they are full
  symmetric tensors in global Cartesian axes. Strain is dimensionless:
  the shear entries are tensor shear, half the engineering shear.
  Cauchy stress is in Pa and tensile-positive. Each `volume_m3` is in m³.
  The solve uses the bundle's coordinate unit and the correct material
  for each cell without a caller-managed node join.
- Recovery uses the same bounded validated cells and constitutive model.
  Results are constant within each linear tetrahedron, not nodal averages
  or interface tractions. Do not interpret staircase-corner or support
  peaks as converged stresses or predictions of strength. Approximate
  cancellation can round nominally zero components; requesting fields is
  not a componentwise accuracy guarantee. Unrepresentable field recovery
  raises `ValueError`; it does not fill fields with zeros or NaNs.

## Read the welded-junction energy share

The current [three-cylinder example](../examples/tet4_static_welded_junction.py)
reports `core_strain_energy_fraction` for each material case. This is the
core's elastic strain energy divided by the entire solid's, a dimensionless
fraction from 0 to 1 (`null` if total energy is zero). Both cases use the
same centroid-based geometric core mask, even when all cells have the same
modulus. The example still requires positive work for its compliance report.

With the mesh, nodal forces, zero prescribed displacements and Poisson ratio
held fixed, this fraction is the local **negative log sensitivity** of
compliance to the core's Young modulus. For example, a fraction of 0.08
means a small 1% modulus increase lowers compliance by about 0.08% at
that material state. It is not a prediction for a large modulus change,
a different load or mesh, or for material strength.

### Check the sign under displacement control

Run [the control comparison](../examples/tet4_junction_load_control.py) on the [conformal two-region CAD fixture](TET4_VTU_BUNDLE.md#optional-conformal-junction-material).
The generator requires Gmsh and Meshio; the comparison reads the resulting
bundle with Meshio and does not run Gmsh.

```sh
python examples/tet4_fused_cylinder_cad.py --junction-sphere --output-dir fused-cad-sphere
python examples/tet4_junction_load_control.py fused-cad-sphere/coarse.zip fused-cad-sphere/coarse.json
python examples/tet4_junction_load_control.py fused-cad-sphere/fine.zip fused-cad-sphere/fine.json
```

Clamp the negative-x cap and apply a total force of (1, 0.27, −0.41) mN
at the positive-x cap. First solve with a 3 GPa junction and 1 GPa rods.
Then remove the applied force and prescribe the complete nodal displacement
profile from that solve at the loaded cap. The initial displacement field
and stored elastic energy are the same for both controls.

Perturb just the junction Young modulus, keeping Poisson ratio, the mesh and
each control fixed. For the uniform junction modulus used here,
`d ln(U)/d ln(E_j)` equals the junction energy share with a negative sign
under force control and a positive sign under displacement control. Here
`U` is total elastic strain energy and `E_j` the junction Young modulus.
The example reports a central finite difference beside each prediction.
The shares are about 0.77% and 0.66% on the two grids, so a small 1% modulus
increase changes total energy by only about 0.007% in this case.
This comparison does not establish a converged peak stress or predict strength.

## Compare two axial voxel phases at fixed stroke

With FiberNet 4.10.2 installed, run from a checkout containing this example:

```bash
python examples/tet4_static_welded_junction.py --axial-phase --tier quick --preflight
python examples/tet4_static_welded_junction.py --axial-phase --tier quick
python examples/tet4_static_welded_junction.py --axial-phase --tier reference
```

This opt-in compares the native sampling lattice with a half-cell shift in
x only. The CAD does not move. Half-width end cells keep the cap planes at
x = −1.5 and +1.5 mm. Within each tier, both phases must have the same
physical grip triangles. Both use homogeneous material (E = 1 GPa,
ν = 0.30), clamp the −x cap in all directions, and move the +x cap by
+3.291840836 nm in x only. The +x cap can move laterally, and there are no
separately applied forces. This is not the example's default traction test.

In axial-phase mode, `--preflight` emits the mesh and structural budget
checks for both phases without solving. A solve reports actuator and support
forces, stiffness, strain energy, independent energy quadrature and the
force and moment balances. The mode writes no VTU files.

| Tier (cell spacing) | Area per cap | Actuator force, native → shifted | Strain energy, native → shifted | Stiffness change |
| --- | ---: | ---: | ---: | ---: |
| Quick (0.25 mm) | 0.75000 mm² | 0.972509 → 0.950075 mN | 1.600672 → 1.563748 pJ | −2.31% |
| Reference (0.1875 mm) | 0.84375 mm² | 1.090175 → 1.079340 mN | 1.794341 → 1.776508 pJ | −0.99% |

Compare phases **within** one row: the two tiers have different cap
footprints. This measures the combined change in the voxelized geometry
and tetrahedral discretization for one axial shift. It is not a mesh
convergence test, a bound over other phases, or a comparison with a
body-fitted geometry. Overall volume is not a correction factor for stiffness.

## Compare three end-cap loads on one fused solid

Use the [CAD sphere procedure](RELEASE_4_21_3.md#compare-the-response-of-a-two-material-fused-junction)
to generate matching coarse and fine bundle/manifest pairs. Install the
optional Meshio reader as described there. Save the block below as
`examples/three_caps.py` beside `tet4_fused_cylinder_thermomechanics.py`.
From the checkout, run:

```sh
python examples/three_caps.py fused-cad-sphere/coarse.zip fused-cad-sphere/coarse.json
python examples/three_caps.py fused-cad-sphere/fine.zip fused-cad-sphere/fine.json
```

Clamp the negative-x cap in all three directions. For each of the three
solves, apply a total of 1 mN outward on just one of the +x, +y and +z caps.
Every other face is free. The surface force uses uniform traction in Pa;
its equivalent nodal forces use stable node IDs. The compliance entry
`C[i, j] = f_i · u_j / F²` gives the area-weighted displacement of cap
`i` along its load direction per unit force on cap `j` (m/N). The symmetric
part compares the same force combinations with the original mesh and with
only the junction modulus tripled.

```python
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh

from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction
from tet4_fused_cylinder_thermomechanics import load_named_mesh

bundle, manifest = map(Path, sys.argv[1:])
mesh, caps, names, _ = load_named_mesh(bundle, manifest, compare_junction=True)
ids = list(map(int, mesh.node_ids))
held = {ids[i]: ("x", "y", "z") for i in np.unique(caps["cap_neg_x"])}
force_N = 0.001
loads = []
for axis, direction in enumerate("xyz"):
    faces = caps["cap_pos_" + direction]
    a, b, c = (mesh.points[faces] * 1e-3).transpose(1, 0, 2)
    area_m2 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    traction = np.eye(3)[axis] * (force_N / area_m2)
    loads.append(nodal_forces_from_traction(
        mesh, [[ids[i] for i in f] for f in faces], traction_Pa=tuple(traction),
    ))

materials = {mid: dict(values) for mid, values in mesh.materials.items()}
materials[names["junction"]]["young_modulus_pa"] *= 3.0

def compliance(model):
    matrix = np.empty((3, 3))
    for j, load in enumerate(loads):
        result = solve_static_tet4(model, forces_by_id=load, fixed_dofs_by_id=held)
        displacement = dict(zip(map(int, result.node_ids), result.displacement_m))
        for i, probe in enumerate(loads):
            matrix[i, j] = sum(np.dot(f, displacement[n]) for n, f in probe.items()) / force_N**2
        np.testing.assert_allclose(
            0.5 * force_N**2 * matrix[j, j], result.strain_energy_J, rtol=2e-8,
        )
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=2e-8 * abs(matrix).max())
    return (matrix + matrix.T) * 0.5

reference = compliance(mesh)
reinforced = compliance(replace(mesh, materials=materials))
percent = 100 * eigh(reference - reinforced, reference, eigvals_only=True)
mix = np.array([0.7, -0.4, 0.5])
print("Reference compliance [m/N]:", reference.tolist())
print("Reinforced compliance [m/N]:", reinforced.tolist())
print("Min/max compliance reduction [%]:", percent[[0, -1]].tolist())
print("Mixed-load energy [pJ]:", (0.5e12 * force_N**2 * np.array(
    [mix @ reference @ mix, mix @ reinforced @ mix],
)).tolist())
```

With the synthetic 1 GPa starting material, the compliance reduction over
these combinations is about 0.54–2.95% on the coarse mesh and 0.48–2.92%
on the fine mesh. The selected signed combination stores about 0.99% and
0.94% less energy at the same forces, respectively. The energy in the
snippet follows from the compliance matrix and linear superposition.
In follow-up solves of the signed load, it agreed with the directly solved
strain energy on both meshes.

The individual cross-cap responses are small: their signs agree, but some
magnitudes differ by up to 75% between the two meshes. Use the global
compliance comparison to interpret this result. Do not use an individual
cross term as a precise sensitivity, or infer local stress, strength,
or mesh convergence from two grids.

## View a static result in ParaView

To compare both welded-junction materials without writing your own script,
use [the packaged example and `--write-vtu-dir` recipe](RELEASE_4_7_0.md#write-both-vtu-results).

To write a **new** visualization file from the same validated solve, install
the optional Meshio dependency (`python -m pip install meshio==5.3.5`) and
use the explicit result exporter:

```python
from fibernet.io.tet4_results import solve_static_tet4_to_vtu

result = solve_static_tet4_to_vtu(
    "static-result.vtu", bundle,
    forces_by_id=nodal_forces, fixed_dofs_by_id=held_axes,
)
```

The file contains the undeformed mesh with coordinates in metres; it is
not automatically warped. In ParaView, open the VTU and select a
**Cell Data** component of `stress_Pa` or `strain` to show per-cell values.
The optional Warp By Vector filter can display displacement separately.
The returned `Tet4StaticResult` includes `cell_fields`.

| Association | VTU arrays |
| --- | --- |
| Point Data | `node_id` (UInt64), `displacement_m` (three global Cartesian components, m) |
| Cell Data | `cell_id`, `material_id` (UInt64); `strain`, `stress_Pa` (nine components); `volume_m3` |

Tensor order is **xx, xy, xz, yx, yy, yz, zx, zy, zz**. Strain is dimensionless
tensor strain, not engineering shear. Cauchy stress is tensile-positive and
reported in Pa. No stress or strain is interpolated to nodes. Open the
result as a visualization, not as input: this is a one-way VTU, without
the versioned input ZIP manifest, and its additional diagnostic arrays are
not permitted by the strict [v1 Tet4 exchange reader](TET4_VTU_BUNDLE.md).
Material IDs label the cells but the file does not archive the material
definitions (Young's modulus or Poisson ratio).

The exporter stages a bounded file (at most 4 MiB after serialization) in
the destination directory and publishes it without replacing another file.
Use a trusted, existing parent directory that remains stable during
export. The exporter resolves the parent once and does not follow an
existing final pathname. This is not a path confinement mechanism. The
filesystem must support atomic hard links; if unsupported, the exporter
fails rather than falling back to an overwrite. This is not a
crash-durability guarantee.

## Assign a different material on the same mesh

Use stable cell IDs, not a saved row mask, to change the material of selected
cells. The example helper works with a validated bundle or a small explicit
fixture like the one below. Run from the source checkout root with FiberNet,
NumPy and SciPy installed; this snippet does not need Meshio, a file, or a mesher.

```python
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission, solve_static_tet4
from examples.tet4_stable_material_retag import mesh_identity, reassign_cells

bundle = Tet4Bundle(
    points=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [0., 0., 1.], [0., 0., -1.]],
    tetrahedra=[[0, 1, 2, 3], [0, 2, 1, 4]],
    node_ids=[0, 10, 20, 30, 40], cell_ids=[0, 2**64 - 1],
    material_ids=[1, 2], length_unit="mm",
    materials={1: {"young_modulus_pa": 1e9, "poisson_ratio": 0.25},
               2: {"young_modulus_pa": 2e9, "poisson_ratio": 0.25}},
)
signature = mesh_identity(bundle)  # record with the selected IDs
variant = reassign_cells(
    bundle, expected_mesh_identity=signature, cell_ids=[2**64 - 1],
    material_id=7, material={"young_modulus_pa": 3e9, "poisson_ratio": 0.25},
)
print(variant.material_ids.tolist())  # [1, 7]
holds = {key: ("x", "y", "z") for key in (0, 10, 20)}
for case in (bundle, variant):
    assert inspect_static_tet4_admission(case, fixed_dofs_by_id=holds).within_factor_budget
    result = solve_static_tet4(case, fixed_dofs_by_id=holds,
                              forces_by_id={40: (0., 0., -0.001)})
    print(f"Node 40 axial displacement: {result.displacement_m[4, 2]:.3g} m")
```

The helper leaves points, connectivity and stable node/cell IDs unchanged.
It preserves an existing target's properties and removes only material
records no longer referenced by any cell. Keep the original selection
signature: it tolerates reordered rows with correctly remapped connectivity,
not a remesh, changed coordinates, units, or local vertex order. It binds the
mesh, not the original phase assignment: record the original materials
separately if the study depends on them. This is an example, not a new solver
entry point. Native preflight and solve checks still apply.
