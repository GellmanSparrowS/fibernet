# Bounded linear Tet4 static solve

Solve small 3D problems with conforming four-node tetrahedra, one isotropic
linear elastic material per cell, nodal forces, and zero Cartesian restraints.
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

Create a [Tet4Bundle](TET4_VTU_BUNDLE.md) in memory or read it with the
optional Meshio adapter. Solving itself does not import or need Meshio.
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
mesh-convergence claim.

## Contract

- Connectivity uses point row indices within the bundle. Forces and
  restraints use stable nonnegative node IDs through `2**64-1`. Forces
  must contain three finite real scalars exactly representable as binary64.
  Each restraint lists distinct Cartesian axes; unspecified axes are free.
- Coordinates are converted to m using the bundle's length unit. Its modulus
  is specified in Pa and Poisson ratio is dimensionless. Supports impose
  zero displacement; an applied load on a held axis contributes to the
  reaction but does not do work.
- The detached `Tet4StaticResult` holds `node_ids`, `displacement_m`,
  `reaction_N`, and `fixed_mask` in input node order. Reaction convention
  is support force on the mesh, `Ku-f` on held axes and zero on free axes.
  The result also holds `strain_energy_J`, `half_external_work_J`,
  `free_residual_inf_N`, `normalized_backward_error`, and
  `relative_energy_work_error`. A small normwise backward residual is not
  a per-component or forward-error guarantee.

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
mesh pieces, or general spatial conformity. It does not support nonzero
prescribed motion, surface tractions as an input format, nodal projection
of stress, contact, nonlinear mechanics, or dynamics. Element-wise strain
and stress are available below. No performance or numerical guarantee is
made across mesh families or platforms.

## Optional cell strain and stress

Request fields during the solve with `return_cell_fields=True`. The default
is `False`; `result.cell_fields` is then `None` and no fields are computed.

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
