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
prescribed motion, surface tractions as an input format, stress export,
contact, nonlinear mechanics, or dynamics. No performance or numerical
guarantee is made across mesh families or platforms.
