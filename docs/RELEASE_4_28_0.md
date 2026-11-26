# FiberNet 4.28.0

## Bring your own Gmsh mesh

Run prescribed-grip elasticity on a connected solid from an ASCII Gmsh 2.2
or 4.1 mesh with linear tetrahedra. Reuse the existing physical volume and
surface names or use the example defaults. The example exports stable node
IDs, displacement in metres, and reactions in newtons, plus a summary with
the input stroke and strain energy. It supports two material groups and
distinct exterior grips. See the [mesh walkthrough](TET4_VTU_BUNDLE.md#run-a-tagged-mesh-above-the-banded-solver-limits).

Install the released solver and obtain the corresponding example files:

~~~sh
python -m pip install "fibernet[solid-iterative]==4.28.0" "meshio==5.3.5"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.28.0"
python -m tarfile -e fibernet-4.28.0.tar.gz .
cd fibernet-4.28.0
python -m examples.tet4_tagged_grip_iterative --output tagged-grip-run
~~~

The example generates its own fixture if you omit `--input`. When
supplying a mesh, declare its coordinate unit with `--length-unit`. Use
`--material-groups "Soft phase" "Stiff phase"` and
`--grip-groups "Fixed end" "Driven end"` to use the names in that file.

## Compare mechanical load cases without repeating setup

Add `solver="amg-cg"` to `solve_static_tet4_load_cases` for up to eight
nodal force cases on one solid with unchanged elastic materials and
zero-displacement restraints. Each case has its own displacement,
reaction, energy and residual checks. The solver validates inputs and
builds the stiffness matrix and AMG hierarchy once. The default banded
solver and the single-case prescribed motion interface are unchanged.

The optional iterative mechanics path also reduces time spent assembling
element stiffness and reuses face connectivity for rigid-mode checks.
In a controlled one-thread example on 186,388 tetrahedra, eight different
force distributions took 45 seconds instead of 76 seconds when each
one prepared its own system. Displacements, reactions and energies
matched within numerical tolerances. This measures one fixed-restraint
force study, not every mesh or coupled simulation. See
[static mechanics](TET4_STATIC.md#solve-several-load-cases-on-the-same-solid).

## Get the initial geometric resistance slope

After downloading and extracting the release as above, run:

~~~sh
python -m examples.tet4_iterative_mechanoelectric_sweep --initial-slope
~~~

The example reports the initial change in relative resistance per
micrometre of grip travel, separating contributions from stretch along
the local current and contraction transverse to it. It uses one DC solve
on the reference geometry at fixed conductivity and electrode voltages.
Omit the flag to compute the five finite-stroke states using separate DC
solves; those states are not inferred from the tangent.
