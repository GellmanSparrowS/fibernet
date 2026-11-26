# FiberNet 4.29.0

## Choose a resolution for the 3D force–resistance example

From the matching source archive, with the optional `solid-iterative`
dependencies installed:

~~~sh
python -m examples.tet4_iterative_mechanoelectric_sweep --refinement 5
python -m examples.tet4_iterative_mechanoelectric_sweep --refinement 5 --initial-slope
~~~

The default mesh contains 28,512 tetrahedra. Choose 67,584 or 132,000
tetrahedra to study resolution or runtime on the same generated specimen.
The finite sweep reports five separate resistance solutions; the
initial-slope mode uses the undeformed electrical solution to measure the
small-stroke geometric sensitivity at fixed conductivity. To compare load
locations at fixed electrical readout, run
`python -m examples.tet4_mechanoelectric_patch_screen --verify`.

## Faster topology validation in iterative DC

The Tet4 CG solver reuses packed connectivity validation from iterative
mechanics. Duplicate cells and non-manifold faces still report the first
invalid input row. In a one-thread benchmark on the 132,000-tetrahedron
example, five DC solves on identical pairs of mesh states took a median
5.72 seconds versus 8.17 seconds with the previous connectivity check;
all returned electrical field arrays and scalars matched exactly. These
times cover the electrical stage only and depend on the mesh and machine.

## Compare loaded pore states against one reference

`fibernet.sim.tet4_pore.solve_matched_pore_geometry_cases` solves the
undeformed reference once for up to eight deformed geometries with
matching topology and the same pressure-gradient driver. It solves and
validates each deformed Stokes state independently. With the
`pore-flow` extra installed, try
`python -m examples.tet4_solid_to_pore_flow --matched-force`.

## Use a homogeneous Gmsh solid and inspect it in ParaView

The tagged-grip example accepts one volume material name and modulus
with `--material-group` and `--young-modulus-gpa`. Its two-material
mode remains available. Add `--write-vtu` to open the computed nodal
displacements and reactions directly in ParaView. See the [tagged-mesh
guide](TET4_VTU_BUNDLE.md#run-a-tagged-mesh-above-the-banded-solver-limits).
