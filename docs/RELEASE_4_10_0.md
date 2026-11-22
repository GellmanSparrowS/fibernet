# FiberNet 4.10.0

## Prescribe a measured displacement in a Tet4 solve

Use `prescribed_displacements_m_by_id` to specify a signed displacement
in **metres** along a global axis at a stable node ID, even if the
geometry uses millimetres. For example, with a `Tet4Bundle` and selected
grip nodes:

```python
from fibernet.sim.tet4 import solve_static_tet4

result = solve_static_tet4(
    bundle,
    forces_by_id={},
    fixed_dofs_by_id=held_axes,
    prescribed_displacements_m_by_id={node_id: {"z": 1.5e-9} for node_id in grip_ids},
)
```

The same optional input is available for structural preflight and
optional Meshio VTU export. Do not list a node-axis in both the zero
restraints and prescribed displacements. Reactions are forces from a
support or actuator **on the mesh**. In a mixed load case, half external
work accounts for both applied forces and moving supports under
proportional loading from zero. Existing force-only calls, Tet4 mesh and
VTU field formats, and the linear elastic model are unchanged. See the
[Tet4 static guide](TET4_STATIC.md#prescribe-a-measured-displacement).

## Compare three layouts at the same stroke

Install the package and, in an empty directory, download and unpack the
matching source distribution:

```bash
python -m pip install "fibernet==4.10.0"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.10.0"
python -m tarfile -e fibernet-4.10.0.tar.gz
cd fibernet-4.10.0
python examples/tet4_static_fitted_placement.py --fixed-cap-stroke-m=3.2918408360037934e-9
```

The script and its frozen input are included in the source distribution,
not installed by the wheel. The example uses the same joined-cylinder
mesh for uniform, center-stiffened, and arm-stiffened material. It fixes
the negative x cap and moves every positive x cap node by the same axial
stroke with transverse motion free. It compares signed grip forces,
stiffness and elastic energy. It needs neither Gmsh nor Meshio. This is an
equal-displacement comparison on one mesh, not uniform traction or a
claim about strength or convergence. See the
[fitted junction guide](TET4_FITTED_PLACEMENT.md#impose-the-same-axial-stroke-on-both-material-placements).
