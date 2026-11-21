# Equal-volume stiffening in a fitted Tet4 junction

This example clamps the −x cap of three joined circular cylinders and
applies a 1 mN resultant in +x to the opposite cap. It compares uniform
material with two placements of the same stiffening volume on the **same**
2,241-tetrahedron mesh, at the junction and along the loaded arm.

With FiberNet 4.9.0 installed, run from a source checkout (or unpack the
matching source distribution **that contains this example**):

```bash
python examples/tet4_static_fitted_placement.py --preflight
python examples/tet4_static_fitted_placement.py
```

The first command checks the inputs and structural factor budget without
solving. The second runs three solves and reports compliance, recovered
strain energy and force/moment balance as JSON. Neither command needs
Gmsh or Meshio. Keep the adjacent `examples/fixtures/fitted_axial_B.json`
file with the script.

Each placement is a sphere of radius 0.35 mm, centered at (0,0,0) or
(0.82,0,0) mm. A cell uses E = (1 + 2φ) GPa and ν = 0.30, where φ is the
part of its volume inside the sphere. Both assignments have the same
integrated sphere volume; neither changes a cell that shares a node
with an end cap. These are volume-averaged properties on cut cells,
not a fitted two-material interface or a claim about peak stress,
material strength or mesh convergence.

The fixture stores only inputs. It comes from the original analytical
three-cylinder CAD and mathematical sphere–tetrahedron integration
prepared for this example. The fractions were integrated before packaging
by adaptive Gauss–Legendre quadrature of slice areas, checked using a
second slicing direction and against the analytic sphere volume. They
are not centroid labels. The repository copy includes the optional
[frozen CAD input](../examples/fixtures/fitted_axial_B.geo) for Gmsh
4.13.1/OpenCASCADE. Remeshing is not required, may change the mesh with
different provider builds, and does not reconstruct the stored fractions
without a separate integration step.
