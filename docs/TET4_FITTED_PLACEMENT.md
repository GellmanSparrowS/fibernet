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

Use `--load-axis y` or `--load-axis z` to clamp the negative cap and
pull the positive cap on that axis with the same 1 mN resultant. Both
choices also work with `--preflight`. The mesh and material placements
stay fixed, including the sphere at x = 0.82 mm. The y and z caps are
smaller in the discrete mesh, so the uniform traction is about
1,361 Pa instead of 1,291 Pa for x.

On this mesh, the arm placement reduces compliance more than the
junction placement under x loading; the ordering reverses under y
and z loading. Swapping y and z preserves the ideal geometry and
material placements, but not the actual triangulation. The y and z
results are close, not identical. These comparisons do not establish
a general location ranking or mesh convergence.

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

## How sensitive is compliance to the modulus contrast?

```bash
python examples/tet4_static_fitted_placement.py --load-axis x --compliance-sensitivity
```

This adds a `compliance_sensitivity` section to the JSON without adding
solves. It is not available with `--preflight`, which does not solve.
For each placement, the positive slopes measure how fast compliance
`C/C0` decreases per unit increase in `r = E_sphere / E_base`. The
baseline `C0` is the uniform result for the selected load. The report
gives the slopes at `r = 1` and `r = 3`, and their interval's average
(compliance reduction divided by 2, not an average of the endpoints).

At fixed nodal force and Poisson ratio, the element modulus is
`E_i/E_base = 1 + (r - 1) phi_i`. The slope is the sum of
`[phi_i / (1 + (r - 1) phi_i)] * U_i(r) / U0` over elements.
Here `phi_i` is the frozen sphere-volume fraction, `U_i(r)` is that
element's elastic energy and `U0` is total uniform elastic energy.
This is not a log derivative or a prediction for another mesh or load.
