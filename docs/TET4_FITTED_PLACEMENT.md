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

## Keep both x end caps and change only the force direction

```bash
python examples/tet4_static_fitted_placement.py --fixed-cap-compliance
```

The −x cap stays clamped and the +x cap receives the same spatially
uniform traction pattern for all three 1 mN bases: x normal and y, z
tangential. This is different from `--load-axis y` or `z`, which move
the clamp and load to another pair of caps. This mode cannot be
combined with `--load-axis`, `--preflight`, or `--compliance-sensitivity`.

The example uses nine sequential solves: three bases for each of the
uniform, center and offset layouts. It reports the work matrix
`W_ij = F_i^T u_j` in joules and the reciprocal compliance matrix
`C = W/F0²` in m/N, with `F0 = 1 mN`. For a unit direction vector
`v = (v_x, v_y, v_z)`, compliance is `v^T C v`.
Generalized eigenvalues bound the offset layout's extra compliance
reduction relative to the uniform layout over all directions.
A near-zero result within the numerical margin is inconclusive.
This result is for the fixed mesh, material model and uniform
traction on one cap. It is not an angular sweep, a bound for other
spatial load patterns, or a mesh-convergence result.

## Impose the same axial stroke on both material placements

For a FiberNet version that provides prescribed Tet4 displacements
(not 4.9.2), run from the source checkout:

```bash
python examples/tet4_static_fitted_placement.py --fixed-cap-stroke-m=3.291840836e-9
```

This opt-in runs **three** solves (uniform, junction, arm) at the same
signed stroke in metres. Negative values impose compression; use the
`--fixed-cap-stroke-m=-3.291840836e-9` form for a negative stroke.
The −x cap is fixed in x, y and z. Each +x cap node moves by the given
amount in x, while its y and z components are free. There are no
separately applied nodal loads. This is not the uniform-traction
boundary condition used in the other modes. The stroke option cannot
be combined with `--load-axis` or other mode options.

The JSON reports the signed x forces exerted **on the mesh** by the +x
actuator and −x support (N), axial stiffness (actuator force divided by
stroke, N/m), and energy (J). Half actuator work is half the actuator
force times the stroke, for proportional loading from zero. It is
checked against integrated elastic energy and force/moment balance.
The three layouts share the same fitted mesh. The comparison does not
establish strength, buckling or mesh convergence.
