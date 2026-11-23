# Experimental Tet4 mechanics at prescribed temperature

Use `fibernet.sim.tet4_thermoelastic.solve_prescribed_temperature_tet4`
to compute one small-strain static elastic snapshot of a conforming
[Tet4Bundle](TET4_VTU_BUNDLE.md). This is a bulk solid calculation, not
the 3D beam or saved electrothermal workflow. It does not solve heat
transfer or feed the deformation back into a temperature field.

```python
from fibernet.sim.tet4_thermoelastic import solve_prescribed_temperature_tet4

result = solve_prescribed_temperature_tet4(
    bundle,
    temperature_K_by_id=nodal_temperature_K,
    stress_free_temperature_K=300.0,
    alpha_per_K_by_material=thermal_expansion,
    forces_by_id={},                  # Or stable node IDs to (Fx, Fy, Fz), in N.
    fixed_dofs_by_id=held_axes,        # Stable node IDs to axes, e.g. ("x", "z").
)
```

Supply absolute kelvin at every stable node ID and signed isotropic
expansion coefficients (1/K) for every declared material ID. Each
tetrahedron has constant material properties; temperature varies
linearly between its four nodes. Geometry follows the bundle's declared
length unit; optional prescribed displacements are always in metres.
Use locally trusted mesh and control data without concurrent mutation.
The [static solver's constraints and size checks](TET4_STATIC.md) still
apply, plus the thermal solver's documented narrower numerical ranges.
These are admission checks, not physical stability or material calibration.

Node and cell output rows are bound to the returned IDs. Strains are
tensors; stresses are tensile-positive **cell means** in Pa, not peak
or interface stresses. Stored energy (J) integrates the elastic strain
against the affine temperature field in each Tet4, including within-cell
temperature variance. Energy is therefore not generally determined by
the mean cell stress alone. The terminal mechanical pairing is not
cumulative external work.

Exactly opposite represented mechanical and thermal load contributions
can sum to zero. The pilot rejects a nonzero contribution lost in
addition or a near cancellation too small relative to those inputs,
rather than presenting a fragile remainder as a reliable load. A
permitted exact zero still undergoes the physical equilibrium and
energy checks.
