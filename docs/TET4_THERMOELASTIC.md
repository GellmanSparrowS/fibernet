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

## Compare materials on a three-cylinder junction

From a source checkout with FiberNet 4.13.0 or newer installed, run:

```bash
python -m examples.tet4_thermoelastic_welded_junction
```

The [example](../examples/tet4_thermoelastic_welded_junction.py) uses a
single conforming mesh of a voxel approximation to the union of three
orthogonal cylinders. It clamps the two x-end caps and holds geometry,
stiffness and a 308–318 K nodal temperature gradient fixed. It compares
uniform expansion with a larger expansion coefficient near the center,
reducing the outer coefficient so the volume integral of the free
linear thermal strain, alpha times temperature rise, stays the same.
The center contains whole tetrahedra whose centroids lie within 0.45 mm
of the junction center; it is not a fitted spherical interface. The
equal integral is not the measured expansion of the clamped structure.

The JSON reports the axial cap reactions, total elastic energy and the
center's share of that energy. It also checks force and moment balance
and the energy bound provided by holding every node at zero displacement.
The numerical values describe this synthetic discrete model, not a
mesh-convergence study or a fitted material.

## Inspect a snapshot in ParaView

With Meshio 5.3.5 installed, use the optional solve-and-write route to
export one diagnostic from the same mesh and temperature field:

```python
from fibernet.io.tet4_results import solve_prescribed_temperature_tet4_to_vtu

result = solve_prescribed_temperature_tet4_to_vtu(
    "thermal_snapshot.vtu",
    bundle,
    temperature_K_by_id=nodal_temperature_K,
    stress_free_temperature_K=300.0,
    alpha_per_K_by_material=thermal_expansion,
    forces_by_id={},
    fixed_dofs_by_id=held_axes,
)
```

The mesh uses undeformed coordinates in metres; display
`temperature_K` as Point Data, or apply **Warp By Vector** to
`displacement_m`. Cell Data includes the global row-major
`mean_stress_Pa` tensor (nine components) and `elastic_energy_J` per
cell, which is integrated energy, not energy density. Stable node,
cell and material IDs are stored as UInt64.

Choose a new .vtu name in a stable local directory that supports
hard links. The writer does not overwrite existing files and limits
each output to 4 MiB. The diagnostic is for visualization, not a restart
file, a strict Tet4 interchange bundle, or an export of arbitrary
previously computed results.

## Hand off a solved electrothermal field on the tagged CAD solid

Run the [tagged fused-cylinder CAD example](TET4_VTU_BUNDLE.md#generate-a-tagged-fused-cylinder-cad-mesh)
first. Only use a generator output directory containing `COMPLETE`.
Then run this example with either matching ZIP and JSON pair; the consumer
checks the pair, not the entire directory. It requires FiberNet 4.21.1
and Meshio; the consumer itself does not use Gmsh.

```sh
python examples/tet4_fused_cylinder_thermomechanics.py fused-cad-output/coarse.zip fused-cad-output/coarse.json
python examples/tet4_fused_cylinder_thermomechanics.py fused-cad-output/coarse.zip fused-cad-output/coarse.json --supports both-clamped
```

Use locally trusted files. The manifest SHA-256 must match the exact
bundle bytes read, and named faces must cover the exterior without
overlap. Both files declare millimetre geometry. This example uses the
single homogeneous solid material from the bundle; its conductivity
inputs (2 S/m and 0.25 W/(m K)) and expansion coefficient (10⁻⁴/K) are
illustrative, not measured.

The two x caps are electrical terminals at 0 and 0.3 V and thermal
boundaries held at 300 K. The remaining exterior exchanges heat with
299.95 K ambient at 100 W/(m² K). The native DC/Joule/Robin call returns
absolute node temperatures. The example passes these temperatures by
stable node ID to the static thermoelastic solve, with 300 K as the
stress-free reference. Temperature and deformation do not feed back
upstream.

For mechanics the default clamps the negative-x cap and leaves the
positive-x cap free; `--supports both-clamped` clamps both. The JSON
reports free-end displacement (the arithmetic mean over positive-x cap
nodes, not an area-weighted mean) or clamp reaction, force and moment
balance, elastic energy, and signed outward heat flow. Negative heat-out
means heat enters the solid. The terminal mechanical pairing is zero here
because no loads or moving supports do mechanical work; stored elastic
energy need not be zero. No mesh convergence or material contrast is
implied.
