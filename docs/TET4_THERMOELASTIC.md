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

### Compare junction conductivity at equal total Joule power

First generate the conformal two-region mesh with the
[optional junction sphere](TET4_VTU_BUNDLE.md#optional-conformal-junction-material).
Use a matching bundle and manifest from a directory containing `COMPLETE`:

```sh
python examples/tet4_fused_cylinder_thermomechanics.py fused-cad-sphere/coarse.zip fused-cad-sphere/coarse.json --compare-junction-conductivity
```

With this explicit flag, the example compares uniform electrical
conductivity (2 S/m) to a lower value (0.5 S/m) in the junction.
The mesh, its two regions, the exterior faces, and the thermal and
elastic properties stay fixed. The example first checks the lower
conductivity at 0.3 V, then runs a second full electrothermal solve with
adjusted voltage and checks that its total Joule power matches baseline.
Both solved temperature fields feed the same mechanical support mode.
The internal material interface is not a convection boundary.

The comparison uses illustrative parameters. The sphere interface and
curved exterior are faceted; each comparison holds its own mesh fixed.
Small differences in displacement or reaction are not evidence of
convergence or validated material behaviour.

### Change junction thermal expansion under uniform heating

Generate one [two-region fused bundle](TET4_VTU_BUNDLE.md#optional-conformal-junction-material)
with Gmsh and Meshio, then run the new comparison on the matching pair
from that completed output. The comparison reads the bundle with Meshio
but does not run Gmsh, electrical transport, or heat transfer.

```sh
python examples/tet4_fused_cylinder_cad.py --junction-sphere --output-dir fused-cad-sphere
python examples/tet4_fused_cylinder_thermomechanics.py fused-cad-sphere/coarse.zip fused-cad-sphere/coarse.json --compare-junction-cte
```

Both cases start stress-free at 300 K and use a prescribed uniform 310 K,
unchanged geometry and elastic properties, and no external mechanical loads.
The arms stay at 10 ppm/K while the junction goes from 10 to 30 ppm/K.
Instead of clamping a whole cap, the script fixes the rigid-body reference:
three degrees at A, two at B and one at C. B's y component follows the
free affine expansion (specified in metres) so equal expansion remains
stress-free. The script checks the resulting support forces and work are
negligible; do not combine this mode with `--supports`.

Compare elastic energy in joules and volume-weighted hydrostatic mean stress
in pascals (positive in tension). Equal expansion gives near-zero stress and
energy. In this example the larger junction coefficient produces mean
compression in the junction and balancing tension in the surrounding arms.
These are global and regional results for an illustrative welded solid,
not peak interface stress, strength, or a convergence claim.

To inspect the 30 ppm/K junction case in ParaView, add
`--write-cte-vtu mismatch.vtu` to the comparison command. This writes only
the mismatch case and still solves each case once. Use a new filename in
an existing trusted directory that supports atomic hard links. Check the
command exits successfully to confirm publication; the stdout JSON reports
only the comparison, not whether the file was published.

The VTU contains undeformed points in metres; view `temperature_K` or warp
by the nodal `displacement_m`. The cell field `mean_stress_Pa` contains nine
row-major components in global axes. For hydrostatic stress in pascals,
take one third of the sum of its xx, yy, and zz components (indices 0, 4,
8); this is a cell mean, not an interface peak. Stable UInt64 IDs and
integrated cell energies are also retained. This is a diagnostic, not a
restart file or Tet4 exchange bundle.

### Change junction stiffness with the temperature held fixed

Use the same two-region CAD mesh to isolate Young's modulus instead of
conductivity. Use a repository checkout that includes the cap-area helper
from PR #363 or newer. The installed FiberNet 4.21.2 core works, but its
source archive's example does not include this helper. Save the Python
block below as `examples/tet4_compare_junction_stiffness.py` in the
checkout, then run this command from the repository root:

```sh
python examples/tet4_compare_junction_stiffness.py
```

Use a completed local CAD output directory and change both paths
together to use a different mesh.

```python
from dataclasses import replace
from pathlib import Path

import numpy as np

from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4_with_convection
from fibernet.sim.tet4_thermoelastic import solve_prescribed_temperature_tet4
from tet4_fused_cylinder_thermomechanics import (
    LEFT, RIGHT, cap_area_mean_x, load_named_mesh,
)

directory = Path("fused-cad-sphere")
mesh, faces, materials, _ = load_named_mesh(
    directory / "coarse.zip", directory / "coarse.json", compare_junction=True,
)
ids = mesh.node_ids
left, right = (np.unique(faces[name]) for name in (LEFT, RIGHT))
electrodes = {**dict.fromkeys(map(int, ids[left]), 0.0),
              **dict.fromkeys(map(int, ids[right]), 0.3)}
exterior = np.concatenate([faces[name] for name in sorted(faces)
                           if name not in (LEFT, RIGHT)])
thermal = solve_dc_thermal_tet4_with_convection(
    mesh, conductivity_S_per_m_by_material=dict.fromkeys(materials.values(), 2.0),
    electrode_voltage_V_by_id=electrodes,
    conductivity_W_per_m_K_by_material=dict.fromkeys(materials.values(), 0.25),
    prescribed_K_by_id=dict.fromkeys(electrodes, 300.0),
    faces_by_node_id=ids[exterior], convection_W_per_m2_K=100.0, ambient_K=299.95,
).thermal
np.testing.assert_array_equal(thermal.node_ids, ids)
temperature = {int(i): float(t) for i, t in zip(thermal.node_ids, thermal.temperature_K)}
junction = materials["junction"]
for factor in (1.0, 2.0):
    properties = {key: dict(value) for key, value in mesh.materials.items()}
    properties[junction]["young_modulus_pa"] *= factor
    elastic_mesh = replace(mesh, materials=properties)
    for mode, held in (("left-clamped", left), ("both-clamped", np.r_[left, right])):
        result = solve_prescribed_temperature_tet4(
            elastic_mesh, temperature_K_by_id=temperature,
            stress_free_temperature_K=300.0,
            alpha_per_K_by_material=dict.fromkeys(materials.values(), 1e-4),
            forces_by_id={}, fixed_dofs_by_id=dict.fromkeys(map(int, ids[held]), ("x", "y", "z")),
        )
        np.testing.assert_array_equal(result.node_ids, ids)
        np.testing.assert_array_equal(result.displacement_m[held], 0.0)
        if mode == "left-clamped":
            np.testing.assert_array_equal(result.reaction_N[right], 0.0)
            displacement_m, _ = cap_area_mean_x(
                mesh.points * 1e-3, faces[RIGHT], result.displacement_m,
            )
            print(f"E_junction x{factor:g}: free-end mean u_x = {displacement_m:.6g} m")
        else:
            reaction_N = result.reaction_N[right, 0].sum()
            print(f"E_junction x{factor:g}: positive-x clamp reaction = {reaction_N:.6g} N")
```

The single temperature solution is reused unchanged for both stiffnesses.
Geometry, Young's modulus in the bulk region, Poisson ratios, thermal
expansion coefficients and external forces also stay fixed; compare like support modes. Clamp
reaction measures the force needed to hold the cap, not a free-end force.
In this synthetic example the reaction change is clearer than the small
free-end displacement difference, which is sensitive to the discrete mesh.
Neither is a material calibration or convergence result.
