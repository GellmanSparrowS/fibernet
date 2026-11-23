# Experimental steady Tet4 conduction

Import `solve_steady_conduction_tet4` from `fibernet.sim.tet4_conduction`. The
solver uses one source-free, steady scalar temperature field over a validated
`Tet4Bundle` with four-node tetrahedra:

```python
result = solve_steady_conduction_tet4(
    bundle,
    conductivity_W_per_m_K_by_material={17: 1.0, 9: 4.0},
    prescribed_K_by_id={hot_node_id: 330.0, cold_node_id: 300.0},
)
```

For a cap, include all node IDs on the cap in the map. At least one prescribed
node is required per connected component; other exterior faces are insulated
in the weak-form sense. This does not impose specified flux, internal sources,
or transient effects. The existing Tet4 material record schema is needed
to validate the bundle; its elastic constants are not used in this solve.

`temperature_K` contains absolute nodal values; `reference_K_by_node` and
`temperature_difference_K` retain the per-component centered representation.
Prescribed temperatures are returned exactly. Out-of-range or unresolved
absolute-temperature results are rejected, not clipped. `flux_W_per_m2` is
the cellwise Fourier flux and `heat_into_solid_W` is the discrete heat entering
through the prescribed nodes (zero on free rows). A positive value means heat
enters; a negative value means heat leaves. The Dirichlet quadratic, terminal
pairing, and free residual pairing have units W K, not joules or watts.
Plain P1 Tet4 does not ensure pointwise heat flux continuity or a discrete
maximum principle on arbitrary meshes.

Prototype limits include 1024 nodes, 8000 tets, absolute inputs and outputs
in [1, 2000] K, conductivity in [1e-6, 1e6] W/(m K), plus geometry, residual,
condition and factor budget checks. It has no reverse thermoelastic feedback.

## Prescribe signed volumetric heating or cooling

Use the separate experimental entry point when the source is already known:

```python
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_source

result = solve_steady_conduction_tet4_with_source(
    bundle,
    conductivity_W_per_m_K_by_material=conductivity,
    prescribed_K_by_id=boundary_temperatures,
    source_W_per_m3_by_cell=source_density,
)
```

The source map uses stable **cell IDs**, not material IDs or row indices.
Supply a value in W/m³ for every cell, including explicit zeros. Positive
values generate heat; negative values remove it. Every connected component
still needs a prescribed temperature. The numerical pilot accepts nonzero
source magnitudes from 1e-8 through 1e8 W/m³ and can reject a load too small
to remain resolved in the equations or in the returned absolute temperatures.

The result retains the source-free fields and adds signed integrated source
per cell and per node (W), total source (W), source pairing (W K), and
`balance_W`, the sum of prescribed-node heat entering the solid and total
source. They balance to the numerical residual; a zero total source does
not mean zero local heat transport. The Dirichlet quadratic equals the
terminal, source and free residual pairings summed using component-centered
temperatures, not stored energy. This is a prescribed source, not a
temperature-dependent law, Joule heating calculation, or transient solve.

## Prescribe signed surface heat flux

Use the separate surface-only entry point to apply one uniform heat flux to
explicit exterior triangles:

```python
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_surface_flux

result = solve_steady_conduction_tet4_with_surface_flux(
    bundle,
    conductivity_W_per_m_K_by_material=conductivity,
    prescribed_K_by_id=boundary_temperatures,
    faces_by_node_id=[(node_a, node_b, node_c), (node_c, node_b, node_d)],
    inward_flux_W_per_m2=100.0,
)
```

Each triple uses three distinct stable **node IDs**. Order within each face
does not matter. The list must be nonempty, faces must not repeat, and each
must belong to exactly one tetrahedron. This is a topological exterior check,
not a check for spatial conformity or overlap. A positive flux adds heat to
the solid and a negative flux removes it; no normal orientation is needed.
This is one scalar in W/m² for all selected faces, not a total in watts, a
per-face map, or a volumetric source. Other exterior faces without a prescribed
temperature or applied flux remain insulated in the weak form.

For a triangle of area A in m², each of its three nodes receives qA/3 in W.
Zero flux is allowed and still validates the selection. Nonzero magnitudes
are limited to 1e-8 through 1e8 W/m² in this numerical pilot. Representational
failures, such as a nonzero face contribution that vanishes in an accumulated
nodal value, are rejected rather than silently discarded.

`surface_by_node_W` and `total_surface_W` report the applied load.
`heat_into_solid_W` reports the remaining prescribed-temperature response,
not the sum of that response and applied flux. A fully prescribed face
contributes to the applied-load ledger and changes the corresponding reaction
but adds no free-row forcing. Each connected component still needs a
prescribed temperature; its applied and prescribed heat balance to the
numerical residual. `balance_W` is the signed sum over the mesh. The surface
pairing uses component-centered temperature in W K, like the terminal
pairing; neither is stored energy.

## Combine volumetric and surface heat loads

Solve the temperature field once when a solid generates heat internally and
also gains or loses heat through selected exterior faces:

```python
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_mixed_loads

result = solve_steady_conduction_tet4_with_mixed_loads(
    bundle,
    conductivity_W_per_m_K_by_material=conductivity,
    prescribed_K_by_id=boundary_temperatures,
    source_W_per_m3_by_cell=source_density,
    faces_by_node_id=[(node_a, node_b, node_c)],
    inward_flux_W_per_m2=-100.0,
)
```

Use the same complete stable-cell-ID map and exterior stable-node-ID triangles
as above. Positive rates add heat; negative rates remove it. Temperatures and
conductivity may differ between connected components; each component needs at
least one prescribed node. The surface load is included on fully prescribed
faces: it changes the local reaction, not the prescribed temperature.

Use `source_by_node_W`, `surface_by_node_W`, `total_source_W`, and
`total_surface_W` to inspect the contributions separately, and
`applied_nodal_heat_W` for the combined physical load before eliminating held
temperatures. The combined value is rounded only after the represented cell
and face factors are summed, so it can differ from the sum of the separately
rounded ledgers. `balance_W` adds both integrated loads and the prescribed-node
response; a small global balance does not imply small local loads or gradients.
`applied_pairing_W_K` is the combined load paired with the component-centered
temperatures; this value, the terminal pairing and the free residual pairing
close the Dirichlet quadratic. The source and surface pairings are also
available separately. The pairings are not stored energy.

The same mesh and rate bounds apply. Near-cancellation that would make a
nonzero nodal or free-row load unresolved, or contributions lost to rounding,
is rejected. This check concerns the represented input geometry and forcing,
not physical model uncertainty. The loads are prescribed; this entry point
does not model temperature-dependent convection, radiation or transients.

## Exchange heat with an environment

A convection boundary changes its heat flux as the surface temperature
changes. Use a constant heat-transfer coefficient when the ambient
temperature is known and the surrounding fluid is not part of the model.
The same calculation can include internal heat generation or removal:

```python
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4_with_convection

result = solve_steady_conduction_tet4_with_convection(
    bundle,
    conductivity_W_per_m_K_by_material=conductivity,
    prescribed_K_by_id=boundary_temperatures,
    source_W_per_m3_by_cell=source_density,
    faces_by_node_id=[(node_a, node_b, node_c)],
    convection_W_per_m2_K=100.0,
    ambient_K=299.95,
)
```

Select exterior triangles by stable node ID and supply one heat density
per stable cell ID, including explicit zeros. The scalar coefficient
applies to all selected faces, in W/(m² K); the ambient is absolute K.
Unselected exterior faces are weakly insulated where no temperature
is prescribed. At least one node per
connected component needs a prescribed temperature. A zero coefficient
keeps the selected faces insulated but still checks the selection.

`convective_heat_into_solid_W_by_node` is integrated against the nodal
shape functions, not a pointwise face flux. Positive heat enters the
solid through convection, negative heat leaves. `heat_into_solid_W` is
the separate prescribed-temperature response. A fully prescribed face
can exchange heat with the ambient: its temperature stays fixed and
the prescribed response accounts for that exchange. Source, ambient
exchange and prescribed response sum to the numerical residual, checked
per component. `convection_pairing_W_K` pairs the convection heat with
component-centered temperature; it is signed and is not stored energy.

The coefficient is zero or in [1e-8, 1e8] W/(m² K) in this bounded solver,
subject to operator and load resolution checks. It uses the consistent
linear triangle integral. It does not fit the coefficient, simulate fluid
flow or radiation, or combine convection and a separately prescribed heat
flux through this entry point.

### Example: material contrast in a welded junction

Run from the repository root. PyPI supplies the solver; the example and
its sibling geometry script both come from the repository checkout.

```sh
python -m pip install "fibernet==4.20.0"
python examples/tet4_dc_convection_welded_junction.py
```

Three welded cylinders share the same 621 nodes and 2,064 tetrahedra in
both runs. At 0.3 V, lowering only the junction's electrical conductivity
from 2 to 0.5 S/m changes the heat balance (rounded to 0.01 µW):

| Case | Joule heat | Convection out | Fixed-temperature caps out |
| --- | ---: | ---: | ---: |
| Uniform conductivity | 53.46 | 57.42 | −3.96 |
| Lower conductivity in junction | 50.97 | 56.78 | −5.81 |

All powers are in µW. Negative cap output means the thermostatted ends
supply heat to the solid. This example uses a shared tetrahedral mesh of a
voxel approximation of the cylinder union; it is not a mesh-convergence
study. The solved temperature does not feed back into electrical
conductivity.
