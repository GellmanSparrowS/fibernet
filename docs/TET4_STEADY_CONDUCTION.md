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
