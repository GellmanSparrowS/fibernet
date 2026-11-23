# Experimental solid DC and one-way Joule heating

Use `solve_dc_tet4` from `fibernet.sim.tet4_dc` for a bounded steady scalar
potential solve on a Tet4 solid. Provide isotropic electrical conductivity
in S/m by stable material ID and signed electrode potential in volts by
stable node ID. Include every node of each prescribed electrode patch.
Other exterior faces are insulated in the weak sense. Every connected
component needs at least one prescribed electrode node; a component with
only one prescribed potential level is passive, with exactly zero current
and Joule heat.

```python
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4

result = solve_dc_thermal_tet4(
    bundle,
    conductivity_S_per_m_by_material={1: 2.0, 2: 8.0},
    electrode_voltage_V_by_id={**left_electrode_V, **right_electrode_V},
    conductivity_W_per_m_K_by_material={1: 0.25, 2: 0.25},
    prescribed_K_by_id=held_temperatures_K,
)
```

The two conductivity maps are physically distinct. The electrical result
reports absolute and component-centered voltage, electrode currents in
amperes (positive into the solid), cell current density in A/m² and
dissipation density in W/m³. It checks that electrode work matches
integrated nonnegative cell power in watts. Centering rejects distinct
represented electrode voltages if they would become indistinguishable.

The same-call wrapper snapshots one validated mesh internally and uses it
for both phases; changing the original mesh between the two computations
does not change the transfer. Do not modify input arrays while they are
being snapshotted. Each cell receives the density `q_J = σ |grad V|²` by stable
cell ID on that same mesh. The wrapper checks cellwise and nodal power
transfer and rejects loads outside the thermal numerical envelope or
unresolved nodal power contributions. It does not accept an external
electrical result, merge meshes, or export a reusable geometry binding.

The thermal solve uses prescribed absolute kelvin and cellwise volumetric
heat only. Electrical conductivity does not depend on temperature; this
is one-way coupling, not an electrothermal feedback calculation.
The model excludes electrode contact resistance, current-controlled
electrodes, and transient heating. A small two-material bar verifies
integrated electrode and cap balances; it does not validate pointwise
temperature peaks on coarse or arbitrary meshes.
