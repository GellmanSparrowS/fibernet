# Experimental direct 3D thermoelasticity

`fibernet.sim.thermoelastic_3d.solve_thermoelastic_network_snapshot_3d`
computes one linear, static snapshot of a welded, straight, circular 3D
beam frame at **prescribed** temperatures. This opt-in Python entry point
does not change the planar facade or the 3D restrictions on saved workflows.
It does not solve for temperature, update geometry, or expose an MCP or
`simulate` mode.

## A minimal free-expansion check

```python
from fibernet import Material, StructureGraph
from fibernet.sim.thermoelastic_3d import solve_thermoelastic_network_snapshot_3d

rod = StructureGraph.from_dict({
    "dimension": 3,
    "nodes": [
        {"id": 10, "position": [0.0, 0.0, 0.0]},
        {"id": 20, "position": [20.0, 0.0, 0.0]},
    ],
    "edges": [{
        "id": 30, "node_i": 10, "node_j": 20, "radius": 1.0,
        "material": Material(youngs_modulus=2e9, poissons_ratio=0.3).to_dict(),
    }],
})
result = solve_thermoelastic_network_snapshot_3d(
    rod, length_unit="mm",
    temperature_K_by_node_id={10: 301.0, 20: 301.0},
    stress_free_temperature_K=300.0,
    expansion_alpha_per_K_by_edge_id={30: 1e-5},
    constraints_by_node_id={10: (0.0,) * 6},
    applicability_limits={
        "endpoint_thermal_strain": 1e-4, "nodal_rotation": 1e-3,
        "transverse_slope_bound": 1e-3, "total_extreme_fiber_strain": 1e-4,
        "endpoint_elastic_extreme_fiber_strain": 1e-4, "twist_shear": 1e-4,
    },
)
tip = result.displacement_m_and_rad[result.node_ids.index(20)]
assert abs(tip[0] - 2e-7) < 1e-15
```

The expected tip displacement is `α ΔT L = 2e-7 m` for this one rod.
The six supplied limits illustrate input format, not validated material limits.

### Inputs

- Graph coordinates and radii use the stated `m`, `mm` or `um`;
  internally, edge Young's modulus uses Pa and Poisson ratio is dimensionless.
  Temperature and expansion maps must cover the corresponding stable IDs exactly.
  Temperatures are absolute kelvin; the coefficient is signed per kelvin.
- Constraint order is `(ux, uy, uz, rx, ry, rz)` in metres and radians,
  with `None` for free components. Optional loads are `(Fx, Fy, Fz)` in
  newtons. Omitted rows are free or unloaded. Applied nodal moments are
  not supported. Only geometry scales with `length_unit`.
- The owner accepts at most 96 nodes, 192 physical edges and 384 free degrees
  of freedom, with a scaled stiffness screen before the dense reduced solve.
  These are size caps, not runtime or factorization-memory guarantees.
  The six positive caller ceilings screen linear kinematics, not stability,
  mesh convergence, constitutive validity or physical failure.

### Results

Node and edge arrays are immutable tuples aligned to `node_ids` and
`edge_ids`; use those IDs, not a presumed sort order. Displacement
components are metres/radians. In `reactions_N_and_Nm` the first three
components are forces, the last three moments: constrained components
are support reactions; free components are equilibrium residuals. Axial
force (N) and nominal axial stress (Pa) are tension-positive. Reported
thermal strain uses the member mean of the two endpoint temperature
increments; the individual endpoint diagnostics retain their IDs.
Storage is the discrete elastic energy about that mean thermal strain,
reported in joules. It excludes along-member temperature variance and
external-force potential. The scaled stiffness ratio is absent (`None`)
when all degrees of freedom are prescribed.

The snapshot is detached from the graph and caller maps. This direct
interface defines no saved-result or JSON contract.
