# Experimental prescribed-temperature thermoelastic snapshots

This opt-in module computes small-displacement, linear, planar welded
Euler–Bernoulli frame response to a prescribed temperature snapshot. It does
not solve temperature or electrical fields, feed deformation back into either
field, or qualify stability. It is not available as a `simulate` facade mode.
The underscored module remains experimental; its API may change.

## Minimal example

```python
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim._thermoelastic import solve_thermoelastic_network_snapshot

graph = StructureGraph.from_dict({
    "dimension": 2,
    "nodes": [
        {"id": 10, "position": [0.0, 0.0, 0.0]},
        {"id": 90, "position": [0.1, 0.0, 0.0]},
    ],
    "edges": [{"id": 70, "node_i": 10, "node_j": 90, "radius": 0.001}],
})
snapshot = solve_thermoelastic_network_snapshot(
    graph,
    length_unit="m",
    E_Pa=1e6,
    expansion_alpha_per_K={70: 1e-6},
    stress_free_temperature_K=300.0,
    temperature_K={10: 301.0, 90: 301.0},
    constraints={10: (0.0, 0.0, 0.0)},
)
result = snapshot.to_dict()
# The free tip extends by alpha * delta_temperature * length = 1e-7 m.
```

Graph lengths and radii use the explicitly supplied `m`, `mm`, or `um` unit.
Constraint translations, applied forces, and moments are always SI: metres,
newtons, and newton-metres; rotations are radians. Constraint tuples are
`(ux, uy, rz)` with `None` for free components. Load tuples are `(Fx, Fy, Mz)`.
No material property is inferred from graph metadata: supply a positive common
Young's modulus and an expansion coefficient for every edge. Negative and zero
expansion coefficients are supported. All absolute temperatures must be
positive kelvin; temperature and coefficient mappings require exact stable-ID
coverage, with integer rather than JSON string keys.

## Two deliberately distinct profiles

- `solve_thermoelastic_snapshot`: two nodes with one or two straight parallel
  members; a single zero clamp; positive x or y root-to-tip direction; uniform
  temperature and equal expansion coefficients; only tip axial loading or
  prescribed axial displacement. Thermal strain is bounded by 0.001 and mean
  total axial strain and nodal rotation by 0.005. Profile identifier:
  `thermoelastic_2d_elementary/2`.
- `solve_thermoelastic_network_snapshot`: one connected supported component,
  2–192 nodes and 1–256 edges, straight members lying in one XY plane, welded
  joints, circular sections, no isolated nodes. Physical parallel edges are
  retained. Profile identifier: `thermoelastic_2d_network/1`.

The network profile interpolates nodal temperature linearly along each member
and uses the endpoint mean for its equivalent axial eigenstrain load. It does
not model a temperature gradient through a cross-section. Nonuniform axial
thermal strain is represented by this element-mean loading approximation;
refine the explicit straight-edge mesh when spatial resolution is needed.
Curved geometry requires explicit subdivision into straight edges. Coincident
coordinates do not imply a joint: topology and identity come from native IDs.

Network admission limits are endpoint thermal strain ≤ 1e-5, mean total axial
strain ≤ 1e-4, absolute nodal rotation and maximum cubic-interpolated transverse
slope ≤ 1e-3 radians, and endpoint total/elastic extreme-fiber strain ≤ 1e-4.
These are numerical applicability checks, not evidence against buckling or
material failure. Supports must remove all rigid modes. Singular solves,
nonfinite arithmetic, failed equilibrium checks, and exceeded limits raise
errors; there is no damping or numerical regularization fallback.

## Interpreting results

`to_dict()` returns a detached JSON-compatible copy. The graph and input maps
are not modified. Node arrays follow sorted `node_ids`; edge arrays follow
sorted `edge_ids`. `endpoints_by_id` retains each member's orientation. Outputs
include SI displacement, reactions, total/thermal/elastic axial strain, axial
stress, bending stress envelope, endpoint moments, edge forces, equilibrium
residuals and bounds, and an input identity hash. Network outputs also include
interpolation-based applicability diagnostics and the balance-origin node ID.

For local x from node i to node j and local y rotated counterclockwise from x,
positive nodal rotation is counterclockwise. Reported `edge_forces` are
`[N, V, (Mi+Mj)/2]`. The scalar moment convention is `M = -EI v''`, so
`V = (Mj-Mi)/L`. The local nodal energy-gradient force vectors are
`qi = (-N, -V, Mi)` and `qj = (N, V, -Mj)`. In particular, V is the negative
of the local i-end transverse nodal force; reported Mj is the negative of the
j-end nodal moment. `sigma_bending` is a nonnegative endpoint envelope and
`sigma_total = abs(sigma_axial) + sigma_bending`, not a signed tensor stress.

The identity hash binds the profile, native geometry and IDs, material and
thermal inputs, supports, and applied loads. A profile change therefore changes
the hash even if the numerical result remains identical. It is not a proof of
physical validity or a substitute for independent validation.

## Compatibility and limits

Existing viscoelastic entry points and helper aliases remain available. The
BeamFrameFEM stress-recovery extension accepts optional axial eigenstrain by
original input edge row; omitting it preserves the existing arithmetic path.
The shared sparse-solve helper preserves the viscoelastic solver-injection seam.
No dependencies or facade modes are added. The exact-source MCP binding must be
updated with the matching package source; the thermoelastic function is not an
MCP tool.

This model excludes 3-D frames, shear deformation, nonlinear kinematics,
geometric stiffness, contact, fracture, viscoelastic thermal response, thermal
transients, electrical response, and any two-way or three-field coupling.
Compressive results are not stability-qualified. The included analytical rod
and independent dense energy-reference tests check the stated linear model,
not applicability to a particular manufactured network or material.
