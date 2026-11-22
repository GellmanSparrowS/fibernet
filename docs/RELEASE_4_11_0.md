# FiberNet 4.11.0

## Direct 3D prescribed-temperature beam snapshot

Use the opt-in Python function
`fibernet.sim.thermoelastic_3d.solve_thermoelastic_network_snapshot_3d`
for a static, small-deformation model of straight circular beams welded at
graph nodes. Supply explicit temperatures in kelvin, a stress-free
temperature, signed expansion coefficients by stable edge ID, and supports.
Assign each edge its isotropic elastic material on the graph. Geometry uses the
declared length unit; force and displacement results use SI.

The detached result keeps node and edge IDs, displacements, rotations, forces,
support reactions and residuals. It reports nominal axial stress and discrete
elastic energy about the member's mean thermal strain. The caller's six
kinematic ceilings and the reduced stiffness check are admission screens,
not guarantees of stability, material validity or continuum stress accuracy.

This is a direct prescribed-temperature solve, not a temperature solver or a
new route for saved electrothermal results. The planar and saved workflows,
the `simulate` facade and MCP tools are unchanged. No new dependency is
required. See the [runnable free-expansion example and API contract](THERMOELASTIC_3D.md).

## Separate source examples

- [3D mixed-material beam FEM](../examples/heterogeneous_3d_graph_workflow.py):
  assign elastic properties by stable ID, solve a small prescribed stretch
  and save the graph and detailed result.
- [3D temperature-dependent DC heating](../examples/electrothermal_3d_cubic.py):
  run a fixed-geometry electrothermal example and check its heat balance. It
  does not run mechanics.

These scripts are in the source distribution and repository, not installed
`examples` modules in the wheel.
