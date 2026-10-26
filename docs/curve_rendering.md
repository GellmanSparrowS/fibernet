# Stored curves in 3D result views

`StructureGraph` stores each physical edge as two endpoint nodes plus optional
absolute `internal_points`. Rendering those samples creates display segments;
it does not create additional physical nodes or edges.

The following result views preserve the complete stored polyline at **zero
additional displayed motion**:

- `render_deformation_3d`
- `render_trajectory_3d`
- `render_stress_3d`
- `render_comparison_3d`
- `render_multi_angle_3d`

Bounds include interior samples, including points beyond the endpoint bounds.
Physical edge IDs/order and explicit result row maps remain unchanged. Parallel
physical edges are not merged merely because they share endpoint nodes.

## Moving curves requires more than endpoint displacement

For an edge with stored interior points, nonzero displayed endpoint movement
raises `ValueError` before a figure or output file is created. Nodal results do
not determine the motion of the interior beam. The renderer does not silently
replace the curve with a chord, retain stale interior points beside moved
endpoints, or invent an interpolation rule.

To inspect a previously saved deformed centerline, supply a graph whose
endpoint and interior coordinates already represent that saved snapshot and
apply **no additional nodal motion**. Label it as saved geometry, not the
undeformed reference. Frame-specific interior-point trajectories are not added
to `SimResult` by this feature.

`displacement_scale=0` disables motion in the deformation view. In
`render_comparison_3d`, the stress panel retains its existing **unscaled**
`deformed_positions` behavior: a nonzero curved stress-panel request therefore
still raises, even when the displacement panel's scale is zero. Edges without
interior samples retain their existing endpoint-based behavior.

## What the colors mean

Displacement coloring uses the mean of the two endpoint nodal displacement
magnitudes, repeated on every display segment of that physical edge. It is
not a measured or reconstructed interior displacement field.

Force coloring uses the absolute supplied per-physical-edge force; stretch
coloring uses the supplied per-edge stretch. These edge values are repeated
on the edge's display segments, not mistaken for additional physical-edge
rows. Node and edge fields retain distinct meanings even when the numbers of
nodes and edges happen to be equal.

This is a geometry-display contract. It does not establish curved-beam solver
support, material calibration, simulation accuracy, convergence, restart
validity, or native-provider compatibility.

## Focused regression checks

The focused tests run plotting in fresh child processes and own their temporary
files. Disable the repository-wide Taichi fixtures for this standalone check:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -o addopts= tests/test_curve_rendering_contract.py
```

The protocol profile always checks geometry/color contracts. The actual
`SimResult` profile runs only when optional Taichi is genuinely absent; otherwise
it skips explicitly without importing that provider. Such a skip is not evidence
of actual-class or native-provider compatibility.
