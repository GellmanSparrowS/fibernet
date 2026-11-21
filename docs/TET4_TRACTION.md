# Constant Tet4 surface traction

Use this explicitly importable helper to turn a known, constant **global
Cartesian traction vector** into the nodal force mapping used by the
[static Tet4 solver](TET4_STATIC.md). Select each triangular surface face
by its three **stable node IDs**, not point-row indices.

```python
from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_loads import nodal_forces_from_traction

forces = nodal_forces_from_traction(
    bundle, [(node_a, node_b, node_c), (node_a, node_c, node_d)],
    traction_Pa=(800., -150., 500.),
)
result = solve_static_tet4(bundle, forces_by_id=forces, fixed_dofs_by_id=held_axes)
```

- Each selected triangle must be a face of exactly one tetrahedron.
  Duplicate triangles, interior faces, missing faces and unknown IDs are
  rejected. Vertex order does not set the direction. This is a **topology**
  check, not a certificate that the physical mesh has no gaps or overlaps.
- The bundle length unit is used to calculate area in m². For each face,
  the helper adds one third of the area times the traction to each of its
  three nodes. The returned dictionary maps exact integer node IDs to
  three global (Fx, Fy, Fz) force components in N. Zero traction gives zero.
- This helper does not use surface normals, compute pressure, distribute
  a specified total force, detect surface regions, or solve a model.
  It uses the solver's point/cell limits and conservative finite numerical
  range, rejecting nonzero areas or force values that cannot be resolved
  safely. It neither establishes element conditioning nor guarantees a
  subsequent solve will be admitted.
