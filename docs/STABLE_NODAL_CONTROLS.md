# Prepare nodal controls by stable ID

`from fibernet.sim import prepare_nodal_controls` is an additive adapter for the
existing ordinary Taichi dynamics and raw BeamFrameFEM row-control APIs. Existing
solver signatures and legacy `fixed_nodes` row semantics do not change.

```python
# graph is the authoritative StructureGraph. IDs need not be contiguous/sorted.
from fibernet.sim import prepare_nodal_controls
prepared = prepare_nodal_controls(
    graph, fixed_node_ids=[101, 8],
    forces_by_id={44: [0.001, 0, 0], 20: [0, -0.001, 0]},
    dimension=3,
)
result = engine.dynamics(graph, **prepared.taichi_kwargs(graph))
result.metadata['nodal_controls'] = prepared.to_dict()
```

Only explicitly supplied force IDs receive nonzero force. Values are complete
translational vectors, not pressure, acceleration or per-edge forces. No unit
conversion occurs. The caller must choose a physically meaningful model, units,
parameters, timestep, duration and other engine controls.

For a straight 2D frame with small displacement, use the existing FEM adapter:

```python
from fibernet.ml.beam_frame_fem import BeamFrameFEM
prepared = prepare_nodal_controls(
    graph, fixed_node_ids=[101, 8],
    forces_by_id={44: [0.001, 0], 20: [0, -0.001]}, dimension=2,
)
fem = BeamFrameFEM(E=1e7, nu=0.3)
inputs = fem.graph_to_fem_input(graph, dim=2)
raw = fem.solve_2d(inputs['edge_index'], inputs['node_pos'], inputs['radii'],
                   **prepared.fem_kwargs(graph), damping=0, deduplicate=False)
result = fem.to_sim_result(raw, graph, node_ids=prepared.node_ids,
                          edge_ids=prepared.edge_ids)
result.metadata['nodal_controls'] = prepared.to_dict()
result.save('result.json', detailed=True)
graph.save_json('graph.json')
```

This frame example uses global E/nu, not heterogeneous edge material laws. Taking
lengths/radii in metres, E in Pa and forces in N is a caller declaration, not
something inferred by the helper. FEM fixed nodes clamp rotational DOFs too;
Taichi fixes translation and has no matching frame rotations. The adapter does
not claim model equivalence. Curved edge geometry/material records remain on the
graph without mutation, but this raw frame example does not model curved beams.

Use result.metadata['node_ids'] and ['edge_ids'] to align results, not the current
iteration order of a reloaded graph. Serialization can sort IDs. Reprepare on a
reloaded or changed graph. `prepared.to_dict()` is a provenance sidecar, not a
restart checkpoint or an executable deserialization interface.

## Invalid inputs and lifecycle

- Nonnegative Python/NumPy integer IDs may be mixed. Strings, floats, booleans,
  negative IDs, duplicate support IDs and unknown IDs are rejected.
- Supplying both fixed_node_ids and fixed_nodes, or both forces_by_id and
  external_force, is rejected even if an input is empty or the values agree.
- dimension must be integer 2 or 3; ordinary Taichi kwargs require 3.
- Invalid vector lengths, masks, boolean components and nonfinite forces fail.
- Any graph serialization or insertion-order change invalidates the prepared
  controls. Call prepare_nodal_controls again. The graph must have a finite,
  JSON-serializable record. Arbitrary nonserializable metadata is unsupported.
- Kwargs are detached copies. Obtain them immediately before the solve and do not
  mutate the graph concurrently. Old row kwargs manually cached by the caller
  cannot be retrospectively validated by the solver.
- Preparation reuses existing accelerated validators and ordering. It imports
  that module (which attempts optional Taichi import) but constructs no engine or
  native runtime. This is not a dependency-free provider discovery API.

No displacement schedules, partial-DOF restraints, applied moments, histories,
LAMMPS controls, unit conversion, cancellation, or restart are added. Existing
owner-specific APIs remain authoritative for those cases.
