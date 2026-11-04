# Final node displacement from two geometries

`compare_lammps_node_displacements` compares a saved LAMMPS result's final
positions with its loaded initial graph and a caller-supplied reference graph.
It does not start LAMMPS or change any inputs.

```python
from fibernet.analysis.geometry_paths import compare_lammps_node_displacements
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.accelerated import SimResult

reference = StructureGraph.load_json('reference.json')
initial = StructureGraph.load_json('initial.json')
# Unprofiled detailed result; profile-bound files require the matching profile.
result = SimResult.load('result_detailed.json')
comparison = compare_lammps_node_displacements(
    reference, initial, result,
    reference_length_unit='mm', initial_length_unit='um',
)
print(comparison['node_ids'])           # sorted physical node IDs
print(comparison['from_initial_m'])     # final minus loaded initial, metres
print(comparison['from_reference_m'])   # final minus reference, metres
```

Save a detailed result with `SimResult.save(..., detailed=True)`; summary exports
omit final positions. For a profile-bound file pass its matching profile name
to `SimResult.load(..., profile=...)`.

Use actual units for each graph. The initial graph's serialized digest and
unit must match the result's recorded source, and the result's explicit node
IDs must cover both graphs exactly. The output arrays are detached and ordered
by returned IDs. The reference is not authenticated: identical IDs do not
prove the graphs represent corresponding physical nodes, coordinate frames or
configurations. The caller owns that correspondence and must provide both
geometries in the same frame. A digest detects an accidental mismatch; it does
not authenticate a fabricated result. Missing final positions or row maps are
rejected rather than guessed.

Calculations use finite float64 coordinates; unit conversion and subtraction
can round near-coincident values. It recomputes from final graph-unit positions;
its values are not guaranteed bitwise equal to `result.displacements`, and
unit round trips can produce a small nonzero difference for an unchanged state.
This is a final-frame geometry comparison,
not a trajectory, stress, reaction force, external work, or physical solver
validation. It does not require recorded native force state. A fixed-node net
force is not generalized to a support reaction by this interface.
