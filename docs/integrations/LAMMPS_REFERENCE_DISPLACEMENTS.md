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

Both displacement arrays are always in metres. In contrast, the saved
`result.displacements` and `result.max_displacement` use the original graph
unit recorded in `result.metadata['field_units']`. The arrays are independent
copies that the caller may edit without changing either graph or the result.
The returned node IDs are physical StructureGraph IDs, not native atom indices.

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
For equal unit factors, subtraction precedes scaling to retain small differences;
components whose raw subtraction overflows use the metre-valued operands instead.
Different unit factors require comparing the converted positions, so cancellation
can lose differences below their converted precision. Nonfinite positions,
unrepresentable metre conversions and nonfinite displacements are rejected.
Representable subnormal differences are retained when the active floating-point
mode supports them. Some native libraries change the calling thread to flush
subnormal outputs or treat subnormal inputs as zero (FTZ/DAZ). The helper does
not change that mode: it rejects conversions or differences that silently become
zero instead of returning a false zero-motion result. This includes differences
between distinct normal coordinates. Integer-bit checks distinguish stored
subnormals from signed zero even in DAZ mode. Rounding between different unit
factors can still make converted coordinates identical, as described above.
The input check runs before dtype promotion, including nested NumPy scalars:
a float32 subnormal can otherwise disappear under DAZ even though its value
would be normal and representable in float64.
This is a final-frame geometry comparison,
not a trajectory, stress, reaction force, external work, or physical solver
validation. It does not require recorded native force state. A fixed-node net
force is not generalized to a support reaction by this interface.
