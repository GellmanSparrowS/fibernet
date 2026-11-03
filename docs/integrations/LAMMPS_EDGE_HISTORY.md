# Edge endpoint history from a LAMMPS result

After preparing a run and parsing its outputs, inspect one StructureGraph edge:

```python
from fibernet.integrations.lammps_backend import edge_length_history

history = edge_length_history(prepared, result, edge_id=77)
# history.steps, history.times_s and history.lengths_m are aligned tuples.
# history.node_i and history.node_j are the original stable graph node IDs.
```

The edge ID is a StructureGraph ID, not a LAMMPS bond index. Length is the
straight distance between its endpoint positions in meters. It is not a curve
arc length, strain, stress, or a material response. This is post-processing;
it does not launch LAMMPS, interpolate frames, or infer missing samples.
The supported prepared runs use fixed nonperiodic boxes; no minimum-image
correction is applied.
For sampled runs the series describes retained frames only, not the unobserved
integration steps between them.

The function regenerates and validates the prepared run, then checks the
result identity, frame schedule, node order and finite position geometry before
extracting values. Matching an identity detects accidental use of a different
prepared run. It does not authenticate a manually constructed result, recheck
the initial state, box, bond reach or fixed constraints, verify forces or
energies, or replace parsing and validation of native output. Use
`parse_outputs` or `parse_output_files` to obtain a validated result first.
