# Full-history harmonic dynamics

Select `profile="harmonic_full_history_v1"` with the existing LAMMPS harmonic
backend when every integration state must be retained. It supports:

- 2–64 nodes and 1–512 straight edges, with no duplicate endpoint pairs,
  in a 3-D nonperiodic box.
- 1–2048 integration steps, with `sample_every=1` and `output_stride=1`.
  Omitting `output_stride` selects 1. The initial state is included, giving
  2–2049 output frames.
- At most two exact bead-mass values and 20 nodes with constant applied loads.
- Explicit SI masses, spring stiffnesses, equilibrium lengths, a maximum bond
  length, and whole-node fixed constraints. Graph geometry retains its declared
  length unit; native state uses SI units.

The profile changes capacity, not the force law or integrator. Every retained
state is checked against the declared box, bond reach, fixed nodes, and stable
IDs. Existing default, large, and sampled profiles are unchanged.

## Prepare without starting a simulation

```python
from fibernet import StructureGraph
from fibernet.integrations import HarmonicBond, RunSpec, prepare_harmonic_beads

graph = StructureGraph(dimension=3)
left = graph.add_node([0.0, 0.0, 0.0])
right = graph.add_node([0.001, 0.0, 0.0])
edge = graph.add_edge(left, right, n_internal=0)
prepared = prepare_harmonic_beads(
    graph,
    length_unit="m",
    maximum_bond_length_m=0.002,
    masses_kg={left: 1e-6, right: 1e-6},  # 1 mg per node
    bonds={edge: HarmonicBond(1.0, 0.001)},
    fixed_node_ids=[left],
    run=RunSpec(1e-6, 2048, 1, ((-0.01, 0.01),) * 3),
    profile="harmonic_full_history_v1",
)
```

Pass the same model options through `fibernet.simulate(graph, mode="dynamics",
backend="lammps", ...)`, together with the required runtime, work directory,
and explicit `ResourceLimits`. Resource admission checks the complete requested
history before starting; selecting a profile does not increase caller-supplied
limits automatically. LAMMPS remains an optional, explicitly supplied dependency.

Use `record_state=True` to retain the state in the returned `SimResult`, and
`result.save(path, detailed=True)` to save it. Reload with
`SimResult.load(path, profile="harmonic_full_history_v1")`. A summary export
cannot restore a full trajectory, and neither export is a restart checkpoint.
Source capabilities are descriptive: the profile alone does not establish
runtime availability, physical validation of a particular network, or GPU/MPI
performance.
