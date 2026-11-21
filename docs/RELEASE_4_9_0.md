# FiberNet 4.9.0

Schedule Taichi dynamics by stable node ID, compare how two materials
store elastic energy in a welded Tet4 junction, and keep verified
runtime digests in LAMMPS CLI results when the executable paths use symlinks.

## Taichi: schedule displacement by node ID

Install the optional acceleration extra on a platform supported by Taichi:

```bash
python -m pip install "fibernet[accel]==4.9.0"
```

This three-node example prescribes the tip's absolute displacement from
its starting position. The waypoint at 2.5 is in **integration steps**,
not seconds; the graph uses metres. The fixed node still uses the raw
solver's row-based support input. The adapter does not convert units or
alter the backend; after graph edits, prepare the schedule again.

```python
from fibernet import simulate
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.nodal_controls import prepare_taichi_schedule_by_id

graph = StructureGraph(dimension=3)
base = graph.add_node([0.0, 0.0, 0.0], merge=False)
unused = graph.add_node([0.0, 0.01, 0.0], merge=False)
mid = graph.add_node([0.001, 0.0, 0.0], merge=False)
tip = graph.add_node([0.002, 0.0, 0.0], merge=False)
graph.remove_node(unused)  # The remaining stable IDs differ from their rows.
graph.add_edge(base, mid, radius=5e-5)
graph.add_edge(mid, tip, radius=5e-5)

schedule = prepare_taichi_schedule_by_id(
    graph, {tip: [(0.0, (0.0, 0.0, 0.0)), (2.5, (1e-7, 0.0, 0.0))]}
)
rows = {node_id: row for row, node_id in enumerate(graph.nodes)}
result = simulate(
    graph, mode="dynamics", backend="taichi", arch="cpu", num_threads=1,
    fixed_nodes=[rows[base]], dt=1e-6, num_steps=4, save_interval=1,
    spring_k=1.0, nodal_masses_by_id={node_id: 1e-6 for node_id in graph.nodes},
    **schedule.taichi_kwargs(graph),
)
print(result.displacements[rows[tip]])
```

Preparation itself works without importing Taichi. Running the dynamics
does require Taichi.

## Tet4: welded junctions and portable inputs

The [welded-junction example](../examples/tet4_static_welded_junction.py)
uses the public traction helper to apply the same 1 mN resultant to both
material cases. Its budget precheck does not replace a solve. The JSON
reports the fraction of elastic strain energy inside the same geometric
core, not peak stress, strength or mesh convergence. The demo needs no
Meshio unless you ask it to write VTUs.

To replay an input mesh with its material assignments, see the
[Tet4 bundle handoff](TET4_VTU_BUNDLE.md). The ZIP holds the mesh and materials.
The companion JSON in the guide records stable face and support IDs,
traction, and a SHA256 checksum of the exact ZIP bytes. The checksum
detects accidental mismatches; it is not a signature. Solver-output VTUs
are for inspection rather than replaying those inputs.

## LAMMPS: named runtime digests

For admitted symlinked executable paths, the CLI result records the
verified Python and native LAMMPS hashes. Runtime pin checks and existing
supported configurations are unchanged; no new MPI support is added.
See the [CLI runtime contract](integrations/LAMMPS_HARMONIC_CLI.md).
