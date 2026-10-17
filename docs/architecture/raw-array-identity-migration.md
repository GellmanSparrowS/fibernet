# Raw FEM array identity

High-level APIs keep
their simple graph-based usage; these explicit maps serve direct array callers.

## Actual ambiguity

`StructureGraph.to_numpy()` sorts node IDs and preserves edge insertion order.
`sim.accelerated._graph_to_arrays()` preserves node insertion order.
The legacy `BeamFrameFEM.to_sim_result(raw, graph)` call reconstructs positions and
connectivity through the latter helper. Thus a perfectly natural
`graph.to_numpy()` → `solve_2d` → `to_sim_result(raw, graph)` journey can attach
correct displacements to wrong positions. Result metadata repairs the
renderer boundary only when the producer's row identity is correct.

## Additive boundary

`to_sim_result` now accepts keyword-only `node_ids=None, edge_ids=None`.
Define these as the **input** node/element row IDs of the direct solve, before
FEM deduplication. This avoids ambiguity with the solver's retained `edge_list`:
output edge identities are `[edge_ids[i] for i in raw['edge_list']]`.
Document that definition prominently; do not call them output field IDs.

When graph and both maps are supplied:

- Validate maps before indexing: integer/nonbool/nonnegative, unique, exact graph
  coverage and raw node count. Validate retained edge indices against input map
- Build positions in supplied node order and endpoint row connectivity in supplied
  edge order, using graph IDs. Compute deformed coordinates, stretches and the
  existing axial energy proxy using those same rows and per-edge radii
- Attach node IDs and retained output edge IDs to SimResult metadata
- Preserve raw arrays, graph ordering and solver numerical calculations

Omitting both maps retains the documented legacy insertion-order assumption.
Supplying only one map fails clearly rather than inferring the other. This bounded
implementation requires graph with explicit IDs: without original
positions, the existing graph-free adapter cannot provide genuine deformed
positions or geometric energy. That limitation deserves a separate status
contract rather than a guessed origin. Do not broaden the first patch into that
redesign.

The maps assert provenance; they cannot prove that arbitrary caller arrays match
the referenced graph geometry/materials. A future copied input snapshot can bind
coordinates, connectivity and identity together. Do not silently accept partial
graph submodels in this first exact-coverage path.

## Executable acceptance gate

Use sparse nodes with reversed insertion records and nonuniform radii. Run the
natural `to_numpy()` journey and explicitly supply `sorted(graph.nodes)` and
`list(graph.edges)`. Solve a fixed/displaced 2D case, adapt, strict-JSON save/reload,
render, and verify the physical fixed node remains fixed. Repeat a supported 3D
case. Permute input element rows to prove `edge_list` is interpreted against the
supplied input IDs. Include parallel edges retained in the graph but deduplicated
by this FEM backend: output IDs describe exactly solved elements, and full graph
stress rendering still rejects incomplete coverage.

Compare existing no-keyword calls byte-for-byte where possible; test malformed,
missing, foreign and duplicate IDs, invalid retained indices, graph/raw immutability,
and mismatched counts. Keep mechanical parallel-edge assembly semantics unchanged.


## Regression coverage

`tests/test_raw_fem_identity.py` executes actual 2D/3D exported-array solves,
strict JSON save/load, and rendered fixed/driven node assertions. It includes
permuted element rows, nonuniform radii and lengths, legacy parallel deduplication,
malformed and incomplete maps, retained-index validation before geometric indexing,
raw-node count mismatches, immutability, and legacy insertion-order equivalence.
The raw numerical solvers, assembly, deduplication and high-level APIs are unchanged.

This is an identity assertion boundary, not validation of arbitrary raw geometry,
connectivity, materials or every raw diagnostic field shape. A copied solver-input
snapshot and a broader raw-result schema remain separate potential work. Legacy
omission remains compatible and does not retroactively verify row provenance.
