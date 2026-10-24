# Taichi geometry acceptance

The graph-based spring/Taichi callers now default to
`geometry_policy="reject_internal_points"`. This is a breaking input-acceptance
change: an edge with any nonempty `internal_points` is rejected, including
collinear, duplicate or backtracking points. Previously these points could be
silently ignored by the endpoint-based mechanics.

Choose `geometry_policy="endpoint_chord"` explicitly only when ignoring those
edge-owned points is the intended model:

```python
result = fibernet.simulate(
    graph, mode="dynamics", backend="taichi",
    geometry_policy="endpoint_chord", **dynamics_options,
)
```

This selects the existing endpoint-chord mechanics. It does not flatten or
modify the graph, preserve curved-fiber mechanics, add bending response, or
establish equivalence to a curve. Materials, units, coefficients, loads and
boundary conditions retain their existing contracts. No automatic opt-in is
performed.

## Actual nodes and physical edges

Intermediate `StructureGraph` nodes connected by independent straight edges
remain accepted. For example, actual-node segmentation from `n_pts_per_side`
is different from edge-owned `internal_points`. This distinction does not
establish mechanical equivalence between those representations. Physical
parallel edges and stable IDs remain distinct; no parallel graph or implicit
reconstruction is created. Geometry preflight reads the existing graph without
mutating its nodes, edges, points, materials or metadata.

## Callers

- `TaichiEngine.dynamics` and `stretch_test` accept the policy as a keyword-only
  argument. Their checks precede owner/array work. Direct construction of an
  engine remains the caller's responsibility.
- `fibernet.simulate` forwards the policy for `backend="spring"` and
  `backend="taichi"`, rejecting incompatible graphs before engine construction.
  Other backends do not accept this option.
- `batch_simulate_from_json` exposes a keyword-only policy. Invalid policy
  values fail before provider construction or output writes. An incompatible
  input graph becomes a per-file failure row; remaining files continue. Graphs
  are parsed one at a time, and the existing resume rules retain failure rows.
- `FiberNetDataset.generate` exposes a keyword-only policy. The selected
  built-in sampling plan is checked before graph generation, provider
  construction or writes. Defaults have not changed: a selected plan containing
  nonzero `n_internal` is incompatible with the strict policy. Choose an
  endpoint-only plan such as `n_internal_range=[0]`, or explicitly select
  `endpoint_chord` if that interpretation is appropriate. Custom factories are
  checked against their actual graphs; a geometry-policy error is not silently
  converted into a partial successful dataset.
- The spring-evaluated RL environments expose explicit policy configuration
  and forward it to the engine. Geometry-policy errors remain errors, not
  low-reward physics outcomes. This does not enable the unrelated root
  `fibernet.train_rl` effective-property evaluator or remove optional RL
  dependency requirements.

Only the exact built-in strings `reject_internal_points` and `endpoint_chord`
are accepted. There is no coercion, alias or implicit selection.

## Provenance and saved results

Successful graph-based Taichi results record `metadata["taichi_geometry"]`:

```json
{
  "schema": "taichi_endpoint_chord/1",
  "policy": "endpoint_chord",
  "representation": "endpoint_chord",
  "ignored_internal_point_edge_ids": [13, 12]
}
```

IDs follow the graph's current edge insertion order. The record is detached
from the graph. Summary and detailed `SimResult` serialization preserve it as
ordinary metadata; loading does not authenticate a claimed solver execution.
Legacy results remain unchanged and do not acquire retrospective provenance.

Batch and Dataset run bindings include the selected policy and the geometry
helper's source fingerprint. Existing unbound or mismatched checkpoints are
not silently migrated. Use a fresh output directory when the binding changes.

The policy changes input acceptance and records interpretation, not the
underlying numerical scheme or its range of physical validity.
