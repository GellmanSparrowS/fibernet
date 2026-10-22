# Solve a DC electrical network through MCP

Use `fibernet_solve_linear_transport` to compute node voltages, oriented edge
currents and Joule power for an approved `StructureGraph`. The adapter calls
FiberNet's public `solve_linear_transport` API and returns its complete result.

## 1. Select a graph and supply its electrical inputs

Get the graph handle and declared geometry unit from `fibernet_catalog`.
Obtain stable node and edge IDs from the approved graph data: the catalog
contains handles, counts and units, not the graph's ID lists.

```json
{
  "graph_id": "<catalog handle>",
  "conductance_S": [
    {"edge_id": 101, "value_S": 2.0},
    {"edge_id": 205, "value_S": 3.0}
  ],
  "dirichlet_voltage_V": [
    {"node_id": 10, "value_V": 1.0},
    {"node_id": 90, "value_V": 0.0}
  ]
}
```

Include every physical edge ID exactly once, including inactive edges with
zero conductance. Conductances are finite nonnegative siemens, not material
conductivity; prescribed voltages are finite volts. Parallel members retain
separate edge IDs. Every positive-conductance component, including isolated
nodes, needs a prescribed voltage. Identical repeated boundary values coalesce;
conflicting values are refused.

IDs must be nonnegative JSON integers at most 2^53−1, not array positions,
booleans, floating-point IDs or numeric strings. Physical S/V values use the
native solver's float64 conversion: equivalent integer and exponent encodings
have the same behavior. Values not representable as finite float64 numbers
receive a structured input error.

The operator must declare the geometry unit as `m`, `cm`, `mm`, `um` or `nm`;
`unspecified` is refused. Coordinates are not rescaled. Supply conductance
explicitly: the solver does not infer it from lengths, radii or materials.
The model has ideal equipotential junctions and constant edge conductances.

## 2. Read the result

Success returns `status: "converged"`, graph/source identity and an artifact
manifest with kind `linear_transport_result`. Read that manifest through
`fibernet://artifacts/{artifact_id}` or `fibernet_artifact_info`. Retrieve its
JSON content with `fibernet_artifact_read`, advancing `offset` until `eof` is
true. Pages are base64-encoded and at most 32 KiB; verify the manifest's byte
size and SHA256 after assembling them.

The content is the native `fibernet-linear-conductance-1` result:

- Node IDs sorted by ID; edge IDs in graph insertion order.
- Endpoint IDs and currents positive from `node_i` to `node_j`.
- Voltages in V, conductances in S, currents/injections in A and powers in W.
- Native residuals, power/current checks, assumptions, tolerances, units,
  input fingerprint and solver provenance.

The fixed native tolerances are `rtol=1e-10`, `current_atol_A=0` and
`power_atol_W=0`. A converged result passed the native finite-output and
conservation checks; those checks are not a forward-accuracy or conditioning
certificate. This route solves steady electrical conduction, not electrical
transients or thermal/mechanical coupling.

The artifact binds the original graph bytes. `fibernet://provenance/1` identifies
the verified native source inventory. The inherited `source_tree` label is an
operator-manifest compatibility tag, not the current source commit. Repeating
identical input performs a native solve and reuses the artifact only when its
complete content, graph source, kind and verified core identity match.

## Limits and recovery

Each electrical call accepts up to 256 nodes, 1,024 physical edges and 256
boundary entries, with a 256 KiB request and 1 MiB result limit. Existing session
limits are one active worker, 128 workers, 64 artifacts and 48 MiB stored data.
Workers have a 512 MiB address-space limit, 10 CPU seconds and an operator wall
timeout of at most 10 seconds.

This is a synchronous MCP request (`taskSupport: "forbidden"`). Cancel it using
normal MCP cancellation. The server terminates and reaps its worker, escalating
to a hard kill when necessary. Failed, timed-out and cancelled calls publish no
result artifact. Closing the session removes its temporary files and artifacts.

Tool errors return `isError=true` and a structured `error` envelope:

- `INVALID_ARGUMENT_OR_HANDLE`: schema, field or handle error.
- `DECLARED_LENGTH_UNIT_REQUIRED` or `DUPLICATE_CONDUCTANCE_ID`.
- `INVALID_TRANSPORT_INPUT`: native graph, ID or boundary refusal.
- `FLOATING_COMPONENT`: a conducting component lacks a voltage anchor.
- `TRANSPORT_SOLVE_FAILED`: native numerical or conservation checks failed.
- `TRANSPORT_GRAPH_LIMIT`, `TRANSPORT_REQUEST_LIMIT` or
  `TRANSPORT_RESULT_LIMIT`: reduce the graph or request size.
- Existing busy, worker timeout, worker failure and provenance errors also apply.

Messages are fixed summaries without local paths. Malformed JSON and transport
security violations, including oversized IDs, strings or frames, can terminate
the session rather than produce a tool response.
