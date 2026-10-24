# Read a checked FiberNetwork projection report through MCP

`fibernet_project_fiber_network_report` reconstructs an approved saved
`StructureGraph`, calls its public `to_fiber_network(profile=...)` method, and
returns the unchanged native admission/loss report as a chunked JSON artifact.

## Request

Select an approved `graph_id` from `fibernet_catalog`. The strict request is:

```json
{"graph_id":"<catalog handle>"}
```

The only optional inputs are:

- `geometry_policy`: `reject_internal_points` (default) or `endpoint_chord`.
  The latter explicitly permits ignoring nonempty stored edge curves.
- `short_edge_policy`: `reject` (default) or `omit_legacy`.
  The latter permits omission at the native raw-coordinate chord threshold.

There are no client paths, input graph uploads, caller-supplied limits or provider
choices. `missing_material_policy` is fixed to `reject`. The existing MCP source
contract requires schema 3.1 and a material object on every edge. An empty object
is valid and the native loader supplies defaults; for example null shear modulus
is recomputed by that loader. Therefore `reject` does not mean the original
source specified every material property. Missing/null whole material records
are source-preflight failures, not reachable native fallback options.

The profile takes its length unit and coordinate frame from the operator manifest.
Units must be `m`, `cm`, `mm`, `um` or `nm`; `unspecified` and frame `unknown` are
refused. These are operator assertions, not certification. Coordinates, internal
points, radii and box dimensions are neither rescaled nor transformed. Conflicting
stored declarations are refused by the native profile.

## Result and identity

Success returns `status: "completed"`, `scope: "checked_projection_report_only"`,
graph/source identity, the effective profile and ceilings, native summary counts,
a request SHA256 and an artifact manifest with kind `fiber_network_projection_report`.
Completed means construction and report publication finished, not solver convergence.

Read the artifact manifest through its `fibernet://artifacts/{artifact_id}` URI or
`fibernet_artifact_info`. Read the complete JSON using `fibernet_artifact_read`,
advancing byte offsets until `eof`. Chunks contain at most 32768 decoded bytes.
Verify assembled byte size and SHA256. The artifact itself is exactly the native
`fibernet.fiber_network_projection.v1` report, not a rewritten subset.

The report includes stable source IDs, directed endpoints, dense fiber-row
correspondence, curve/omission/material dispositions, counts, explicit losses and
qualification exclusions. IDs are local to the source graph, not global identities.
All IDs retain the existing exact JSON integer range, 0 through 2^53-1.

- `source_sha256` hashes original approved source bytes.
- `request_sha256` hashes the canonical effective profile, including fixed limits;
  combine it with source/core identity when identifying the invocation inputs.
- Artifact SHA256 hashes canonical report content, not the source or invocation.
- `fibernet://provenance/1` identifies the verified native Python source inventory
  in this live session. The historical `source_tree` label is not the current Git tree.

Repeated admitted input executes the native operation again and may reuse the
same artifact when content, source, kind and verified core identity agree. No task
ID, persistent job, network object handle or lossless roundtrip is provided.
The authoritative source is the saved graph after the existing native 3.1 loader,
not an arbitrary pre-serialization in-memory object.

## Fixed bounds

- 256 nodes, 1024 physical edges.
- 4096 source points, 8192 output centerline points, 16384 pairwise crosslinks.
- 1 MiB native metadata/material admission and 1 MiB report admission.
- 4 KiB effective request; 2 MiB private worker envelope including provenance.

Native admission occurs after the existing bounded source reconstruction. These
payload/count limits are not claims of bounded loader peak memory or measured
worst-case performance. The original source byte/depth/string/ID preflight still
applies. Workers retain 512 MiB address space, 10 CPU seconds and an operator wall
deadline at most 10 seconds. One worker may be active; there is no queue.

Existing session limits remain: 128 worker invocations, 64 artifacts, 48 MiB stored
source/artifact bytes, and 64 KiB protocol replies. Catalog and artifact chunk reads
do not consume the worker-invocation budget. Bounded request/result files are
additional transient overhead, not an aggregate-process memory promise.

## Errors, cancellation and publication

Tool errors set `isError=true` and return fixed, path-free codes:

- `INVALID_ARGUMENT_OR_HANDLE`, `DECLARED_LENGTH_UNIT_REQUIRED`, `DECLARED_FRAME_REQUIRED`.
- `PROJECTION_POLICY_REJECTED` (domain): optional diagnostics list each native
  violation group, total count, first at most 128 edge IDs, and a truncation flag.
- `INVALID_PROJECTION_INPUT` or `PROJECTION_NUMERIC_RANGE` (domain).
- `PROJECTION_GRAPH_LIMIT`, `PROJECTION_REQUEST_LIMIT`, `PROJECTION_LIMIT`,
  `PROJECTION_RESULT_LIMIT` (resource).
- `PROJECTION_OUTPUT_CONTRACT_FAILURE` or `PROJECTION_WORKER_OUTPUT_REJECTED`
  (internal); existing busy, worker failure/timeout and provenance refusals also apply.

Malformed source JSON/schema fails startup. Some malformed transport frames
terminate the session rather than produce a tool response. Native error messages,
paths and uncontrolled metadata are never echoed as error instructions.

This is a synchronous request with `taskSupport: "forbidden"`. Normal MCP
cancellation observed while the worker is active terminates its process group,
escalates to SIGKILL if necessary, and reaps it before releasing the active slot.
The publication commit point is successful managed artifact storage, after worker
provenance and new-tool response validation. Failed/cancelled/timed-out operations
publish no new report artifact and do not remove earlier deduplicated artifacts.
Cancellation arriving after completed publication does not undo it. There is no
persistent status/cancellation endpoint. Session shutdown removes ephemeral files;
export required reports before disconnecting.

## Verification scope

The returned legacy FiberNetwork is constructed internally, then discarded.
There is no full-network transfer, network wire encoding, mesh/OBJ export, FE
solve, physical validation, manufacturing certification or new physics model.

`tests/test_projection.py` uses a real MCP SDK stdio client and the checked-in
125-node/804-edge saved scene. It compares all canonical report bytes against a
frozen native reference whose original save/load check included payload, insertion
order, curve shapes and every material field. No fixture generation, topology or
geometry analysis, OBJ pipeline, training or physical simulation is rerun.

The two saved JSON fixtures are stored as `.json.gz.b64` text files to keep
repository transport small. `tests/projection_fixtures.py` bounds the encoded
read, validates canonical base64, caps gzip output, requires one complete member
with no trailing data, and checks exact original byte lengths and SHA256 values.
The source decodes to 675,329 bytes and the report to 458,163 bytes, without
regeneration or JSON reserialization. Tests materialize the source only in their
private temporary directory before supplying the ordinary approved graph path.
This packaging is test-only; it changes neither the MCP schema nor native loading.

Separate small saved-byte cases compare the same loader on both sides. Checks
cover strict loss diagnostics, ID mapping, source normalization, bounds, chunking,
publication, provenance and worker ownership. The hard-kill test deliberately
SIGSTOPs the actual worker to exercise escalation; it is not physical or numerical
cancellation validation. Source-extracted/synthetic controls are labelled separately
from actual SDK execution. Installed-wheel checks also compare the frozen report.
