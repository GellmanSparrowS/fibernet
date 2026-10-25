# Prepare deformed transport through MCP

`fibernet_prepare_deformed_transport` is a synchronous, prepare-only tool. It
calls `fibernet.sim.deformed_transport.prepare_deformed_transport` once with an
existing `StructureGraph`. It does not accept mechanical results, execute a
mechanical or electrical solver, infer conductance from material data, transform
coordinate frames, or implement a separate graph or physics model.

## Request

Supply a catalog `graph_id`, an explicit `length_unit` (`m`, `mm`, or `um`), and:

- `translations_m`: records `{ "node_id": 42, "value_m": [0.0, 0.0001] }`, one
  per source node. The two components are prescribed translations in the
  source graph x/y axes, always in metres.
- `reference_conductance_S`: records `{ "edge_id": 307, "value_S": 2.0 }`, one
  per physical edge. Values must be finite and strictly positive.

IDs must be JSON integers from 0 through 9007199254740991, never strings,
booleans, or floating-point representations. Duplicate IDs and incomplete row
counts are rejected before launching a worker. Equal row counts do not prove
membership: the native owner checks complete stable-ID maps. Sparse IDs,
input row ordering, and distinct parallel-edge IDs are retained without
renumbering or collapsing edges. Source graph list order, not request row order,
defines native insertion order.

The declared unit must match the operator catalog. Stored unit/frame labels
must also satisfy native admission. An operator frame label is not an
authenticated coordinate frame; it is not inserted into undeclared metadata.
No coordinate rescaling or repair is performed by the adapter.

The source must round-trip through `StructureGraph.from_dict` and `to_dict`
without changing data, except for the serializer's node/edge list sorting.
Native insertion order must match the source lists. Legacy default-filled or
lossy records are refused rather than corrected.

## Bounds

The tool uses fixed native admission limits of 1536 nodes, 2048 physical edges,
1 MiB aggregate metadata/material data and 2 MiB native report data. Independent
adapter caps are 4 MiB source bytes, 1 MiB serialized request and 4 MiB complete
result bundle. Maximum counts do not guarantee that every payload fits.
Existing electrical-tool limits are unchanged.

The existing strict JSON transport also applies: depth 16, strings at most
4096 UTF-8 bytes, finite numbers, no duplicate object keys and exact safe
integers. No preparation-specific conversion weakens these rules. Malformed
JSON or global transport-limit violations may terminate the transport; admitted
but invalid tool arguments return structured errors. Nested bundle and worker
envelope levels count toward the same depth bound.

The existing single-worker supervisor provides CPU/address-space/file bounds
and an operator-configured wall timeout. These are not an end-to-end latency,
aggregate service-lifetime CPU, peak-memory or physical-validity guarantee.

## Result and resource reads

Success has status `prepared`, not `converged` or a physical certificate. It
returns one `deformed_transport_preparation` artifact with schema
`fibernet.mcp.deformed_transport_bundle/1`, containing:

- `derived_graph`: the complete native graph record.
- `conductance_S`: typed edge-ID/value records in native edge insertion order.
- `binding_report`: the complete, unmodified native preparation report.
- `core_provenance`: the verified native Python-source manifest identity.
- `source_sha256`: the SHA256 of the managed source's original bytes.

The raw source hash and the native canonical graph/order hashes are distinct
identities. The producer's legacy `source_tree` compatibility marker is not a
claim about the latest repository commit; use verified core provenance for the
actual native source identity.

`fibernet://artifacts/{artifact_id}` and `fibernet_artifact_info` return the
manifest. Read the bytes with `fibernet_artifact_read`, following `next_offset`
until `eof`, and verify `total_bytes` and `sha256`. Each read verifies artifact
integrity. The geometry, conductances and report form one immutable byte
snapshot with no mutation endpoint. The graph is not automatically registered
as another catalog graph. Native objects are not claimed to be frozen Python
objects, and the report binds preparation time, not later caller edits.

## Publication and cancellation

All worker identity, result/bundle and prospective reply validation occurs
before the final cancellation checkpoint. Publication then performs one
synchronous bounded file write and registry commit with no intervening await.
A failed write removes its owned partial file and publishes no descriptor.
Deduplicated descriptors receive the same prospective reply validation.

Cancellation already delivered at the final checkpoint publishes nothing.
Cancellation after commit cannot erase a completed artifact, and a disconnected
client is not proof that no commit occurred. There is no job ID, status/cancel
tool, queue, durable job or resumable calculation. Standard MCP request
cancellation terminates and reaps the worker process group, escalating when
necessary, and removes owned request/output temporary files.

Failures contain path-free category/code/message/retryable fields. Native
admission failures map to `PREPARATION_REJECTED`; resource limits, unavailable
core dependencies and native invariant failures remain distinguishable. No
artifact is returned on a failed preparation.

## Portable contract checks

The retained procedural graph and complete native preparation bundle are
stored as deterministic gzip under `tests/fixtures/preparation/`, with encoded
and exact decompressed byte identities and scope in `manifest.json`. They are
immutable expected values, not regenerated by the
tests. `tests/test_preparation.py` runs all checks without a native preparation
call or solver. Missing or mismatched fixture bytes fail rather than skip.

From the repository root, with the adapter's documented test dependencies:

```sh
PYTHONPATH=adapters/mcp/src python -B adapters/mcp/tests/test_preparation.py
```

The existing bounded CI suite also includes this standalone test. Fixture
paths may be supplied through `FIBERNET_PREPARATION_TEST_SOURCE` and
`FIBERNET_PREPARATION_TEST_BUNDLE` together, using either raw JSON or gzip.
Bounded decoding must recover the same retained bytes and identities.
The fixture checks establish wire/storage/delegation
contracts against saved native outputs, not fresh physical validation.
