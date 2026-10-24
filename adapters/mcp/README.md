# FiberNet MCP adapter

A separate optional, private Linux stdio adapter exposing seven bounded native
operations. It uses MCP Python SDK 1.29.0 and negotiated protocol 2025-11-25;
compatibility is checked on Linux CPython 3.12. Other combinations are unvalidated.
This unreleased implementation has no HTTP service, arbitrary code execution,
URL fetching, model-supplied filesystem paths, graph generation or deployment.

## Install the matching core and adapter

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m pip install ./adapters/mcp
```

Install the native core from this matching checkout, not merely a package named
FiberNet 4.3.0. The current manifest binds all 242 native Python source files and
the complete `.py` inventory. The fixed label
`6eabb154cd37d95c7c447a568b1b737c3ec883c8` remains the historical native-core snapshot
label used by v1 outputs/operator configuration. It is not the current enclosing
repository tree, commit or a unique identity for a changed manifest. Older labels
`195494f987805cc14dcf66f4713565e2e5c53b08` and
`d04536e2261c9898b8b1eca91cb33d887cab977d` remain historical too.

Runtime verification enforces source bytes and inventory, not the current Git tree.
Any listed native-source change requires a separately validated binding refresh;
never replace the manifest merely to bypass a refusal. Use the additive verified
provenance resource for the precise current manifest identity. The adapter has its
own distribution; no base FiberNet dependency or Python requirement is changed.
It requires Python >=3.10 and exactly mcp 1.29.0. No tool installs dependencies.

## Operator configuration and local launch

Use an operator-owned JSON manifest with absolute regular-file paths and no symlink
components. For the included topology fixture:

```json
{"source_tree":"6eabb154cd37d95c7c447a568b1b737c3ec883c8","graphs":[{"path":"/absolute/path/to/voronoi_4340.json","sha256":"fb40544438cd4e7ca9bc60d8f10c6363c169b0aa7ce99f4e0a3d45b64629c584","length_unit":"unspecified","frame_id":"unknown"}]}
```

```sh
mkdir -p local-mcp-output
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m fibernet_mcp.server --manifest manifest.json --output-root local-mcp-output
```

The manifest is operator-controlled, never a tool argument. This fixture has
4340 nodes and 5073 physical edges, suitable for topology, not the tighter
projection/electrical operation limits. Declare actual units/frame for those tools.
Sessions are ephemeral; export results before disconnecting.

## Seven tools and resources

Exact input/output schemas are in `src/fibernet_mcp/tool_schemas.json`. tools/list
prunes only unreachable local definitions to keep its response bounded.

- `fibernet_catalog`: approved graph summaries and global source/reply limits.
- `fibernet_validate`: serialization/topology validation, not solver readiness.
- `fibernet_analyze_topology`: native summary and complete ID-aligned artifact.
- `fibernet_artifact_info`: small artifact manifest.
- `fibernet_artifact_read`: base64 byte chunks, at most 32768 decoded bytes.
- `fibernet_solve_linear_transport`: existing steady DC conductance result;
  see [ELECTRICAL.md](ELECTRICAL.md) for its specific inputs and physical scope.
- `fibernet_project_fiber_network_report`: checked compatibility projection report
  only; see [PROJECTION.md](PROJECTION.md). No network transfer, mesh export or FE solve.

Resources are `fibernet://capabilities`, `fibernet://provenance/1`, and artifact
manifests at `fibernet://artifacts/{artifact_id}`. No other scheme is read.
resources/read never returns unbounded artifact contents; use chunks and verify
byte size/SHA256. structuredContent is authoritative; text is a concise pointer.
Unknown tools/resources use protocol errors. Stable tool errors contain no raw
exception, path or uncontrolled metadata. Metadata remain data, not instructions.

SourceSHA hashes original input bytes; artifact SHA hashes compact sorted JSON
result bytes. They need not match. Rounded native fingerprints are not immutable
identities. Operator unit/frame declarations are not source-schema certification.

## Verified native provenance

`fibernet://provenance/1` returns exactly:

```json
{"schema_version":"fibernet.core-provenance/1","canonical_core_manifest_sha256":"<64 lowercase hex characters>","source_file_count":242,"scope":"fibernet_python_sources"}
```

Its digest uses the same strictly parsed core_manifest.json object that verified
all native source files: sorted keys, compact separators, ASCII JSON escaping,
UTF-8, no trailing newline. Duplicate keys, malformed digests and invalid relative
paths are refused. Formatting/key order do not change this canonical digest.
The historical Producer source_tree field is unchanged, not reinterpreted.

The server retains startup identity. Each worker freshly verifies sources before
scientific imports and returns a private provenance envelope. Missing/different
identity is rejected before results or artifacts are published. Retrieval and
artifact deduplication bind to that same live-session identity. The resource is
not standalone attestation for previously exported artifacts.

Scope excludes adapter source, dependencies, interpreter/environment, non-Python
package data, graphs, repository and Git history. Verification is not continuous
attestation or protection against replacement after verification/compromised code.
No install paths, raw manifests, graph data or environment values are exposed.

## Bounds, lifecycle and trust

Server address-space soft limit is 384 MiB, inherited hard ceiling 512 MiB. Workers
have 512 MiB address space, 10 CPU seconds, ≤10 wall seconds, 16 MiB file output, no core
dumps and one BLAS thread. One worker is active, with no queue; the session allows
128 worker invocations. Catalog/chunk reads do not consume worker invocations.
Timeout/cancellation/EOF terminate the actual child, escalate after 0.25 s if needed,
and reap before clearing active state. No detached/restart-persistent tasks exist.

Global source limits: 16 MiB, 10000 nodes, 30000 edges, JSON depth 16, strings 4096 UTF-8 bytes,
256 internal points per edge, exact integer IDs through 2^53-1. Original-byte
preflight rejects duplicates, nonfinite values and coercible malformed IDs/values
before native reconstruction. Native 2D boxes may have a zero third component.
Schema-invalid source fails startup; native/topology-invalid admitted source may
produce an invalid report. Operation-specific tighter ceilings are documented above.

Incoming line cap 20 MiB precedes JSON decoding; JSON cap 16 MiB then applies. Replies
are ≤64 KiB. Stored source/artifact bytes are ≤48 MiB, with ≤32 graphs/64 artifacts.
The bounded transient request/result files are additional overhead. Private owned
session files use no-follow directory-relative descriptors and hashes on retrieval.
Cleanup removes only owned files; optional operator audit never overwrites files.

This is fixed trusted code, not a hostile-process sandbox, multi-tenant security
boundary, aggregate-memory guarantee or protection against a same-UID/compromised
operator/runtime/kernel. There are no network operations.

## Reproduce verification

Use test-only psutil/pytest via `python -m pip install './adapters/mcp[test]'` in an
authorized test environment. From a matching checkout:

```sh
python adapters/mcp/tests/provenance_contract.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/bounded_suite.py
```

For source-only core testing, also put the absolute core root in PYTHONPATH and
set FIBERNET_CORE_PATH. Acceptance creates disposable evidence under the adapter;
normal installed operation does not write into the package. The full suite is
bounded 120 CPU seconds/300 wall seconds/60 MiB adapter data, samples aggregate RSS
against 1 GiB, accounts terminal waited-child CPU and checks/kills/reaps descendants.
Sampling is not a hard aggregate-memory sandbox. Server/worker OS caps remain.

Actual SDK sessions verify initialize/tools/resources, chunk integrity, native
parity, errors, timeout/cancellation/kill/EOF and cleanup. Source-extracted controls
are identified as such, not actual protocol sessions. Projection reference reuse
is documented in PROJECTION.md; its checks are not additional physical validation.

Separate Linux CPython 3.12 CI builds/installs core and adapter wheels and checks
binding, MIT notice and installed-client topology/DC/projection report results.
`distribution.py --wheels /absolute/path` checks an installed pair; optional
`--installed-root /absolute/path/to/prefix` checks an explicit installed prefix and
refuses the source checkout. No graphical-client or other-platform validation is claimed.
