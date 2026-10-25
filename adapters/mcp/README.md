# FiberNet MCP adapter

A separate optional Linux stdio adapter exposing seven bounded native graph, electrical, prepare-only and artifact tools. It uses Python SDK 1.29.0 and negotiated protocol 2025-11-25, with compatibility checks on Linux CPython 3.12; other SDK/protocol/platform combinations are not validated. This is an unreleased compatibility implementation. It exposes no electrothermal or mechanical execution, generation, arbitrary code execution, URL fetching or remote HTTP service.

## Install with the matching native core

The adapter requires Python >=3.10 and exactly mcp 1.29.0. Install FiberNet separately from the matching source checkout, then this adapter into the same virtual environment. A package merely named FiberNet 4.3.0 is not sufficient: all 246 native Python source hashes must match core_manifest.json. The required native core snapshot is 6eabb154cd37d95c7c447a568b1b737c3ec883c8. This fixed label identifies the native-only repository projection for this adapter build, excluding the MCP adapter and its workflow; it is not the current enclosing source tree or commit. Test-only maintenance may change the repository projection without changing the 246 bound runtime module bytes. Runtime compatibility checks enforce those module hashes, not the current Git tree; the snapshot label remains the historical producer/core snapshot label used by v1 outputs and operator configuration. It is not a unique identity for a subsequently changed core manifest; use the additive verified provenance resource below for that narrower identity. This whole-core pin includes unrelated modules: any change to a listed native module requires a refreshed manifest and renewed compatibility validation. No narrower API compatibility guarantee is implemented. Do not replace that manifest to bypass a mismatch; a different source requires an independently validated adapter build.

From the repository containing this adapter:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m pip install ./adapters/mcp
```

The first install is the native core from this checkout, not a PyPI version assumption. The second installs this independent distribution and its fixed SDK dependency. Native optional physics/ML dependencies are not required. Use a matching Python interpreter for the MCP client process. No server tool installs dependencies. No base FiberNet dependency or Python-version requirement is changed.

Earlier adapter builds used snapshot labels `195494f987805cc14dcf66f4713565e2e5c53b08` and `d04536e2261c9898b8b1eca91cb33d887cab977d`. Those labels remain historical; they do not identify this core. Operator manifests must use the snapshot label for their matching adapter build. A label mismatch is an explicit compatibility error, not permission to bypass core hash verification.

## Configure and launch locally

Create an operator-owned JSON file such as manifest.json. Paths must be absolute regular-file paths without symlink components. Replace the placeholder path and SHA with your chosen graph's actual bytes; for the included fixture, use the SHA below.

```json
{"source_tree":"6eabb154cd37d95c7c447a568b1b737c3ec883c8","graphs":[{"path":"/absolute/path/to/voronoi_4340.json","sha256":"fb40544438cd4e7ca9bc60d8f10c6363c169b0aa7ce99f4e0a3d45b64629c584","length_unit":"unspecified","frame_id":"unknown"}]}
```

```sh
mkdir -p local-mcp-output
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m fibernet_mcp.server --manifest manifest.json --output-root local-mcp-output
```

This command speaks MCP over stdin/stdout; configure it as a local stdio server in an MCP client. Startup configuration is operator-controlled, not a tool argument. The included synthetic fixture has 4340 nodes and 5073 physical edges. No model-supplied filesystem paths are accepted. Export desired artifacts before disconnecting: sessions are ephemeral.

## Wire contract

Exact schemas: src/fibernet_mcp/tool_schemas.json. tools/list uses the same schemas with unreachable local definitions removed to fit the reply limit.

The seventh tool, `fibernet_prepare_deformed_transport`, returns one immutable native geometry/conductance/binding bundle without running a solver. Its exact ID, unit, source-round-trip, limit and cancellation contract is documented in [Deformed transport preparation](DEFORMED_TRANSPORT.md). Existing electrical limits are unchanged.

- fibernet_catalog: approved graph summaries and limits
- fibernet_validate: valid/invalid serialization-and-topology report, not solver readiness
- fibernet_analyze_topology: native scalar summary plus complete ID-aligned JSON artifact
- fibernet_artifact_info: manifest only
- fibernet_artifact_read: offset/length byte chunks, base64, maximum32768 bytes per chunk
- fibernet_solve_linear_transport: the existing bounded steady-DC electrical tool; see [Electrical method and limits](ELECTRICAL.md)
- fibernet_prepare_deformed_transport: native prepare-only geometry/conductance/binding bundle; see [Deformed transport preparation](DEFORMED_TRANSPORT.md)

resources/read supports fibernet://capabilities, fibernet://provenance/1 and fibernet://artifacts/{artifact_id}; the latter returns a small manifest, not unbounded artifact contents. No other resource schemes are read. Complete data are reconstructed through chunks and verified against SHA256. structuredContent is authoritative; text content is a concise pointer rather than duplicating up to44KiB base64. Errors carry stable categories/codes and no raw exception/path/metadata. Unknown tool names and unknown resources use protocol errors. Validation may successfully report an invalid graph; analysis of it returns a domain error.

SourceSHA hashes original input bytes. ArtifactSHA hashes compact sorted JSON result bytes, and is not expected to equal the sourceSHA. Native rounded fingerprint is never an immutable identity. Units/frame are explicitly operator-asserted or unknown; native schema3.1 does not verify them. Metadata are treated as data and never incorporated into tool instructions.

## Verified core provenance resource, version 1

`fibernet://provenance/1` is an additive read-only resource. At its introduction, the five tool schemas,
v1 `Producer` fields, capabilities contents and legacy operator `source_tree` gating
were unchanged. The historical label `6eabb154cd37d95c7c447a568b1b737c3ec883c8`
is not being reinterpreted as a protocol-only tag or asserted to identify a modified
runtime manifest. The new resource returns exactly four fields:

```json
{"schema_version":"fibernet.core-provenance/1","canonical_core_manifest_sha256":"<64 lowercase hex characters>","source_file_count":246,"scope":"fibernet_python_sources"}
```

The digest is SHA256 of UTF-8 JSON for the same strictly parsed `core_manifest.json`
object used to verify all core Python source bytes and the complete `.py` inventory.
Canonicalization uses sorted object keys, compact separators `,` and `:`, ASCII
JSON escaping (`ensure_ascii=True`) and no trailing newline. Duplicate keys,
invalid source-relative paths and malformed SHA256 values are rejected. Formatting
or key-order changes do not change the digest; any source binding change does.
This is a canonical manifest digest, not the SHA256 of the raw JSON file.

The server retains the identity verified at startup. Each fixed worker freshly
verifies its own source files and returns that identity in a private envelope.
The server rejects absent/different identities before publishing worker results
or artifacts. Artifact deduplication and retrieval are bound to the same verified
identity internally; no extra provenance fields enter the closed v1 tool or
Producer shapes. Clients can associate this resource with artifacts from that
same live session. It is not a standalone attestation for exported older artifacts.

Scope is only the listed FiberNet Python source files. It excludes the adapter,
non-Python package data, dependencies, interpreter, environment, graphs, whole
repository and Git history. Verification occurs at server startup and before
worker scientific imports; this is not continuous runtime attestation or protection
against source replacement after verification or a compromised process. The
resource exposes no install paths, raw manifests, graph data or environment values.
No additional simulation/solver capability or dependency is introduced.

## Bounds and trust model

The supported entrypoint fails closed off Linux and requires the exact SDK. Server address-space soft limit384MiB with inherited hard ceiling512MiB; worker512MiB,10CPU seconds,10wall seconds,16MiB file output, no core dumps, one BLAS thread. At most one worker is active, no queued work; at most128 worker invocations per session. Timeout/cancellation/EOF terminate the actual fixed-operation child, escalate after0.25s to SIGKILL if needed, and reap it before clearing active state. Partial worker output is discarded; completed artifacts are only published after successful bounded result parsing. The CLI may reduce, never increase, the10second deadline.

Graph limit16MiB,10000nodes,30000edges, depth16, UTF-8 strings4096bytes, at most256 internal points per edge, exact integer IDs up to2^53-1. Strict preflight checks original JSON before native reconstruction. It rejects duplicate JSON/entity keys, nonfinite values and coercible string/bool/fractional numeric inputs. Native2D box_size may have a zero third component. Schema-invalid sources fail startup. Schema-valid but native/topology-invalid sources remain reachable for invalid reports.

Incoming line hard limit20MiB before JSON decoding; stricter JSON budget16MiB then applies. Outgoing protocol replies<=64KiB. Session source/artifact storage<=48MiB, at most32 graphs/64 artifacts. The one transient worker result is additionally bounded16MiB. The measured smoke artifact footprint stays below its separate60MiB test budget. Files are exclusively created under a private server-owned session directory, opened via no-follow directory-relative descriptors, and checked against stored hashes when read. Every input path component rejects symlinks; no arbitrary client paths exist. Session-owned sources/artifacts are ephemeral and removed at shutdown. Export required results before disconnecting. Optional bounded operator audit records worker termination evidence; no audit or cleanup operation overwrites preexisting files.

This is not an OS hostile-process sandbox or multi-tenant authentication boundary. A same-UID operator can inspect local files; hashes/confinement detect selected tampering but do not confer cryptographic authorization or protect against a compromised Python/runtime/kernel. It runs only fixed trusted native code and has no network operations. The supervisor does not support detached/restart-persistent jobs.


## Reproduce acceptance checks

Install test-only psutil with `python -m pip install './adapters/mcp[test]'`. Run from the repository root using the matching installed native core:

```sh
python adapters/mcp/tests/provenance_contract.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/bounded_smoke.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/core_binding.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/adversarial.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/transport_edges.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/cleanup_failures.py
```

For a source-only native checkout, set FIBERNET_CORE_PATH to its absolute root and include that root in the parent PYTHONPATH too. Acceptance creates a fixture manifest and disposable evidence beneath the adapter directory, so this source-test directory must be writable. Normal server operation does not require writing into the installed package. The independent SDK client negotiates initialize, tools/list/call and resource retrieval, rebuilds chunked artifacts by SHA, checks every native topology field, and exercises error/timeout/cancellation/kill/EOF behavior. Additional tests cover byte budgets, tampering, source changes, malformed input and owned-file cleanup failures. Core-binding and adversarial checks pair exact validation-boundary refusals with real SDK positive controls on a valid output root; a nonzero startup exit alone is not sufficient evidence. The existing steady-DC electrical checks are described in [Electrical method and limits](ELECTRICAL.md); these protocol checks do not qualify electrothermal or mechanical solves or a rendered graphical client.

The `provenance_contract.py` check uses temporary synthetic source files, isolated callback/worker bodies and real JSON Schema validation. It checks canonical manifest identity, strict source verification, worker/cache mismatch refusal, privacy and immutable v1 output fixtures without launching an MCP session or importing scientific providers. The existing SDK acceptance and installed-distribution sessions also check the additive resource against an independently canonicalized core manifest; they do not add sessions or topology workers for this check.

The complete `bounded_suite.py` runner rejects runs exceeding120CPU seconds (terminal kernel waited-child usage plus supervisor usage, with one CPU second reserved for report/exit work),300wall seconds or60MiB of adapter-directory data, including reserved report space. It samples process-tree RSS against1GiB and monitors intermediate CPU/disk use; these observations are not hard aggregate-memory limits. A Linux per-process child subreaper captures short-lived and orphaned descendants, checks terminal resources even on failure, and rejects any worker-exit leak before bounded kill/reap cleanup. The nested acceptance smoke is an additional sampled monitor. Existing server/worker OS limits and the CI job/command wall deadlines remain separate safeguards. Run the full suite with `PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/bounded_suite.py`. Raw logs distinguish unsuccessful attempts from passing runs. Scientific and authorization limits above apply even when protocol checks pass.

## Separate adapter CI and wheel checks

The Linux Python 3.12 adapter workflow builds and installs separate core and adapter wheels, verifies the embedded 246-file binding and exact MIT notice, and exercises actual installed SDK initialize/catalog and worker topology/chunk/native-parity calls. It also runs the bounded source acceptance, binding-refusal, adversarial, transport and cleanup scripts. The main cross-platform core test workflow does not collect these standalone adapter scripts. The fixed SDK/protocol and Linux-only support limits still apply; this is not evidence for a graphical MCP client or another operating system.

The standalone adapter wheel includes the repository MIT copyright and permission notice. To check an already built and installed pair, run `python adapters/mcp/tests/distribution.py --wheels /absolute/path/to/wheels` from the matching source checkout. An explicit `--installed-root /absolute/path/to/pip-target` can check a separate `pip --target` installation. The check refuses to treat the source checkout as the installed distribution.

## Historical native-core compatibility notes

The following notes describe earlier core additions and their five-tool adapter
surface. The current seven-tool surface is listed above. The saved-electrothermal
bridge changes the native source binding, not the MCP tool surface; it adds no
electrothermal or mechanical MCP execution tool.

The paired core includes explicit high-level LAMMPS dynamics dispatch and its backend-tagged SimResult state schema. Geometry-facing result arrays retain declared graph units while native state is SI; stable IDs and explicit equilibrium lengths remain authoritative. The MCP tool surface remains the existing five graph/topology/artifact tools. SDK startup and saved-network parity checks do not execute the high-level physics dispatch or establish a native LAMMPS trajectory result.

The paired core adds connected physical-edge-once manufacturing trails and stable-ID
route mappings. The existing five MCP graph/topology/artifact tools are unchanged;
no manufacturing or machine-control tool is exposed. Compatibility checks cover
the updated core hashes, SDK startup and topology artifact parity.

The paired core also exports explicit one-way Joule load assembly and heating.
The existing five MCP graph/topology/artifact tools are unchanged; no
electrical or thermal execution tool is added.

The paired core also exports quasistatic 2-D material-history viscoelasticity.
The existing five MCP tools are unchanged; no viscoelastic execution tool or
additional optional dependency is introduced.

The paired Python core includes bounded pure-fiber LAMMPS CG preparation and
execution. The existing five MCP tools are unchanged: no new physics-task or
live native cancellation endpoint is exposed by this adapter.

The paired core supports small-strain dynamic axial viscoelasticity using the
existing Taichi integrator and explicit material history. The existing five MCP
graph/topology/artifact tools are unchanged; no dynamic-material execution tool
is exposed by this adapter.

The paired core separates automatic SB3 evaluation from its training environment
and clarifies the distinct root random-rollout helper. This refresh changes the
whole-core source binding, not the five MCP tools: no RL execution tool or new
optional native dependency is added.
