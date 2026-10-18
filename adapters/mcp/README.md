# FiberNet MCP adapter

A separate optional Linux stdio adapter exposing five bounded native graph and artifact tools. It uses Python SDK 1.29.0 and negotiated protocol 2025-11-25, with compatibility checks on Linux CPython 3.12; other SDK/protocol/platform combinations are not validated. This is an unreleased compatibility implementation. It exposes no simulation, generation, arbitrary code execution, URL fetching or remote HTTP service.

## Install with the matching native core

The adapter requires Python >=3.10 and exactly mcp 1.29.0. Install FiberNet separately from the matching source checkout, then this adapter into the same virtual environment. A package merely named FiberNet 4.3.0 is not sufficient: all 217 native Python source hashes must match core_manifest.json. The matched native source tree is 13b4fb0657681418c9ec2e524e2394eda7846d9b. This label identifies the native-only repository projection, excluding the MCP adapter and its workflow; it is not the enclosing combined source tree or commit. This whole-core pin includes unrelated modules: any change to a listed native module requires a refreshed manifest and renewed compatibility validation. No narrower API compatibility guarantee is implemented. Do not replace that manifest to bypass a mismatch; a different source requires an independently validated adapter build.

From the repository containing this adapter:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m pip install ./adapters/mcp
```

The first install is the native core from this checkout, not a PyPI version assumption. The second installs this independent distribution and its fixed SDK dependency. Native optional physics/ML dependencies are not required. Use a matching Python interpreter for the MCP client process. No server tool installs dependencies. No base FiberNet dependency or Python-version requirement is changed.

## Configure and launch locally

Create an operator-owned JSON file such as manifest.json. Paths must be absolute regular-file paths without symlink components. Replace the placeholder path and SHA with your chosen graph's actual bytes; for the included fixture, use the SHA below.

```json
{"source_tree":"13b4fb0657681418c9ec2e524e2394eda7846d9b","graphs":[{"path":"/absolute/path/to/voronoi_4340.json","sha256":"fb40544438cd4e7ca9bc60d8f10c6363c169b0aa7ce99f4e0a3d45b64629c584","length_unit":"unspecified","frame_id":"unknown"}]}
```

```sh
mkdir -p local-mcp-output
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m fibernet_mcp.server --manifest manifest.json --output-root local-mcp-output
```

This command speaks MCP over stdin/stdout; configure it as a local stdio server in an MCP client. Startup configuration is operator-controlled, not a tool argument. The included synthetic fixture has 4340 nodes and 5073 physical edges. No model-supplied filesystem paths are accepted. Export desired artifacts before disconnecting: sessions are ephemeral.

## Wire contract

Exact schemas: src/fibernet_mcp/tool_schemas.json. tools/list uses the same schemas with unreachable local definitions removed to fit the reply limit.

- fibernet_catalog: approved graph summaries and limits
- fibernet_validate: valid/invalid serialization-and-topology report, not solver readiness
- fibernet_analyze_topology: native scalar summary plus complete ID-aligned JSON artifact
- fibernet_artifact_info: manifest only
- fibernet_artifact_read: offset/length byte chunks, base64, maximum32768 bytes per chunk

resources/read supports fibernet://capabilities and fibernet://artifacts/{artifact_id}; the latter returns a small manifest, not unbounded artifact contents. No other resource schemes are read. Complete data are reconstructed through chunks and verified against SHA256. structuredContent is authoritative; text content is a concise pointer rather than duplicating up to44KiB base64. Errors carry stable categories/codes and no raw exception/path/metadata. Unknown tool names and unknown resources use protocol errors. Validation may successfully report an invalid graph; analysis of it returns a domain error.

SourceSHA hashes original input bytes. ArtifactSHA hashes compact sorted JSON result bytes, and is not expected to equal the sourceSHA. Native rounded fingerprint is never an immutable identity. Units/frame are explicitly operator-asserted or unknown; native schema3.1 does not verify them. Metadata are treated as data and never incorporated into tool instructions.

## Bounds and trust model

The supported entrypoint fails closed off Linux and requires the exact SDK. Server address-space soft limit384MiB with inherited hard ceiling512MiB; worker512MiB,10CPU seconds,10wall seconds,16MiB file output, no core dumps, one BLAS thread. At most one worker is active, no queued work; at most128 worker invocations per session. Timeout/cancellation/EOF terminate the actual fixed-operation child, escalate after0.25s to SIGKILL if needed, and reap it before clearing active state. Partial worker output is discarded; completed artifacts are only published after successful bounded result parsing. The CLI may reduce, never increase, the10second deadline.

Graph limit16MiB,10000nodes,30000edges, depth16, UTF-8 strings4096bytes, at most256 internal points per edge, exact integer IDs up to2^53-1. Strict preflight checks original JSON before native reconstruction. It rejects duplicate JSON/entity keys, nonfinite values and coercible string/bool/fractional numeric inputs. Native2D box_size may have a zero third component. Schema-invalid sources fail startup. Schema-valid but native/topology-invalid sources remain reachable for invalid reports.

Incoming line hard limit20MiB before JSON decoding; stricter JSON budget16MiB then applies. Outgoing protocol replies<=64KiB. Session source/artifact storage<=48MiB, at most32 graphs/64 artifacts. The one transient worker result is additionally bounded16MiB. The measured smoke artifact footprint stays below its separate60MiB test budget. Files are exclusively created under a private server-owned session directory, opened via no-follow directory-relative descriptors, and checked against stored hashes when read. Every input path component rejects symlinks; no arbitrary client paths exist. Session-owned sources/artifacts are ephemeral and removed at shutdown. Export required results before disconnecting. Optional bounded operator audit records worker termination evidence; no audit or cleanup operation overwrites preexisting files.

This is not an OS hostile-process sandbox or multi-tenant authentication boundary. A same-UID operator can inspect local files; hashes/confinement detect selected tampering but do not confer cryptographic authorization or protect against a compromised Python/runtime/kernel. It runs only fixed trusted native code and has no network operations. The supervisor does not support detached/restart-persistent jobs.


## Reproduce acceptance checks

Install test-only psutil with `python -m pip install './adapters/mcp[test]'`. Run from the repository root using the matching installed native core:

```sh
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/bounded_smoke.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/core_binding.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/adversarial.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/transport_edges.py
PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/cleanup_failures.py
```

For a source-only native checkout, set FIBERNET_CORE_PATH to its absolute root and include that root in the parent PYTHONPATH too. Acceptance creates a fixture manifest and disposable evidence beneath the adapter directory, so this source-test directory must be writable. Normal server operation does not require writing into the installed package. The independent SDK client negotiates initialize, tools/list/call and resource retrieval, rebuilds chunked artifacts by SHA, checks every native topology field, and exercises error/timeout/cancellation/kill/EOF behavior. Additional tests cover byte budgets, tampering, source changes, malformed input and owned-file cleanup failures. Core-binding and adversarial checks pair exact validation-boundary refusals with real SDK positive controls on a valid output root; a nonzero startup exit alone is not sufficient evidence. No physics solve or rendered graphical-client integration is part of these tests.

The complete `bounded_suite.py` runner rejects runs exceeding120CPU seconds (terminal kernel waited-child usage plus supervisor usage, with one CPU second reserved for report/exit work),300wall seconds or60MiB of adapter-directory data, including reserved report space. It samples process-tree RSS against1GiB and monitors intermediate CPU/disk use; these observations are not hard aggregate-memory limits. A Linux per-process child subreaper captures short-lived and orphaned descendants, checks terminal resources even on failure, and rejects any worker-exit leak before bounded kill/reap cleanup. The nested acceptance smoke is an additional sampled monitor. Existing server/worker OS limits and the CI job/command wall deadlines remain separate safeguards. Run the full suite with `PYTHONPATH=adapters/mcp/src python adapters/mcp/tests/bounded_suite.py`. Raw logs distinguish unsuccessful attempts from passing runs. Scientific and authorization limits above apply even when protocol checks pass.

## Separate adapter CI and wheel checks

The Linux Python 3.12 adapter workflow builds and installs separate core and adapter wheels, verifies the embedded 217-file binding and exact MIT notice, and exercises actual installed SDK initialize/catalog and worker topology/chunk/native-parity calls. It also runs the bounded source acceptance, binding-refusal, adversarial, transport and cleanup scripts. The main cross-platform core test workflow does not collect these standalone adapter scripts. The fixed SDK/protocol and Linux-only support limits still apply; this is not evidence for a graphical MCP client or another operating system.

The standalone adapter wheel includes the repository MIT copyright and permission notice. To check an already built and installed pair, run `python adapters/mcp/tests/distribution.py --wheels /absolute/path/to/wheels` from the matching source checkout. An explicit `--installed-root /absolute/path/to/pip-target` can check a separate `pip --target` installation. The check refuses to treat the source checkout as the installed distribution.
