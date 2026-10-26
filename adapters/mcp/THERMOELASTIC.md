# Prescribed-temperature planar network snapshot

`fibernet_solve_thermoelastic_network_snapshot` is the eighth tool. It adds real
mechanical execution to the optional Linux stdio adapter, using exactly one call
to `fibernet.sim.thermoelastic.solve_thermoelastic_network_snapshot`. It does not
call an example, generate a graph, implement another solver, or solve a heat
field. Uniform temperature equal to the reference is an ordinary zero-thermal-
strain input, not an implicit static alias with hidden temperature defaults.

## Admitted model

Only native profile `thermoelastic_2d_network/1` is exposed: 2–192 nodes,
1–256 physical edges, one connected planar welded Euler–Bernoulli network,
solid circular sections from edge radii and a caller-specified uniform `E_Pa`.
Parallel physical members retain distinct stable IDs. No `size_limits` override,
backend selector, elementary profile, time history or automatic repair exists.

The native owner rejects curved members (straight ordered internal points are
allowed), non-XY-planar geometry, invalid supports and cases outside its motion,
strain and thermal-strain bounds. These are admission checks, not a material,
junction, shear, mesh, buckling or stability certificate. Per-edge material and
arbitrary shape metadata are **not inferred as constitutive input**: this profile
uses the explicit uniform modulus and circular radii. No thermal solve, feedback,
geometric stiffness or numerical regularization is added.

## Input and units

The source must round-trip through the native StructureGraph loader without
losing fields or stable-ID insertion order. A shortened JSON graph that passes
topology validation may still be refused here rather than silently filling or
dropping serialized fields.

Every top-level field below is required; additional fields are rejected:

| Field | Contract |
|---|---|
| `graph_id` | Session-local handle returned by `fibernet_catalog`; never a path or URL. |
| `length_unit` | `m`, `mm` or `um`, exactly matching the operator's graph declaration. Only native code converts graph coordinates/radii into metres. |
| `E_Pa` | Finite positive uniform Young's modulus; no material inference. |
| `stress_free_temperature_K` | Finite positive absolute reference temperature. |
| `temperature_K` | Complete records `{node_id, value_K}`; positive absolute kelvin for every node. |
| `expansion_alpha_per_K` | Complete records `{edge_id, value_per_K}`; finite signed coefficient for every physical edge. |
| `constraints` | Records `{node_id, ux_m, uy_m, theta_rad}`. Each component is a finite SI value or `null` for a free DOF. Omitted nodes are free. |
| `loads` | Records `{node_id, Fx_N, Fy_N, Mz_Nm}`; finite SI nodal loads. `[]` means no applied load. |

IDs are nonnegative JSON integers in the exact interoperable range, not strings,
bools, coordinates or array-row indices. Duplicate and unknown IDs are rejected
before dictionary conversion. Temperatures and coefficients require full ID
coverage. The existing strict JSON finite-number/depth/integer-range contract
continues to apply; no numeric string conversion is performed.

Constraint values remain m/rad and loads remain N/N·m even for `mm`/`um` graphs.
All components refer to the graph's XY frame. `frame_id`, `frame_provenance` and
`unit_provenance` come from the operator catalog, are included in request binding,
and cannot be overridden by a tool argument. They remain assertions (or an
unknown frame), not verified physical calibration or a coordinate transform.

## Result, identity and lifetime

Success returns `status: solved`, the exact native profile, source SHA256,
canonical request SHA256, native input SHA256, declared unit/frame provenance,
and an artifact descriptor of kind `thermoelastic_network_snapshot`.

The artifact contains the complete **unmodified 36-field native snapshot**,
including stable IDs, original/SI geometry, inputs, masks, displacements,
reactions, stresses, edge forces, moments, residual bounds and applicability
fields. Arrays use the native sorted node/edge ID order, not assumed row IDs.
The adapter checks complete finite layout and source/request/provenance binding;
it does not recalculate equilibrium or certify physics independently.

`fibernet://artifacts/{artifact_id}` and `fibernet_artifact_info` return the
manifest, not result bytes. Use `fibernet_artifact_read` with byte offsets and
lengths up to 32768; concatenate base64-decoded chunks and check the whole-file
SHA256. Handles/artifacts expire when the session closes. The native snapshot
is a JSON result, not a native restart/checkpoint contract.

## Bounds and errors

New-route limits are 1 MiB source graph, 64 KiB request and 1 MiB native result.
Count/source/request admission occurs before the scientific worker; source
integrity is rechecked with bounded reads. These limits do not predict sparse
factorization fill. Existing process CPU/address-space/file limits, 10-second
worker timeout, session limits and cancellation cleanup still apply.

Errors use `isError: true` and `structuredContent.error` with bounded symbolic
codes: invalid wire input/unit/source binding is a contract error; unsupported
native model/motion/strain/support input is `SNAPSHOT_DOMAIN_REJECTED`; resource
caps use `SNAPSHOT_*_LIMIT`; dependency/invariant failures are internal errors.
Raw native exceptions, filesystem paths and input tables are not exposed.
There is no fallback, regularization or automatic retry.

Worker result, provenance, full snapshot and reply schema validation precede a
final cancellation checkpoint and the sole synchronous artifact commit. Failure
or cancellation before commit publishes no result; after commit, a response-
delivery failure does not roll back the published session artifact. Existing tools retain their contracts;
the historical provenance fixture is unchanged and tested through an exact
inverse of the additive tool extension.

## Validation coverage

Contract tests cover typed records, forwarding, publication and cancellation with
synthetic/stubbed results. An actual installed SDK call for an 81-node/208-edge
network at 300 K with zero thermal expansion reproduced all 36 native snapshot
fields: the canonical 222,998-byte artifact matched the retained F1 result
byte-for-byte. The reference result and F2 were not recomputed for this check.

A separate installed SDK call for a 25-node/56-edge network used a saved
nonuniform temperature field from 300 to 300.327082162793 K, a 300 K stress-free
reference and an expansion coefficient of 1e-6/K. All 36 native fields matched
the retained frame-10 mechanical snapshot byte-for-byte in a 63,809-byte
canonical artifact. The case exercises stable-ID temperature transfer and the
complete mechanical result with nonzero thermal strain in all 56 members.

Metadata discovery and topology checks remain separate from these mechanical
execution checks. Neither case solves a new thermal field, feeds deformation
back into electrical or thermal calculations, or establishes stability.
