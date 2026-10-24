# Checked FiberNetwork projection

The optional `FiberNetworkProfile` adds explicit input admission and a detached
loss report to `StructureGraph.to_fiber_network()`, while preserving existing
no-profile usage. `StructureGraph` remains the authoritative geometry/topology
model; the bridge produces the existing endpoint-chord representation.

```python
from fibernet import FiberNetworkProfile

# Existing return type and behavior are unchanged.
legacy_network = graph.to_fiber_network()

# Explicit unit/frame declarations activate checked admission and a report.
network, report = graph.to_fiber_network(profile=FiberNetworkProfile(
    length_unit="mm", coordinate_frame="lab_xyz"))

# Deliberately permit these two losses; missing materials still reject.
network, report = graph.to_fiber_network(profile=FiberNetworkProfile(
    length_unit="mm", coordinate_frame="lab_xyz",
    geometry_policy="endpoint_chord", short_edge_policy="omit_legacy"))
```

The keyword defaults to `None`. A supplied profile always returns a two-tuple;
without one the method returns only `FiberNetwork`. `FiberNetworkProfile`,
`FiberNetworkProjectionLimits` and `FiberNetworkProjectionError` are exported
from `fibernet`, `fibernet.core`, and their owning core module.

## Acceptance and explicitly selected losses

- `geometry_policy="reject_internal_points"` rejects every nonempty edge-owned
  point array, including collinear, duplicate and backtracking samples.
  `"endpoint_chord"` explicitly permits ignoring them. Missing and empty arrays
  remain distinguishable. Actual intermediate graph nodes remain real nodes.
- `short_edge_policy="reject"` rejects chords below the existing raw-coordinate
  threshold `1e-12`. `"omit_legacy"` explicitly permits omission. Exactly `1e-12`
  is retained, subject to the legacy floating calculation. This threshold is
  not derived from graph merging tolerance or physical units.
- `missing_material_policy="reject"` rejects missing whole material records on
  every source edge, even omitted edges. `"legacy_default"` permits the existing
  fresh `Material()` fallback only for retained edges. Missing omitted edges
  are reported without constructing/copying a material for them.
- Malformed geometry/material/metadata does not become acceptable through a
  loss policy. Every source edge is validated, including omitted candidates.
- Empty, isolated, disconnected and odd-degree graphs remain eligible. No
  route closure, even-degree requirement or topology repair is introduced.

Required unit/frame labels are nonempty exact caller assertions, not recognized
or certified units. They jointly cover stored XYZ node coordinates, internal
points, radius and box dimensions. Declaring units does not rescale coordinates
or transform the coordinate frame; the selected geometry policy controls curve
projection. Geometry declarations are interpreted only at each
graph/node/edge metadata dictionary root, or directly in a dictionary stored
under the exact key `geometry` (which may itself be nested). Within those
scopes the following declarations must agree: `length_unit`, `coordinate_unit`, `coordinate_units`,
`position_unit`, `radius_unit`, `internal_points_unit`, `curve_unit`,
`box_size_unit`, `coordinate_frame`, and `units` as a string or dictionary with
`length`, `coordinates`, `position`, `radius`, `internal_points`, `curve` or
`box_size`. Arbitrary descendants do not inherit geometry scope: for example,
`load.units="N"`, `stress.units="Pa"`, `load.coordinate_frame="force_local"`
and `load.length_unit="m"` do not declare graph geometry. Use
`geometry.units="mm"` or `provenance.geometry.radius_unit="mm"` for explicit
nested declarations. Other named physical units and material names are not
interpreted. Existing manufacturing route-adapter declaration behavior is unchanged.

## Data and numerical scope

Checked mode accepts an ordinary `StructureGraph` with ordinary SNode/SEdge
records, real NumPy coordinate arrays and JSON-native metadata. Node positions
have shape `(3,)`, curves `(M, 3)`, and box dimensions `(3,)` or None. Nonnegative
box dimensions may include zero axes. Full XYZ is preserved even when the
declared graph dimension is 2. Masked arrays, nonfinite values, malformed shapes
and inexact source-to-binary64 normalization are refused. Positive radius and
positive integral segmentation are required.

IDs must be matching nonnegative integral mapping keys/record IDs, excluding
booleans. NumPy integer IDs normalize to Python integers without an int64 cap.
Endpoints must exist and differ; opposite-direction parallel physical edges
are never collapsed. Identity is local to this graph, not global.

The existing norm and segment rule are used unchanged. Checked admission rejects
numeric overflow, coordinate-difference loss and a computed zero distance for
distinct stored endpoints, rather than calling these authorized omissions.
The zero-distance distinctness guard uses stored binary64 bit identities,
treating signed zeros as equal and retaining nonzero subnormal distinctions.
It does not substitute a different norm/segmentation algorithm. Interpolated
centerline arithmetic retains ordinary binary64 precision, not exact real
arithmetic. The caller's NumPy error settings are restored.

Checked materials must be ordinary `Material` instances with only their known
dataclass fields, no custom copy hooks/dynamic fields, finite stored numbers,
an optional finite `(6, 6)` stiffness tensor, and bounded JSON-native extra data.
Optional None fields, including an explicitly mutated `shear_modulus=None`,
remain None: the bridge copies rather than reconstructs or normalizes them.
This validates stored representation, not constitutive admissibility or
effective solver properties. Existing custom-material behavior remains available
through the unchanged no-profile branch.

## Report and ownership

The detached JSON-native report has schema
`fibernet.fiber_network_projection.v1`. Its ordered edge records associate stable
edge IDs and original directed endpoint IDs with dense `fiber_row` values or
null for omitted edges. They include point presence/count, source/output segment
counts, material disposition, stored-origin status, optional missing-material
field names and source metadata field names. Summaries list
all nonempty-curve IDs, curves actually projected into retained fibers, omissions,
missing/defaulted materials and nodes with no retained incident fiber.

Graph metadata is copied to the network. Node/edge metadata field names and the
loss of boundary flags, tolerance and ID-allocation counters are recorded, not
reconstituted. The report contains no full source graph, curve point arrays or
material snapshot. Preserve the native StructureGraph 3.1 source separately.

Materials remain independently detached per fiber, even when source edges share
one Material. Network box/metadata and report containers do not alias source
storage. No graph copy, source mutation or metadata annotation is required.
The report binds only by this call's returned pairing, at conversion time. It
has no integrity hash and does not track later network mutations. Concurrent
source mutation and hostile class monkeypatching are unsupported.

Crosslinks retain the legacy pairwise `welded` representation at shared source
node IDs. Coincident-but-distinct nodes are not welded. `strength=inf` and
`stiffness=inf` remain legacy sentinels, not qualified physical properties.
Boundary flags are not solver constraints. Copying box dimensions does not set
`FiberNetwork.periodic`; periodic-looking source metadata remains metadata only.

## Bounds and errors

`FiberNetworkProjectionLimits` defaults:

- 10,000 nodes; 20,000 physical edges.
- 100,000 source points: nodes plus all stored internal points.
- 1,000,000 output centerline points; 100,000 pairwise crosslinks.
- 4 MiB metadata/material payload; 16 MiB report payload.
- JSON nesting at most 100 levels.

Limits are positive platform-sized integers, excluding booleans. Callers may
explicitly provide a different frozen limits object and own the increased cost.
Complete admission occurs before Fiber/Crosslink construction. Output samples
are counted using the exact legacy segment selection; crosslinks are counted
as the sum of `degree*(degree-1)/2` after omissions.

The metadata/material budget charges each graph/node/edge metadata object and
each source edge's complete known material-field record once per logical
occurrence, including omitted edges and repeated references. It does not charge
the same logical record again merely because validation precedes its output
copy. Retained missing materials explicitly replaced by defaults are charged
once per actual fallback; omitted missing materials incur no material charge.
Thus repeated retained materials cannot evade the bound through aliasing.
Counts conservatively use compact ASCII-escaped JSON bytes; integer digit
estimates may overcount by one digit. The active Python integer-to-string limit
is also enforced without decimal encoding. Tensor records are fixed at 6 by 6.
The independent report budget includes duplicated ID summaries and field names.
Report labels/IDs/records/field lists are incrementally charged before appending
them, followed by a final envelope consistency walk. Cycles, excessive nesting,
huge strings/integers and invalid JSON objects are
rejected before whole-payload serialization or copying. These are admission
bounds, not a measured peak-RSS guarantee.

Malformed inputs raise ValueError/TypeError; numerical arithmetic failures raise
OverflowError; budget excess raises MemoryError. Policy rejection raises
FiberNetworkProjectionError with ordered stable-ID tuples in `.violations`.
No partial network is returned and the source remains unchanged.

## Downstream boundary

This is a bridge report, not a mesh-export report. The current endpoint-cylinder
mesh helper has a separate `1e-10` cutoff; OBJ also rounds vertices to six decimal
places and drops material/identity fields. A retained bridge fiber therefore
need not survive OBJ. Existing OBJ bytes and threshold behavior are unchanged.
Nothing here certifies printability, bonded junctions, curved geometry, solid
mechanics, or a lossless roundtrip through `from_fiber_network`.
