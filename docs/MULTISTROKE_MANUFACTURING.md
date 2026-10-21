# Edge-once manufacturing trails

`manufacturing_trails_from_graph` closes the routing gap for connected networks
with odd-degree vertices. It produces the minimum number of edge-once strokes
without adding any physical fiber, duplicating any physical traversal, or editing
the authoritative `StructureGraph`. A single segment or open chain is one open
stroke. A spatial Y or tetrahedral K4 is two strokes. A six-arm star is three.
An all-even connected graph remains one closed stroke.

This is route correspondence only. It does not certify printability, stiffness,
extrusion, collision avoidance, supported deposition, nonzero fiber length or
machine safety. It neither creates solids/endcaps nor plans travel between strokes
or generates G-code. A trail can end at an existing branch junction. Treat stroke
breaks as explicit discontinuities; never connect consecutive strokes in a plot
or downstream process merely because their rows are adjacent.

## Complete 3D example

With FiberNet's declared dependencies installed, run from the checkout:

```sh
python -m examples.multistroke_manufacturing NEW_OUTPUT_DIRECTORY
```

The parent directory must exist and the output directory must be new. The example
saves an authoritative graph, a CSV and a JSON mapping. Its three fibers connect
(0,0,0) metres to (0.01,0,0), (0,0.01,0.01), and (0,-0.01,0.01). The four odd
vertices give two open strokes and exactly five CSV data rows. The CSV is in
millimetres; source coordinates and 0.0001 m radii remain intact in the snapshot.
No physical solver or optional manufacturing/meshing dependency is used.

The equivalent API pattern is:

```python
from fibernet import manufacturing_trails_from_graph, export_mapped_trails

trails, snapshot = manufacturing_trails_from_graph(
    graph, graph_id="my-graph", length_unit="m", coordinate_frame="lab_xyz"
)
mapping = export_mapped_trails(
    "fresh-trails.csv", trails, snapshot, sidecar_path="fresh-trails.json"
)
for stroke in range(trails.stroke_count):
    a, b = trails.stroke_node_ptr[stroke:stroke + 2]
    source_node_ids = snapshot.node_ids[trails.route_nodes[a:b]]
    # Plot/process only this stroke before starting a new path.
```

Use `StructureGraph.add_edge(..., allow_parallel=True)` to retain distinct
physical fibers with the same endpoints. Keep the source graph or its saved file.
The returned `GraphBatch` is a detached numerical/provenance view, not a complete
graph persistence format and not a second editable graph owner.

## API and immutable result

Public names are available from `fibernet`, `fibernet.gen`, or
`fibernet.gen.manufacturing_trails`:

- `ManufacturingTrails`: factory-only frozen result; direct construction is refused
- `manufacturing_trails_from_graph(graph, *, graph_id, length_unit, coordinate_frame)`
  returns `(trails, snapshot)`
- `export_mapped_trails(csv_path, trails, snapshot, *, sidecar_path,
  source_input_sha256=None)` returns a fresh JSON-native copy of the sidecar

For E physical edges and S strokes, result fields are:

| Field | Meaning |
| --- | --- |
| `schema_version` | `fibernet.manufacturing_trails.v1` |
| `algorithm_version` | `sorted_odd_matching_euler_split.v1` |
| `route_nodes` | E+S dense node rows, flattened by stroke |
| `route_edges` | E dense physical-edge rows, flattened by stroke |
| `stroke_node_ptr` | S+1 offsets into `route_nodes` |
| `stroke_edge_ptr` | S+1 offsets into `route_edges` |
| `closed` | S Boolean endpoint-equality flags |
| `stroke_count` | S, derived from `closed` |
| `normalized_source_sha256` | Bound GraphBatch payload and typed array bytes |
| `route_binding_sha256` | Bound route arrays, schema/algorithm and source hash |

Pointers use half-open slices. Each stroke contains at least one physical edge
and one more node than edge. Node/edge index and pointer arrays are int64; flags
are Boolean. Arrays have detached immutable bytes backing, and frozen attributes
reject ordinary reassignment. Export validates the hashes and structural
invariants again. These checks are accidental-mutation safeguards, not a security
boundary against code that deliberately rewrites private state or hashes.

The snapshot preserves sorted stable node IDs, insertion-ordered physical edge
IDs, ordered source incidence, coordinates, per-edge radii with availability,
stored materials and their recorded origin, meshing segments, boundary-face
flags and metadata. Its forward/reverse ML messages do not represent additional
physical fibers. Material records are provenance, not effective solver material;
no constitutive or material-unit conversion is performed. Later changes to the
original graph do not change the detached pair; adapt again to export new geometry.

## Scope and input validation

Only an ordinary `StructureGraph`, one connected component including every node,
at least one node and physical edge, and the existing physical graph budgets
(180,000 nodes; 240,000 edges) are accepted. Empty/disconnected graphs and isolates
are refused. Disconnected routing is outside this version.

Explicit source `length_unit` is exactly `"m"` or `"mm"`; `coordinate_frame` and
`graph_id` must be nonempty strings. Nested geometry-unit/frame metadata must
agree. Source coordinate/radius storage must satisfy the existing bridge's
binary64 representability checks. Unsupported floating storage, nonfinite values
and inexactly representable integers are refused. Every nonempty internal-point
array is refused, including collinear points; missing and explicitly empty records
remain distinguishable. Curved-edge sampling requires a separate future schema.

Metres are converted to millimetres only in CSV coordinates, using binary64
multiplication by 1000; millimetres use factor 1. No translation, rotation, fitting
or `PrintSettings` is applied. The reused guards reject nonfinite conversion,
loss of a nonzero coordinate and collapse of distinct physical-edge endpoints.
Conversion is not guaranteed bit-reversible. Radii, metadata and source geometry
retain their original values and units in the sidecar snapshot.

This adapter does not introduce a stricter physical zero-length policy. The
existing radius-only snapshot/closed-route path can represent coincident endpoint
coordinates and manually stored self-loop incidence, although `add_edge` normally
refuses creating self-loops. Such topology is not physical qualification. A
self-loop traversal reports source direction +1 by convention.

## Algorithm and guarantee

Count incidence with multiplicity, so three parallel edges give degree three.
Let O be the number of odd-degree vertices. At least O/2 open trails are necessary,
since each odd vertex must be an endpoint. Pair sorted odd node rows in private
scratch storage, append temporary edge rows starting at E, and run the existing
edge-row-aware Euler routine. Rotate to begin just after a temporary edge, split
at every temporary edge, and remove all temporary edges. Each odd node has only
one temporary incident edge, so no empty stroke is created. This achieves
`max(1, O/2)` strokes and every odd vertex appears exactly once as a trail endpoint.
All-even graphs use the unchanged Euler itinerary directly.

Every physical row is emitted once, and no temporary edge appears in route
arrays, CSV, mapping or the source snapshot. No shortest-path augmentation or
`networkx.eulerize` is used. Determinism is for the same ordered snapshot and
algorithm version; arbitrary edge insertion reorderings need not share one
canonical route. Production routing uses the existing FiberNet routine and does
not require a newer NetworkX API. Existing Python >=3.9 and NetworkX >=3.0
declarations remain unchanged. NetworkX is an independent test oracle.

## New CSV and JSON schemas

CSV header:

```text
stroke,step,node_row,node_id,incoming_edge_row,incoming_edge_id,source_edge_direction,x_mm,y_mm,z_mm
```

`stroke` is zero-based; `step` resets to zero inside each stroke. `node_row` and
`incoming_edge_row` are dense snapshot rows; `node_id` and `incoming_edge_id` are
stable source IDs. At each stroke's initial row, incoming edge row/ID are -1 and
direction is 0. All corresponding JSON values are null. Subsequent direction is
+1 from stored `node_i` to `node_j`, or -1 in reverse. Stable IDs are nonnegative,
so the -1 sentinel cannot name a physical edge. Each stroke has E_s+1 rows; the
CSV has E+S data rows. There is no connection or travel segment across strokes.

The JSON schema is `fibernet.manufacturing_trails_mapping.v1`. It includes algorithm
identity, count/parity, closure flags and separate node/edge boundaries, every
route row's stable mapping, the complete detached GraphBatch v2 payload and typed
storage hashes, source/unit/frame/material provenance, bound route/source hashes,
and the actual CSV SHA256. `source_input_sha256`, when supplied, must be a lowercase
64-character hex digest. The adapter records it as caller asserted, not independently
file-verified; the example calculates it from the graph file it just saved.

Consumers should verify `route_csv_sha256` against the CSV before using the pair.
Both destinations must be distinct fresh paths with existing parent directories;
existing files and symlinks are refused. Full validation/serialization occurs
before exclusive reservations. The CSV is atomically published and byte-checked,
then JSON is atomically published last. This is not an atomic two-file transaction:
a valid sidecar plus matching CSV hash is the completion record. On failure, owned
empty reservations or a complete CSV can remain. The exporter neither removes final
paths nor automatically retries. Concurrent destination/parent replacement and
concurrent source mutation are unsupported; inspect incomplete outputs before
choosing new destinations.

## Compatibility and checks

`ManufacturingNetwork`, `manufacturing_from_graph`, `export_route`, and
`export_mapped_route` retain their existing connected/all-even closed-route
contracts, legacy CSV column meanings and bytes. New trails are not substituted
into `ManufacturingNetwork` or accepted as solid-generation input. The new schema
has explicitly different column semantics and must be handled separately.

Focused tests use real NumPy, authoritative StructureGraph/GraphBatch, and an
independent keyed NetworkX multigraph oracle. They cover open segment/3D chain,
3D Y, K4, six-arm star, cycle+branch, sparse-ID mixed-direction parallel fibers,
even ring/multigraph exact legacy CSV parity, units and provenance, graph/snapshot
nonmutation and determinism, immutable arrays, invalid data, and failure/race
publication behavior. The tests use self-contained fixtures and the declared dependencies:

```sh
python -m unittest discover -s tests -p test_manufacturing_trails.py
python -m unittest discover -s tests -p test_manufacturing_mapping.py
```

These are small routing/export checks, not a whole-toolkit rerun, large-network
performance benchmark, physical simulation or fabrication qualification.
