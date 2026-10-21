# Stable graph identities in manufacturing routes

Use the explicit adapter when a `StructureGraph` has sparse/reordered IDs or
when a route must carry source material, radius, units and metadata. The existing
`ManufacturingNetwork` still owns dense numerical rows; no second graph model is
introduced. The two new package exports are:

```python
network, snapshot = fibernet.manufacturing_from_graph(
    graph, graph_id="specimen-a", length_unit="m", coordinate_frame="specimen")
mapping = fibernet.export_mapped_route(
    "route.csv", network, snapshot, sidecar_path="route.mapping.json",
    source_input_sha256=trusted_saved_input_sha256)
```

The input SHA is optional. If supplied, it is retained as a caller assertion;
this function does not open or independently verify the source input file.
`graph_id` is also caller asserted. `normalized_source_sha256` verifies the
factory's detached GraphBatch v2 payload and its typed numerical array bytes.
It is a different identity from a saved input-file hash, a graph's rounded
fingerprint, or a claim about physical evidence.

## Geometry and ownership

`manufacturing_from_graph` uses the existing `graph_batch` contract:

- Node rows follow sorted stable node IDs; physical edges keep insertion order,
  direction and their own stable IDs, including every parallel edge
- Geometry is unchanged. Coordinates and radii must be exactly representable
  in the supported binary64 snapshot profile; unsupported narrowing is refused
  Floating source storage must already be binary64; float16/float32/extended
  floating storage is conservatively refused, including exact-looking values
  Integer source values are accepted only when exactly representable in binary64
- Radius availability, original stored material records/origins, meshing
  segments, geometric boundary flags and detached graph/node/edge metadata are
  retained in the GraphBatch v2 snapshot
- Nonempty internal-point geometry is refused rather than silently straightened
  Missing versus explicitly empty internal-point records remain distinguishable
- The existing manufacturing constructor requires connected, all-even topology
  The adapter never reconnects components, duplicates edges, splits junctions,
  repairs topology or applies `PrintSettings`

The returned network and snapshot do not alias caller geometry or metadata.
Their numerical arrays are read-only. The factory binds the complete snapshot
and network/reference/edge/route arrays with private digests. Export refuses
unbound legacy networks, a different same-geometry snapshot, or changed arrays.
A read-only flag and digest check protect ordinary ownership mistakes, not
adversarial mutation by arbitrary code in the same Python process.

This is a lossless handoff for the supported GraphBatch geometry, stable IDs and
stored provenance. It is not a complete `StructureGraph` persistence format:
merging tolerance, bounding-box conventions, ID-allocation counters and private
runtime caches are not reconstituted by this route API. Keep the authoritative
source graph/input when those are needed.

## Explicit units and stored material

Source `length_unit` must be exactly `"m"` or `"mm"`. Unknown or unsupported
units refuse. Explicit length/coordinate/position/radius unit declarations in
graph, node or edge metadata, including nested provenance, must agree with the
argument. Supported declaration keys are `length_unit`, `coordinate_unit`,
`coordinate_units`, `position_unit`, `radius_unit`, and `units` as a length-unit
string or an object with `length`, `coordinates`, `position` or `radius` fields.
Existing `coordinate_frame` declarations must also agree. Other named physical
units, such as a material modulus in Pa, are not converted or inferred.

Output route centers are always mm: declared m uses a fixed isotropic factor
1000; declared mm uses factor 1. No bounding-box fit, translation, rotation or
print-envelope rescale is performed. Original coordinates and radii remain in
the snapshot in their source units. Converted coordinates use binary64
arithmetic; bit-exact reversibility is not promised. Nonfinite converted values,
loss of a nonzero coordinate to zero, or distinct physical-edge endpoints that
collapse after conversion are refused before output creation. Stored binary64
bit identities detect this loss even when caller floating-point modes flush
subnormal arithmetic; signed zeros agree.

Material records are stored provenance, with their existing declared or unknown
origin. The mapping does not claim that a solver used those values, substitute
effective solver E/nu, infer material units from names, or certify constitutive
validity. Radius is retained in the sidecar; CSV alone does not encode thickness.

## CSV identity and completion contract

The new helper calls the unchanged legacy `export_route`. Its CSV remains:

```text
step,incoming_edge_id,node_id,x_mm,y_mm,z_mm
```

Despite those legacy column names, `node_id` and `incoming_edge_id` in the CSV
are dense row ordinals, not authoritative StructureGraph IDs. The initial
incoming-edge value is -1. The explicit JSON sidecar maps every route step to
both the dense row and stable ID, retains original physical edge direction,
and identifies forward/reverse traversal. All physical multiedges remain
separate rows and stable IDs. Its source snapshot includes the complete
GraphBatch v2 fields and typed-array identities.

The sidecar also records source/output units, conversion, source and route CSV
hashes, metadata, stored material and radius availability, and the scope of the
claims. The returned mapping is a fresh JSON-native copy; changing it cannot
change the network, snapshot or already written sidecar.

Both destination names must be different, absent, and have existing parent
directories. Paths, network/snapshot bindings, conversion and the complete
CSV/mapping serialization are validated before the first write. Both final
names are then exclusively reserved. CSV publication uses the existing atomic
single-file exporter; the sidecar is published last with the same writer.
There is no atomic two-file transaction. A valid sidecar whose CSV SHA256
matches the actual CSV bytes is the completion record.

On failure, a newly owned empty reservation or a completed CSV can remain.
A competing creation during reservation fails closed and is never overwritten.
There is no automatic retry or deletion of final output paths. Only sibling
staging files owned by the existing atomic writer are cleaned up. Preserve and
inspect incomplete outputs before choosing fresh names for any later attempt.
Pre-existing files, symlinks and same-path aliases are refused. Concurrent
replacement of owned destinations or their parent directories after validation
and reservation is unsupported.

Legacy `export_route(path, network, centers)` behavior is unchanged, including
its existing replacement semantics. The stricter fresh-output rule applies
only to `export_mapped_route`.

A closed route and correct provenance do not establish a printable solid,
slicer-ready toolpath, manufacturing feasibility or predictive accuracy. This
API does not train, predict, generate new simulation labels or solve physics.
Routing and export checks do not qualify a numerical scene or fabrication process.

## Connected odd-degree networks

The additive [multi-stroke adapter and export](MULTISTROKE_MANUFACTURING.md) now
support connected odd-degree networks without adding physical fibers. It uses a
separate immutable result and new stroke-aware CSV/JSON schema. The closed-route
APIs and legacy bytes described above remain unchanged.
