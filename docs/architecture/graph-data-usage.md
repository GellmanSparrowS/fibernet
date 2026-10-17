# Detached graph data for learning adapters

`fibernet.ml.graph_data.graph_batch` produces a versioned NumPy `GraphBatch`
snapshot from existing `StructureGraph` objects. It does not edit graphs, train
models or require Torch/PyG. The lazy exports `fibernet.ml.graph_batch` and
`fibernet.ml.GraphBatch` expose the same API. Existing `ml.gnn.graph_from_structure`
and every existing model/feature/artifact schema are unchanged; those models do
not automatically consume this new view.

```python
from fibernet.ml import graph_batch

view = graph_batch(
    [graph_a, graph_b],
    graph_ids=['specimen-a', 'specimen-b'],
    length_unit='mm',
    coordinate_frame='specimen-local',
    edge_features=['radius', 'chord_length', 'polyline_length'],
)
```

## Identity and axes

Schema: `fibernet.graph_batch.v1`. Graph IDs are caller-supplied, nonempty and
unique within the batch. They are names, not verified source provenance or
independence declarations. Node and physical-edge IDs remain local to each graph;
the same ID may occur in different graphs. IDs must be nonnegative int64-range
integers, not booleans; dictionary keys must agree with record IDs.

- `node_ptr` and `edge_ptr` delimit each graph's rows, including empty graphs
- `node_ids` uses sorted IDs within each graph; `positions` is always `(N,3)` XYZ
- `edge_ids` uses physical-edge insertion order; `edge_index` is `(2,E)` with
  global dense node rows preserving `SEdge.node_i -> node_j`
- `message_index` is `(2,2E)`: forward then reverse for each physical edge
- `message_parent_rows` identifies the global physical-edge row;
  `message_parent_ids` repeats its original local ID
- `message_orientation` is +1/-1 relative to stored SEdge direction, not sorted
  node IDs. Parallel members, including reversed members, stay separate
- `internal_ptr` delimits each physical edge's rows in `(P,3)` `internal_points`;
  `internal_available` distinguishes None from an explicitly empty polyline

There are no added self-loops, deduplication, node merging or adjacency
normalization. A physical self-edge already in the source also produces two
messages; this is a representation rule, not physical-validity approval.

## Features, units and availability

Only requested edge features are assembled, in caller order. The default is no
features: `(E,0)` values and availability arrays. Unknown or duplicate names fail.

- `radius` copies the existing edge value. None/missing uses numerical zero with
  `edge_feature_available=False`; a present zero remains available. The source
  graph may have assigned defaults; this adapter cannot certify their provenance
- `chord_length` is the full XYZ endpoint distance
- `polyline_length` sums XYZ segments from node_i through stored internal points
  to node_j. Without internal points it equals chord length. This is discretized
  path length, not an analytic smooth-curve arc length

Distances use a scaled Euclidean calculation rather than squaring unscaled
coordinates; representable tiny/subnormal and large distances are retained.
Polyline lengths use accurate floating-point summation. A distance or total
length outside finite float64 range raises ValueError, never an available zero
or infinity. Nonzero source coordinates or selected scalars that would convert
to zero in float64 also reject rather than becoming available zeros. Ordinary
float64 rounding remains; no epsilon cutoff, rescaling of
source geometry or physical minimum length is introduced.

Lengths and radius are interpreted in the caller-declared `length_unit`.
`coordinate_frame` describes the shared frame convention. Both declarations are
required nonempty strings; they are not validated against a units registry and no
conversion takes place. Each graph's declared dimension is retained, but XYZ is
never silently projected. Callers must establish consistent units/frame across
the batch; split incompatible inputs into separate batches or explicitly convert
upstream. This is representation validation, not dimensional/constitutive or
planarity certification.

Node/internal coordinates must be finite real numeric arrays of the stated
shape. MaskedArray coordinates and internal points are rejected before conversion,
including arrays whose masks are entirely false; unavailable coordinates cannot
be silently interpreted as valid geometry. Selected present features must be
finite real scalars and remain finite after float64 conversion. Unselected radius
values are not inspected. Metadata must be JSON-native with finite numbers and
string keys; cycles/depth >100 and unsupported objects fail. Metadata is captured
for every graph, sorted node and insertion-ordered physical edge. Existing
`unit_id`/`material_id` values in metadata survive as source assertions. No unit
cell IDs, material laws, boundary conditions, targets or availability assertions
for those absent schemas are invented. Material records themselves are outside
this first version; metadata names do not substitute for constitutive data.

## Snapshot ownership

All arrays are detached copies marked non-writeable; dataclass fields are frozen.
Metadata dictionaries recursively become read-only mappings and lists become
tuples. Thus metadata container types are intentionally normalized, not a
byte-for-byte or original-type-preserving graph serialization. Mutating the
source graph later does not change this view, or vice versa.

This prevents ordinary accidental mutation, not hostile mutation: NumPy owners
can deliberately re-enable write flags. Direct `GraphBatch(...)` dataclass
construction bypasses builder validation; use `graph_batch` at the boundary.
Read-only mapping proxies are not a persistence format. Persist the authoritative
graph and explicit declarations; this view has no checkpoint serializer.
Concurrent source mutation during snapshot creation is outside the contract.

## Opt-in stored provenance (v2)

The default remains v1. To retain additional authoritative source records:

```python
view = graph_batch(
    [graph_a], graph_ids=['specimen-a'], length_unit='mm',
    coordinate_frame='specimen-local', include_provenance=True,
)
stored = view.provenance
material = stored.edge_material_records[0]  # immutable mapping, or None
```

`include_provenance` must be an actual bool. True selects
`fibernet.graph_batch.v2`; the attached frozen `GraphBatchProvenance` record uses
`fibernet.graph_batch.provenance.v1`. The same GraphBatch, IDs, row ordering,
geometry, messages and metadata are retained. This is a detached view extension,
not another graph owner. The current `graphs_from_batch` GNN adapter accepts only
v1 and explicitly rejects v2; no model automatically consumes this provenance.
`GraphBatchProvenance` is also available through the lazy `fibernet.ml` export.

The payload aligns with the enclosing view's node/physical-edge rows and ptrs:

- `node_boundary_face_flags`: readonly bool `(N,6)`, in the exact order
  x_min, x_max, y_min, y_max, z_min, z_max
- `node_boundary_face_available`: bool `(N,)`; absent/None uses six false
  placeholders with availability=False. A present all-false record is available
- `edge_material_records`: tuple of immutable authoritative Material JSON
  records, or None when missing. `edge_material_available` is bool `(E,)`
- `edge_segments`: readonly int64 `(E,)` meshing counts, with a separate
  `edge_segments_available` mask. Missing/None uses zero with availability=False;
  present values must be positive nonbool int64-range integers
- `node_boundary_face_origin`, `edge_material_origin`, `edge_segments_origin`:
  tuples containing `stored_origin_unknown` or `missing`

Boundary-face flags describe geometric cell membership used for periodic
welding. They are **not mechanical boundary conditions**, displacement/rotation
constraints, forces or loads. Those require explicit separate caller records,
DOF/frame definitions and units; none are inferred here. Segments are FEM
discretization provenance, not a constitutive or curve-shape feature.

Material records use the existing StructureGraph schema 3.1 serialization
validation and are detached/frozen recursively, including stiffness tensors and
extra metadata. A material name or metadata material_id is not a guessed source
of constitutive parameters. This initial profile accepts the native Material
class, not custom subclasses/serializers. Unsupported objects, nonfinite or
malformed records and cyclic extra data fail; no missing material is replaced
with a generic Material. Serialization validation does not certify a supported
physical law, tensor symmetry/positivity, units or a model's consumption.

The records describe values already stored in the graph. A constructor-assigned
default or derived shear modulus may be indistinguishable from an explicit input;
`stored_origin_unknown` makes no claim about that history. Optional properties
omitted by Material.to_dict remain absent; a retained null is unavailable, while
an explicit numeric zero remains zero. No per-property numeric feature matrix,
normalization or conversion is added. The graph's existing unit declarations and
original internal-point geometry are unchanged.

With `include_provenance=False` or the default, the three optional source fields
are not inspected or validated, preserving v1 behavior even for records not
supported by the opt-in profile. The GraphBatch dataclass has one new trailing
optional field, `provenance=None`, so existing positional construction still
works. Dataclass field introspection gains that field. Neither dataclasses.asdict
nor pickle was or is a GraphBatch persistence contract: persist StructureGraph
and explicit declarations instead. StructureGraph JSON serialization is unchanged.

All new arrays are detached and marked non-writeable, with the same ordinary
write-protection limits as v1. Direct dataclass construction still bypasses
validation. Tests cover heterogeneous records with identical material names,
parallel/reversed edges, repeated local IDs across graphs, empty/missing records,
source mutation, default behavior and strict v2 adapter refusal without Torch.

## Holdout boundary and validation

This API does not split datasets and does not infer independent source groups
from graph IDs, seeds or hashes. Continue the existing explicit native Dataset
`split='group', groups=...` workflow. This data view does not change existing
splitting code.

`tests/test_graph_data.py` uses pure-data fixtures for 2D/3D (including equal XY,
different Z), sparse/reordered IDs, reversed parallel edges, internal paths,
empty/multiple graphs, missing feature masks, invalid input, detached arrays and
metadata, and optional-backend import isolation. These checks establish the data
contract only; they are not physics validation or model-performance evidence.
