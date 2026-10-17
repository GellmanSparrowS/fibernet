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
an explicit numeric zero remains zero. This provenance snapshot itself adds no
per-property numeric feature matrix, normalization or conversion. The graph's existing unit declarations and
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

## Explicit stored material scalar features

`material_features_from_batch` in `fibernet.ml.graph_data` derives a frozen,
non-Mapping `MaterialFeatureView` from an existing v2 snapshot. Both names also
have lazy `fibernet.ml` exports. The view references that same snapshot; it is
not another graph owner, topology copy, persistence format or FiberGNN input.
The default `graph_batch` v1 behavior and `graphs_from_batch` are unchanged.

```python
from fibernet.ml.graph_data import material_features_from_batch

# Caller assertion: these stored property values use the units declared below.
features = material_features_from_batch(
    view,
    fields=('youngs_modulus', 'shear_modulus', 'density', 'poissons_ratio'),
    units={'youngs_modulus': 'Pa', 'shear_modulus': 'Pa',
           'density': 'kg/m^3', 'poissons_ratio': '1'},
    scales={'youngs_modulus': 1e9, 'shear_modulus': 1e9,
            'density': 1000., 'poissons_ratio': 1.},
    material_assumption='isotropic_scalar',
    origin_policy='retain_stored_origin_unknown',
)
physical_values = features.values       # readonly float64 (physical edges, fields)
physical_available = features.available # readonly bool, same shape
assert features.snapshot is view
```

Select a nonempty, duplicate-free ordered subset of exactly those four fields.
Each units/scales dict must contain exactly the selected keys. Only the shown
native base-unit declarations are supported; no units are converted or inferred
from geometry length_unit or a material name. A single declaration cannot prove
that historical records used consistent units. Per-edge/per-graph mixed-unit
declarations are unsupported; normalize and verify the source records explicitly
before snapshotting. Geometry with unknown length units stays unknown, so this
material-only table is not a unit-resolved physical model input.

Each finite, strictly positive scale is a fixed divisor in the corresponding
property unit. No fitting, centering, clipping, imputation or G derivation occurs.
Any externally fitted scale must be fitted only on training sources and then
frozen; the function cannot certify that external procedure. Explicit stored G
remains independent of E/nu. The isotropic assumption is the caller's assertion,
not a check of physical isotropy, stability, ranges or E/G consistency. Finite
negative values and genuine zeros can be represented; a future constitutive
consumer must define its own physically valid domain.

`schema_version` is `fibernet.material_feature_view.v1`. `feature_names`,
`property_units` and `scales` preserve the exact selected column order. Physical
rows follow the existing edge_ptr/edge_ids ordering. Parallel/reversed members
stay distinct; local IDs may repeat across graphs. A stable edge identity is the
graph ID plus local edge ID. `snapshot.message_parent_rows` maps both directed
messages to their physical edge's values/masks/status; no duplicate message
table is stored and no members are merged.
An empty canonical graph collection, or graphs without edges, yields `(0, F)`
value/mask arrays. This data-only conversion does not impose a model's pooling
requirement for nonempty graphs.

`property_status` is an immutable tuple of row tuples with four possible states:

- `material_missing`: no Material record; unavailable with +0 placeholder
- `property_missing`: the record lacks this key; unavailable with +0 placeholder
- `property_null`: an allowed stored null; unavailable with +0 placeholder
- `stored_origin_unknown`: an available stored value, including genuine signed zero

The available mask and the cellwise null-status mask have different meanings.
Null E/density/nu is malformed; null G is allowed without recomputation. Missing
keys never construct Material defaults. A constructor default or derived G
remains `stored_origin_unknown`, even if its value equals a known default/formula.
Serialization may already have omitted optional nulls or rounded/coerced numbers;
the view cannot recover or certify information lost before the snapshot.

Conversion/division uses ordinary binary64 rounding. Packed-bit checks reject
new nonfinite results, loss of a nonzero value to either signed zero, and sign
loss; genuine signed zero is retained. Scales must remain positive and nonzero.
No floating-point mode is changed. This is not exact rational normalization or
a general guarantee of FTZ/DAZ arithmetic accuracy, and no float32/Torch device
conversion is provided.

Non-null stiffness tensors, nonempty custom extra, unknown keys and unsupported
types fail closed. A finite 6x6 tensor is not a validated anisotropic law. Known
unselected native strengths/thermal/electromagnetic fields remain provenance
only, with serialization-form checks. Names are identifiers, not feature values
or a source of guessed properties. Geometric boundary-face flags, meshing
segments, curves, coordinates, IDs and metadata remain in the snapshot and are
not appended to this numeric table. Mechanical conditions still require explicit
caller records with units, frames, DOFs and stable IDs.

The accepted declaration types are exact built-in list/tuple for fields, dict
for units/scales, str names and int/float scalars (not bool, NumPy scalar wrappers
or custom subclasses). Canonical snapshot records already contain built-in
serialized values. Relevant identity arrays must be exact base ndarrays with
canonical dtype/shape and read-only flags; custom array/coercion hooks are not
accepted. Exact GraphBatch/GraphBatchProvenance types and schema tags are required.

Use a factory-produced or equivalently trusted canonical snapshot. Direct
construction receives schema/alignment/value checks, not factory-origin
authentication or hostile-object isolation. MappingProxyType alone cannot prove
that its backing mapping is safe; a deliberately forged custom-backed proxy is
outside this trust boundary. Frozen/read-only data prevents ordinary accidental
mutation, not intentional re-enabling of array writes. There is no new serializer.

Current FiberGNN does not consume these material values or masks. Passing this
typed view to it is unsupported; `graphs_from_batch` still rejects v2. A later
model bridge requires explicit profile/schema guards and tested mask-aware edge
feature consumption, rather than adding dictionary keys that a model ignores.
Stored material features also do not make current quick FEM/Taichi/reduced or
transport backends consume per-edge Material automatically; see
[material consumption](material-consumption-contract.md). No physical labels,
predictions, training or performance evidence are produced by this conversion.

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
