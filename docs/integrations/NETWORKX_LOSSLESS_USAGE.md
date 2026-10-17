# Native NetworkX interchange

`StructureGraph` remains FiberNet's source graph. The additive methods
`g.to_networkx_lossless()` and `StructureGraph.from_networkx_lossless(G)` exchange
complete native records with an undirected NetworkX MultiGraph. Existing
`to_networkx()`/`from_networkx()` are unchanged analysis projections.

## Example: topology algorithms and supported edits

```python
import networkx as nx
from fibernet import StructureGraph

g = StructureGraph(dimension=3, tolerance=1e-5)
a = g.add_node([0, 0, 0])
b = g.add_node([1, 0, 2])
g.add_edge(a, b)
g.add_edge(b, a, allow_parallel=True)

G = g.to_networkx_lossless()        # MultiGraph even without parallel edges
assert nx.is_connected(G)
assert nx.shortest_path(G, a, b) == [a, b]  # unweighted topology query
G.nodes[a]['_fibernet']['metadata']['score'] = 0.75
G.nodes[a]['_fibernet']['position'] = [0.1, 0.2, 0.3]
G.edges[a, b, 0]['_fibernet']['radius'] = 0.2
restored = StructureGraph.from_networkx_lossless(G)
assert restored.nodes[a].metadata['score'] == 0.75
assert restored.edge_array().tolist() == [[a, b], [b, a]]
```

The native import has no dimension/tolerance overrides: both come from its
record. This avoids silently ignoring an explicit override. For arbitrary
NetworkX graphs and the old import defaults, use the existing projection API.

## Native v1 schema

Every graph/node/edge attribute mapping has exactly one key, `_fibernet`.
Unknown or missing attributes/record fields raise. There are no flat `pos`,
`radius`, or material aliases with competing authority.

- Graph record: `format='fibernet.structure_graph.networkx'`, `version=1`,
  `structure_graph_version='3.1'`, `dimension`, `tolerance`, `box_size`,
  `next_node_id`, `next_edge_id`, `node_order`, `edge_order`, `metadata`
- Node record: `id`, `position`, `boundary`, `metadata`
- Edge record: `id`, `node_i`, `node_j`, `radius`, `segments`, `material`,
  `internal_points`, `metadata`

Node labels equal stable node IDs. Edge keys equal stable edge IDs, unique
across the entire graph. Record IDs must agree with those keys. The explicit
order lists preserve source dictionary insertion order; NetworkX edge iteration
order is not used as source row order. Ordered endpoints preserve each member's
orientation independently of NetworkX's undirected endpoint enumeration.

All three coordinates are preserved, including for graphs labeled dimension 2.
Units, periodicity and other physical declarations are retained only when the
source provides them in existing fields/metadata; this bridge adds no physics,
unit conversion, curve discretization or material-law inference.

Material data uses the existing safe schema-3.1 fields, with no pickle or Python
class loading. Only the native Material class is supported. A manually mutated
null shear modulus is rejected because Material construction would recompute it.
Material serialization's existing optional-field/default semantics still apply.

Metadata must be JSON-compatible dictionaries: string keys and finite JSON
values, nested lists/dictionaries, at most 100 nesting levels. Cycles, tuples,
NumPy arrays/scalars, arbitrary objects and nonfinite metadata numbers are
rejected rather than dropped or stringified. Typed numeric structural fields
must be finite; IDs/counters/segments are nonnegative integers excluding bool.
Tolerance must be positive, including for empty graphs. Numeric geometry and
material values must be exactly representable as float64; oversized integers
or wider-precision values that would be rounded are rejected. The guarantee is
schema-3.1 numerical records and float64 geometry, not arbitrary mutated NumPy
dtype/precision preservation. Metadata integers are retained as integers.
Positions/box are three-coordinate lists, boundary is six bools, and curves are
null or ordered `(M,3)` data. Empty native curves return shape `(0,3)`.
Source malformed array shapes are rejected. Physical constitutive validity is
not inferred by these serialization checks.
Masked arrays/scalars, including arrays with all-false masks, are rejected
before source serialization can erase their masks. This includes positions,
box coordinates, curves and material tensors.

## Editing boundary and strict rejection

Supported entity edits change current namespaced positions, boundaries, radii,
segments, materials, curves, or metadata. They are read from current records;
there is no hidden whole-graph snapshot that can override those edits. Mutable
records, arrays and material values are independently owned across export and
import.

Store algorithm results under metadata, for example:

```python
for nid, score in nx.degree_centrality(G).items():
    G.nodes[nid]['_fibernet']['metadata']['score'] = score
```

`nx.set_node_attributes(G, scores, 'score')` adds an unsupported flat attribute
and consequently makes strict import fail. Move that result into metadata and
remove the flat attribute before importing. A lossless label does not mean any
NetworkX transformation is importable.

Ordinary node/edge addition, deletion, relabeling, reconnection and proper
subgraph extraction leave native identity/order/topology declarations
inconsistent and are rejected. Counter values must exceed all retained IDs;
the original counters preserve non-reuse of previously deleted IDs. Directed
graphs, simple Graph inputs, loops, duplicate IDs, stale coverage, inconsistent
endpoints and unknown schema versions fail clearly. Missing entities are never
restored from a stale snapshot. An all-entity view with the complete unchanged
records is still the same logical graph.

These checks validate a self-contained record, not edit history. A caller who
deliberately rewrites every declaration into another valid native envelope has
constructed another valid record; v1 does not authenticate its prior history.

Native graph JSON remains the existing persistence format. Its canonical
`to_dict()` order is sorted by IDs; native NetworkX additionally records and
restores live insertion order. NumPy outputs and legacy FiberNetwork/PyG
feature projections are separate computational views, not interchangeable
lossless archives.
