# Import sampled fibers with recorded welds

Use `convert_sampled_welded_fibers` to build a native `StructureGraph` from
sampled legacy `FiberNetwork` geometry. Each pair of consecutive centerline
samples becomes a straight edge. This preserves the sampled path, including
bends and nonuniform sample spacing. Use the separate
[straight-fiber importer](MULTI_WELDED_JUNCTION_IMPORT.md) for exact interior
welds on its supported straight fibers.

## Convert a generated ZigZag network

```python
from fibernet.core.sampled_weld_import import convert_sampled_welded_fibers
from fibernet.gen import ZigZagGenerator

network = ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network()
assert (network.num_fibers, network.num_crosslinks) == (45, 90)
graph, report = convert_sampled_welded_fibers(network)
assert (graph.num_nodes, graph.num_edges) == (891, 900)
assert len(report["crosslink_junction_node_ids"]) == 90
graph.save_json("zigzag_structure.json")
```

## Accepted geometry and weld records

- Inputs are nonperiodic, two- or three-dimensional networks of circular
  fibers without section parameters, with positive finite radii and ordinary
  `Material` records. Each fiber needs at least two distinct consecutive
  samples in an XYZ array. Coordinates must be finite, exactly representable
  in float64 and within the graph's coordinate range. For 2D networks, z
  coordinates must be zero.
- Each welded crosslink needs positive infinite strength and stiffness. It
  references two distinct **fiber list indices** as Python integers. For a
  centerline of length `n`, the attachment parameter must equal the canonical
  float64 value `k / (n - 1)` for an existing sample index `k`. This includes
  endpoints. The two identified samples must exactly match the recorded
  position. Welds between samples are unsupported.
- Only explicit records join sample occurrences. The same record, including
  a reversal, has no additional effect on graph topology. Each report row
  still identifies its junction. Unwelded crossings and coincident samples
  remain disconnected from one another. Parallel edges retain their own IDs.
- The limits are 1,024 fibers, 4,096 crosslinks, 10,000 samples per fiber and
  100,000 samples overall. Memory use also depends on copied metadata and
  material contents.

## Geometry, identity and metadata

For each source list index, `report["fibers"][index]` gives the original
`source_fiber_id`, ordered `sample_node_ids` and ordered `segment_edge_ids`.
Stored fiber IDs are labels and may repeat. The graph allocates new node and
edge IDs. `report["crosslink_junction_node_ids"]` follows the source crosslink
list order.

Radius, the optional box and independent material records are retained. A
two-coordinate box is normalized to `[x, y, 0]`. Each edge uses the native
default of four segments. The source fiber's discretization and weld strength
or stiffness are not carried as graph properties.

`report["source_metadata"]` holds a detached copy of the legacy metadata. The
graph records the import format. Source metadata such as `weld_triplets` is
not applied as controls for the new IDs. Prepare controls using the new graph.
Native graph JSON has the usual material restrictions, and the report can
contain arbitrary copied Python metadata. Geometry is carried in the supplied
units without conversion. The import describes geometry and topology and
does not establish mechanical equivalence.
