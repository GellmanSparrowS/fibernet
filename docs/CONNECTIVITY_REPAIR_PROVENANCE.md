# Post-tile 3D connectivity repair

`pattern_3d` runs a post-tile repair step for HCP and the TPMS units `gyroid`,
`schwarz_p`, `schwarz_d`, `iwp`, `neovius` and `lidinoid`. The repair can add
physical edges that were not in the tiled design. Inspect those additions on
the returned `StructureGraph`, before treating connectivity as a design property:

```python
from fibernet import pattern_3d, StructureGraph

graph = pattern_3d(unit="hcp", box=(2., 2., 2.), grid=(1, 2, 1))
report = graph.metadata["post_tile_connectivity_repair"]
for edge_id in report["added_edge_ids"]:
    edge = graph.edges[edge_id]
    print(edge_id, edge.node_i, edge.node_j,
          edge.metadata["post_tile_connectivity_repair"])

graph.save_json("hcp.json")
restored = StructureGraph.load_json("hcp.json")
assert restored.to_dict() == graph.to_dict()
```

## What the report means

The graph and added-edge metadata use the reserved key
`post_tile_connectivity_repair`, with metadata record `version=1`. They are plain
JSON data inside native graph schema 3.1, not a separate graph or
serialization format.

The graph report describes the most recent invocation of this repair stage:

- `added_edge_ids` lists newly created physical edge IDs in addition order.
  Look up their endpoints, radius, material and geometry in `graph.edges`.
  The stage preserves pre-existing parallel edges and their metadata.
- `added_node_ids`, `merged_node_ids` and `removed_edge_ids` are empty because
  this stage only adds edges. They do not claim that earlier tiling performed no
  welding, or that other generator stages made no changes.
- `components_before` and `components_after` count topological components at the
  stage boundaries. Unknown counts are `null`, not an inferred zero.
- `status="not_needed"` means there were zero or one components. The
  algorithm returns before its degree-one pass in this case.
- `status="completed"` means the repair traversal finished, not that all
  components were joined. Check `components_after` and the added-edge list.
- `status="skipped_missing_dependency"` names missing SciPy or NetworkX in
  `missing_dependency`. `incomplete` remains if an exception interrupts repair.
  Exceptions propagate; partial additions are not rolled back.

Each newly added edge has its own reason, `distance_at_addition`, and
`distance_limit`. `connect_disconnected_components` labels component bridges;
`connect_degree_one_node` labels the subsequent augmentation of degree-one
candidates. Duplicate addition attempts return an existing edge and are not
reported as a new edge or a second event. A later no-op invocation replaces the
graph-level report but retains earlier edge labels.

## Coordinate units and repair rules

Distances use Euclidean raw graph coordinates. `coordinate_space` is
`graph_coordinates`, `length_unit` is `null`, and `unit_conversion_applied` is
false. A null unit means the generator does not know a physical unit, not that
these distances are dimensionless. Use a consistent externally declared length
unit for the box, coordinates and radii. There is no automatic m/mm conversion
or scaling of thresholds to the box size.

The public generation path uses a strict component-bridge limit of 3.0
and degree-one limit of 6.0 in those coordinate units. The degree-one candidates
and their neighbors come from the graph at the start of that pass; there is no
boundary-node filter. The pass can add edges within one component, or connect
components farther apart than the initial 3.0 cap.

Reports are historical observations at this generation stage. Distances are
not recomputed after users move or scale a graph. `tile_3d` copies this report
without remapping its recorded edge IDs; graph merging does not combine repair
reports. After tiling, merging or other ID-reassigning operations, these records
are not a live lineage index for the derived graph. Inspect the report on the
generated graph or its native save/load copy. Native save/load and graph copy
retain the reports and stable IDs.

This coverage does not include 2D Voronoi bridge repair, general tiling welds,
the unused nearby-node weld helper, or legacy `FiberNetwork` generators. Absence
of this key does not prove that no other generation stage repaired a structure.
Topological connectivity is not mechanical stability, and added bridge edges
are not a physics qualification or an intended-design guarantee.
