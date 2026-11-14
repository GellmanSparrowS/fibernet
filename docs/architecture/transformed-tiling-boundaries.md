# Boundary labels for transformed tiling

`fibernet.core.tiling.tile_with_transforms` places a unit cell on an XY grid,
applies any per-cell callback before translation, and welds nearby nodes. Its
boundary labels use the same nominal outer XY planes as `fibernet.tile_2d`.
This example is a synthetic geometry check, not a mechanical simulation.

```python
from fibernet import StructureGraph
from fibernet.core.tiling import tile_with_transforms
from fibernet.core.transforms import rotate

cell = StructureGraph(dimension=2, box_size=(1, 1))
cell.add_polyline([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
graph = tile_with_transforms(
    cell,
    grid=(2, 2),
    cell_transforms={(1, 0): lambda g: rotate(g, 90, center=(0.5, 0.5))},
)
assert (graph.num_nodes, graph.num_edges) == (9, 12)
assert len(graph.get_boundary_nodes()) == 8
assert len(graph.get_internal_nodes()) == 1
```

## Which nodes are labeled

Each node carries six flags in the order `(x_min, x_max, y_min, y_max, z_min,
z_max)`. If the effective cell dimensions are `w, h` and the grid is `nx, ny`,
the first four flags compare the placed node's coordinates with `x=0`,
`x=nx*w`, `y=0`, and `y=ny*h`. A flag is true only when the absolute coordinate
distance is strictly less than the effective welding tolerance. An explicit
`box_size` controls both spacing and these planes; otherwise the existing cell
box or bounding-box fallback is used. Existing tolerance selection is unchanged:
a truthy `tolerance` override is used, otherwise `unit_cell.tolerance` is used.

Labels are recomputed after the callback and grid translation, rather than
copied from input labels. Internal cell seams are not outer faces. Welding
retains the first stored node and its labels, rather than taking the union of
labels from all nearby candidates. The coordinates and labels therefore refer
to the same representative.

These are independent coordinate-plane tests, not a convex hull, an actual
geometry bounding box, or a clipping operation. A transform can move all nodes
off the nominal planes, leaving no flagged nodes. A node outside the nominal
rectangle can still be flagged if it lies on one of those planes. Stored edge
interior points do not create boundary nodes.

This API tiles two grid axes even when the input graph has `dimension=3`.
It preserves that dimension and the placed Z coordinates, keeps the output box
Z extent at zero, and sets both Z flags false. Use `tile_3d` for the existing
three-axis grid operation. Callback inputs and ownership rules are unchanged:
callbacks receive the source cell, so use non-mutating callbacks such as the
built-in transforms to keep the source unchanged.

## Compatibility

Earlier `tile_with_transforms` results incorrectly left every output boundary
flag false. Geometry, topology, material records and callback invocation are
unchanged by this correction, but boundary-node selections, internal-node
selections, rendering highlights and boundary-derived ML features can change.
Boundary flags also participate in serialized graph/provenance records.
Regenerate affected derived data and review models or workflows that depended
on the previous labels before mixing results across revisions. Existing saved
graphs are not modified automatically.

These labels do not impose forces, displacements, periodic constraints or any
other solver boundary condition. Select and validate the controls required by
the chosen solver explicitly.
