# Scale-invariant 3D rotation axes

`fibernet.core.transforms.rotate` interprets an explicit 3D axis as a
direction, not a displacement or a scale. Pass three finite real components,
with at least one nonzero component. The vector need not have unit length.

```python
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.transforms import rotate

graph = StructureGraph(dimension=3)
node = graph.add_node([1.0, 0.0, 0.0])
rotated = rotate(graph, 90.0, axis=[0.0, 0.0, 1e300])
np.testing.assert_allclose(rotated.nodes[node].position, [0.0, 1.0, 0.0], atol=1e-15)
np.testing.assert_array_equal(graph.nodes[node].position, [1.0, 0.0, 0.0])
```

Positive multiples of the same axis give the same rotation within floating-point
roundoff. Reversing the direction reverses the rotation sense. Angles are in
degrees, with the right-hand rule. An omitted axis uses the z direction.
For 2D graphs the axis argument remains ignored and rotation stays in the xy-plane.

Normalization scales the components before computing their norm, so very large
or subnormal finite nonzero axes remain usable. Malformed, complex, nonfinite
and all-zero explicit 3D axes raise `ValueError` before the graph is copied,
including for an empty graph. The same rules apply through `make_rotate` when
the resulting transform is executed, including inside `compose`.

The result remains a detached graph. Node IDs, edge IDs, radii, materials,
boundary flags and existing box metadata retain their prior rotation behavior;
node positions and internal curve points rotate about the requested center.
This correction does not redefine periodic cells, boundary semantics, or
validation of the angle and center arguments.

Earlier revisions squared the raw axis components while normalizing. A large
finite axis could silently shrink the geometry; a tiny axis could produce
nonfinite values and fail later. Regenerate results produced with such axes
rather than treating the old output as a rigid rotation.
