# StructureGraph curve serialization

`StructureGraph.to_dict()` and `save_json()` retain each edge's stored internal
points. `from_dict()` and `load_json()` restore their coordinate-array shape:

- `null` restores `None`, representing absent internal-point data.
- `[]` restores an empty NumPy array with shape `(0, 3)`.
- A nonempty list of XYZ rows restores an array with shape `(M, 3)`.

An explicitly empty curve therefore remains compatible with endpoint stacking
and the lossless NetworkX adapter after a dictionary or JSON round trip.
Malformed coordinate-array shapes raise `ValueError` during loading.

```python
import numpy as np
from fibernet.core.structure_graph import StructureGraph

restored = StructureGraph.from_dict(graph.to_dict())
edge = restored.edges[edge_id]
if edge.internal_points is not None:
    polyline = np.vstack([
        restored.nodes[edge.node_i].position,
        edge.internal_points,
        restored.nodes[edge.node_j].position,
    ])
```

Use `to_networkx_lossless()` and `from_networkx_lossless()` when exchanging
native graph records, including stable IDs, parallel physical edges, endpoint
orientation, curves, materials and metadata. The legacy `to_networkx()` and
`from_networkx()` methods provide a simpler projected graph view.
