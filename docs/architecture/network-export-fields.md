# Network analysis and visualization exports

`FiberNetwork.to_networkx()` uses fiber list positions as node keys, matching
crosslink indices and sparse adjacency rows. Every fiber is included, including
isolated fibers. The original `fiber_id` is available as a node attribute and
can repeat. The return value remains a simple undirected `networkx.Graph`;
for repeated crosslinks the last record supplies the edge attributes.

## VTP result fields

```python
from fibernet.api import export

export(network, 'results.vtp',
       point_data={'displacement': displacement_at_samples},
       cell_data={'axial_force_N': force_per_fiber})
```

The arrays follow the existing sampled geometry:

| Field | Shape | Ordering |
| --- | --- | --- |
| Point scalar | `(N,)` | Fiber list, then centerline samples |
| Point vector | `(N, 3)` | Same samples, three components |
| Cell scalar | `(F,)` | One whole-fiber polyline per fiber |

Here `N` counts all samples and `F` counts fibers. Coincident samples on different
fibers remain distinct. Supply values in that ordering, using result IDs and
interpolation when the solver uses a different discretization.

Floating fields are written as Float64, signed integers as Int64, unsigned
integers as UInt64, and boolean indicators as UInt8. Integer lists retain their
exact values through the 64-bit range. A list mixing integers and floats must
represent every integer exactly as Float64. Field names are printable ASCII
tokens of 1–255 characters, without whitespace or `%`; XML metacharacters are
escaped. All fields are validated before opening the destination.

Legacy `.vtk` files retain one cell per adjacent sample segment and Float32
attribute formatting. Their cell arrays therefore have shape `(M,)`, where
`M` is the segment count. Exporting `.vtp` without fields retains the existing
geometry-only output.
