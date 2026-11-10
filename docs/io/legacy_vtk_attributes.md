# Legacy VTK point and cell attributes

`fibernet.io.vtk.to_vtk` writes a legacy ASCII VTK `POLYDATA` view of a
`FiberNetwork`. Each consecutive centerline pair is a separate line cell.
Coincident points in different fibers remain separate; graph stable IDs,
welded crosslinks, physical units and solver laws are not encoded.
Keep the authoritative `StructureGraph` and detailed simulation JSON separately.
For straight `StructureGraph` interchange with IDs and declared field meanings,
see the optional [VTU adapter](../integrations/MESHIO_USAGE.md).

## Attribute admission

Point fields must have shape `(N,)` for scalars or `(N, 3)` for vectors, where
`N` is the sum of all fiber centerline point counts. Cell fields must be scalar
arrays of shape `(M,)`, where `M` is the total number of consecutive centerline
segments. Points and cells follow fiber order, then point/segment order within
each fiber. These are export rows, not graph node/edge IDs.

Pass `None` or a mapping from field names to arrays. Names must be
printable ASCII tokens of 1–255 characters without whitespace, control
characters or `%`, such as
`temperature_K` or `displacement_m`. VTK readers decode percent escapes in
names and can reject tokens longer than 255 bytes. Strings, complex arrays, object arrays,
NaN, infinity and values whose magnitude exceeds the largest finite float32
are rejected. Integer, unsigned integer, floating-point and boolean indicator
arrays are accepted. Active masked values are rejected, including nested
masks; existing all-false masked arrays remain accepted.

All attribute checks finish before opening the destination. Invalid attributes
therefore leave an existing file unchanged and do not create a missing file.
This is input admission, not atomic file replacement: geometry errors or I/O
failures after opening can still leave partial output. Neither the geometry
contract nor `to_vtk_xml` changes in this fix.

## Precision and compatibility

Coordinates retain `double` declarations and `.17g` formatting. Attributes
retain the legacy `float` declaration and `.6e` formatting, including numeric
0/1 for boolean indicators. This is a lossy visualization format: values can
round, and sufficiently small magnitudes can underflow in float32 readers.
The exporter does not normalize or rescale fields. Valid existing arrays keep
the same output bytes; previously accepted nonfinite/complex data and malformed
names now raise `ValueError` rather than producing invalid attribute records.
Use native JSON or an appropriate higher-precision export when precision or
identity retention matters.

## Example

```python
import numpy as np
import fibernet as fn
from fibernet.io.vtk import to_vtk

# Geometry coordinates are supplied in metres in this example.
graph = fn.pattern_2d(unit="honeycomb", grid=(2, 2), box=(1, 1), radius=.01)
graph.save_json("graph.json")
network = graph.to_fiber_network()  # Legacy endpoint-chord projection.
n_points = sum(len(f.centerline) for f in network.fibers)
n_segments = sum(max(len(f.centerline) - 1, 0) for f in network.fibers)

# Synthetic fields demonstrate array layout, not a thermal simulation.
to_vtk(network, "network.vtk",
       point_data={"temperature_K": np.full(n_points, 293.15)},
       cell_data={"selected": np.ones(n_segments, dtype=bool)})
```

A nodal displacement array from `SimResult` cannot be passed directly as a
legacy VTK point array: the converted fibers have their own sampled point rows.
Map solver rows through its identity metadata, then explicitly interpolate to
the chosen export points. Endpoint interpolation is a visualization convention,
not a recovered beam-interior solution. Likewise repeat or interpolate each
physical edge field across that fiber's segments in the declared order.
