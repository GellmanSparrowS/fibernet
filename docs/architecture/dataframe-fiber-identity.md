# Fiber identity in point tables

`fibernet.io.to_dataframe` exports one row per centerline point. Keep
`fiber_id` and `point_idx` with the coordinates when saving or filtering a table.

When stored labels differ from fiber list positions, the export appends
`fibernet_fiber_index`. This reserved column distinguishes fibers that share a
label and preserves their original order. Tables whose labels already match
positions retain their existing columns.

```python
from fibernet.io import to_dataframe, from_dataframe
import pandas as pd

points = to_dataframe(network)
points.to_csv("network_points.csv", index=False)
loaded = pd.read_csv("network_points.csv", float_precision="round_trip")
restored = from_dataframe(loaded)
```

With the reserved column present, import groups fibers by ascending integer
position, then sorts each centerline by its integer `point_idx`. Positions may
have gaps after filtering. A named index level can hold either field; for
example, `points.set_index(["fibernet_fiber_index", "point_idx"])`. Keep each
field in one place, as a column or an index level. Each fiber position needs one
consistent `fiber_id`; point indices must be unique within that fiber.

Without the reserved key, existing external tables keep their `fiber_id`
grouping behavior, including categorical group order. Imported fibers receive
sequential IDs in group order. This point-table adapter carries centerlines,
radii and the exported material fields; use the native network format when
saving crosslinks, box information and network metadata.
