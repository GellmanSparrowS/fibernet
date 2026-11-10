# Finite-width contact units and raster limits

`ContactConfig.compute` measures overlap of straight endpoint segments thickened
to capsules of the supplied width. Width and coordinates use the same caller
length unit. It excludes pairs sharing a node ID, retains distinct physical edge
rows, and does not infer welding, contact forces, friction, or finite-width pore
space. In `SnapshotFeatureExtractor`, these are the `contact_*` raster fields,
not the separate centerline crossing and collinear-overlap fields.

The raster calculation translates all supplied positions and divides both
positions and width by their common scale (the greater of the XY extent and
width). Projection and distance tests use this dimensionless canvas. This
removes the former absolute squared-length floor, which could erase contact
after only a change of length unit. Pixel spacing is restored to caller units;
overlap area is restored to squared units. Counts and coverage are dimensionless.
The bounding canvas still includes isolated positions, as before.

```python
import numpy as np
from fibernet.analysis import ContactConfig

positions = np.array([[0., 0.], [2., 0.], [1., -.5], [1., .5]])
edges = np.array([[0, 1], [2, 3]])
for scale in (1., 1e-16):
    result = ContactConfig(width=.2 * scale, resolution=128).compute(
        positions * scale, edges)
    assert result['contact_pair_count'] == 1
    assert result['contact_coverage_ratio'] == .0625
    print(result['contact_overlap_area'] / scale / scale)
# Both print approximately 0.0392832785665571.
```

Before this correction, the second example reported no contact. Recompute
affected saved raster features consistently before comparing datasets or using
them with a fitted model, and retain the producing FiberNet revision. Public
signatures, output keys, width convention, adjacency exclusion, resolution and
pixel budgets are unchanged. Rounding at a pixel boundary may also change a
classification at ordinary scales.

This remains a finite-resolution approximation. Features smaller than a pixel,
near-threshold tangencies, tiny features compared with the global canvas, and
coordinate precision already lost before the call can change raster decisions.
Uniform scaling is reliable for well-resolved geometry, not an exact-predicate
guarantee. A nonrepresentable coordinate/width normalization, physical pixel
spacing, positive pixel area, or total overlap area raises `ValueError` rather
than silently returning nonfinite or underflowed measurements. Zero width and
empty edges retain their zero-result short circuit.

This correction does not normalize the other snapshot structure or centerline
contact statistics. The separate [pore scaling contract](SNAPSHOT_PORE_SCALING.md)
still applies to planar faces. It is not evidence of a mechanically validated
contact model or improved resolution.
