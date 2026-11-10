# Snapshot pore units and numerical scale

Pores here are bounded faces of planar centerlines, not finite-width voids or a
mechanical response. The pore pipeline translates the filtered positions and
uses the larger XY span of edge endpoints as its length scale before detecting intersections,
splitting segments and walking faces. It converts pore areas and perimeters
back to the caller's units squared and caller's units, respectively. Counts,
aspect ratios and coverage are dimensionless. A zero span uses unit scale.

All active components share this scale. Distant disconnected components can
make small faces fall below normalized tolerances. Disconnected components
nested within a bounded face supply its inner boundaries, including holes and
zero-width interior branches. Isolated nodes do not set the scale, but remain
in the analysis-region bounding box when
no region is supplied.

Coverage divides the sum of bounded-face areas, with holes removed, by the supplied region
area, or the retained-node bounding-box area without a region, with an area floor
of 1e-12 in normalized coordinates. Coverage is capped at 1. The legacy key
`porosity` denotes this centerline-face coverage, not finite-width material
porosity. Bounded faces partition the enclosed planar area rather than counting
nested disks twice. The perimeter includes outer and inner boundaries. Each
side of an interior branch is part of that boundary, whether the branch is
connected to the surrounding loop or is a separate component. Two pores are
neighbors when they share an edge, not merely a vertex. Aspect ratios retain
the existing axis-aligned bounding box of each face's outer boundary.

## Nested-loop example

The two closed square centerlines below delimit a 12-unit annular face and a
4-unit inner face. They share the inner square's boundary. The annular
perimeter is 16 + 8 = 24, and the inner perimeter is 8.

```python
import numpy as np
from fibernet.analysis import SnapshotFeatureExtractor

positions = np.array([[0, 0], [4, 0], [4, 4], [0, 4],
                      [1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                  [4, 5], [5, 6], [6, 7], [7, 4]])
features = SnapshotFeatureExtractor().extract(positions, edges,
                                             rect=[0, 0, 5, 5])
np.testing.assert_allclose(features["pore_areas"], [12, 4])
np.testing.assert_array_equal(features["pore_neighbor"], [1, 1])
assert features["pore_perim_mean"] == 16
assert features["porosity"] == 0.64
```

The earlier implementation reported overlapping areas `[16, 4]`, no pore
neighbors and a mean perimeter of 12 for this example. Stored pore features,
plots or trained models must retain their producing source revision; recompute
affected nested-component inputs consistently before comparing or combining
old and corrected values. Feature names, output shapes and public signatures
are unchanged. Structure and optional finite-width contact fields are unchanged.

## Numerical scope

The existing geometric tolerances are applied to these normalized coordinates.
This reduces dependence on the caller's choice of length unit for
well-conditioned geometry, but it is not an exact geometric predicate: tiny
features relative to the overall span, near-collinear edges, nearly coincident
points, or precision already lost in the input may change topology. With active
edges, unrepresentable coordinate/area scales, normalized region areas and restored
pore measurements raise `ValueError`. The square of the active span must itself
be finite and nonzero, even when individual face areas might be representable.
A supplied region still filters edges by midpoint;
faces are not clipped to its boundary.
Results near the former absolute tolerances may differ from earlier releases.

This change concerns pore metrics only. Structure and optional contact metrics
retain their existing calculations and tolerances; scaling all input coordinates
does not establish scale invariance for those other outputs. With contact
features enabled, intersections are computed separately for pores and contact.
Existing graph budgets remain in force; this is not a performance improvement.
