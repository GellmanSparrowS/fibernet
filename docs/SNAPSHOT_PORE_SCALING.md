# Snapshot pore units and numerical scale

Pores here are bounded faces of planar centerlines, not finite-width voids or a
mechanical response. The pore pipeline translates the filtered positions and
uses the larger XY span of edge endpoints as its length scale before detecting intersections,
splitting segments and walking faces. It converts pore areas and perimeters
back to the caller's units squared and caller's units, respectively. Counts,
aspect ratios and coverage are dimensionless. A zero span uses unit scale.

All active components share this scale. Distant disconnected components can
make small faces fall below normalized tolerances; normalization does not fix
the existing face walker's treatment of nested disconnected loops. Isolated
nodes do not set the scale, but remain in the analysis-region bounding box when
no region is supplied.

Coverage divides the sum of positive bounded-face areas by the supplied region
area, or the retained-node bounding-box area without a region, with an area floor
of 1e-12 in normalized coordinates. Coverage is capped at 1. The legacy key
`porosity` denotes this centerline-face coverage, not finite-width material
porosity or a union-area calculation for nested disconnected cycles.

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
