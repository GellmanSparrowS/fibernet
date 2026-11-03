# Snapshot pore units and numerical scale

Pores here are bounded faces of planar centerlines, not finite-width voids or a
mechanical response. The pore pipeline translates the filtered positions and
uses the larger XY span of edge endpoints as its length scale before detecting intersections,
splitting segments and walking faces. It converts pore areas and perimeters
back to the caller's units squared and caller's units, respectively. Counts,
aspect ratios and coverage are dimensionless. A zero span uses unit scale.

The existing geometric tolerances are applied to these normalized coordinates.
This reduces dependence on the caller's choice of length unit for
well-conditioned geometry, but it is not an exact geometric predicate: tiny
features relative to the overall span, near-collinear edges, nearly coincident
points, or precision already lost in the input may change topology. Excessive
or unrepresentable coordinate/area scales raise an error rather than returning
non-finite pore measurements. A supplied region still filters edges by midpoint;
faces are not clipped to its boundary.
Results near the former absolute tolerances may differ from earlier releases.

This change concerns pore metrics only. Structure and optional contact metrics
retain their existing calculations and tolerances; scaling all input coordinates
does not establish scale invariance for those other outputs. With contact
features enabled, intersections are computed separately for pores and contact.
Existing graph budgets remain in force; this is not a performance improvement.
