# Legacy sampled-network trimming

`fibernet.core.transform.trim_to_box(network, box_min, box_max,
remove_outside=True)` returns a `FiberNetwork`. The signature, default and
sampled geometry rules are unchanged:

- `True` keeps each **whole fiber** with at least one centerline sample inside
  the inclusive box. A segment crossing the box without an inside sample is
  still omitted.
- `False` keeps only inside samples, in their original order, and drops fibers
  with fewer than two surviving samples. It does not insert boundary crossings
  or split multiple retained runs. Consecutive retained points can therefore
  form a new chord over a removed run, as in the original operation.
- Output fiber IDs are dense list indices. The output keeps dimension and
  periodic flags and uses `box_max - box_min` for `box_size`. This operation
  does not otherwise transfer network metadata or change units.

## Crosslinks follow retained attachments

Previously every call discarded all crosslinks, even when no fiber was removed.
The operation now preserves surviving records in input order, including parallel
or repeated records, and remaps both fiber indices. These are source **list
indices**, not `Fiber.fiber_id` labels. Links touching a removed fiber are omitted.

In whole-fiber mode, parameters remain unchanged. A link may remain outside the
box because its complete attachment fibers remain too. Crosslink positions are
not used as a second geometry filter.

In sampled mode, parameters follow the normalized sample-index convention used
by legacy contact detection: parameter `p` denotes sample coordinate
`p * (N - 1)`, not normalized arc length. An attachment survives at a retained
vertex or inside an original segment whose two endpoints are retained. Its
parameter is recomputed on the shortened sample list. If either attachment is
removed, the crosslink is omitted. In particular, a link is not moved onto a
new chord connecting separated retained runs. Nonfinite or out-of-range
parameters do not identify surviving attachments. Small floating-point roundoff
when reconstructing an integer sample index is tolerated.

Positions, type, strength, stiffness and additional crosslink attributes are
copied, without inferring new junctions, projecting positions, or redetecting
contacts. Crosslink records and mutable values are detached from the input.
The input is not modified. Existing fiber ownership behavior remains unchanged:
whole-fiber mode deep-copies fibers; sampled mode constructs fibers with the
source material and cross-section parameter objects.

## Native graph coordination and migration

Keep `StructureGraph` as the authoritative graph. Its existing
`to_fiber_network()` bridge returns sampled endpoint chords and pairwise endpoint
junctions. Those junctions now survive a legacy trim when their attachments
survive. The bridge is still approximate; trimming does not restore native IDs,
curves, boundary labels, isolated nodes or all metadata. The separate
`StructureGraph.from_fiber_network()` endpoint projection is also approximate
and does not import arbitrary interior crosslinks.

This correction intentionally changes crosslink counts, degree/connectivity
results, serialization and any downstream calculation that consumes those
links. Recompute derived legacy outputs rather than mixing old empty-link
results with corrected results. If an application intentionally needs a
geometry-only network, clear the **returned** network's `crosslinks` explicitly.
The correction does not certify a mechanical model or alter solver behavior.

The regressions cover retained/discarded fibers, dense index remapping,
nonuniform sample spacing, segment interiors, separated runs, repeat records,
copy isolation, roundoff and a native graph bridge/crop/serialization journey.
