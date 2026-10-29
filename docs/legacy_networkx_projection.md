# Legacy NetworkX projection

`fibernet.graph.from_networkx(G, material=None, pos_attr="pos")` returns a
legacy `FiberNetwork`. It is not the native `StructureGraph` interchange API.

## Supported edge-incidence contract

- Each edge in an undirected `networkx.Graph` or `MultiGraph` becomes one
  separate, straight, two-point fiber. Parallel edges are not merged.
- Every original node shared by two or more edge endpoints creates all pairwise
  welded crosslinks at that node. A degree-2 node joins its two fibers; a
  three-edge branch creates three pairwise crosslinks, not three separate
  junction positions. A degree-1 node remains a fiber endpoint without a
  crosslink. Maximal paths are not combined into a single fiber.
- Node identity determines connectivity. Distinct nodes at the same position
  are not welded merely because their coordinates match.
- Every node must have a 2D or 3D position in `pos_attr`. Two-coordinate positions
  are extended with zero Z. The legacy `FiberNetwork.dimension` default remains
  3; no unit recognition or coordinate-frame conversion occurs.
- Edge `radius` defaults to `0.1`; edge `youngs_modulus` overrides the default
  material's modulus. The material contributes its name, modulus, Poisson ratio
  and density. Fiber sampling, segment defaults and legacy fiber IDs are not
  changed by the connectivity repair; crosslinks reference fiber list indices.

For a node of degree `d`, the representation needs `d * (d - 1) / 2` crosslinks.
This is the same junction convention used by `StructureGraph.to_fiber_network`,
but this importer does not use that bridge's centerline resampling policy.

## Deliberate legacy limits

This conversion preserves the supported edge incidence, not a complete source
graph record. Isolated nodes have no representation in `FiberNetwork` and are
omitted. Original IDs, graph/node/edge metadata, boundary flags, curve points,
edge material objects/records and additional default material properties are
not transferred. Only the attributes listed above are read.

Directed graphs retain edge fibers but lose direction: crosslinks describe
undirected incidence. Self-loops and coincident-endpoint edges still create
degenerate zero-length fibers; they are not made simulation-valid by import.
These are existing projection limits, not newly enforced rejection policies.

`fibernet.graph.to_networkx` separately exports crosslink positions and fiber
segments between crosslinks. It does not reconstruct original leaf nodes,
isolates or source IDs and is not an inverse. For example, a three-node path
imports as two fibers joined by one crosslink; exporting that result yields
one crosslink node and no edges, not the original path. JSON loading through
`load_graph_json` uses the same legacy importer; JSON graph saving does not
provide a lossless material/curve/parallel-edge protocol.

## Native preservation route

Keep `StructureGraph` as the authoritative structure. Its
`to_networkx_lossless()` / `StructureGraph.from_networkx_lossless()` pair uses
the native record envelope to preserve native IDs, isolated nodes, parallel
members, curves, full material records and metadata. The lossless reader is
for graphs produced by that native protocol, not arbitrary NetworkX graphs.
The similarly named `StructureGraph.from_networkx()` is also a projection;
passing an arbitrary rich graph through it does not recover discarded fields.
