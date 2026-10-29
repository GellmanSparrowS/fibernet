# Legacy endpoint summary and native graph analysis

`fibernet.api.analyze(graph, kind='topology')` directly returns existing `TopologyMetricsV1` for StructureGraph. Explicit kind is required; it does not promise geometry, material analysis or spectrum. No new result wrapper or schema is introduced. Wrong kind/type raises clearly.

`fibernet.api.analyze(network)` with FiberNetwork and omitted/None kind is a legacy FiberNetwork endpoint summary, **not a general crosslinked-network analysis**. The repaired domain is nonperiodic, crosslink-free networks with finite float64 centerlines, nondegenerate endpoint chords, no duplicate endpoint pairs and no distinct endpoints merged by the archived six-decimal rounding. Interior intersections are not junctions. Explicit crosslinks are refused rather than assigned to arbitrary nearest endpoints. Build explicit StructureGraph incidence for native graph analysis; there is no automatic conversion.

## Existing native route (unchanged)

StructureGraph is the authoritative representation for ordered and disordered networks alike. These existing analyzers operate on it directly:

- `analyze_topology(graph) -> TopologyMetricsV1`: dimensionless physical edge/unique-connection counts, separate degrees, isolates, components, cycle rank, bridges and articulation IDs. Preserves stable node/edge identities and physical multiplicity; does not infer periodic equivalences. Empty largest-component fraction is `None`.
- `analyze_geometry_paths(graph, length_unit=..., coordinate_frame=..., node_pairs=...) -> GeometryPathMetricsV1`: explicit caller-unit full XYZ chord/polyline lengths, per-edge straightness and budgeted selected shortest paths. Curves are stored polylines, not smoothed reconstruction. Unreachable distances and undefined ratios are `None` with statuses; no automatic meter assumption. Retains edge IDs and multiplicity, including parallel curves.
- `analyze_spectrum(graph, adjacency='physical_multiplicity') -> LaplacianSpectrumV1`: bounded, unweighted combinatorial Laplacian, explicit optional unique-connection projection, raw eigenvalues and numerical residual checks. No mechanical/material-weighted eigenfrequency claim. Empty eigenvalues are empty; largest/Fiedler values follow the existing undefined-value contract.

Use the explicit kind for StructureGraph; do not collapse physical multiedges, infer node IDs from positions, or replace materials/curves with defaults to make a summary run. These are analyses, not serializers: material and metadata remain on the original graph, unmodified. Native validators reject malformed IDs/endpoints/self-loops; geometry declares resource budgets and numerical failures, spectrum declares dense limits. Dependency/input/numerical failures propagate, never become zero-valued reports.

## Legacy return

Exactly the existing ordered twelve keys remain:

`num_fibers, num_crosslinks, dimension, nematic_order, mean_length, total_length, mean_tortuosity, num_nodes, num_edges, mean_degree, is_connected, num_components`.

Counts/dimension are Python integers, is_connected a bool, other fields floats. Lengths use the supplied coordinate unit (no new unit conversion). Other quantities are dimensionless. There is no volume_fraction field. Arc length includes all centerline segments. Tortuosity is arc/chord; orientation is the unweighted endpoint-chord nematic statistic with the existing dimension-dependent convention. It is not a local curved-tangent orientation statistic. 2D fibers must be XY-planar. Radii/materials do not enter these fields and are not validated as a mechanical model.

Empty sums/counts and means, including nematic order and tortuosity, are zero **by API convention, not inferred physical isotropy**; is_connected is False. Two exact shared endpoints are connected even without an explicit Crosslink record, consistently with the archived endpoint projection. Crosslink count in the supported domain is necessarily zero. Chord <= 1e-12 is rejected because the legacy direction property otherwise fabricates an x-axis direction. Distinct endpoints that collide at six decimals are rejected; the threshold uses caller coordinate units and therefore this legacy boundary is not scale invariant. Use native topology for explicit adjacency at other scales.

This repair directly reuses archived morphology and topology, but not their full reports, density, centralities or all-pairs paths. Compensated `math.fsum` aggregation is a deliberate small numerical improvement over historical sequential/NumPy reduction; final-bit compatibility is not promised. Invalid inputs raise TypeError/ValueError; floating-point failures may raise FloatingPointError/OverflowError, missing dependencies ImportError. No catch-all fallback, graph mutation or lossy conversion occurs.

The 2D guard is per fiber: different constant-Z layers are permitted and remain disconnected unless full XYZ endpoints match. It does not certify that the entire network occupies one plane.

`StructureGraph.from_fiber_network` is explicitly approximate, not a lossless migration path. General legacy crosslinks require authoritative explicit incidence/curve/ID construction; this summary does not infer it.
