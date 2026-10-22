# Stable identity in detailed simulation results

`SimResult.compute_detailed(graph, stiffness=...)` calculates geometric edge stretches, axial forces and elastic energy from saved result arrays and the reference `StructureGraph`.

When result metadata contains `node_ids` and `edge_ids`, these lists define the row order of node and edge arrays. Reordering graph records, including a graph JSON round-trip, preserves the computed metrics. A stiffness vector follows `edge_ids`; a scalar applies to every edge. Distinct physical edges retain separate rows, including edges sharing endpoints.

Both identity lists must be supplied together, contain unique nonnegative integer IDs, and exactly cover the graph. Graph records must agree with their keys and connectivity. Result arrays must match the reference coordinate shape. Reference edges must have positive finite geometric lengths, and detailed metrics require at least one edge. Invalid inputs or nonfinite derived metrics raise `ValueError` before changing the result.

Results without identity metadata retain historical graph-insertion ordering. For those results, preserve the original node and edge order when recomputing details.

Use the original reference geometry and topology. Matching IDs establish row correspondence, not proof that the reference graph has remained unchanged. Native axial-viscoelastic results retain their dedicated material-state evaluator. LAMMPS results retain explicit equilibrium-length semantics and reject this geometric-rest recomputation.
