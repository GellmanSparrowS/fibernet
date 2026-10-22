# Result identity and mechanical channels

`SimResult.metadata['node_ids']` labels displacement, deformed-position and trajectory rows. `metadata['edge_ids']` labels axial-force and stretch rows. IDs identify physical edges: two edges with the same endpoints may have different sections/materials and must not be matched by endpoints alone.

## Recomputing geometric metrics

`result.compute_detailed(graph)` uses these maps rather than the graph's current insertion order. Both maps must be supplied together. Node IDs must cover the graph exactly; edge IDs may select a unique ordered subset. Output arrays retain result edge order, and `n_edges` counts that selection. Graph serialization may change insertion order without changing these identities.

A scalar spring stiffness applies to every selected edge. An array is in the result's `edge_ids` order, not the current graph order. The axial convention is `F = k * (L/L0 - 1)` with geometric rest length; `k` is force-like, not force per extension. `edge_forces` is signed; `max_force` is its maximum absolute axial value. These are not nodal/contact forces or bending moments.

Invalid maps, inconsistent metric-input rows/counts, invalid stiffness and nonfinite metric inputs reject before metrics are changed. Existing zero default counts may be populated. Trajectories are retained, not recomputed. Historical results with neither map retain the old all-node/all-edge graph insertion-order assumption. Do not use a differently ordered graph with those files. A partially mapped result is rejected rather than guessed during recomputation; historical partial-map files remain loadable, while duplicate-ID files retain the loader's existing rejection.

For `mode='fem_beam_frame'`, omit stiffness. This recomputes geometric stretch/displacement metrics only and preserves solver force/energy channels and `energy_kind='axial_stress_proxy'`. Explicit spring stiffness is rejected before mutation. Geometric stretch is not a substitute for a beam constitutive strain or total beam energy.

## FEM resultants

New conversions retain the existing numeric `metadata['beam_resultants']` array and add `metadata['beam_resultant_contract']` with schema version, dimension and ordered components:

- 2-D: signed axial `N` (force), shear `V` (force), and `(M_i + M_j)/2` (moment). The last column is a mean, not either endpoint moment.
- 3-D: axial `N` (force), shear `V_y` (force), shear `V_z` (force). The current producer recovers only axial `N`. The two retained shear slots are legacy placeholders explicitly marked unavailable, not measurements of zero shear.

`N` here is an axial-force symbol, not an assertion of newton units. Quantity kinds distinguish force from moment; no SI or other unit system is inferred from the graph. Existing nonlinear approximate status stays approximate, and existing field/solver status remains authoritative. This metadata does not recover missing forces or establish any contact-force channel.

A component status of `available` means that its column is provided or recovered under the recognized producer layout; it does not establish solver validity, numerical accuracy, equilibrium or convergence. Producer-authored `field_status` and `solver_status` remain separate: they are preserved when present and remain absent when absent, and an existing `approximate` qualification is not upgraded.

Legacy files lacking this component contract remain readable, including historical three-column FEM `edge_forces` exports. The loader retains that original array separately and extracts axial force as before; it does not invent dimension-specific availability for old records.
