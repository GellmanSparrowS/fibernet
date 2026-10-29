# Result identity and mechanical channels

`SimResult.metadata['node_ids']` labels displacement, deformed-position and trajectory rows. `metadata['edge_ids']` labels axial-force and stretch rows. IDs identify physical edges: two edges with the same endpoints may have different sections/materials and must not be matched by endpoints alone.

## Recomputing geometric metrics

`result.compute_detailed(graph)` uses these maps rather than the graph's current insertion order. Both maps must be supplied together. Node IDs must cover the graph exactly; edge IDs may select a unique ordered subset. Output arrays retain result edge order, and `n_edges` counts that selection. Graph serialization may change insertion order without changing these identities.

A scalar spring stiffness applies to every selected edge. An array is in the result's `edge_ids` order, not the current graph order. The axial convention is `F = k * (L/L0 - 1)` with geometric rest length; `k` is force-like, not force per extension. `edge_forces` is signed; `max_force` is its maximum absolute axial value. These are not nodal/contact forces or bending moments.

Invalid maps, inconsistent metric-input rows/counts, invalid stiffness and nonfinite metric inputs reject before metrics are changed. Existing zero default counts may be populated. Trajectories are retained, not recomputed. Historical results with neither map retain the old all-node/all-edge graph insertion-order assumption. Do not use a differently ordered graph with those files. A partially mapped result is rejected rather than guessed during recomputation; historical partial-map files remain loadable, while duplicate-ID files retain the loader's existing rejection.

For `mode='fem_beam_frame'`, omit stiffness. This recomputes geometric stretch/displacement metrics only and preserves solver force/energy channels and `energy_kind='axial_stress_proxy'`. Explicit spring stiffness is rejected before mutation. Geometric stretch is not a substitute for a beam constitutive strain or total beam energy.

## Effective spring coefficients

New ordinary spring/Taichi dynamics results add `metadata['spring_coefficient_provenance']`, with nested schema `spring_coefficient_provenance/1`. Saved JSON retains this record under the same key in its historical flat result namespace; there is no new top-level result schema. The record describes coefficients, not convergence, numerical accuracy or a restart state. FEM, LAMMPS and explicit dynamic-viscoelastic contracts are unchanged.

The record has `law='geometric_rest_axial/1'` and `quantity_kind='force'`. The law is the existing `F = k * strain`, `energy = 0.5 * k * L0 * strain**2`, using the existing geometric-rest metric convention. `k` is force-like, not force/extension stiffness. This does not assert newtons or infer units from graph Material. Existing caller-consistent unit declarations, when present, remain unchanged; no units are converted or invented.

Two roles prevent recomputed metrics from rewriting simulation history:

- `simulation` records the original effective input: `source='spring_k'` when supplied, otherwise `source='stiffness'`. The signature cannot distinguish omitted stiffness from an explicitly supplied equal default.
- `current_metrics` initially contains only `{'source': 'simulation'}`, referencing that descriptor without duplicating its values. After `compute_detailed(graph, stiffness=...)` actually replaces force/energy metrics, this role instead contains its own descriptor with `source='compute_detailed.stiffness'`. Original simulation inputs are retained. With `stiffness=None`, or without deformed positions to recompute force/energy, coefficient provenance is unchanged.

A coefficient descriptor includes `representation`, `edge_count` and `availability`:

- `representation='scalar'`, `availability='available'`, `value=<float>` retains one value even in a summary.
- `representation='per_edge'`, `availability='available'`, `values=<array>`, `row_map='metadata.edge_ids'` aligns values with the existing result edge-ID map, never a later graph's insertion order. An equal-valued explicit array remains per-edge. There is no second ID list or graph.
- Default summaries omit per-edge values and use `availability='unavailable'`, `reason='omitted_from_summary'`, retaining count and row-map reference. This is a deliberate loss of coefficient values, not zero stiffness. Detailed exports retain available values. Loading a summary and later exporting detailed data cannot recover them.

For example, with result edge IDs `[90, 3]`, `spring_k=[2, 7]` and legacy `stiffness=100000`, the simulation descriptor records `[2, 7]` in that order. Recomputing with `stiffness=[11, 13]` changes only the current-metric descriptor and metrics; the simulation descriptor remains `[2, 7]`. In a summary both independent per-edge value arrays are omitted. A scalar role remains available independently of the other role.

Consumers should first inspect this versioned record, then resolve a `current_metrics` simulation reference before reading its availability and scalar value or per-edge values. Historical files without the record remain readable, but their effective simulation coefficient is unknown: never infer it from `metadata['stiffness']`. That legacy field is intentionally unchanged and may describe an inactive argument. Old consumers that ignore the new record cannot be repaired retroactively.

Explicit metric recomputation of a legacy file records only the new metric model, with `simulation={'availability': 'unknown', 'reason': 'not_recorded'}`. For legacy results with neither ID map, existing insertion-order behavior is preserved without adding IDs. A scalar metric coefficient is still identifiable; a per-edge metric descriptor instead has `row_map=None`, `availability='unavailable'`, `reason='missing_edge_ids'`, and no values, because no persistent stable-ID pairing can be asserted.

Export/load reject unsupported versions, unknown fields or sources, invalid representations, nonfinite or negative coefficients, mismatched counts, and missing/duplicate required edge maps. Recompute validates a replacement descriptor before committing metrics or metadata. It does not automatically reuse a saved coefficient, infer one from Material, or modify trajectories, sampled dynamics state or energy ledgers. Existing state/energy consistency checks still apply: changing the metric energy does not rewrite the original sampled simulation energy and can therefore make that result ineligible for export.

Result/state IDs are strict unique integers and may be signed. Recomputing against an in-memory graph checks actual membership, not the sign of an ID. This does not broaden the separate existing nonnegative-ID rules for native `StructureGraph.from_dict` input or explicit FEM conversion maps. Signed result persistence and signed graph-JSON persistence are different contracts.

### Ownership and cost

The producer retains its already normalized, detached float64 per-edge buffer only after existing metric computation finishes. Caller-array mutations cannot change provenance. Scalar provenance retains no broadcast array. Initially the incremental retained coefficient storage is approximately `8E` bytes for per-edge input, or constant-size for a scalar. A later independent per-edge metric recomputation can require another `8E` bytes; these two different coefficient sets are intentional, not duplicate authorities. They share the same edge-ID map.

Summary validation scans available coefficients for validity but neither copies the normal float64 buffer nor converts it to a Python list. Its added serialized coefficient descriptor is constant-size per role. The whole historical summary is not constant-size: its existing edge IDs and other metadata remain. Detailed serialization creates one list/text payload per independent available coefficient array; detailed load restores detached float64 arrays. Existing metric work arrays and temporary validation masks/ID validation storage remain bounded `O(E)` and are not a claim of constant-memory validation.

## FEM resultants

Raw `solve_2d`, `solve_3d` and `solve_2d_nonlinear` dictionaries include the same `beam_resultant_contract` component metadata as the adapter. Nonlinear recovery retains its approximate qualification. Each contract is detached; changing the raw component contract does not change the converted component contract.

New conversions retain the existing numeric `metadata['beam_resultants']` array and add `metadata['beam_resultant_contract']` with schema version, dimension and ordered components:

- 2-D: signed axial `N` (force), shear `V` (force), and `(M_i + M_j)/2` (moment). The last column is a mean, not either endpoint moment.
- 3-D: axial `N` (force), shear `V_y` (force), shear `V_z` (force). The current producer recovers only axial `N`. The two retained shear slots are legacy placeholders explicitly marked unavailable, not measurements of zero shear.

`N` here is an axial-force symbol, not an assertion of newton units. Quantity kinds distinguish force from moment; no SI or other unit system is inferred from the graph. Existing nonlinear approximate status stays approximate, and existing field/solver status remains authoritative. This metadata does not recover missing forces or establish any contact-force channel.

A component status of `available` means that its column is provided or recovered under the recognized producer layout; it does not establish solver validity, numerical accuracy, equilibrium or convergence. Producer-authored `field_status` and `solver_status` remain separate: they are preserved when present and remain absent when absent, and an existing `approximate` qualification is not upgraded.

Legacy files lacking this component contract remain readable, including historical three-column FEM `edge_forces` exports. The loader retains that original array separately and extracts axial force as before; it does not invent dimension-specific availability for old records.

Generic field-prediction training accepts caller-supplied numeric targets; it does not ingest this contract or mask unavailable components. Callers must select and qualify their labels, rather than treating placeholder or unknown components as measurements. The existing `physical_dataset` labels are force-curve summaries, not these three resultant columns.
