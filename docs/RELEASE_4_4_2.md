# FiberNet 4.4.2

FiberNet 4.4.2 improves native graph editing, numerical geometry, input
admission and saved-analysis workflows. Python >=3.9 and the four core
dependencies remain unchanged. This release covers the changes since the
published 4.4.1 package, not just the latest graph metadata correction.

## Changes since 4.4.1

- `StructureGraph.to_dict()` now deep-copies graph, node and edge metadata.
  Editing a serialized design no longer edits the source graph through shared
  nested containers. Schema 3.1, stable IDs and geometry remain unchanged.
- Preserve per-member radius/length associations in cylinder volume and
  lateral-area features. Normalize finite-width contact raster geometry and
  widths together, restoring reported spacing and area to caller units.
- Apply scalar graph scaling consistently to geometry and radii for NumPy
  numeric scalars and zero-dimensional arrays. Apply positive Kagome lattice
  perturbations and their seed; recompute transformed-tiling boundary labels.
- Correct ionic bond forces to mean force on the first endpoint, consistent
  with spring and covalent crosslinks and the negative first-endpoint energy
  gradient.
- Add explicit per-edge isotropic material selection for linear beam FEM,
  validate beam radii and representable circular sections, and document small
  network FEM/result/export workflows. The native result's axial energy proxy
  is not a newly introduced full beam-energy recovery result.
- Reject malformed legacy VTK attributes before opening the output, nested
  masked learning geometry before batch construction, invalid native 3D cell
  dimensions before factory work, and unprofiled harmonic LAMMPS tasks whose
  mandatory trajectory rows already exceed the explicit output budget.
- Share electrical/thermal graph admission for stable identities and finite
  real coordinate provenance. Edge conductances and other model coefficients
  remain explicit inputs; admission does not infer them from materials/radii.
- Store the reported recruitment `threshold_curve` in float64 to match decision
  precision; recruitment decisions are unchanged. Add a versioned,
  pickle-free reduced-recruitment replay archive with explicit graph IDs,
  analysis settings and float64 energies.
- Require `manifold3d>=3.2.0` in the manufacturing/full extras, matching the
  native methods used by solid geometry workflows.

## Migration

- Use `graph.metadata`, node metadata or edge metadata to edit the live graph.
  Use `StructureGraph.from_dict(record)` to make a new graph from an edited
  export. Code that previously wrote back through exported metadata aliases
  must change. Python `copy.deepcopy` governs custom objects; non-copyable
  metadata can fail at export and non-JSON values are not made serializable.
  See [snapshot editing and compatibility](structure_dict_snapshots.md).
- Revalidate historical features/models affected by heterogeneous radii,
  contact unit scaling or boundary labels. Matching feature names do not make
  corrected numerical values interchangeable. See the
  [radius workflow](heterogeneous_radius_workflow.rst),
  [contact units](FINITE_WIDTH_CONTACT_SCALING.md) and
  [tiling boundary contract](architecture/transformed-tiling-boundaries.md).
- Kagome inputs with positive perturbation now produce displaced geometry.
  Ionic nonzero forces reverse sign relative to the old first-endpoint result;
  their energies and magnitudes are unchanged. See the
  [perturbation](architecture/lattice-perturbation-contract.md) and
  [crosslink force](architecture/crosslink-force-contract.md) contracts.
- NumPy scalar factors now resize radii as well as coordinates. Vector scaling
  remains coordinate-only. Specify units and material selection explicitly;
  the library does not automatically convert them. See
  [graph scaling](graph_scaling.rst) and the
  [beam FEM contract](METHODS_BEAM_FEM_CONTRACT.md).
- Inputs rejected by the stricter admission paths need correction rather than
  casting away masks, imaginary parts or invalid IDs. Transport graph float64
  rounding is distinct from the exact-representability contract of the separate
  prepare-only API. See [transport input semantics](integrations/LINEAR_TRANSPORT_USAGE.md).
- Existing original fields in older reduced-recruitment archives remain
  readable; the new fields require regeneration for full replay. The archive
  supports saved analysis, not resuming
  an integration checkpoint or guaranteeing cross-platform reproduction. See
  [saved replay](REDUCED_RECRUITMENT_REPLAY.md) and
  [recruitment thresholds](METHODS_TENSILE_RECRUITMENT.md).

The repository MCP adapter's exact source manifest is updated for this core
revision; the adapter and desktop application are separate distributions and
are not published by the core release. Stored source-bound artifacts may need
explicit revalidation after source changes. No optional provider execution or
physical validity is implied by graph admission, persistence or packaging.
Historical `Unreleased` changelog entries remain as development history;
this document defines the 4.4.2 release scope.
