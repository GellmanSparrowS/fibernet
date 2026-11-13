# FiberNet 4.4.1

FiberNet 4.4.1 improves input preservation, numerical geometry and graph
analysis. Python >=3.9 and the four core dependencies remain unchanged.

## Changes since 4.4.0

- Preserve input CSV files when saving trained models. Model destinations
  that identify the input entry, including symlink, case-insensitive and
  Unicode-normalization aliases, are rejected before training. Separately
  replaceable hardlink entries remain supported under the documented checks.
- Restore nonzero k-core values in `GraphFeatureExtractor3D` and preserve its
  60-feature layout. `kcore_frac` retains its maximum-core-number / mean-degree
  definition, including isolates, and can exceed one.
- Normalize explicit 3D rotation axes without overflowing or underflowing
  their norm. Reject malformed, complex, nonfinite and all-zero axes.
- Correct open-chain snapshot straightness and normalize planar pore geometry
  before geometric classification, reducing sensitivity to coordinate units.
- Stabilize LAMMPS geometry projection across numerical ranges; add stable
  edge-length histories and final-displacement comparisons against an explicit
  reference geometry.
- Validate stored graph curves before JSON writes and detach periodic cell
  inputs from caller-owned arrays.
- Restore persisted graphs and verify saved artifact bindings when resuming
  the tutorial pipeline. Add storage-only planning to the crosslinked CG example.
- Add the optional `image` dependency group for legacy OpenCV-backed features
  and include it in `full`. OpenCV remains outside the core dependencies.

## Migration

- Regenerate affected 3D k-core features and retrain and validate models before
  combining corrected values with historical all-zero data. Matching feature
  names and ordering do not make those numerical values interchangeable.
  See [k-core definitions and migration](architecture/graph-feature-3d-kcore.md).
- Recompute geometry previously rotated about finite axes whose direct norm
  overflowed or underflowed. Ordinary valid axes remain compatible within
  floating-point roundoff. The 2D axis argument is still ignored. See the
  [rotation-axis contract](architecture/rotation-axis-contract.md).
- Snapshot straightness and pore results can change where the prior behavior
  was incorrect or sensitive to units. Existing near-degenerate geometry and
  tolerance limits remain.
- Legacy tutorial checkpoints without required artifact bindings need a fresh
  data directory. Exact source-bound artifacts and MCP adapters must be
  regenerated or independently revalidated against the current source binding.
- Install `fibernet[image]==4.4.1` only when a legacy image feature needs OpenCV.
  OpenCV distributions share the `cv2` namespace; do not install multiple
  distributions into the same environment.

The LAMMPS additions perform geometry-only post-processing or storage planning.
They do not authenticate native results, qualify external providers, or validate
physical predictions. The core release does not publish the separate MCP adapter
or desktop application. Historical `Unreleased` changelog headings are retained
as development history; this document describes the 4.4.1 release scope.
