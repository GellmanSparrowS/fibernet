# Shared reference-fiber displacement profile

## Computational method

The deformation profile of a reference fiber is an ordered sequence of two-component offsets. The first component acts along the local fiber tangent and the second along its in-plane normal. Both are dimensionless fractions of the **individual fiber's initial length**. For n input control points, their parameter positions are uniformly spaced in the open interval (0, 1): s_{i} = (i + 1)/(n + 1), with zero-based i. A request for m interior points evaluates linear interpolation at t_{j} = (j + 1)/(m + 1). Values outside the input control-point interval use the nearest endpoint value, matching NumPy's interpolation rule. An empty profile yields zero offsets; a zero-point request yields an empty profile. The implementation accepts only finite two-column input and nonnegative output counts.

For a square unit cell, the resulting offsets are multiplied by the edge length L and transformed in the ordered sequence AB, BC, CD, DA. The successive transformations are identity, +90°, 180°, and −90° rotations in the reference plane. For general non-square fibers, the APP uses each edge's own tangent, normal, and length; that larger construction remains in the APP pending migration. The library module exposes the profile as a validated Python object as well as callable compatibility helpers, and the APP imports those helpers from the same source.

## Reproducibility and limits

The profile defines geometry, not material response. Its dimensionless scaling avoids applying the same absolute displacement to short and long fibers, but a large profile can still cause self-intersection or violate manufacturing clearances. Those constraints require separate validation. A fixed profile and edge orientation produce deterministic offsets; stochastic perturbations elsewhere in structure generation remain controlled by the factory seed. Four real-trajectory golden cases covering square, bow, chiral, and reentrant networks produced 40 arrays numerically identical to the pre-migration reference. This demonstrates refactor equivalence for those cases, not universal geometric validity. The numerical cache version is e2v9 because the shared profile implementation is now part of the source identity.

## Public entry point

The module is `fibernet.gen.spectrum` and contains the `FiberSpectrum` class. The APP retains its established profile function names as aliases to this module. The public class can be used without importing Qt or FiberScope; full manufacturable network compilation is not yet available from this module alone.
