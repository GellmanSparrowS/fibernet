# Small-dense combinatorial spectra, schema 1

```python
from fibernet.analysis import analyze_spectrum
result = analyze_spectrum(graph, adjacency="physical_multiplicity")
assert result.schema == "fibernet.spectrum/1"
```

`LaplacianSpectrumV1` is a frozen typed result for a finite, loop-free,
undirected `StructureGraph`. This is a dimensionless graph Laplacian spectrum,
not a stiffness spectrum, mechanical-stability certificate or periodic band
structure. Geometry, radius, material, arbitrary metadata weights, boundary
flags and box sizes do not enter the calculation. Geometric/material validity
is neither checked nor certified by this topology-only calculation.

## Matrix definition and identity

The combinatorial Laplacian is `L = D - A`. Explicit `adjacency` policies are:

- `physical_multiplicity` (default): each physical edge contributes one unit of
  adjacency and one unit to each endpoint's degree. Parallel edges accumulate
- `unique`: each distinct unordered endpoint pair contributes exactly one unit

There are no implicit edge weights and no normalization by degree. Self-loops
are rejected. Node keys/record IDs, edge keys/record IDs and endpoints must be
nonnegative integers, excluding bool and float; NumPy integer scalars are
accepted and output IDs are Python integers. Endpoints must exist.

`node_ids` is sorted and defines Laplacian row order. `edge_ids` preserves all
physical edges in insertion order even in unique mode. The graph is not mutated.
Eigenvalues describe the whole graph; they are not assigned to particular nodes
or edges. Eigenvectors are used internally for diagnostics and are not returned.

## Returned values and undefined cases

`raw_eigenvalues` is the full sorted NumPy `eigh` output, with small signed
roundoff retained. Values are not clipped and there is no threshold-based
numerical zero count. `n_components` and `topological_zero_count` both report
exact connectivity counts; the latter is the theoretical Laplacian nullity,
not an eigensolver classification.

- Empty graph: spectrum `()`, components/nullity zero, `lambda_max=None`,
  `fiedler_value=None`, solver `analytic_empty`
- Single isolated node: spectrum `(0.0,)`, components/nullity one,
  `lambda_max=0.0`, `fiedler_value=None`, solver `analytic_singleton`
- Two or more nodes: full `numpy.linalg.eigh` solve. `lambda_max` is the largest
  raw value, also for disconnected graphs. For a disconnected graph,
  `fiedler_value=0.0` is an exact topological fact. For a connected graph it is
  the second-smallest raw eigenvalue, subject to the acceptance checks below

The analytic empty/singleton branches report zero norms and tolerances because
no numerical eigensolve was performed. All outputs, norms and tolerances are
dimensionless. Physical coordinate scaling does not rescale this spectrum;
repeating every edge k times rescales the physical-multiplicity spectrum by k
but leaves the unique-adjacency spectrum unchanged.

## Numerical acceptance policy

Let `eps` be float64 machine epsilon, `N` the number of nodes, and
`laplacian_inf_norm = ||L||∞`, the maximum absolute row sum. With original raw
eigenpairs `(w, V)` (sorted together), returned diagnostics are:

- `residual_inf_norm = ||L V - V diag(w)||∞`
- `residual_tolerance = 64 eps N max(1, ||L||∞)`
- `orthogonality_inf_norm = ||Vᵀ V - I||∞`
- `orthogonality_tolerance = 64 eps N`

These are explicit engineering acceptance tolerances, not rigorous mathematical
error bounds or a certificate of relative accuracy for each small eigenvalue.
No eigenvalue is modified to satisfy them. Masked solver arrays are rejected
before conversion, including arrays with all-false masks. Wrong output shape, non-real/nonfinite
solver data, a raw eigenvalue below `-residual_tolerance`, or either diagnostic
above its tolerance raises `ArithmeticError`. A connected graph with a
nonpositive numerical second eigenvalue also raises `ArithmeticError` instead
of reporting spurious disconnection. `numpy.linalg.LinAlgError` propagates.
Failures are never silently replaced by zero or an empty spectrum.

## Resource scope

This version intentionally implements only bounded small-dense analysis:

- Defaults: `max_nodes=256`, `max_edges=20000`, `max_matrix_bytes=4194304`
- Absolute limits: 512 nodes and 20000 physical edges, even when caller budgets
  are larger. Budgets must be positive integers, not bool or float
- Planned dense array workspace is conservatively estimated as eight N-by-N
  float64 arrays (`64 N²` bytes). The matrix budget checks this estimate before
  dense allocation. It is not a bound on process RSS, BLAS/LAPACK workspace or
  Python/NetworkX overhead
- Exceeding configured or absolute limits raises `MemoryError`. Invalid inputs
  raise `ValueError`, or `TypeError` for a non-StructureGraph

Full eigenvectors are needed internally, and dense solve cost grows cubically.
Successful tests on analytic graphs, including a 512-node path, do not establish
large sparse graph support. Sparse/partial spectra need a separate contract.

## Compatibility

Legacy 94/60-dimensional extractors, geometry/path metrics, snapshot features,
GraphBatch, tutorials and model schemas are unchanged. New training columns
would require an explicitly versioned feature schema and retraining; do not
silently replace values expected by existing artifacts. This API does not
claim that its graph eigenvalues predict mechanical strength.
