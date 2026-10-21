# Native dataset generation: sampling v2 and exact-run resume

`FiberNetDataset.generate` now forms the ordered Cartesian product
`units × grid_range × n_internal_range × radius_range`, explicitly passes each
radius to the generator, and uses the same base seed for every radius. The method
does not have `samples_per_config`. Multiple radii are paired geometry requests,
not an implicit repeated-realization count. Seed must be an integer in
`[0, 2**32-1]` and is no longer reduced modulo 10000. For built-in square graphs,
actual generated edges and extracted radius features are checked with engine
sentinels; this does not validate simulated material response.

This deliberately corrects historical behavior: radius VALUES were ignored and
only list length selected successive seeds. Old multi-radius results are not
numerically equivalent to v2. Old datasets remain explicitly readable; their
sample IDs are not relabeled. New IDs begin `v2-` followed by full SHA256 of the
canonical sample specification including box, radius, seed and sampling version.
A sample ID is a locator, not a standalone experiment identity.

## Sampling controls

Omitted ranges retain documented defaults. An explicit empty range means zero
jobs and produces a bound empty dataset without constructing the engine. Duplicate
full specs (including forced ID collisions) reject before generation. Grids are
positive integer pairs; internal counts are nonnegative integers; box lengths and
radii are finite positive numbers. Booleans are not numeric controls. An optional
positive integer `max_samples` applies the existing RandomState(seed) shuffle then
truncation to the complete plan; even an oversized cap is part of the binding.
The entire pre-truncation declaration and selected ordered plan are bound.

## Checkpoint contract

`checkpoint.json` has schema `fibernet.generate-checkpoint/1`, a full run binding,
its SHA256 and a sample payload digest. The run binding includes the sampling
policy, ordered plan, declarations, explicit solver controls, inherited solver
policy, source hashes for generation/tiling/transforms/graph/material/features/
labels/solver, and Python/NumPy/SciPy/NetworkX/Taichi versions. Whole-file hashes
are conservative: even a source edit with no numeric effect can require a new
save directory. Missing optional-package metadata is recorded as `not-installed`.
These hashes are consistency checks, not signatures or protection from malicious
in-process changes, concurrent mutation or tampering by someone able to rewrite
both payload and digests.

A resume must match exactly. Changing box, radii, order, maximum sample count,
solver settings, implementation or runtime identity rejects before engine
construction and writes. There is no append/expanded-plan resume policy in v1.
`verbose` and `checkpoint_every` may change without changing the experiment;
checkpoint_every must remain a positive integer. Stiffness retains the backend's
finite nonnegative domain. Damping and target stretch are checked for finite scalar
encoding only; binding does not add new physical ranges or validate unusual values. Completed and empty matching
checkpoints return without constructing the engine or rewriting saved files.
Partial resumes retain accepted samples and compute only missing jobs. A failed
job is not an accepted row: it is retried on a later matching resume. Nonfinite
features/labels cannot enter a new accepted row. This differs from batch CSV
failure-row handling and should not be inferred from that API.

Legacy/unknown checkpoints, corrupt payloads, contradictory sample params,
duplicate IDs or existing native Dataset files without a checkpoint fail closed.
Use a new directory for a new experiment. Existing datasets can still be loaded
explicitly; do not backfill old checkpoints with today's configuration. This also
means a matching completed checkpoint does not repair missing/corrupt native
Dataset output files automatically: inspect/load those artifacts separately.

## Extension and saving boundary

With any `save_dir`, actual registered factory objects must match the known
built-in functions. A public custom registration cannot be certified by static
package source hashes and is rejected before generating a new checkpoint or
reusing one. Fresh calls with `save_dir=None` retain custom factory support, with
no checkpoint-resume assurance. There is no caller-identity override API.
Arbitrary internal monkeypatching/hot reload is outside the binding contract.

Native Dataset's existing three-file save/load and rollback implementation is
unchanged. Checkpoint writes use a unique private sibling staging file; a preexisting
`checkpoint.json.tmp` sidecar is untouched. Existing ordinary checkpoint mode is
copied only to the stage; symlink/nonregular/special-mode checkpoint targets are
rejected. Cleanup cannot mask a publication error or turn a committed checkpoint
into failure. This does not make checkpoint plus native Dataset publication one
transaction, and adds no concurrent-writer, hostile-directory, ACL/ownership or
power-loss durability guarantee.

Tests use actual tiny graph generation, actual feature extraction/persistence,
and a synthetic engine only. They test contract/dispatch safety, not Taichi or
physical convergence.
