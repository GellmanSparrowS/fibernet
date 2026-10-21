# Shared physical-label dataset stream

The existing `fibernet` library now owns the bounded, atomically checkpointed dataset stream used by FiberScope. The APP supplies its existing structure factory, feature and label callbacks; a public `PlanarPhysicalDataset` adapter supplies only library components. In both cases, the stream stores a 14-column geometry descriptor matrix, three target columns, per-sample generation specifications, the next seed and the most recent failure records in a compressed NPZ. A checkpoint is loaded only when its serialized configuration exactly matches the requested one. Temporary files are created beside the target and replaced atomically after each accepted or rejected attempt, so an interruption does not leave a partially overwritten committed dataset.

For the public planar adapter, each candidate samples five two-dimensional reference-line displacements uniformly from a caller-specified amplitude interval and an independent perturbation within its specified range. The candidate is built with the shared planar manufacturing graph constructor. Graph size is bounded by configurable node and edge limits before features or physics are evaluated. The 14 descriptors match the APP's shared physical-surrogate schema. In physical mode, the shared reduced stretch solver evaluates 24 load increments and 1500 time steps up to a stretch ratio of 1.8. Labels are the maximum raw reaction, a linear slope fitted over 5–40% of the simulated stretch span, and the trapezoidal integral of raw reaction over stretch. In generate mode the three labels are explicitly NaN; in surrogate mode they come from a supplied fitted model and a new path is required for each model state. The acquisition selector can choose among bounded candidates using the library's shared rules; candidate pools are capped at 128 and requests at 2000 samples per call.

The APP still preserves its historical serialized configuration fields and can resume a checkpoint written before the migration: a five-sample file was reopened by the new shared core, extended to seven samples, and its complete arrays and specifications exactly matched a fresh seven-sample run with the same seed. A same-seed square physical label from the public library adapter matched the APP's features and all three labels with zero numerical difference. These checks establish local implementation parity, not validity of the reduced solver as a predictor of measured material properties. Randomly split predictions from such data can leak network-family similarity; topology-, geometry- and experiment-level holdouts are required for scientific claims. Source hashes and schema identifiers protect resume compatibility but do not replace versioned external data and calibration.

## Row and checkpoint integrity

Each accepted row owns a finite `(14,)` feature vector, a `(3,)` label vector,
and a JSON-serializable generation specification. Physics and surrogate labels
must be finite; generate-mode labels remain explicitly NaN. Callback outputs
are staged and copied before any row is appended, so a rejected specification
or reused callback buffer cannot corrupt earlier rows. Checkpoint loading checks
matrix shapes, finite features, mode-specific labels, and matching specification
counts before exposing the saved arrays. Existing valid checkpoint formats are
retained; invalid historical checkpoints are rejected rather than silently
reshaped or treated as training data.

The planar adapter also binds checkpoints to its implementation source hash.
These integrity and graph-indexing changes therefore require a new cache path
for checkpoints created by an earlier implementation. Do not remove the hash
check to mix old and new labels. This does not change the NPZ schema of the
callback-driven stream or the intentional generate-mode NaN convention.

New checkpoints also record the effective acquisition pool size. Changing it
for an active sampler is rejected because it changes seed consumption and
candidate selection. Random sampling always uses one candidate, so changing an
unused pool-size argument does not invalidate its checkpoint. Legacy callback
stream checkpoints without this metadata still load, with a warning for active
sampling that sequence compatibility cannot be verified. The requested size is
recorded on the next save; the loader cannot infer a missing historical value.
