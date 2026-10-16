# Explicit source-group training with train_gnn

The existing `fibernet.ml.gnn.train_gnn` accepts `split='group', groups=...` for
CPU float32 scalar FiberGNN training. This is an opt-in mode of the same trainer,
not another training framework. It requires scikit-learn for GroupShuffleSplit;
the dependency is imported only on this path. There is no added mandatory backend.

```python
import torch
from fibernet.ml.gnn import FiberGNN, graphs_from_batch, train_gnn
from fibernet.ml.training import save_model_bundle

inputs = graphs_from_batch(training_snapshot, source_groups=source_groups)
torch.manual_seed(42)  # Before model construction; the trainer does not reseed Torch.
model = FiberGNN(node_dim=4, hidden=8, n_layers=1, n_outputs=1,
                 layer_type='gcn', dropout=0.2)
result = train_gnn(model, inputs, scalar_targets,
                   split='group', groups=source_groups,
                   val_split=0.25, epochs=20, batch_size=32,
                   early_stopping=5)
record = result['training_data']
save_model_bundle(model, 'own-model.pt',
                  feature_names=record['feature_names'], target_names=['property'],
                  metadata={'training_data': record})
```

`training_snapshot`, targets and groups must exclude any final scientific test
set. Group labels are explicit assertions of common origin, not certified
independence. Augmentations, load steps, remeshes and multiple fidelity results
from one source should stay in that source's group.

## Defaults and compatibility

New schema-tagged XYZ+dimension dictionaries require explicit `split='group'`
and aligned groups. Missing split, a row-mode request, mixed schemas, unknown
schemas or mismatched source_group labels fail before optimizer creation. Tagged
graph IDs must be unique within the training collection. Separately constructed
GraphBatches may reuse graph IDs: choose collection-unique IDs when creating those
snapshots. Graph ID uniqueness does not prove independent scientific origin.

Unmarked legacy dictionaries with `split=None, groups=None` keep the historical
fixed-seed row-permutation algorithm, numerical behavior and two-key return value.
They may opt into group mode with explicit groups, but the trainer records their
feature semantics as `caller-legacy` rather than guessing names. Providing groups
without explicit group mode is an error.

Both legacy and group modes retain the existing per-epoch `model.train()` call,
then evaluate validation rows under `model.eval()` and `no_grad()`. This mode
transition is unchanged. Regression hooks verify training/gradient state across
two epochs with nonzero dropout, including re-entry after validation.

Group splitting uses `GroupShuffleSplit(n_splits=1, test_size=val_split,
random_state=42)`. The fraction refers to groups, so realized row fractions may
differ. At least two distinct groups and nonempty train/validation sets are
required. No learned preprocessing is added; any external fitted scaler must fit
only training groups and its state must accompany the experiment.

## Validation is not a final test set

Optimizer steps use only train rows and targets. Validation rows are evaluated
without gradients, but their losses drive early stopping. They therefore influence
model selection and are not an untouched final test set. Evaluate final-test source
groups only after choosing the model/configuration, outside this training call.

`best_val_loss` is the minimum observed validation loss. The supplied model remains
the last trained state, not a restored best checkpoint. The metadata explicitly
records `best_state_restored=False`. With early_stopping=0 and a fixed full-run
seed, changing validation labels does not affect trained parameters; validation
history still changes. That invariance is not promised when early stopping is on.

## Supported inputs and preflight

The first group-mode contract accepts the exact supported FiberGNN class, finite
CPU float32 model parameters and node tensors, and scalar model output width 1.
Targets must be a finite real vector `(n_graphs,)`; boolean, complex, masked,
nonfinite, float32-overflow and nonzero-underflow targets are rejected. Tagged
inputs must have the four ordered names `(x, y, z, dimension)`, compatible model
width, consistent explicit units/frame and aligned source groups. Endpoints must
be graph-local int64 rows; GCN adjacency must be finite CPU float32 sparse COO.
These checks do not authenticate arbitrary caller-built dictionaries as snapshots.

Epochs and batch_size must be positive nonboolean integers; early_stopping a
nonnegative nonboolean integer; lr a finite positive real; and 0<val_split<1.
Multi-layer GAT must have a hidden width divisible by its positive head count.
These new-mode checks run before optimizer creation and parameter updates. They
do not promise transactional rollback for arbitrary failures during optimization.
The legacy constructor and unmarked row branch retain their previous behavior.

GCN, GAT and SAGE still do not consume edge feature values, masks or physical IDs
as learned inputs. XYZ is not automatically translation/rotation/scale invariant.
GAT's effective attention dropout remains 0.1 even when FiberGNN(dropout=0), as
explained in [adapter usage](gnn-graph-batch-usage.md). The new record captures
actual per-layer dropout and aggregation settings rather than inferring them from
a single constructor argument.

## Training record and checkpoint boundaries

Group-mode results add JSON-native `training_data` alongside history and
best_val_loss. It records input schema/names, IDs where declared, units/frame,
asserted groups, train/validation row indices and group sets, split seed and
fractions/counts, effective model configuration, training settings and epochs run.
It stores no whole graph snapshots or tensors. Caller-origin/provenance metadata
can accompany this record in the existing model bundle; the record itself is not
proof of measured labels or independent specimens.

The split seed does not seed model initialization or training dropout. Set and
record experiment seeds yourself. The trainer does not mutate the global RNG
seed on your behalf.

Use existing save_model_bundle/load_model_bundle with matching model constructor
settings and verify stored input schema/names before inference. Loader checks are
not automatic. Load only your own trusted same-runtime bundles; no cross-version
portability is established. Existing torch.save is non-atomic and does not promise
to preserve an old file on failure. Bundle format and trust behavior are unchanged.

Synthetic tests verify source-group isolation, every-epoch train/eval states,
train-target alignment, fixed-epoch validation-label perturbation invariance and
same-runtime checkpoint predictions. They establish integration behavior, not
predictive superiority or physical validation.
