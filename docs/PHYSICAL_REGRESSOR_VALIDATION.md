# Explicit validation rows for physical regression

`PhysicalRegressor.train` keeps its historical random row holdout by default.
Related loading conditions, material variants, and other observations from one
source network can therefore occur in both training and validation. That split
may be appropriate for a within-network prediction question, but it does not
establish performance on unseen source networks.

First reserve genuinely untouched source networks as the final test set. For
the remaining development rows, assign all variants of each source network to
one partition and pass the chosen validation row indices explicitly:

```python
from fibernet.ml import PhysicalRegressor

# X_dev and Y_dev contain only development rows; each Y row has three targets.
source = ["network-A", "network-A", "network-B", "network-B", "network-C", "network-C"]
validation_rows = [i for i, group in enumerate(source) if group == "network-C"]
model = PhysicalRegressor(key="mlp", seed=0)
history = model.train(X_dev, Y_dev, validation_indices=validation_rows)
```

`validation_indices` accepts a list, tuple, or one-dimensional NumPy array of
distinct, in-range integer indices (masked arrays are rejected). The wrapper
still requires at least five total samples, with at least two rows in validation
and at least two in its complement. Validation order follows the supplied indices;
training rows retain input order. The stored index arrays are detached from the input.
Both model paths fit normalization on training rows only. The explicit rows
replace the random split; `val_frac` must still be strictly between 0 and 0.5. After a
successful fit, `model.train_indices` and `model.val_indices` describe this
partition.

These are caller-supplied row indices, not verified provenance. FiberNet does
not infer whether networks or variants are related. Validation informs model
selection and early stopping, so it is not an untouched final test set. Preserve
the prior exposure history and keep final-test networks out of all fitting and
model-selection decisions. A split alone does not establish scientific accuracy.

## Missing observations

Supply complete observed feature and target values to `train` and complete feature
values to `predict`. NumPy masked arrays, including masked rows inside a sequence,
raise `ValueError` before fitting or prediction. This includes wrappers whose masks are entirely false. First identify complete
rows from both masks, then convert only the selected complete rows to ordinary
arrays:

```python
complete = ~(np.ma.getmaskarray(X).any(axis=1) |
             np.ma.getmaskarray(Y).any(axis=1))
X_observed = np.asarray(X[complete], dtype=float)
Y_observed = np.asarray(Y[complete], dtype=float)
```

Use the same `complete` selector for group labels, then choose validation indices
within the filtered dataset. Nonfinite values still require separate handling.
A rejected refit leaves the previously fitted model available.

## Training array shapes

Pass features as `X` with shape `(N, D)` and labels as `Y` with shape `(N, 3)`.
This regressor fits three targets together; record their meanings and units in
one fixed column order. A two-column label table, a one-dimensional target
vector, or a transposed table reports the expected and received shapes before
creating or replacing a fitted backend. For named single-target regression,
use `fibernet.train_model` with an explicit dataset target and split.
