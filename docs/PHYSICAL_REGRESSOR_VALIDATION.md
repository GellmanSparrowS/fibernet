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
