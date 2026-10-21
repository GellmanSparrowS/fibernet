# Safe training-column selection

`fibernet.train_model` validates selected CSV inputs before splitting data or fitting
an estimator. The convenience API supports `lr`, `rf`, and `gb` with the same
split, random seed, estimator settings, metrics, and saved pickle structure as
before. These checks protect the column-selection boundary; they are not a
scientific model-validation procedure.

## Dataset exports

```python
import fibernet as fn

dataset.to_csv("dataset.csv")
metrics = fn.train_model("dataset.csv", target="label_max_axial_force_N")
```

With `features=None`, automatic selection is allowed only when the CSV contains
all four structural headers (`sample_id`, `unit`, `grid_x`, `grid_y`), at least
one `feat_` column and one `label_` column with nonempty suffixes, no columns
outside those namespaces and structural headers, and the requested target is a
`label_` column. Header order is unrestricted. Only `feat_` columns are selected,
in CSV order. IDs, grid metadata, and every measured `label_` are excluded.
This recognizes a naming convention, not file provenance, units, or valid
scientific inputs.

The default target remains `max_force`. A Dataset export usually does not have
that name, so pass the actual `label_...` target explicitly. A missing-target
error lists available label names. The API never guesses a target, even when
only one label exists.

## Arbitrary CSVs and batch simulation outputs

```python
metrics = fn.train_model(
    "results.csv",
    target="max_force",
    features=["grid_x", "grid_y", "n_nodes", "n_edges", "strain"],
    model_type="rf",
    save_path="model.pkl",
)
```

This is an intentional compatibility change: arbitrary and batch-shaped CSVs
now require explicit `features=[...]`. An old `train_model("results.csv")` call
no longer guesses inputs from other columns. Numeric solver results such as
energy or displacement can leak outcomes; there is no numeric-column fallback
or supposedly safe automatic blacklist. The example's feature list is a caller
choice for that exporter, not a general guarantee that those features are safe.
Choose predictors available before the outcome and available at inference.

Explicit features must be a nonempty list of unique, nonempty string column
names that exist in the CSV. Their order is preserved exactly. The target and
**all** `label_` columns are forbidden as features, including labels other than
the chosen target and labels in arbitrary CSVs. Explicit selection does not
bypass this reserved namespace. Chained-label workflows are outside this
convenience API. Other explicit numeric names, including metadata identifiers,
are not automatically banned; the caller owns their scientific suitability.

## Rejected input

- Raw duplicate headers, even in unselected columns, before pandas can rename
  them; empty or whitespace-only header names
- Short or long logical records, with the record number reported; malformed
  CSV quoting. Quoted commas and newlines are supported, and blank physical
  lines are skipped (a quoted empty field is a real record)
- Missing or empty targets and invalid feature lists, including bare strings,
  empty lists, repeated names, missing names, target names, and `label_` names
- Selected nonnumeric or non-real data, missing values, and positive or negative
  infinity, in any row, including rows that would be held out

Headers use ordinary comma-separated CSV quoting and UTF-8 with optional BOM;
initial blank lines are skipped. Numeric strings parsed as numeric by pandas
are valid. No invalid strings are coerced to zero, rows dropped, or missing
values filled. Unselected text and missing metadata values do not prevent
valid explicit training. Raw header validation applies to the entire file's
header, regardless of selection.

## Artifact and inference responsibilities

The optional pickle remains `{model, features, metrics}`. `model` is the bare
sklearn estimator and accepts `model.predict(X)` with an ndarray. The saved
feature list remains the authoritative order for constructing prediction inputs:

```python
X = prediction_frame[artifact["features"]].to_numpy()
predictions = artifact["model"].predict(X)
```

Equal-width reordered arrays are still accepted by sklearn and can yield wrong
predictions. This change does not introduce an inference wrapper, unit/schema
validation, dataset migration, or new artifact contract. Callers still own
feature provenance, confounding, appropriate train/test splits, units, and
availability at inference.

`FiberNetDataset` may already have serialized a missing feature or label as zero.
Once that zero appears as an ordinary finite numeric value in the CSV, this
boundary cannot distinguish it from a genuine measurement. Fix or validate the
upstream dataset; this change deliberately does not alter `dataset.py` or claim
to detect already-filled missing values.
