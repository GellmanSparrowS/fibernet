# Train directly from a FiberNetDataset

`fibernet.train_model` (the easy sklearn API) accepts either an existing CSV path
or an explicit `FiberNetDataset`. Native calls preserve raw feature and target
names; they do not export/reparse a temporary CSV. The separate Torch training
loop at `fibernet.ml.training.train_model` is unchanged.

## Minimal native call

```python
from fibernet import train_model
from fibernet.ml.dataset import FiberNetDataset

dataset = FiberNetDataset.load("my-dataset")
metrics = train_model(
    dataset,
    target="max_axial_force_N",  # Exact existing raw target name
    model_type="lr",            # lr, rf, or gb
    test_size=.3,
    split="row",               # Explicit row holdout, not origin validation
    save_path="new-model.pkl",
)
```

The first argument retains the historical keyword `csv_path`, so existing
`train_model(csv_path="results.csv", ...)` calls still work. Native calls may
also use `csv_path=dataset`; there is no `dataset=` alias. A directory string is
not automatically loaded as a Dataset. Load it explicitly first.

`features=None` uses `dataset.feature_names` in its stored order. An explicit
nonempty list selects raw names in the requested order. Do not add `feat_` or
`label_`. Target defaults to literal `max_force`; if that name is absent, pass
the real target explicitly. The function never guesses the first target.

Only selected features and the selected target are validated. Missing values,
text, complex numbers, masked values and nonfinite selected numbers fail before
splitting/fitting. Unselected unavailable labels and opaque metadata/params do
not become numerical inputs. Selected features cannot have any declared target
name. Native namespaces, rather than CSV prefixes, identify features/labels.
Declared feature/target lists must contain unique nonempty string names. Sample
IDs must be nonempty strings at this training boundary; duplicate IDs are kept
with their distinct original row indices. Dataset persistence is unchanged.

## Choose the dependency unit deliberately

Native calls require `split="row"` or `split="group"`. There is no implicit
native row-split default.

- Row mode shuffles source rows with seed 42. Related runs of the same network
  can land in both partitions; unique sample IDs do not prevent that leakage
- Group mode holds out entire caller-declared groups with seed 42. It checks
  labels, row alignment and train/test disjointness; it cannot verify that the
  supplied labels describe genuine independent networks

```python
# origin_ids must come from your reviewed lineage and match dataset.samples.
metrics = train_model(
    dataset, target="max_axial_force_N", model_type="rf",
    split="group", groups=origin_ids, test_size=.25,
)
```

`groups` must be a list/tuple of nonempty strings, one per row, with at least two
distinct labels. Row mode rejects supplied groups rather than ignoring them.
In group mode `test_size` is a fraction of groups, not rows; unequal-size groups
produce a different held-out row fraction. Both native modes require a finite
fraction strictly between 0 and 1, at least one training row and at least two
test rows because the result includes R². Unsplittable cases fail before fit.

Grouping is never inferred from `sample_id`, unit/grid, a params hash, exact
feature equality or a graph fingerprint. In the retained ten-square reference,
unit/grid gives only one group, while IDs and parameter pairs are all distinct.
None provides an automatic independent-origin holdout. Its example deliberately
uses a row split for an API check, not a scientific generalization claim.

Explicit groups do not prevent every form of leakage: target-derived features,
repeated tuning on the same holdout, or incorrectly declared origins remain the
caller's responsibility. Keep the original Dataset and its generation lineage.

## Preserve selected source assertions

Native results always retain source row indices, sample IDs, actual train/test
indices, method and seed. Group results also retain the aligned group vector.
They do not automatically copy every parameter or metadata object.

```python
metrics = train_model(
    dataset, target="max_axial_force_N", model_type="lr",
    split="row", test_size=.3,
    metadata_keys=["length_unit", "target_unit", "target_meaning"],
)
provenance = metrics["training_data"]
```

Every selected metadata key must exist in every row. Selected values are
snapshotted recursively without coercion and must contain only JSON-native
null, bool, string, int, finite float, list, or string-keyed dict values.
NumPy scalars/arrays, tuples, sets, custom objects, nonfinite floats and cycles
are rejected before fitting or saving. Shared noncyclic structures are allowed
and copied. Later caller mutation does not change the returned snapshot.
Unselected metadata is not traversed. `None` or `[]` selects no metadata.

Null is valid in retained availability metadata, but is not a valid selected
numerical feature/target. For the reference, the three keys above preserve “m”,
“N” and the maximum-member-force interpretation. Optional mechanics declarations
such as E, nu or loading conditions must be selected explicitly when needed.

These are preserved source assertions, not verified units or physics. There is
no inferred feature-unit schema, metadata normalization, or `verified` flag.
A Dataset has no durable source locator; the artifact does not invent one.
Unselected metadata, params and full numerical data remain only in the source
Dataset. Saving only a model does not archive those omitted values. Relative
result paths still need their original external base directory.

## Result and trusted saved artifact

Native metrics contain the existing `r2`, `rmse`, `n_samples`, `features`, plus
raw `target` and `training_data`:

```python
{
    "source_kind": "FiberNetDataset",
    "rows": [
        {"row_index": 0, "sample_id": "square-00",
         "metadata": {"length_unit": "m", "target_unit": "N",
                      "target_meaning": "maximum absolute member axial force, not summed grip reaction"}},
        # Remaining source rows, in their original order
    ],
    "metadata_keys": ["length_unit", "target_unit", "target_meaning"],
    "split": {
        "method": "row", "random_state": 42,
        "train_rows": [0, 7, 2, 9, 4, 3, 6], "test_rows": [8, 1, 5],
    },
}
```

The `metadata` row key is absent when no metadata is selected. Group mode adds
`split["groups"]`. This record has O(rows) size; it is not a complete archived
dataset or proof of identity after the original data changes.

A saved pickle keeps the existing v1 envelope:
`model`, `features`, `metrics`, `artifact_version`, `target`. Its `metrics` equals
the returned dictionary, including native provenance. The model remains the
bare sklearn estimator. No automatic serialization security or physical-schema
checking is introduced. Load only a trusted artifact you created or otherwise
have reason to trust; never load an untrusted pickle.

Both native and CSV training now write and close a complete sibling pickle
before replacing the destination. Ordinary serialization, write, or replacement
failures preserve the previous file bytes or its absence. Existing targets must
be regular files without symlinks or special permission bits, and the parent
directory must already exist. This single-file replacement does not guarantee
crash durability, concurrent-writer safety, or preservation of inode ownership
and ACLs. See [model artifact persistence](model-artifact-persistence-usage.md)
for ordinary permissions, cleanup warnings, and the complete limits.

```python
# artifact is an already-loaded, trusted dictionary.
from fibernet import predict_named
rows = [{name: sample.features[name] for name in artifact["features"]}
        for sample in dataset.samples]
result = predict_named(artifact, rows)
identified_predictions = [
    {"row_index": i, "sample_id": sample.sample_id, "prediction": prediction}
    for i, (sample, prediction) in enumerate(zip(dataset.samples, result["predictions"]))
]
```

`predict_named` is unchanged: validation is `named_columns_only`; it does not
consume a Dataset, validate units or propagate IDs. Preserve the same row order
when joining predictions to samples. Predictions on training-source rows are
not an independent evaluation. Bare `.predict(ndarray)` still cannot detect
semantically reordered equal-width columns.

## CSV and optional dependencies

CSV calls keep their original default split, four metrics keys, v1 envelope,
model settings and fitted predictions. Complete Dataset CSVs still use prefixed
`feat_`/`label_` names. No prefix translation or native provenance is retrofitted
into old artifacts. Non-default native-only `split`, `groups` or
`metadata_keys` options on CSV calls are rejected.

Native fitting needs scikit-learn, with pandas confined to the CSV path. Native
mapping-based inference needs no pandas. Torch is neither imported nor needed
for this API. The existing `ml` installation extra still includes pandas and
sklearn; packaging has not changed. Fresh-process tests block pandas/Torch
imports across LR/RF/GB native fitting and named inference.

See the [reference research journey](reference-research-journey.md) for the
native end-to-end example. Row splitting tests held-out rows; meaningful origin
generalization requires valid caller-supplied independent-origin groups.
Historical saved journeys and their prefixed model artifacts are unchanged.
