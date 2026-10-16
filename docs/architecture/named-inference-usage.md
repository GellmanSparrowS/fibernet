# Named inference from trusted training artifacts

`fibernet.predict_named` is the exact callable exported from `fibernet.easy`.
Pass an already-loaded trusted artifact dictionary. It does not accept paths,
pickle bytes or arrays and does not deserialize anything. Never load untrusted
pickles: deserialization can execute arbitrary code before this API is called.

```python
import fibernet as fn

# artifact is a trusted dictionary previously saved by fn.train_model.
result = fn.predict_named(artifact, {"feat_a": 2.0, "feat_b": 3.0})
# For a full source table containing IDs, labels or metadata:
result = fn.predict_named(artifact, table, extra_columns="ignore")
```

Input may be a pandas DataFrame, one mapping, or a nonempty list/tuple of
mappings. Pandas is optional for record input and imported lazily for tables.
The artifact's nonempty unique ordered `features` list is authoritative. Source
column/key order can differ. Row order is retained. Every row must contain all
selected features. Names must be nonempty strings and unique, including names
of ignored extras. DataFrame duplicates are rejected. A normal dict has already
collapsed equal keys before the wrapper sees it; overwritten keys cannot be
reconstructed.

Extra columns reject by default. Explicit `extra_columns="ignore"` permits
heterogeneous extra keys across records, and does not numeric-validate their
values. Selected values must be finite real numeric scalars; Python/NumPy booleans
retain binary compatibility. Valid pandas nullable numeric/boolean values work,
but selected missing/null/NaN/infinite/complex/text values reject. No text coercion,
normalization, imputation or row dropping occurs. Inputs and stored names are not
mutated. Invalid inputs reject before the estimator is called.

The estimator receives one ordered numeric matrix, once. Its output must be a
finite real vector of exact shape `(n_rows,)`; scalar, column-vector and multitarget
outputs reject rather than being squeezed. An estimator MaskedArray with any
active mask rejects before conversion, so masked values cannot silently become
predictions. An entirely unmasked MaskedArray is accepted if its shape and values
otherwise meet the same contract. Selected input masked values reject before
prediction, like other missing values. The JSON-ready result contains:

- `predictions`: one list of numbers in input row order
- `target`: the stored name, or `None` for legacy artifacts without one
- `features`: the actual ordered feature names, copied from the artifact
- `validation`: `"named_columns_only"`

## Envelope compatibility

New `train_model` saves add `artifact_version: 1` and the actual training target
column to the existing `model`, `features`, `metrics` dictionary. Version 1
requires a nonempty target name distinct from features. Unknown or invalid
versions, including bool and float versions, reject. Unversioned legacy trusted
three-key dictionaries work and report unknown target; a valid declared legacy
target is retained. Invalid declared target names reject, never get guessed from
metrics or changed to `max_force`.

Existing pickle/sklearn callers can still read new saves, but bytes and exact
key sets change. Training selection, fitting, split, prediction and metrics are
unchanged by this envelope addition. Exact-key compatibility tests are updated
explicitly, with their originals retained as baseline evidence.

This validates column names only, not scientific meaning, units, provenance,
physical/feature schemas, model fitness or calibration. Version 1 labels the
artifact envelope only. Bare estimators remain available for callers accepting
responsibility for unnamed matrices, including equal-width wrong-order inputs.
The [reference journey](reference-research-journey.md) exercises both boundaries.
