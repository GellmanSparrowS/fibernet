# Optional CPU XGBoost scalar regression

`fibernet.train_model(..., model_type='xgb')` uses an optional XGBoost scalar
regressor through the existing easy API. It accepts CSV or a `FiberNetDataset`,
keeps explicit native row/group splitting, and saves the same version-1 artifact
envelope consumed by `predict_named`. No graph model or registry is added.

```python
from fibernet import train_model

metrics = train_model(
    dataset,
    target='max_axial_force_N',
    features=['length', 'radius'],  # Existing raw Dataset names, in desired order
    model_type='xgb',
    split='group',
    groups=reviewed_source_groups,
    test_size=.25,
    metadata_keys=['origin'],
    save_path='model.pkl',
)
```

Names/metadata keys must actually exist in the input. Caller-supplied groups must
represent independent sources; unique sample IDs are not evidence of independence.
See [native Dataset training](native-dataset-training-usage.md) for existing
selected-column, source-assertion and holdout rules. CSV retains its existing row
holdout behavior; use the native Dataset interface for explicit group holdout.

## Optional dependency and bounded defaults

The package is imported only when `model_type='xgb'` is selected. If unavailable,
the call raises an actionable ImportError, never substitutes another model. Base
FiberNet imports and LR/RF/GB paths do not require XGBoost. No base dependency or
base Python requirement was changed.

The validated pilot uses **xgboost-cpu 3.4.1 / Python 3.12.14 / sklearn 1.8.0**,
NumPy 2.3.5 and SciPy 1.17.0 on Linux x86_64. This wheel declares Python >=3.12;
it does not provide evidence of XGBoost support on every Python version supported
by base FiberNet. Install a compatible optional distribution in your environment;
other versions/platforms need their own validation. The
[official CPU package](https://pypi.org/project/xgboost-cpu/3.4.1/) is Apache-2.0.
No install or dependency upgrade occurs inside `train_model`.

Fixed initial defaults are 32 trees, maximum depth 3, learning rate .1,
`reg:squarederror`, histogram trees, `device='cpu'`, `n_jobs=1`, seed 42 and
verbosity 0. This slice adds no hyperparameter tuning surface. There is no learned
scaler/imputer/preprocessor; fitting consumes only training rows. Heldout data is
not passed to early stopping or `eval_set`. Only finite scalar `(N,)` prediction
vectors are scored; masked arrays, nonfinite values and other axes reject before
saving. LR/RF/GB validation and estimator configurations remain unchanged.

## Named inference and persistence limits

The saved dictionary still has exactly `model`, `features`, `metrics`,
`artifact_version`, and `target`. The estimator is an XGBRegressor, and the stored
feature order remains authoritative. Load only your own/trusted pickle and pass
the already-loaded dictionary to `fibernet.predict_named`; mapping key order may
differ, but missing columns and unexpected columns reject under the existing
policy. No automatic deserialization is added.

Publication uses the existing [owned-stage artifact writer](model-artifact-persistence-usage.md).
This is a **same-tested-environment pickle snapshot**, not a durable cross-version
XGBoost model format. The official [Model IO guide](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)
warns that pickle snapshots are version-sensitive; native JSON/UBJSON export plus
FiberNet column metadata would require a separate export contract. Existing
LR/RF/GB artifact formats are untouched.

## Evidence and boundaries

`tests/test_easy_xgboost.py` uses an explicitly synthetic 24-row/8-group fixture,
with selected columns in reversed order, and exercises an actual CPU fit in a
30-second subprocess with a 1-GiB address-space limit on Linux. It checks:

- disjoint source groups and a spy confirming only stored training rows/targets
  reach the real estimator, without holdout eval/early-stopping arguments
- source Dataset immutability and retained synthetic-origin assertions
- same-runtime save/load/named-order prediction equality, independent training
  equality and heldout-label perturbation invariance
- independently recomputed R²/RMSE, missing/extra named-column rejection, CSV fit
- explicit missing-package failure, invalid prediction rejection before file
  replacement, and legacy paths that never import XGBoost

The real-backend test skips when the optional package is absent and the resource
wrapper is currently Linux-only. A skip is not a successful fit. Synthetic
interface checks establish neither physical validity nor predictive superiority.
