# CSV prediction workflows

Choose the entry point that matches the CSV rather than inferring input columns
from every numeric value. Both entry points train a model; neither merely loads
an existing predictor.

## Batch simulation CSV

`batch_simulate` writes simulation outputs and configuration columns, not `feat_`
columns. Use the explicit-column training entry point:

```python
from fibernet import train_model

metrics = train_model(
    "results.csv", target="max_force",
    features=["grid_x", "grid_y", "strain"],
)
```

Select inputs appropriate to the study. Do not include the target or other
simulation outputs as predictors simply because they are numeric.

## Prefixed-feature CSV

`predict_from_csv` selects varying columns beginning with `feat_` by default:

```python
from fibernet.ml.utils import predict_from_csv

result = predict_from_csv(
    "features.csv", target="max_force", model_type="ridge",
    output_dir="prediction_results",
)
print(result["importance_plot"])
```

For CSV exported by `FiberNetDataset.to_csv`, pass the exact target column, for
example `target="label_max_force"`. No target is inferred. The selected feature
prefix must exclude the target and all reserved `label_` columns. Target overlap
is rejected even when the target is constant. No matching varying features is
an error before training or output-directory creation; batch CSV callers should
use `train_model` above. Nonfinite
values become missing, missing-target rows are dropped, and selected missing
features are filled with zero.

## Saved outputs and importance

The return value contains the model, metrics, cross-validation, feature names,
target and sample count. The `importance_plot` dictionary is
also included in saved `metrics.json` when an output directory is requested:

- `not_requested`: no output directory was requested.
- `written`: this call successfully wrote `importance.png`.
- `skipped`: the estimator has no native importance vector or coefficients. Its
  `reason` explains that `importance.png` was not updated; any prior file stays
  untouched and must not be treated as output of this call.

RF and GB expose tree feature importance. Ridge exposes absolute coefficients;
with the built-in Ridge default those coefficients are in standardized feature
space, not original physical units, causal effects or physical sensitivities.
The pipeline axis says "model feature space": other StandardScaler settings may
center without scaling or leave the original values intact. Only the final
estimator is inspected. Arbitrary transforming
pipelines and multioutput coefficient matrices are not silently mapped to the
original feature names.

Default RBF SVR and MLP do not expose native importance. Their optional importance
plot is skipped, while model, metrics, prediction and residual outputs continue.
Calling `plot_feature_importance` directly for an unsupported estimator still
raises a clear error. Malformed importance vectors, plotting errors and write
errors propagate; they are not converted into successful skips.

Outputs are individual writes, not an atomic artifact bundle. A failed call can
leave partial outputs. Use a new output directory when separating runs, and
inspect `importance_plot` rather than assuming an old image belongs to the latest
run.

## Plot ownership

The four `fibernet.ml.utils` plot helpers return a Matplotlib `Figure`. With the
default `show=False`, this is a detached Agg-backed figure: edit its axes or call
`fig.savefig(...)` to export it. The helper does not switch your plotting backend,
register the new figure with pyplot, or change your current pyplot figure. This
replaces the earlier pyplot-registration behavior; later `plt.show()` will not
display a detached figure. Use `show=True` for pyplot-managed interactive display
with your selected backend. Returned figures are not cleared by the helper.
