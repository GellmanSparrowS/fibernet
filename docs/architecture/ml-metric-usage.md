# Point metrics for training and cross-validation

New train_model histories use history_schema_version 2 and
metric_definition "point_regression_uniform_output_v1". This definition uses the
shared regression scorer: uniform-average per-output R² and RMSE across all
entries. Exact constant targets score R²=1; inexact constants score 0. A
validation set with fewer than two rows has R²=None and valid RMSE.

## Declaring a prediction adapter

Both train_model and cross_validate_torch accept the keyword-only
metric_prediction argument. It is a callable from a raw output Tensor to a
point-prediction Tensor shaped (N,) or (N,K), preserving batch order, row count,
output order, and device. It runs under no_grad on a detached copy of the raw
output, before any scalar-axis normalization is reflected in that input.
Callable errors are chained to a clear ValueError; non-Tensor, nonfinite,
wrong-device, empty, complex/bool, wrong-rank, or incompatible shapes fail explicitly.
Batch and output ordering are the caller's responsibility and cannot be
inferred from tensor values.

The adapter is used only for scoring. The loss still receives the existing
model prediction and target inputs, including the established scalar-axis
normalization. It never receives the adapter result. Validation loss still
controls scheduling, early stopping, checkpoints, and best-state restoration.
CV applies the same adapter again when scoring each restored model.

For the built-in FiberUncertaintyNet, the documented layout is K means followed
by K log variances, and gaussian_nll_loss expects targets shaped (N,K):

    from fibernet.ml.models import FiberUncertaintyNet, gaussian_nll_loss
    from fibernet.ml.training import train_model, cross_validate_torch

    K = 2

    def predictive_means(raw):
        if raw.ndim != 2 or raw.shape[1] != 2 * K:
            raise ValueError("Expected K means followed by K log variances")
        return raw[:, :K]

    model = FiberUncertaintyNet(n_features=X_train.shape[1], n_outputs=K)
    history = train_model(
        model, X_train, y_train, X_val=X_val, y_val=y_val,
        loss_fn=gaussian_nll_loss, metric_prediction=predictive_means,
    )
    scores = cross_validate_torch(
        lambda: FiberUncertaintyNet(n_features=X.shape[1], n_outputs=K),
        X, y, loss_fn=gaussian_nll_loss,
        metric_prediction=predictive_means,
    )

This example declares the known model layout; no automatic 2K slicing, model
class detection, or MC-dropout inference is performed. Flat (N,) targets are
not the documented contract of this Gaussian loss.

## Availability and compatibility

Every newly generated train_model history has one r2 entry, rmse entry, and
val_metric_status object per completed epoch:

- Available: status="available", r2="available", rmse="available"
- Singleton: status="partial", r2="insufficient_samples", rmse="available"
- No validation: both metrics None, status="unavailable",
  reason/r2/rmse="validation_unavailable"
- Incompatible identity prediction: both None, status="unavailable",
  reason/r2/rmse="prediction_shape_mismatch", with prediction_shape and
  target_shape for the observed validation batch
- Nonfinite identity predictions or targets: both None, status="unavailable",
  reason/r2/rmse="nonfinite_prediction_or_target"

Unsupported identity prediction dtypes are marked unsupported_prediction_dtype.
Computed nonfinite scores (including arithmetic overflow from finite inputs) are
recorded as None with a per-metric nonfinite_metric reason and partial or
unavailable status. This recording-boundary treatment does not change the
shared scorer or CV's existing nonfinite-score semantics.

Only (N,) and (N,1) scalar axes are interchangeable. No NumPy broadcasting
defines a metric. An absent adapter allows valid custom-loss training to
continue with unavailable metrics. CV continues to reject incompatible
identity predictions because its result promises point scores. Explicit
adapter errors are never silently downgraded to unavailable metrics.

History scores describe each epoch before final best-state restoration.
The CV score describes the restored model; it need not equal the last epoch.
Console output labels missing R² as unavailable. The existing plot continues
to show loss and learning rate, with no invented metric series.

Metadata survives to_dict, JSON save/load, and model bundle round trips.
metric_prediction records only "identity" or "explicit_callable"; callable
code is never serialized or reconstructed. Generic TrainingHistory objects,
including VAE reconstruction/KL histories, remain unmarked. Legacy histories
load their original values without assigning the new definition retroactively.

This is not a broad strict-JSON migration: existing no-validation Infinity
best_val_loss, legacy nonfinite numbers, and CV singleton NaN semantics remain
unchanged. New singleton history R² is None, exported as JSON null.
