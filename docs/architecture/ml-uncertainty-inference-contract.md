# MC-dropout inference contract

Implemented boundary: `FiberUncertaintyNet.predict_with_uncertainty` in
`fibernet/ml/models.py`. This is separate from the training and cross-validation
metric adapter. It changes neither the network architecture nor `forward`.

## Inputs and returned values

- `x` is a tensor shaped `(batch, n_features)` on the model's device with a
  compatible dtype. A batch containing one row is supported.
- `n_mc` defaults to 20 and must be a `numbers.Integral` value of at least 2.
  Python and NumPy integer counts are accepted. Booleans, floating-point counts,
  strings, and scalar tensors are rejected with `TypeError`. Integral values
  below 2 raise `ValueError`.
- Count validation happens before any forward call or training-mode change.
- The result is `(mean, std)`, each shaped `(batch, n_outputs)`, with the model's
  existing output dtype and device. The method does not move or cast inputs or
  parameters. Outputs have no autograd graph.

## Calculation and inference state

Each pass produces mean `mu_k` and clamped log variance `log_var_k`, as before.
The returned mean is the average of the sampled means. The squared returned
standard deviation is:

```text
sample_variance(mu_k, denominator=n_mc - 1) + mean(exp(log_var_k))
```

The sample-variance convention is preserved. Requiring at least two draws avoids
the undefined one-draw sample variance; this patch does not silently substitute
a population-variance estimator or set epistemic variance to zero.

Prediction uses `torch.no_grad()`. The model is evaluated with only its
`nn.Dropout` modules in training mode. BatchNorm uses stored running statistics,
including for one-row inputs, and does not update running means, variances, or
batch counters. Parameters, existing gradients, and inputs remain unchanged.

Every module's exact prior `training` flag is restored in `finally`, both after a
successful return and when a forward call raises. This preserves mixed caller
state, such as a training parent with explicitly frozen BatchNorm children.
The enclosing autograd setting is also restored.

MC sampling consumes the normal PyTorch RNG stream. The method does not restore
RNG state, promise identical unseeded repeated calls, or make a calibration or
physical uncertainty claim. Callers may seed their own experiments. This method
temporarily changes module modes, so overlapping prediction/training calls on
the same model instance are not supported.

## Compatibility changes

- Previously, `n_mc=1` and `True` returned a NaN standard deviation. They now fail
  explicitly, with `ValueError` for 1 and `TypeError` for `True`.
- Zero and negative counts previously failed later in `torch.stack`; they now
  fail before changing model state.
- Integer scalar tensors previously happened to work through Python `range`;
  they are now rejected. Convert an intended scalar count to a Python integer.
- Prediction no longer updates BatchNorm statistics or leaves the model in a
  different mode. Its numeric predictions can therefore differ from the old
  training-mode calculation. This is the intended inference correction.
- Returned predictions no longer retain gradients. Use the existing `forward`
  path when differentiable outputs are required.

## Verification

`tests/test_ml_uncertainty_contract.py` covers single-row CPU float32/float64
multioutput prediction, untouched BatchNorm state, exact mode restoration after
success and injected errors, count validation before state changes, stochastic
dropout, no autograd saved tensors, unchanged parameters and existing gradients,
and exact agreement with a manually constructed seeded dropout-only oracle.

The checks establish the software boundary above. They do not establish learned
model quality, probabilistic calibration, CUDA behavior, or uncertainty validity
for a particular physical material or dataset.
