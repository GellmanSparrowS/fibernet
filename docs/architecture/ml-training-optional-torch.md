# Optional PyTorch in the ML training API

`fibernet.ml.train_model` and `fibernet.ml.training.train_model` resolve to the
PyTorch training API. They are distinct from `fibernet.train_model`, the tabular
training helper in `fibernet.easy`.

Importing `fibernet.ml` does not itself load the training owner. Explicit access
to its four training aliases (`train_model`, `TrainingHistory`,
`save_model_bundle`, `load_model_bundle`) loads `fibernet.ml.training`.

When importing the top-level `torch` package raises `ModuleNotFoundError` with
`name == "torch"`, the owner remains importable without PyTorch. Its six existing
fallback entry points (`TrainingHistory`, `train_model`, `cross_validate_torch`,
`save_model_bundle`, `load_model_bundle`, `get_lr_scheduler`) raise
`ImportError("PyTorch required: pip install torch")` when used.

Other failures while importing Torch propagate unchanged, including missing
transitive dependencies, missing symbols and initialization errors. Once the
top-level Torch import succeeds, failures loading `torch.nn`, `torch.optim`, or
`torch.utils.data` also propagate unchanged—even if a submodule failure names
`torch` itself. These failures indicate that the installed provider could not
initialize; installing the already-present package is not necessarily the fix.
The original exception type, message, name and cause remain available for diagnosis.

This changes failure timing for broken installations: resolving the training
owner fails immediately instead of succeeding with missing-provider stubs.
Genuinely absent PyTorch retains the fallback behavior. Signatures, lazy alias
ownership and successful-provider training behavior are unchanged.

`tests/test_training_optional_torch.py` exercises the full training-owner source
and ML lazy facade, including its unchanged utility and dataset modules, with
real NumPy and scoped synthetic Torch imports. It checks missing
root behavior, broken root and submodule diagnostics, and successful definition
of the API. It does not import a real Torch installation, train models, simulate
structures, or read/write model artifacts. Synthetic successful imports establish
source definition compatibility only, not compatibility with an installed Torch
version or successful training.

The test binding uses a temporary root `fibernet` package shell to select the
canonical source tree without executing the root package initializer. Its real
`fibernet.ml` facade and siblings are loaded; this is not validation of the root
package's eager import behavior. Module registry entries and the import hook are
restored after each case; parent attributes belong only to the temporary modules.
