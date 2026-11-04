# Optional Torch: GAN, PINN and diffusion imports

`fibernet.ml.gan`, `fibernet.ml.pinn` and `fibernet.ml.diffusion` distinguish an absent optional Torch package from a broken installation, consistently with the model, GNN, VAE and training owners.

Only `ModuleNotFoundError` naming exactly `torch` from the initial `import torch` selects the existing unavailable-backend branch. Its classes/functions remain importable and raise `ImportError("PyTorch required: pip install torch")` when used. PINN retains its existing `generate_collocation_points` fallback as well as its three facade aliases. GAN/diffusion helpers that did not have absent-provider stubs still do not acquire them.

All other root import failures and all subsequent `torch.nn`/`torch.nn.functional` import failures propagate unchanged, including failures naming `torch` at a submodule stage. Explicit owner import or lazy facade alias resolution therefore exposes a broken provider's original exception, missing-module name and cause instead of representing it as absent.

There are no model-body, signature, numerical/training, call-local dependency or export-list changes. The ML facade remains lazy. This is a three-owner change, not a claim that every other ML owner already follows this policy.

## Regression scope

`tests/test_ml_remaining_optional_torch.py` follows the existing owner import regression pattern: real NumPy and actual selected ML facade/utils/dataset/owner source files, a temporary root-package shell, and scoped synthetic Torch import boundaries. It covers absent-root stubs and lazy alias identity; broken root and submodule exception identity/name/cause; and unchanged OSError/RuntimeError propagation.

The healthy branch supplies an empty `nn.Module` and an identity `no_grad` decorator solely to define the existing classes and methods. GAN has four definition-time `@torch.no_grad()` uses and diffusion has three. No healthy classes are instantiated; no model, loss, training, sampling, collocation, structure-generation or artifact-I/O function is called. The decorator does not implement or validate gradient-mode behavior.

These are source-definition tests, not real-Torch numerical compatibility, whole-root package import or installation qualification. Run them in an isolated interpreter without Torch preloaded. Actual execution status belongs in the delivery report.
