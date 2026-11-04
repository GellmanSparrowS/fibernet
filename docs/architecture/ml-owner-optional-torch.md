# Optional Torch: model, GNN and VAE imports

`fibernet.ml.models`, `fibernet.ml.gnn` and `fibernet.ml.generative` distinguish an absent optional Torch package from a broken installation, consistently with `fibernet.ml.training`.

When the initial `import torch` raises `ModuleNotFoundError` naming exactly `torch`, existing unavailable-backend classes/functions remain importable. Using these stubs raises `ImportError("PyTorch required: pip install torch")`. GNN's unconditional `graphs_from_batch` and `predict_gnn` helpers also check the provider before accessing inputs; they are not fallback stubs.

Other root-import failures, and **all** `torch.nn`/`torch.nn.functional` import failures, propagate unchanged at explicit owner import or lazy alias resolution (for example `from fibernet.ml import FiberMLP`). This preserves the original exception, missing-module name and cause. A broken provider is no longer silently represented as an absent optional dependency. Even a later submodule failure naming `torch` propagates.

No model architecture, numerical/training behavior, call-local dependency policy or export list changes. In particular GNN's call-local scikit-learn group-split error handling is separate. The ordinary ML facade remains lazy. Other ML owners are outside this three-owner change.

## Regression scope

`tests/test_ml_optional_torch_owners.py` uses real NumPy and actual selected ML facade/utils/dataset/owner files under a temporary root-package shell, with synthetic Torch import boundaries. The healthy case supplies only an empty `nn.Module` base for defining classes; no models are constructed or evaluated. Tests cover absent-root stubs/early guards and lazy alias identity, broken root/submodule exception identity/name/cause, and OSError/RuntimeError controls. They do not prove real-Torch numerical compatibility or a whole-package installation. Validation status belongs in the delivery report; these tests must be run in an isolated interpreter without Torch preloaded.
