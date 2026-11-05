"""GAN/PINN/diffusion import contracts; synthetic Torch, no model execution."""
import builtins
from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "fibernet"
ALIASES = {
    "gan": ("FiberGAN", "FiberWGAN", "FiberCGAN", "GANTrainer"),
    "pinn": ("PhysicsLoss", "PINNModel", "train_pinn"),
    "diffusion": ("FiberDiffusion", "ConditionalFiberDiffusion",
                  "DiffusionTrainer", "GraphDiffusionGenerator"),
}
STUBS = {
    "gan": ALIASES["gan"],
    "pinn": ALIASES["pinn"] + ("generate_collocation_points",),
    "diffusion": ALIASES["diffusion"],
}
HEALTHY_ONLY = {
    "gan": ("_MiniBatchStddev", "_build_mlp", "FiberGenerator", "FiberDiscriminator",
            "ConditionalGenerator", "ConditionalDiscriminator"),
    "pinn": (),
    "diffusion": ("NoiseSchedule", "SinusoidalTimeEmbedding", "ScoreNetwork",
                  "ConditionalScoreNetwork"),
}
STAGES = ("torch", "torch.nn", "torch.nn.functional")


def forbidden(*args, **kwargs):
    raise AssertionError("No numerical, training, generation or I/O body is permitted")


class RemainingMLOptionalTorchTests(unittest.TestCase):
    @contextmanager
    def binding(self, failure=None, stage="torch"):
        """Execute actual ML facade/utils/dataset and requested owner sources.

        Only the root fibernet package is a temporary path-selecting shell.
        Real NumPy is required. Torch/nn/functional are synthetic import objects;
        nn.Module is an empty class for class-definition-only checks. no_grad is
        an identity decorator for GAN/diffusion definitions, not a gradient-mode
        implementation. No Tensor, optimizer, layers or model code is supplied.
        All sys.modules/import-hook/bytecode settings restore on every exit.
        """
        self.assertFalse(any(n == "torch" or n.startswith("torch.") for n in sys.modules),
                         "Run in isolation: no real Torch may already be loaded")
        root = types.ModuleType("fibernet")
        root.__path__ = [str(ROOT)]
        torch = types.ModuleType("torch")
        torch.nn = types.ModuleType("torch.nn")
        torch.nn.functional = types.ModuleType("torch.nn.functional")
        torch.nn.Module = type("Module", (), {})
        torch.no_grad = lambda: lambda function: function
        attempts = []
        original = builtins.__import__

        def controlled(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "torch" or name.startswith("torch.")):
                attempts.append(name)
                if name == stage and failure is not None:
                    raise failure
                if name not in STAGES:
                    raise AssertionError(f"Unexpected Torch import: {name}")
                return torch
            return original(name, globals, locals, fromlist, level)

        with patch.dict(sys.modules), patch.object(sys, "dont_write_bytecode", True), \
                patch("builtins.__import__", new=controlled):
            for name in list(sys.modules):
                if name == "fibernet" or name.startswith("fibernet."):
                    del sys.modules[name]
            sys.modules["fibernet"] = root
            spec = importlib.util.spec_from_file_location(
                "fibernet.ml", ROOT / "ml" / "__init__.py",
                submodule_search_locations=[str(ROOT / "ml")])
            facade = importlib.util.module_from_spec(spec)
            sys.modules["fibernet.ml"] = facade
            spec.loader.exec_module(facade)
            self.assertEqual(attempts, [])
            for name in ("utils", "dataset"):
                loaded = sys.modules[f"fibernet.ml.{name}"]
                self.assertIs(loaded.np, np)
                self.assertEqual(Path(loaded.__file__).resolve(),
                                 (ROOT / "ml" / f"{name}.py").resolve())
            yield facade, attempts

    def resolve(self, facade, owner_name, route):
        name = f"fibernet.ml.{owner_name}"
        if route == "direct":
            owner = importlib.import_module(name)
        else:
            getattr(facade, route)
            owner = sys.modules[name]
        self.assertEqual(Path(owner.__file__).resolve(),
                         (ROOT / "ml" / f"{owner_name}.py").resolve())
        self.assertIs(owner.np, np)
        return owner

    def test_missing_root_preserves_stubs_and_aliases(self):
        for name, aliases in ALIASES.items():
            for route in ("direct", aliases[0]):
                with self.subTest(owner=name, route=route):
                    missing = ModuleNotFoundError("injected absent root", name="torch")
                    with self.binding(missing) as (facade, attempts):
                        owner = self.resolve(facade, name, route)
                        self.assertFalse(owner.HAS_TORCH)
                        for alias in aliases:
                            self.assertIs(getattr(facade, alias), getattr(owner, alias))
                        for symbol in STUBS[name]:
                            with self.assertRaises(ImportError) as caught:
                                getattr(owner, symbol)(forbidden, unused=forbidden)
                            self.assertEqual(str(caught.exception),
                                             "PyTorch required: pip install torch")
                        for symbol in HEALTHY_ONLY[name]:
                            self.assertNotIn(symbol, vars(owner))
                        self.assertEqual(attempts, ["torch"])

    def assert_failure(self, owner_name, failure, stage, route):
        expected_name = getattr(failure, "name", None)
        cause = RuntimeError("retained original cause")
        failure.__cause__ = cause
        with self.binding(failure, stage) as (facade, attempts):
            with self.assertRaises(type(failure)) as caught:
                self.resolve(facade, owner_name, route)
            self.assertIs(caught.exception, failure)
            self.assertIs(caught.exception.__cause__, cause)
            self.assertEqual(getattr(caught.exception, "name", None), expected_name)
            self.assertEqual(attempts, list(STAGES[:STAGES.index(stage) + 1]))
            self.assertNotIn(f"fibernet.ml.{owner_name}", sys.modules)
            for alias in ALIASES[owner_name]:
                self.assertNotIn(alias, vars(facade))

    def test_root_failures_preserve_original_diagnostics(self):
        for owner, aliases in ALIASES.items():
            for kind in (ModuleNotFoundError, ImportError):
                names = ("torch._C", "dependency_of_torch", None)
                if kind is ImportError:
                    names += ("torch",)
                for name in names:
                    with self.subTest(owner=owner, kind=kind, name=name, route="direct"):
                        self.assert_failure(owner, kind("injected broken root", name=name),
                                            "torch", "direct")
            with self.subTest(owner=owner, route=aliases[0]):
                self.assert_failure(owner, ImportError("injected lazy failure", name="torch"),
                                    "torch", aliases[0])

    def test_submodule_failures_including_missing_root_are_preserved(self):
        for owner in ALIASES:
            for stage in STAGES[1:]:
                for kind in (ModuleNotFoundError, ImportError):
                    for name in ("torch", stage, "dependency_of_torch", None):
                        with self.subTest(owner=owner, stage=stage, kind=kind, name=name):
                            self.assert_failure(owner, kind("injected submodule failure", name=name),
                                                stage, "direct")

    def test_nonimport_failures_are_preserved(self):
        for owner in ALIASES:
            for stage in STAGES:
                for kind in (OSError, RuntimeError):
                    with self.subTest(owner=owner, stage=stage, kind=kind):
                        self.assert_failure(owner, kind("injected initialization failure"),
                                            stage, "direct")

    def test_healthy_definitions_and_aliases_without_model_execution(self):
        for name, aliases in ALIASES.items():
            with self.subTest(owner=name):
                with self.binding() as (facade, attempts):
                    owner = self.resolve(facade, name, aliases[0])
                    self.assertTrue(owner.HAS_TORCH)
                    self.assertEqual(attempts, list(STAGES))
                    for alias in aliases:
                        self.assertIs(getattr(facade, alias), getattr(owner, alias))
                    for symbol in STUBS[name] + HEALTHY_ONLY[name]:
                        self.assertEqual(getattr(owner, symbol).__module__, f"fibernet.ml.{name}")
                    cls = getattr(owner, aliases[0])
                    self.assertIs(cls.__bases__[0], owner.nn.Module)
                    self.assertNotIn("a", cls.__init__.__code__.co_varnames)
                    self.assertEqual(attempts, list(STAGES))


if __name__ == "__main__":
    unittest.main()
