"""Torch import diagnostics without importing Torch or running model code."""
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
ALIASES = ("train_model", "TrainingHistory", "save_model_bundle", "load_model_bundle")
STUBS = ALIASES + ("cross_validate_torch", "get_lr_scheduler")
STAGES = ("torch", "torch.nn", "torch.optim", "torch.utils.data")


def forbidden(*args, **kwargs):
    raise AssertionError("No model, array, data-loader or I/O operation is permitted")


class TrainingOptionalTorchTests(unittest.TestCase):
    @contextmanager
    def binding(self, failure=None, stage="torch"):
        """Bind whole source modules with scoped synthetic import dependencies.

        The selected ML facade, utility, dataset and training source files
        execute with real NumPy. A temporary root-package shell selects the
        source tree; Torch imports are synthetic and cannot reach a real
        installation. sys.modules and the import hook are restored on all exits.
        """
        root = types.ModuleType("fibernet")
        root.__path__ = [str(ROOT)]
        torch = types.ModuleType("torch")
        torch.nn = types.ModuleType("torch.nn")
        torch.optim = types.ModuleType("torch.optim")
        data = types.ModuleType("torch.utils.data")
        data.DataLoader = data.TensorDataset = forbidden
        attempts = []
        original = builtins.__import__

        def controlled(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "torch" or name.startswith("torch.")):
                attempts.append(name)
                if name == stage and failure is not None:
                    raise failure
                if name not in STAGES:
                    raise AssertionError(f"Unexpected Torch import: {name}")
                return data if name == "torch.utils.data" else torch
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
            self.assertEqual(attempts, [])  # facade itself stays lazy
            for name in ("utils", "dataset"):
                loaded = sys.modules[f"fibernet.ml.{name}"]
                self.assertIs(loaded.np, np)
                self.assertEqual(Path(loaded.__file__).resolve(),
                                 (ROOT / "ml" / f"{name}.py").resolve())
            yield facade, attempts

    def resolve(self, facade, route):
        if route == "direct":
            return importlib.import_module("fibernet.ml.training")
        getattr(facade, route)
        return sys.modules["fibernet.ml.training"]

    def test_missing_root_preserves_all_stubs_and_lazy_aliases(self):
        for route in ("direct",) + ALIASES:
            with self.subTest(route=route):
                missing = ModuleNotFoundError("injected absent root", name="torch")
                with self.binding(missing) as (facade, attempts):
                    owner = self.resolve(facade, route)
                    self.assertFalse(owner.HAS_TORCH)
                    self.assertEqual(attempts, ["torch"])
                    for alias in ALIASES:
                        self.assertIs(getattr(facade, alias), getattr(owner, alias))
                    for name in STUBS:
                        with self.assertRaises(ImportError) as caught:
                            getattr(owner, name)(forbidden, unused=forbidden)
                        self.assertEqual(str(caught.exception),
                                         "PyTorch required: pip install torch")
                    self.assertEqual(attempts, ["torch"])

    def assert_failure(self, failure, stage, route):
        expected_name = getattr(failure, "name", None)
        cause = RuntimeError("retained original cause")
        failure.__cause__ = cause
        with self.binding(failure, stage) as (facade, attempts):
            with self.assertRaises(type(failure)) as caught:
                self.resolve(facade, route)
            self.assertIs(caught.exception, failure)
            self.assertIs(caught.exception.__cause__, cause)
            self.assertEqual(getattr(caught.exception, "name", None),
                             expected_name)
            self.assertEqual(attempts, list(STAGES[:STAGES.index(stage) + 1]))
            self.assertNotIn("fibernet.ml.training", sys.modules)
            self.assertNotIn("train_model", vars(facade))

    def test_root_internal_import_failures_are_preserved(self):
        for kind in (ModuleNotFoundError, ImportError):
            names = ("torch._C", "dependency_of_torch", None)
            if kind is ImportError:
                names += ("torch",)
            for name in names:
                for route in ("direct",) + ALIASES:
                    with self.subTest(kind=kind, name=name, route=route):
                        self.assert_failure(kind("injected broken root", name=name),
                                            "torch", route)

    def test_submodule_failures_including_missing_torch_are_preserved(self):
        for stage in STAGES[1:]:
            for kind in (ModuleNotFoundError, ImportError):
                for name in ("torch", stage, "dependency_of_torch", None):
                    for route in ("direct",) + ALIASES:
                        with self.subTest(stage=stage, kind=kind, name=name, route=route):
                            self.assert_failure(kind("injected broken submodule", name=name),
                                                stage, route)

    def test_nonimport_failures_are_preserved(self):
        for stage in STAGES:
            for kind in (OSError, RuntimeError):
                with self.subTest(stage=stage, kind=kind):
                    self.assert_failure(kind("injected initialization failure"), stage, "train_model")

    def test_successful_import_defines_real_api_without_running_it(self):
        with self.binding() as (facade, attempts):
            owner = self.resolve(facade, "train_model")
            self.assertTrue(owner.HAS_TORCH)
            self.assertIs(owner.np, np)
            self.assertEqual(attempts, list(STAGES))
            for alias in ALIASES:
                self.assertIs(getattr(facade, alias), getattr(owner, alias))
            for name in STUBS:
                self.assertEqual(getattr(owner, name).__module__, "fibernet.ml.training")
            self.assertIn("model", owner.train_model.__code__.co_varnames)
            self.assertEqual(owner.TrainingHistory.__init__.__code__.co_argcount, 1)
            self.assertIs(owner.DataLoader, forbidden)
            self.assertIs(owner.TensorDataset, forbidden)


if __name__ == "__main__":
    unittest.main()
