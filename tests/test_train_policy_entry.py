"""Public policy-entry contracts; no real policy learning or simulation."""

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
FACADES = ("__init__.py", "easy.py", "rl/__init__.py")


def resolver(path, owner, failure=None):
    """Execute facade declarations/resolver only, with an inert owner loader."""
    tree = ast.parse((ROOT / "fibernet" / path).read_text(encoding="utf-8"))
    calls = []

    def load(name):
        calls.append(name)
        assert name == "fibernet.rl.algorithms"
        if failure is not None:
            raise failure
        return SimpleNamespace(train_rl=owner)

    def importer(name, *args, **kwargs):
        assert name == "importlib"
        return SimpleNamespace(import_module=load)

    namespace = {"__name__": "inert_facade"}
    namespace["__builtins__"] = {
        "__import__": importer, "dict": dict, "getattr": getattr,
        "globals": lambda: namespace, "AttributeError": AttributeError,
    }
    statements = [node for node in tree.body if
                  (isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id in
                      {"_LAZY_EXPORTS", "_LAZY_MODULES", "_RENDER_EXPORTS"}
                      for target in node.targets)) or
                  (isinstance(node, ast.FunctionDef) and node.name == "__getattr__")]
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)
    assert calls == []
    return namespace, calls


@pytest.mark.parametrize("path", FACADES)
def test_policy_entry_is_original_owner_not_a_wrapper(path):
    def owner(env, *, algorithm="PPO", **kwargs):
        raise AssertionError("Resolving the entry must not call the trainer")

    namespace, calls = resolver(path, owner)
    assert namespace["__getattr__"]("train_policy") is owner
    assert calls == ["fibernet.rl.algorithms"]
    if path != "rl/__init__.py":
        assert namespace["train_policy"] is owner


@pytest.mark.parametrize("path", FACADES)
@pytest.mark.parametrize("kind", (ImportError, AttributeError, RuntimeError))
def test_policy_entry_preserves_owner_load_failure(path, kind):
    failure = kind("inert owner error")
    namespace, calls = resolver(path, object(), failure)
    with pytest.raises(kind) as caught:
        namespace["__getattr__"]("train_policy")
    assert caught.value is failure
    assert calls == ["fibernet.rl.algorithms"]
    assert "train_policy" not in namespace


def test_real_facades_resolve_one_owner_without_optional_provider_imports():
    # Separate process exercises actual root imports without polluting other tests.
    # Blocking providers makes this independent of whether optional extras exist.
    code = r'''
import builtins
import importlib
import inspect
import sys

original_import = builtins.__import__
requests = []
blocked = {"torch", "gymnasium", "gym", "stable_baselines3"}
def without_providers(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        requests.append(name)
        raise ModuleNotFoundError("deliberately unavailable: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_providers

import fibernet as fn
assert "fibernet.rl.algorithms" not in sys.modules
assert requests == [], requests
from fibernet import easy, rl
assert "fibernet.rl.algorithms" not in sys.modules
assert requests == [], requests
first = fn.train_policy
assert "fibernet.rl.algorithms" in sys.modules
from fibernet.rl import algorithms
owner = algorithms.train_rl
assert first is owner
for facade in (fn, easy, rl):
    assert "train_policy" in facade.__all__
    assert facade.train_policy is owner
    assert inspect.signature(facade.train_policy) == inspect.signature(owner)
assert fn.train_rl is easy.train_rl
assert rl.train_rl is owner
assert fn.train_rl is not owner
assert requests == [], requests
replacement = object()
algorithms.train_rl = replacement
try:
    for facade in (fn, easy, rl):
        importlib.reload(facade)
        assert facade.train_policy is replacement
finally:
    algorithms.train_rl = owner
    for facade in (fn, easy, rl):
        importlib.reload(facade)
        assert facade.train_policy is owner
assert fn.train_rl is easy.train_rl
assert rl.train_rl is owner
assert requests == [], requests
try:
    fn.train_policy(object())
except ImportError as error:
    assert "stable-baselines3 required" in str(error)
else:
    raise AssertionError("Missing provider must not fall back to random rollout")
assert requests == ["stable_baselines3"], requests
'''
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
