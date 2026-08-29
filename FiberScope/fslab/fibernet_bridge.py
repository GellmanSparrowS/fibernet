"""Load fibernet.gen.pattern WITHOUT executing fibernet/__init__ (which pulls
taichi/torch). A stub package with __path__ is registered first so submodule
imports resolve straight to the repo files (pure python + numpy + networkx)."""
import importlib
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT = os.path.normpath(os.path.join(os.path.dirname(_ROOT), "fibernet"))
FIBERNET_ROOT = os.environ.get("FIBERNET_ROOT", _DEFAULT)


def ensure_fibernet():
    if getattr(sys, "frozen", False):
        # vendored light fibernet (core+gen, thin __init__) collected by
        # PyInstaller; no taichi/torch in the bundle
        return importlib.import_module("fibernet.gen.pattern")
    if "fibernet" not in sys.modules:
        pkg_dir = os.path.join(FIBERNET_ROOT, "fibernet")
        pkg = types.ModuleType("fibernet")
        pkg.__path__ = [pkg_dir]
        sys.modules["fibernet"] = pkg
    return importlib.import_module("fibernet.gen.pattern")
