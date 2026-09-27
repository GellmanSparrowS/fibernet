"""Load fibernet.gen.pattern WITHOUT executing fibernet/__init__ (which pulls
taichi/torch). A stub package with __path__ is registered first so submodule
imports resolve straight to the repo files (pure python + numpy + networkx)."""
import ast
import importlib
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_ROOT)
_DEFAULT = (_PARENT if os.path.isdir(os.path.join(_PARENT, "fibernet", "core"))
            else os.path.join(_PARENT, "fibernet"))
FIBERNET_ROOT = os.environ.get("FIBERNET_ROOT", _DEFAULT)


def _public_exports(init_file):
    """Read the declared public names without importing optional modules."""
    source = open(init_file, encoding="utf-8").read()
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__all__"
                        for target in node.targets)):
            return set(ast.literal_eval(node.value)) | {"__all__", "__version__"}
    return {"__version__"}


def ensure_fibernet():
    if getattr(sys, "frozen", False):
        # vendored light fibernet (core+gen, thin __init__) collected by
        # PyInstaller; no taichi/torch in the bundle
        return importlib.import_module("fibernet.gen.pattern")
    if "fibernet" not in sys.modules:
        pkg_dir = os.path.join(FIBERNET_ROOT, "fibernet")
        init_file = os.path.join(pkg_dir, "__init__.py")
        public_names = _public_exports(init_file)
        pkg = types.ModuleType("fibernet")
        pkg.__path__ = [pkg_dir]
        pkg.__file__ = init_file
        pkg.__package__ = "fibernet"

        def load_public(name):
            if name not in public_names:
                raise AttributeError("module 'fibernet' has no attribute %r" % name)
            original = dict(pkg.__dict__)
            pkg.__dict__.pop("__getattr__", None)
            try:
                with open(init_file, encoding="utf-8") as fh:
                    code = compile(fh.read(), init_file, "exec")
                exec(code, pkg.__dict__)
            except Exception:
                pkg.__dict__.clear()
                pkg.__dict__.update(original)
                raise
            if name not in pkg.__dict__:
                raise AttributeError("module 'fibernet' has no attribute %r" % name)
            return pkg.__dict__[name]

        pkg.__getattr__ = load_public
        sys.modules["fibernet"] = pkg
    return importlib.import_module("fibernet.gen.pattern")
