"""Build FiberScope exe (onedir).

Vendors a LIGHT fibernet package (core + gen only, thin __init__) so the
bundle never pulls taichi/torch, then ships only the five fibernet modules
the app actually uses as hidden imports. A broad exclude list keeps the
analysis graph from dragging the (huge) ML stack of this machine into the
bundle; every excluded import is guarded/optional at runtime.

Usage: python scripts/build_exe.py
Smoke: dist\FiberScope\FiberScope.exe --smoke   (exit 0 + "[smoke] PASS")
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fslab.fibernet_bridge import FIBERNET_ROOT  # noqa: E402

SRC = os.path.join(FIBERNET_ROOT, "fibernet")
VENDOR = os.path.join(ROOT, "build_vendor", "fibernet")

if os.path.isdir(os.path.join(ROOT, "build_vendor")):
    shutil.rmtree(os.path.join(ROOT, "build_vendor"))
os.makedirs(VENDOR)
for sub in ("core", "gen"):
    shutil.copytree(os.path.join(SRC, sub), os.path.join(VENDOR, sub),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
with open(os.path.join(VENDOR, "__init__.py"), "w", encoding="utf-8") as fh:
    fh.write('"""vendored light fibernet (core+gen) for FiberScope bundle"""\n')

# ONLY the modules pattern_2d needs (module-level graph: numpy only).
# networkx/scipy reach the bundle through pattern.py function-level imports.
HIDDEN = [
    "fibernet.core.material",
    "fibernet.core.structure_graph",
    "fibernet.core.transforms",
    "fibernet.core.tiling",
    "fibernet.gen.pattern",
]

# big packages of the dev machine that the static graph reaches through
# guarded/optional imports (scipy array_api_compat -> torch, etc.).
EXCLUDES = [
    "torch", "torchvision", "torchaudio", "torio", "functorch",
    "transformers", "accelerate", "einops", "safetensors", "datasets",
    "tokenizers", "huggingface_hub", "peft", "trl", "diffusers",
    "sentencepiece", "timm", "bitsandbytes", "deepspeed",
    "sklearn", "optuna", "pulp", "alembic", "sqlalchemy",
    "IPython", "ipywidgets", "ipykernel", "jupyter_client", "jupyter_core",
    "notebook", "qtconsole",
    "matplotlib", "skimage", "h5py", "taichi", "tensorflow", "keras",
    "pandas", "cv2", "lxml", "pygments", "pydot", "imageio", "fsspec",
    "pooch", "tifffile", "sympy", "triton", "onnx", "onnxruntime",
]

ICON = os.path.join(ROOT, "assets", "icon.ico")

cmd = [
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
    "--name", "FiberScope",
    "--windowed",
    "--icon", ICON,
    "--paths", os.path.join(ROOT, "build_vendor"),
    "--add-data", os.path.join(ROOT, "data") + os.pathsep + "data",
    "--add-data", os.path.join(ROOT, "assets") + os.pathsep + "assets",
]
for h in HIDDEN:
    cmd += ["--hidden-import", h]
for e in EXCLUDES:
    cmd += ["--exclude-module", e]
cmd.append(os.path.join(ROOT, "run.py"))
subprocess.check_call(cmd, cwd=ROOT)
print("build done: dist/FiberScope/FiberScope.exe")
