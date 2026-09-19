"""Build the FiberScope Windows exe (single file by default).

Pipeline
  1. vendor a LIGHT fibernet (core + gen only, thin __init__) so the bundle
     never pulls taichi/torch;
  2. write FiberScope.spec: Analysis with a broad exclude list, then an
     explicit binary/data filter that drops the unused Qt modules, the
     test-only pywin32 and the packaging metadata, and swaps the
     collected ICU for the Windows system one;
  3. run PyInstaller (onefile EXE, no UPX, windowed, version resource) with a
     sanitized PATH, so no foreign dev-machine DLL gets swept in;
  4. gate: runtime-dep probe, frozen --smoke, portability audit and a
     clean-room smoke from %TEMP% with a stripped environment.

Usage:
    python scripts/build_exe.py                # onefile + every gate
    python scripts/build_exe.py --onedir       # inspectable bundle (debug)
    python scripts/build_exe.py --no-smoke     # skip the frozen run
    python scripts/build_exe.py --no-gate      # skip probe/audit/clean-room
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fslab.fibernet_bridge import FIBERNET_ROOT  # noqa: E402
from fslab.version import (APP_NAME, APP_VERSION, VERSION_TUPLE, ORG_ZH,
                           AUTHOR, AUTHOR_SIGNATURE)  # noqa: E402

SRC = os.path.join(FIBERNET_ROOT, "fibernet")
VENDOR = os.path.join(ROOT, "build_vendor", "fibernet")
SPEC = os.path.join(ROOT, "FiberScope.spec")
VERFILE = os.path.join(ROOT, "build_version.txt")

# ONLY the modules pattern_2d needs (module-level graph: numpy only).
HIDDEN = [
    "fibernet.core.material",
    "fibernet.core.structure_graph",
    "fibernet.core.transforms",
    "fibernet.core.tiling",
    "fibernet.gen.pattern",
    "sklearn.neural_network", "sklearn.linear_model", "sklearn.neighbors",
    "sklearn.ensemble",
    "fast_simplification", "manifold3d",
    "skimage.measure._marching_cubes_lewiner",
]

# big packages of the dev machine that the static graph reaches through
# guarded/optional imports (scipy array_api_compat -> torch, etc.).
EXCLUDES = [
    "pyvista", "vtk", "vtkmodules", "trimesh",
    "stable_baselines3", "gymnasium",
    "torch", "torchvision", "torchaudio", "torio", "functorch",
    "transformers", "accelerate", "einops", "safetensors", "datasets",
    "tokenizers", "huggingface_hub", "peft", "trl", "diffusers",
    "sentencepiece", "timm", "bitsandbytes", "deepspeed",
    "optuna", "pulp", "alembic", "sqlalchemy",
    "IPython", "ipywidgets", "ipykernel", "jupyter_client", "jupyter_core",
    "notebook", "qtconsole",
    "matplotlib", "h5py", "taichi", "tensorflow", "keras",
    "pandas", "cv2", "lxml", "pygments", "pydot", "imageio", "fsspec",
    "pooch", "tifffile", "sympy", "triton", "onnx", "onnxruntime",
    # Optional accelerators the static graph reaches but nothing calls:
    # pyqtgraph.functions_numba drags numba + llvmlite (120 MB of LLVM) and the
    # vendored fibernet helpers drag networkx / Cython.  Measured clean by
    # scripts/check_runtime_deps.py (all 15 units + every tab, 5/5 suites).
    "numba", "llvmlite", "networkx", "Cython", "cython",
    # dev/test-only or unused at runtime (see also DROP below)
    "psutil", "yaml", "setuptools", "pkg_resources", "wheel",
    "win32com", "win32comext", "pythonwin", "pywin",
    # QtOpenGL + QtOpenGLWidgets must NOT be excluded: pyqtgraph.Qt.
    # OpenGLHelpers imports both unguarded (GraphicsView, PlotCurveItem).
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets",
    "PySide6.QtPdf", "PySide6.QtTest", "PySide6.QtSvg", "PySide6.QtXml",
    "PySide6.QtCharts", "PySide6.QtMultimedia", "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets", "PySide6.QtWebSockets",
    "PySide6.Qt3DCore", "PySide6.QtConcurrent", "PySide6.QtSql",
    "PySide6.QtPositioning", "PySide6.QtSensors", "PySide6.QtSerialPort",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtLocation",
    "PySide6.QtStateMachine", "PySide6.QtTextToSpeech",
    "PySide6.QtDataVisualization", "PySide6.QtGraphs", "PySide6.QtHttpServer",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSpatialAudio",
    "PySide6.QtVirtualKeyboard", "PySide6.QtDesigner", "PySide6.QtHelp",
    # pydoc must stay: scipy._lib._docscrape and pyqtgraph.parametertree
    # both import it at module level (measured by check_runtime_deps).
    "tkinter", "_tkinter", "unittest", "doctest", "pdb",
]

# binary/data entries dropped after Analysis (lowercase, '/' separators).
# Everything here is either an unused Qt module, a software-GL fallback or
# dev-machine baggage; the frozen --smoke run is the safety net.
DROP = (
    "pyside6/qt6quick", "pyside6/qtquick", "pyside6/qt6qml", "pyside6/qtqml",
    "pyside6/qt6pdf", "pyside6/qtpdf",
    "pyside6/qt6test", "pyside6/qttest", "pyside6/qt6svg", "pyside6/qtsvg",
    "pyside6/qt6xml", "pyside6/qtxml", "pyside6/qt6charts",
    "pyside6/qt6multimedia", "pyside6/qtmultimedia", "pyside6/qt6webengine",
    "pyside6/qtwebengine", "pyside6/qt6websockets", "pyside6/qt6webchannel",
    "pyside6/qt6bluetooth", "pyside6/qt6nfc", "pyside6/qt6positioning",
    "pyside6/qt6location", "pyside6/qt6sensors", "pyside6/qt6serialport",
    "pyside6/qt6serialbus", "pyside6/qt6sql", "pyside6/qtsql",
    "pyside6/qt6concurrent", "pyside6/qt6designer", "pyside6/qt6help",
    "pyside6/qt6linguist", "pyside6/qt6statemachine", "pyside6/qt6scxml",
    "pyside6/qt6remoteobjects", "pyside6/qt6texttospeech",
    "pyside6/qt6datavisualization", "pyside6/qt6graphs",
    "pyside6/qt6spatialaudio", "pyside6/qt6httpserver", "pyside6/qt6grpc",
    "pyside6/qt6virtualkeyboard", "pyside6/qt63d", "pyside6/qt3d",
    "pyside6/translations/", "pyside6/qml", "opengl32sw.dll",
    "pyside6/plugins/multimedia", "pyside6/plugins/sqldrivers",
    "pyside6/plugins/networkinformation", "pyside6/plugins/tls",
    "pyside6/plugins/position", "pyside6/plugins/sensors",
    "pyside6/plugins/iconengines", "pyside6/plugins/platforminputcontexts",
    "pyside6/plugins/generic", "pyside6/plugins/accessiblebridge",
    "psutil", "yaml/", "_yaml", "pythonwin/", "win32/", "win32com/",
    "win32comext", "pywin32_system32/", "pythoncom", "pywintypes",
    "setuptools", "pkg_resources", "wheel-", "importlib_metadata",
    # trailing '/' matters: a bare "cython" also matched and dropped
    # scipy/linalg/_cythonized_array_utils.pyd (frozen ModuleNotFoundError)
    "lib2to3/", "numba/", "llvmlite/", "networkx/", "cython/",
    # image-format plugins whose Qt module is dropped above (orphan imports)
    "pyside6/plugins/imageformats/qpdf.dll",
    "pyside6/plugins/imageformats/qsvg.dll",
    # MSVCP140* / VCRUNTIME140* must stay: Qt6Gui imports MSVCP140_1.dll and a
    # clean judge machine has no VC++ redistributable installed.
)

VERSION_INFO = '''# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004,
    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('080404b0', [
        StringStruct('CompanyName', '{org}'),
        StringStruct('FileDescription', '{name} - fibrous metamaterial explorer'),
        StringStruct('FileVersion', '{ver}'),
        StringStruct('InternalName', '{name}'),
        StringStruct('LegalCopyright', '{author}'),
        StringStruct('OriginalFilename', '{name}.exe'),
        StringStruct('ProductName', '{name}'),
        StringStruct('ProductVersion', '{ver}'),
        StringStruct('Comments', 'GOAI Track 3 - AI for Research'),
    ])]),
    VarFileInfo([VarStruct('Translation', [2052, 1200])])
  ]
)
'''

SPEC_TMPL = '''# -*- mode: python ; coding: utf-8 -*-
# generated by scripts/build_exe.py - do not edit by hand

DROP = {drop!r}
SYS_ICU = {sysicu!r}
PRIVATE_DLLS = {private!r}


def _keep(name):
    n = str(name).replace("\\\\", "/").lower()
    return not any(pat in n for pat in DROP)


def _is_icu(name):
    b = str(name).replace("\\\\", "/").lower().rsplit("/", 1)[-1]
    return b.startswith("icu") and b.endswith(".dll")


a = Analysis(
    [{entry!r}],
    pathex=[{vendor!r}],
    binaries=[],
    datas=[({data!r}, 'data'), ({assets!r}, 'assets'), ({training!r}, 'training_runtime')],
    hiddenimports={hidden!r},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes!r},
    noarchive=False,
    optimize=0,
)
# Qt6Core binds to the PLAIN ucnv_* exports of icuuc.dll.  Any ICU swept in
# from PATH (conda / poppler) exports only version-suffixed names such as
# ucnv_open_78, and the frozen app then dies at `import PySide6.QtWidgets`
# with ERROR_PROC_NOT_FOUND.  Drop every collected ICU and ship the Windows
# system build instead - the one Qt6Core binds against in development.
a.binaries = [b for b in a.binaries if _keep(b[0]) and not _is_icu(b[0])]
a.datas = [d for d in a.datas if _keep(d[0])]
a.binaries += [(dst, src, "BINARY") for dst, src in SYS_ICU]
# Same-named wheel DLLs may be collected under another package by dependency analysis.
# Preserve the directory explicitly registered by the fast_simplification bootstrap.
a.binaries += [(dst, src, "BINARY") for dst, src in PRIVATE_DLLS
               if dst not in [item[0] for item in a.binaries]]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    {coll}
    [],
    name='{name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[{icon!r}],
    version={verfile!r},
)
{collect}'''


def vendor_fibernet():
    if os.path.isdir(os.path.join(ROOT, "build_vendor")):
        shutil.rmtree(os.path.join(ROOT, "build_vendor"))
    os.makedirs(VENDOR)
    for sub in ("core", "gen"):
        shutil.copytree(os.path.join(SRC, sub), os.path.join(VENDOR, sub),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runtime = os.path.join(ROOT, "build_vendor", "training_runtime")
    os.makedirs(os.path.join(runtime, "scripts"))
    for source, target in ((os.path.join(ROOT, "fslab"), "fslab"), (VENDOR, "fibernet")):
        shutil.copytree(source, os.path.join(runtime, target),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(os.path.join(ROOT, "scripts", "rl_worker.py"), os.path.join(runtime, "scripts"))
    with open(os.path.join(runtime, "fibernet", "__init__.py"), "w", encoding="utf-8") as fh:
        fh.write('"""Light training runtime."""')
    with open(os.path.join(VENDOR, "__init__.py"), "w", encoding="utf-8") as fh:
        fh.write('"""vendored light fibernet (core+gen) for FiberScope"""\n')


def system_icu():
    """(dest, src) pairs for the Windows system ICU.

    System32 icuuc.dll exports the plain ucnv_* symbols Qt6Core imports
    (measured: 542 exports, 0 version-suffixed) and forwards to icu.dll; both
    are shipped so the payload never depends on the target machine's ICU.
    """
    sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                          "System32")
    out = [(fn, os.path.join(sysdir, fn)) for fn in ("icuuc.dll", "icu.dll")
           if os.path.exists(os.path.join(sysdir, fn))]
    if not any(dst == "icuuc.dll" for dst, _src in out):
        raise SystemExit("[build] System32 icuuc.dll not found under %s"
                         % sysdir)
    return out


def build_env():
    """PATH limited to this interpreter's tree + the Windows system dirs.

    PyInstaller resolves binary dependencies by sweeping PATH, so a polluted
    dev PATH injects foreign DLLs into the payload (measured: a poppler ICU of
    35 MB plus its libcrypto/libssl, and that ICU broke QtWidgets at startup).
    """
    # a conda env root is the very folder that holds python.exe
    envroot = os.path.dirname(os.path.abspath(sys.executable))
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    keep = [envroot, os.path.join(envroot, "DLLs"),
            os.path.join(envroot, "Library", "bin"),
            os.path.join(envroot, "Library", "usr", "bin"),
            os.path.join(envroot, "Library", "mingw-w64", "bin"),
            os.path.join(envroot, "Scripts"),
            os.path.join(sysroot, "System32"), sysroot,
            os.path.join(sysroot, "SysWOW64")]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(k for k in keep if os.path.isdir(k))
    for key in ("PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH",
                "QT_QPA_PLATFORM"):
        env.pop(key, None)
    return env


def cleanroom_env(room):
    """Environment of a clean judge machine: System32-only PATH, no conda."""
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    local = os.path.join(room, "appdata")
    os.makedirs(local, exist_ok=True)
    return {
        "PATH": os.path.join(sysroot, "System32") + os.pathsep + sysroot,
        "SystemRoot": sysroot, "SystemDrive": sysroot[:2],
        "TEMP": local, "TMP": local, "LOCALAPPDATA": local, "APPDATA": local,
        "USERPROFILE": room, "HOMEDRIVE": sysroot[:2],
        "HOMEPATH": os.path.basename(room), "USERNAME": "judge",
        "COMPUTERNAME": "CLEANROOM", "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "NUMBER_OF_PROCESSORS": os.environ.get("NUMBER_OF_PROCESSORS", "8"),
    }


def private_dlls():
    from pathlib import Path
    from importlib.util import find_spec
    package = find_spec('fast_simplification')
    folder = Path(package.origin).parent.parent / 'fast_simplification.libs'
    files = sorted(folder.glob('*.dll'))
    if not files:
        raise RuntimeError('fast_simplification private DLL directory is missing')
    binaries = [('fast_simplification.libs/' + file.name, str(file)) for file in files]
    manifold_root = Path(find_spec('manifold3d').origin).parent
    binaries.extend((file.name,str(file)) for file in manifold_root.glob('msvcp140-*.dll'))
    return binaries


def write_spec(onefile: bool):
    # onefile: EXE() must swallow the binaries and data itself
    coll = "a.binaries,\n    a.datas,\n    " if onefile else ""
    coll2 = "" if onefile else (
        "coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,\n"
        "                 upx_exclude=[], name='%s')\n" % APP_NAME)
    text = SPEC_TMPL.format(
        drop=DROP, entry=os.path.join(ROOT, "run.py"),
        vendor=os.path.join(ROOT, "build_vendor"),
        data=os.path.join(ROOT, "data"), assets=os.path.join(ROOT, "assets"),
        training=os.path.join(ROOT, "build_vendor", "training_runtime"),
        hidden=HIDDEN, excludes=EXCLUDES, coll=coll, collect=coll2,
        name=APP_NAME, icon=os.path.join(ROOT, "assets", "icon.ico"),
        verfile=VERFILE, sysicu=system_icu(), private=private_dlls())
    with open(SPEC, "w", encoding="utf-8") as fh:
        fh.write(text)


def write_version_file():
    with open(VERFILE, "w", encoding="utf-8") as fh:
        fh.write(VERSION_INFO.format(t=VERSION_TUPLE, ver=APP_VERSION,
                                     org=ORG_ZH, name=APP_NAME, author=AUTHOR_SIGNATURE))


def exe_path(onefile: bool):
    if onefile:
        return os.path.join(ROOT, "dist", APP_NAME + ".exe")
    return os.path.join(ROOT, "dist", APP_NAME, APP_NAME + ".exe")


def _read_sidecar(exe):
    side = os.path.join(os.path.dirname(exe), "smoke_result.txt")
    txt = ""
    if os.path.exists(side):
        with open(side, encoding="utf-8", errors="replace") as fh:
            txt = fh.read().strip()
        os.remove(side)
    return txt


def smoke(exe, timeout=600):
    t0 = time.time()
    r = subprocess.run([exe, "--smoke"], capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=timeout)
    dt = time.time() - t0
    txt = _read_sidecar(exe)
    print("[smoke] exit=%d  %.1fs  %s" % (r.returncode, dt, txt[:200]))
    if r.returncode != 0:
        print((r.stdout or "")[-800:])
        print((r.stderr or "")[-800:])
    return r.returncode == 0, dt


def cleanroom_smoke(exe, timeout=900):
    """Copy the payload into a fresh temp dir and smoke it with no dev env.

    Proves the build carries its own dependencies: System32-only PATH, no
    PYTHON*/CONDA*/QT_* variables, writable TEMP (onefile unpacks there).
    """
    room = tempfile.mkdtemp(prefix="fsclean_")
    try:
        if os.path.isdir(exe):
            dst = os.path.join(room, os.path.basename(exe.rstrip("\\/")))
            shutil.copytree(exe, dst)
            target = os.path.join(dst, os.path.basename(exe))
        else:
            target = os.path.join(room, os.path.basename(exe))
            shutil.copy2(exe, target)
        t0 = time.time()
        r = subprocess.run([target, "--smoke"], capture_output=True,
                           text=True, encoding='utf-8', errors='replace', timeout=timeout, env=cleanroom_env(room),
                           cwd=room)
        dt = time.time() - t0
        txt = _read_sidecar(target)
        print("[cleanroom] exit=%d  %.1fs  %s" % (r.returncode, dt, txt[:200]))
        if r.returncode != 0:
            print((r.stdout or "")[-600:])
            print((r.stderr or "")[-600:])
        return r.returncode == 0, dt
    finally:
        shutil.rmtree(room, ignore_errors=True)


def dep_gate():
    """Fail fast when a planned exclusion is actually needed at runtime."""
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "scripts", "check_runtime_deps.py"),
                        "--only", "probe,gui_smoke"],
                       cwd=ROOT, capture_output=True, text=True)
    for line in [l for l in (r.stdout or "").strip().splitlines() if l][-4:]:
        print("   ", line)
    if r.returncode != 0:
        print((r.stderr or "")[-1500:])
    return r.returncode == 0


def audit(target):
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "scripts", "audit_dist.py"),
                        "--path", target],
                       cwd=ROOT, capture_output=True, text=True)
    print((r.stdout or "").strip()[-1800:])
    if r.returncode != 0:
        print((r.stderr or "")[-800:])
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onedir", action="store_true",
                    help="build an inspectable folder bundle instead")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip dep probe / portability audit / clean-room run")
    args = ap.parse_args()
    onefile = not args.onedir

    if not args.no_gate:
        t0 = time.time()
        print("[gate] runtime-dep probe (excluded packages blocked) ...")
        if not dep_gate():
            print("[gate] FAIL - an excluded package is needed at runtime")
            return 1
        print("[gate] dep probe ok in %.0fs" % (time.time() - t0))

    vendor_fibernet()
    write_version_file()
    write_spec(onefile)
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", SPEC]
    t0 = time.time()
    subprocess.check_call(cmd, cwd=ROOT, env=build_env())
    print("[build] %.0fs" % (time.time() - t0))

    exe = exe_path(onefile)
    if onefile:
        mb = os.path.getsize(exe) / 1e6
        print("[size] %s = %.1f MB (single file)" % (exe, mb))
    else:
        total = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _d, fs in os.walk(os.path.dirname(exe))
                    for f in fs)
        print("[size] %s folder = %.1f MB" % (os.path.dirname(exe),
                                              total / 1e6))
    if args.no_smoke:
        return 0

    failed = []
    ok, dt = smoke(exe)
    print("[gate] frozen smoke %s in %.1fs" % ("PASS" if ok else "FAIL", dt))
    if not ok:
        failed.append("smoke")
    if not args.no_gate:
        target = exe if onefile else os.path.dirname(exe)
        if not audit(target):
            failed.append("audit")
        ok, dt = cleanroom_smoke(exe)
        print("[gate] clean-room smoke %s in %.1fs"
              % ("PASS" if ok else "FAIL", dt))
        if not ok:
            failed.append("cleanroom")
    if failed:
        print("[result] GATE FAILED: %s" % ", ".join(failed))
        return 1
    print("[result] all gates PASS -> %s" % exe)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
