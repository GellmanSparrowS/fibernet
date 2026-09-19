"""Portability audit of the frozen build (onefile exe or onedir folder).

Rule: every DLL/PYD the payload imports must be either bundled inside the
payload, present in System32/SysWOW64, or an api-set stub (api-ms-*/ext-ms-*).
Anything else is MISSING and will fail on a clean judge machine with
0xc00007b, "entry point not found", or a silent exit with no window.

Usage:
    python scripts/audit_dist.py                       # dist/FiberScope.exe
    python scripts/audit_dist.py --path dist/FiberScope
    python scripts/audit_dist.py --top 15              # size breakdown rows
"""
import argparse
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pefile  # noqa: E402

PE_EXT = (".dll", ".pyd", ".exe")
ICU_VERSIONED = re.compile(r"^u[a-z0-9_]+_\d+$")
SYS_DIRS = [os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), d)
            for d in ("System32", "SysWOW64")]


def _system_dlls():
    names = set()
    for d in SYS_DIRS:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith(".dll"):
                    names.add(fn.lower())
    return names


def icu_verdict(name, data):
    """(ok, why) for a bundled ICU dll.

    Qt6Core imports the PLAIN ucnv_* symbols.  A conda/poppler ICU only
    exports version-suffixed names (ucnv_open_78), which is exactly the
    ERROR_PROC_NOT_FOUND that kills the frozen app at QtWidgets import; the
    Windows system ICU exports the plain names, so it is the one to ship.
    """
    low = os.path.basename(name).lower()
    if low.startswith("icudt"):
        return False, "conda/poppler data dll (system ICU has none)"
    try:
        pe = pefile.PE(data=data, fast_load=True)
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
        exp = set()
        d = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
        if d:
            exp = {e.name.decode() for e in d.symbols if e.name}
        pe.close()
    except Exception as e:
        return False, "unparsable: %s" % e
    if "ucnv_open" in exp:
        return True, "system build, plain ucnv_open (%d exports)" % len(exp)
    ver = sorted(n for n in exp if ICU_VERSIONED.match(n))
    return False, "versioned-only exports %s" % (ver[:3] or "(none)")


def collect_pes(path):
    """Return {name: bytes} for every PE in the payload."""
    out = {}
    if os.path.isdir(path):
        for dp, _dirs, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(PE_EXT):
                    full = os.path.join(dp, fn)
                    rel = os.path.relpath(full, path).replace("\\", "/")
                    with open(full, "rb") as fh:
                        out[rel] = fh.read()
        return out
    from PyInstaller.archive.readers import CArchiveReader
    r = CArchiveReader(path)
    for name in r.toc:
        if name.lower().endswith(PE_EXT):
            try:
                out[name.replace("\\", "/")] = r.extract(name)
            except Exception as e:      # pragma: no cover - defensive
                print("  ! cannot extract %s: %s" % (name, e))
    return out


def analyze(pes, top=12):
    sysdlls = _system_dlls()
    bundled = {os.path.basename(n).lower() for n in pes}
    imports = defaultdict(set)
    missing = defaultdict(set)
    icu, upx, non_x64, tiny_runtime = [], [], [], []
    min_win = 0.0
    sizes = defaultdict(int)
    for name, data in pes.items():
        sizes[name.split("/")[0].lower()] += len(data)
        low = os.path.basename(name).lower()
        if low.startswith("icu") and low.endswith(".dll"):
            ok, why = icu_verdict(name, data)
            icu.append((name, ok, why))
        try:
            pe = pefile.PE(data=data, fast_load=True)
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
        except Exception as e:
            print("  ! unparsable %s: %s" % (name, e))
            continue
        if getattr(pe, "FILE_HEADER", None) is not None:
            machine = pe.FILE_HEADER.Machine
            if machine not in (0x8664, 0xAA64):
                non_x64.append("%s (0x%x)" % (name, machine))
            for sec in pe.sections:
                nm = sec.Name.rstrip(b"\x00").decode("latin1", "replace")
                if nm.startswith("UPX"):
                    upx.append(name)
                    break
        oh = getattr(pe, "OPTIONAL_HEADER", None)
        if oh is not None:
            ver = float("%d.%02d" % (oh.MajorSubsystemVersion,
                                     oh.MinorSubsystemVersion))
            min_win = max(min_win, ver)
        if low.endswith(".dll") and len(data) < 10 * 1024:
            tiny_runtime.append("%s (%d B)" % (name, len(data)))
        groups = [getattr(pe, "DIRECTORY_ENTRY_IMPORT", None) or [],
                  getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", None) or []]
        for grp in groups:
            for entry in grp:
                dep = entry.dll.decode("utf-8", "replace").lower()
                imports[name].add(dep)
                dl = os.path.basename(dep)
                if dl in bundled or dl in sysdlls:
                    continue
                if dl.startswith(("api-ms-", "ext-ms-")):
                    continue
                missing[name].add(dep)
        pe.close()
    return dict(n=len(pes), missing=missing, icu=icu, upx=sorted(set(upx)),
                non_x64=non_x64, tiny_runtime=tiny_runtime,
                min_win=min_win, sizes=sizes, total=sum(sizes.values()),
                n_imports=sum(len(v) for v in imports.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    path = args.path or os.path.join(ROOT, "dist", "FiberScope.exe")
    if not os.path.exists(path):
        onedir = os.path.join(ROOT, "dist", "FiberScope")
        path = onedir if os.path.isdir(onedir) else path
    if not os.path.exists(path):
        print("no build to audit:", path)
        return 2
    print("[audit] %s" % path)
    pes = collect_pes(path)
    r = analyze(pes, top=args.top)
    print("  PEs %d, imports %d, payload %.1f MB, min Windows %.2f"
          % (r["n"], r["n_imports"], r["total"] / 1e6, r["min_win"]))
    print("  size by group:")
    for k, v in sorted(r["sizes"].items(), key=lambda kv: -kv[1])[:args.top]:
        print("    %-28s %7.1f MB" % (k, v / 1e6))
    problems = 0
    if r["missing"]:
        problems += 1
        print("  MISSING dependencies:")
        for name, deps in sorted(r["missing"].items())[:20]:
            print("    %s -> %s" % (name, ", ".join(sorted(deps))))
    else:
        print("  MISSING: none")
    for name, ok, why in r["icu"]:
        print("  icu %-14s %-3s %s" % (name, "OK" if ok else "BAD", why))
    if any(not ok for _n, ok, _w in r["icu"]):
        problems += 1
        print("  ICU CONFLICT: ship the system ICU (plain ucnv_* exports) "
              "or none at all")
    for label, items in (("UPX packed", r["upx"]),
                         ("non-x64", r["non_x64"]),
                         ("tiny runtime dlls", r["tiny_runtime"])):
        if items:
            problems += 1
            print("  %s: %s" % (label, ", ".join(items[:8])))
    verdict = "PORTABLE" if not problems else "NOT PORTABLE"
    print("  verdict: %s" % verdict)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())