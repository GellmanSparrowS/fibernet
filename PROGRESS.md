# FiberNet — Task Progress

## Latest Update (2026-07-17): CI 全绿 ✅

### Phase 18: macOS CI Fix — Taichi Version Check ✅
- **Root cause**: Taichi background thread calls `urllib.request.urlopen` → `getproxies_macosx_sysconf` → signal 6 abort
- **Fix**: monkey-patch `taichi._version_check` at conftest.py module level (before any test imports taichi)
- **Result**: ALL 12 CI jobs pass (Linux×4 + macOS×4 + Windows×4)
- Commit: `b2d9fad`

### Phase 17: Cross-Platform CI ✅
- Windows: skip Taichi simulation tests (no fork + SNode exhaustion)
- macOS: `TI_DISABLE_VERSION_CHECK=1` (not sufficient alone, see Phase 18)
- Linux: `--forked` (isolated subprocesses)

### Phase 16: Professional Bilingual README ✅
- `README.md` (EN) + `README_CN.md` (CN) with language toggle
- `scripts/generate_readme.py` for maintainability
- Commit: `77bba69`

### Phase 15.5: Cleanup ✅
- 77 obsolete files removed (32K+ lines)
- Commit: `6b2fb70`

### Phase 15: CI Fix — Taichi SNode Exhaustion ✅
- Cached `compute_forces()` + `clear_field_cache()` classmethod
- Added `pytest-forked`
- Commit: `b5844b7`

## Git Log
```
b2d9fad fix: monkey-patch Taichi version check at conftest module level
27d392e fix: cross-platform CI — skip Taichi on Windows, disable version check
77bba69 docs: professional bilingual README with language toggle
6b2fb70 chore: cleanup obsolete directories and files
b5844b7 fix: resolve Taichi SNode exhaustion segfault in CI tests
```

## Status
- ✅ CI: 189 passed, 6 skipped (all 12 jobs green)
- ✅ Cleanup: 77 obsolete files removed
- ✅ README: professional bilingual (EN/CN + toggle)
- ℹ️ PyPI: v4.0.5 (unchanged — only updates on `git tag v*`)

## Library Version: v4.0.5 (on PyPI)

### Phase 19: Lab Homepage HTML ✅
- Created `fibernet_cn.html` (Chinese, academic style, no emoji)
- Created `fibernet_en.html` (English version)
- Output: `/media/sf_share/`
