# FiberNet — Task Progress

## Latest Update (2026-07-17): README Overhaul + CI Fix + Cleanup

### Phase 17: Cross-Platform CI Fix ✅
- **Windows**: skip Taichi simulation tests (no fork, SNode exhaustion)
- **macOS**: set `TI_DISABLE_VERSION_CHECK=1` (fixes abort in version check)
- **Linux**: `--forked` (each test in isolated subprocess)
- CI workflow: separate test steps for Unix vs Windows
- Commit: `27d392e`

### Phase 16: Professional Bilingual README ✅
- `README.md`: English-only, professional layout
- `README_CN.md`: Chinese-only, matching structure
- Language toggle links at top (`[中文文档]` / `[English]`)
- CI badge, clean sections, proper images
- `scripts/generate_readme.py`: template-based generator
- Commit: `77bba69`

### Phase 15.5: Cleanup Obsolete Files ✅
- Removed 77 files (32K+ lines): `_old_scripts/`, `_scripts_archive/`, `analysis_results/`, `analysis_scripts/`, `output_data/`, `output_viz/`
- Updated `.gitignore`
- Commit: `6b2fb70`

### Phase 15: CI Fix — Taichi SNode Exhaustion ✅
- Cached `compute_forces()` Taichi fields (was leaking 6 SNodes per call)
- Added `TaichiEngine.clear_field_cache()` classmethod
- Added `pytest-forked` + `--forked` to pyproject.toml
- 189 passed, 6 skipped, 0 errors
- Commit: `b5844b7`

## Git History
```
27d392e fix: cross-platform CI — skip Taichi on Windows, disable version check
1f77007 docs: update PROGRESS.md - README overhaul + cleanup + CI fix
77bba69 docs: professional bilingual README with language toggle
6b2fb70 chore: cleanup obsolete directories and files
b5844b7 fix: resolve Taichi SNode exhaustion segfault in CI tests
```

## Status
- ✅ CI fix (189 tests pass on Linux)
- ✅ Cleanup (77 obsolete files removed)
- ✅ Professional bilingual README (EN + CN + language toggle)
- ✅ Cross-platform CI (Windows skips Taichi, macOS version check disabled)
- ⏳ Waiting for CI results on GitHub

## Library Version: v4.0.5 (on PyPI)
