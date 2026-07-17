# FiberNet — Task Progress

## Latest Update (2026-07-17): README Overhaul & CI Fix

### Phase 16: Professional Bilingual README ✅
- **README.md**: English-only, clean professional layout
- **README_CN.md**: Chinese-only, matching structure  
- **Language toggle**: `[中文文档](README_CN.md)` / `[English](README.md)` links at top
- **CI badge** added to header
- **scripts/generate_readme.py**: template-based generator for consistency
- All code examples verified working
- Commit: `77bba69` (push pending — network timeout)

### Phase 15: CI Fix — Taichi SNode Exhaustion ✅
- **Root cause**: `compute_forces()` leaked 6 Taichi fields per call (no caching)
- **Fix**: Cached `compute_forces` fields + `TaichiEngine.clear_field_cache()` classmethod
- **Test isolation**: Added `pytest-forked` + `--forked` to pyproject.toml
- **Result**: 189 passed, 6 skipped, 0 errors
- Commit: `b5844b7`

### Phase 15.5: Cleanup Obsolete Files ✅
- Removed: `_old_scripts/`, `_scripts_archive/`, `analysis_results/`, `analysis_scripts/`, `output_data/`, `output_viz/`
- Removed: `test_readme_blocks.json`, `README_UPDATE_PLAN.md`
- Updated `.gitignore` to prevent re-tracking
- 77 files deleted, 32K+ lines removed
- Commit: `6b2fb70`

## Current Status

### Completed ✅
- CI fix (Taichi segfault → 189 tests pass)
- Cleanup (77 obsolete files removed)
- Professional bilingual README (EN + CN with language toggle)
- README generator script for maintainability

### Pending ⏳
- **Push to GitHub** (network timeout — retry `git push origin main`)
- Check CI passes on GitHub after push

## Git Log (recent)
```
77bba69 docs: professional bilingual README with language toggle
6b2fb70 chore: cleanup obsolete directories and files
b5844b7 fix: resolve Taichi SNode exhaustion segfault in CI tests
820466d docs: update PROGRESS.md - Phase 14 README verification
c1080c7 docs: fix README Block 3 (keyword args, displacement count)
```

## Files Changed
- `README.md` — rewritten (English, professional layout)
- `README_CN.md` — new (Chinese, matching structure)
- `scripts/generate_readme.py` — new (template-based generator)

## Library Version: v4.0.5 (on PyPI)
