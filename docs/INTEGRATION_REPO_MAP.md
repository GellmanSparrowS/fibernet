# Existing FiberNet repository: integration and release map

The canonical upstream is `https://github.com/GellmanSparrowS/fibernet.git`.
The 4.2.0 integration was merged into that repository's `main` through PR #18
on 2026-09-27. The package retains its existing `fibernet` name. No separate
Python distribution or GitHub repository has been created.

`FiberScope/` is the APP source used for cross-end validation. The library
wheel is distributed by PyPI, while the Windows executable belongs in a
GitHub Release asset. Git-tracked source files do not include the frozen APP.

The tested homepage source is `docs/README_VNEXT.md`; the root `README.md`
is generated from it with corrected paths by `scripts/promote_homepage.py`.
The source-backed media live in `docs/media/`.
