# Existing FiberNet repository: local integration map

The canonical upstream is `https://github.com/GellmanSparrowS/fibernet.git`.
`E:/GOAI/FiberNetUnified` is a local checkout of that **same repository**, based
on public commit `aeea216` and branch `codex/fibernet-unification`. The package
keeps its existing `fibernet` name; this integration branch declares version
4.2.0. No separate Python distribution or GitHub repository has been created.

The older `E:/GOAI/fibernet` directory is part of a different dirty local Git
layout, so this isolated checkout prevents integration work from mixing with
unrelated changes. `FiberScope/` in this checkout is the APP source used for
cross-end validation. The user authorized publishing this integration source
and homepage to the existing GitHub repository on 2026-09-24. A source push is
separate from publishing a PyPI wheel or a new frozen APP release.

The tested homepage source is `docs/README_VNEXT.md`; the root `README.md`
is generated from it with corrected paths by `scripts/promote_homepage.py`.
The source-backed media live in `docs/media/`.
