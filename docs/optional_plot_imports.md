# Deferred plotting imports

Importing `fibernet`, its core types, `fibernet.rl.env`, or the Easy API no
longer loads Matplotlib, initializes a plotting backend, or runs Matplotlib's
font discovery. The normal Python package namespace is used throughout.

Top-level rendering exports and the graph/publication exports in `fibernet.viz`
resolve to their original implementation objects on explicit attribute access
or `from ... import name`. Their signatures, implementation code, themes, and
plotting-time errors are unchanged. Wildcard imports from `fibernet` and
`fibernet.viz` still resolve all names in their existing `__all__` lists and
therefore request plotting. `dir()` includes the deferred names without loading
the rendering modules. Failed imports are not cached by the package resolver.

The `fibernet.viz.render_network_3d` export continues to select the publication
renderer, which historically overwrote the visualization-module alias. The Easy
API imports its renderer inside `show()`; direct imports of its historical
`render_graph` and `render_deformation` re-exports also remain available.
Its explicit `__all__` preserves every historical wildcard name, including the
two rendering re-exports and incidental public imports. Importing the module
does not resolve those names; an explicit wildcard import does.

This change does not alter Matplotlib's existing Agg selection when a rendering
module is requested. It introduces no import wrappers, substitute packages,
backend/cache configuration, or global interpreter changes. Matplotlib remains
listed in package installation requirements; changing dependency metadata is a
separate decision. The runtime import boundary can nevertheless be tested with
Matplotlib unavailable.

## Focused verification

`tests/test_lazy_plot_imports.py` is usable with unittest or pytest. Its real-RL
import check runs with an existing Gymnasium installation and skips cleanly
when Gymnasium is absent. A separate core-import check blocks both Gym variants
and plotting libraries and still runs without either dependency. It launches fresh
interpreters so previous plotting imports cannot mask eager initialization.
The tiny figure check builds only a two-node, one-edge graph and performs no
scientific solve, RL episode, training, or campaign work.

