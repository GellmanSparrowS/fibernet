# Analysis and optional OpenCV support

Importing `fibernet.analysis` or calling `analyze_topology(graph)` (including
`fibernet.api.analyze(graph, kind="topology")`) does not probe OpenCV. Selecting
`GraphFeatureExtractor` loads its implementation and checks image dependencies.
The analysis, root and ML exports resolve to the same class, with unchanged
constructor and extraction signatures. `__all__` and `dir()` retain its name.

A missing `cv2` module retains the existing unavailable-feature reports and
`FeatureUnavailableError` from default extraction. `allow_unavailable=True`
continues to opt into the historical placeholder results. A broken OpenCV
installation now propagates its original import error rather than being reported
as an absent optional dependency; this includes missing nested dependencies.

The analysis module caches the class only after successful resolution. Its
module dictionary no longer contains that class before selection. Wildcard
imports and introspection that resolves every attribute can still load it.
Reloading the facade clears its alias, not the cached implementation module.
This boundary does not change image algorithms or establish native OpenCV
numerical compatibility.
