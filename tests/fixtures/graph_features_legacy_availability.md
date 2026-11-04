# Frozen legacy reference for feature availability parity

`graph_features_legacy_availability.py` is a byte-identical test-only copy of
`fibernet/analysis/graph_features.py` from FiberNet main commit
`d9105222c813fe5f82fd6f32bc401650f5d39993` (PR84; this file itself was unchanged).
Git blob: `d4448c7fc24324c901624df359ab3401c416d156`.
SHA256: `f8b9ec2130bf8387cf8c57c7c7c33d0ec11d11d8952f8ab5753765d98040ade9`.
License: project MIT license in repository `LICENSE`, copyright FiberNet Contributors.

The complete module is retained because tests compare all94 original values and
exercise the original shared raster/statistical helpers. Reducing it would edit
the independent oracle or change its imports/initialization; it is not replaced
with the implementation under test. Only small explicit graphs and controlled
mock branches are used. This reference does not qualify native image accuracy.

The public test uses this fixture by default; no private path or network is
required. `FIBERNET_BASELINE_FEATURE_FILE` is an optional local gate override.
The fixture belongs to tests, not the `fibernet` runtime package, and does not add
to its246 source files. Existing MANIFEST.in includes tests' `.py` and fixtures'
`.md` files for sdists; wheels intentionally contain only runtime packages.
Source-checkout/sdist testing and detached installed-package testing are distinct.
