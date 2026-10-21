# FiberScope generated-input cache contract

`run_stretch` builds the structure once, optionally welds intersections, and
selects a cache using that final graph. `cache_tag(factory, cfg, *, graph)` now
requires the already-built post-weld graph; it does not build or weld anything.
There is no configuration-only fallback. Existing factory/configuration, engine
version/source identity, and frame cap remain part of the key.

The `g1_` filename namespace binds a versioned SHA-256 digest of Engine2 inputs:
- Sorted stable node IDs and engine-converted XY float64 coordinates
- Physical edge IDs in insertion order and directed endpoint row mappings
- `metadata.weld_triplets` mapped to node rows, preserving row order

Arrays use canonical little-endian int64/float64, explicit names/dtypes/shapes,
and C-order bytes without decimal rounding. IDs must be integral (not booleans
or floats), fit signed int64, and reference existing nodes. Masked inputs,
invalid shapes and nonfinite XY are rejected before lookup. Weld-triplet order
and duplicates conservatively affect the key even though the solver may
canonicalize them. Unused Z, radius, material and curve fields are not asserted
to be physical inputs of this solver and are not included in this digest.

Old configuration-only NPZ files remain untouched and cannot match the new
namespace. Cache format, pickle policy, atomic writer and default cache
selection are unchanged. This is input identity, not cache authentication.

`StretchRun.cache_hit` and `StretchRun.cache_key` are appended runtime fields
with defaults `False` and `''`; they are not persisted in NPZ. Only a successful
load returned by `run_stretch` sets `cache_hit=True`. A missing/unreadable file,
fresh solve or write failure remains `False`; GUI status uses this returned
value. Use `run.cache_key` to locate `run_<cache_key>.npz`, rather than rebuilding
a graph to guess its filename. A nonempty key identifies the selected path,
not proof that a file was successfully written.

With no usable cache directory, the key is empty and the factory need not
implement `key()`. With a usable directory, `use_cache=False` still bypasses
reads and writes the fresh result, preserving existing behavior. A caller
manually invoking `cache_tag` is responsible for supplying the exact final graph.

After source changes, run the official maintenance command:
`python FiberScope/scripts/bump_engine_version.py`.

Regression: `python -m pytest -q tests/test_fiberscope_geometry_cache.py` uses
only synthetic results, real disposable NPZ files and pre-physics sentinels.
Worker success/failure dispatch is tested by executing its production method
with signal stubs; this does not constitute a rendered Qt GUI test.
