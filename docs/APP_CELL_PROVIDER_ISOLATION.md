# FiberScope cell providers and cache identity

FiberScope's visible cell labels are unchanged. `StructureFactory(unit='kagome')`
continues to use FiberScope's existing geometry. Library
`pattern_2d(unit='kagome')` keeps the library's separate existing definition;
initializing FiberScope no longer replaces it. This change does not certify either
shape as a scientific standard-cell definition.

APP cells use qualified internal registry keys. The public label remains in APP
configuration and, when present, `metadata['pattern']['unit']`; additive
`metadata['generator_identity']` records provider, visible label, effective lookup
key, legacy definition version and requested topology. Cell/points providers include
an ordered geometry digest. Classic library providers instead record callable
module/name and source digest where inspectable; this is not a complete fingerprint
of closure captures or transitive dependencies.
For topnet26 it also identifies the manufacturing compiler. APP JSON exports carry
this identity; graph-native JSON preserves it through normal metadata serialization.
Workflow records are not claimed to acquire full provider provenance automatically.

Custom cells with an exact existing APP key retain their existing override priority;
square/spectrum presets retain their point-provider priority. Case-only ambiguous
keys are rejected with the affected names for explicit migration. No file is silently
renamed or rewritten. Reload, deletion and editor preview changes invalidate stale
intermediate-node counts. Deleted cells cannot be recovered accidentally from old
qualified registry entries.

Code that relied on importing FiberScope to register a bare custom/library name
must use StructureFactory, or explicitly call the public library register_unit API
for its own factory. Bare compatibility aliases are deliberately not provided.
Qualified names are a convention, not a security boundary; list_units may expose
them. Concurrent custom edits and deliberate qualified-name overrides are not
isolated by this change.

The engine fingerprint changes with this source change. New caches and dataset
configuration IDs are expected; old checkpoint resume must reject a different
configuration rather than relabel historical data. Feature dimensionality/schema
is unchanged. This source version records `e2v14 / a35e2dffdbee` for new outputs only.
Historical `e2v13 / bf791d21316c` results remain unchanged; different source builds
may have different fingerprints.
