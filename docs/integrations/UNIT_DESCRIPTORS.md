# Unit declarations (descriptor-only v1)

`fibernet.gen.unit_contract.UnitContracts` is an explicit in-memory catalog. It
reads the existing 2D/3D unit registry and never invokes a factory, generates a
graph, imports a named plugin, or loads source paths. No root alias is added.
Existing `register_unit` / `register_unit_3d` registrations and legal overrides
continue to work. No descriptor is installed automatically for built-ins.

```python
from fibernet.gen.unit_contract import UnitContracts, SCHEMA_VERSION

catalog = UnitContracts()
print(catalog.describe('square', 2))  # undeclared; parameters null; capabilities unknown
catalog.declare('square', 2, {
    'descriptor_schema': SCHEMA_VERSION,
    'plugin_id': 'my-lab/square-declaration', 'plugin_version': '1',
    'kind': 'unit_generator', 'dimensions': [2],
    'representation': 'StructureGraph', 'parameter_schema_version': '1',
    'parameters': None, 'capabilities': 'unknown',
    'randomness': 'unknown', 'id_policy': 'unknown',
})
```

This is an author declaration, not verified behavior. Even a declaration claiming
StructureGraph or a capability is not checked by executing the factory. Descriptor
fields are exactly those shown. Dimensions are a nonempty unique list of integer
2/3 values including the selected registry dimension. Version/ID fields are
nonblank strings; the module does not verify semantic versions or namespace
ownership. `randomness` and `id_policy` accept `unknown` or a finite JSON object
of author-supplied policy details. Capabilities accept `unknown` or a map of names
to `supported`, `unsupported`, or `unknown`. Unknown is preferable to guessing.

## Small parameter schema

`parameters=None` explicitly means unknown. Resolution then raises; no hidden
signature inference or guessed defaults. A declared schema must have type `object`:

```python
schema = {
    'type': 'object',
    'properties': {
        'radius': {'type': 'number', 'minimum': 0, 'default': 0.1,
                   'units': 'unscaled'},
        'seed': {'type': 'integer', 'minimum': 0},
    },
    'required': ['seed'], 'additionalProperties': False,
}
# Put schema into the descriptor's parameters field and declare it explicitly.
# result = catalog.resolve_parameters('square', 2, {'seed': 42})
# result contains detached raw and effective parameters; it does not call pattern_2d.
```

This illustrative schema is a caller-chosen subset, not an automatically verified
schema for all square factory options. It must not be presented as the actual
factory's complete parameter acceptance rules.

Supported types: object, array, string, boolean, integer, finite number. All support
optional `default`, nonempty `enum`, and nonblank `units`. Numbers support inclusive
`minimum`/`maximum`. Arrays require `items`; optional nonnegative integer
`minItems`/`maxItems` specify length (nested arrays express fixed shapes). Objects
require `properties` and explicit boolean `additionalProperties`; optional
`required` lists unique declared names. A default satisfies required membership.
Defaults are applied recursively when a containing object is present or defaulted;
an absent optional object is not created without its own default. Unknown schema
keywords reject. There are no references, union types, regexes, callbacks or unit
conversions. Explicit null is allowed as an undeclared/open extra JSON value, not
as a typed parameter. Bool is never an integer/number. Enum comparisons use strict
canonical JSON representation (thus integer 1 and float 1.0 are distinct enums).

`canonical_json` only accepts built-in dict/list/string/bool/int/finite-float/null,
with string dictionary keys. Cycles and nesting beyond 100 levels reject with
ValueError; shared noncyclic containers are allowed and detached. It sorts keys without reordering lists or coercing
numbers, names or units. Nonfinite numbers, tuples, NumPy scalar objects and custom
serialization hooks reject. Raw/effective records and returned descriptions are
detached; changing them cannot change the stored declaration.

## Callable identity and restoration

A declaration captures the exact registered callable object. Every lookup and
parameter resolution checks current object identity. A replacement raises a stale
descriptor error until explicitly redeclared or the exact original object is
restored. Restoring that same object restores its original declaration association;
this catalog does not observe registration history. A replace/restore between
lookups cannot be detected. In-place mutation of a callable/closure/global state
cannot be detected either. Missing or noncallable registry entries raise before any
callback. Dimension registries and catalogs are separate. Name lookup uses the
existing lowercased key and also reports original requested spelling; it does not
change pattern's historically case-sensitive postprocessing behavior.

## Evidence and resume boundary

All descriptions and resolved parameters have `resume_eligible=False`.
`verify_source_bytes(bytes, expected_sha256)` checks only explicitly supplied bytes
against a lowercase SHA256 digest. A mismatch raises; a match returns
`supplied_bytes_match`, `callable_link_verified=False`, `resume_eligible=False`.
It reads no files and does not establish that those bytes implement the registered
callable. It does not change catalog status. Identical function code or names can
capture different state. Neither this helper nor a declaration certifies closure
purity, determinism, physical behavior, dependency closure or checkpoint safety.
Native `FiberNetDataset.generate` custom-factory checkpoint rejection stays intact.
No new descriptor is inserted into native sample IDs or experiment bindings.

## Scope and validation

This catalog only inspects unit declarations. It does not implement transforms, welding lineage, clipping,
defects, material assignment or legacy disordered-graph conversion. Physical
members, parallel edge IDs and bidirectional ML message edges retain their
existing separate meanings; no metric or legacy ML feature is redefined.

The focused tests are pure metadata/registry fixtures with factories that fail if
invoked. They cover seven gates: detached strict JSON; invalid schema rejection;
undeclared/missing diagnostics; replacement/restoration identity; closure and byte
limits; unchanged generator/Graph sources; unchanged custom-resume refusal. They
are not graph-generation or scientific validation runs.
