# Native JSON input

Configuration and native graph/network JSON readers reject repeated object keys
with a `ValueError` identifying the key. This applies to
`ExperimentConfig.from_json`, `StructureGraph.load_json`,
`FiberNetwork.load_json`, `fibernet.api.load` for JSON, and the captured JSON
input parser used by `batch_simulate_from_json`. The batch API records a
parsing error as a failed per-file result; it does not necessarily raise that
error to its caller. Its resume-source fingerprint includes the shared parser.

The check applies at every object depth, including objects in lists. Repeated
keys are rejected even when their values match; escaped spellings that decode
to the same key also count as duplicates. Different objects may use the same
key normally. Older versions kept the last value of a repeated key; remove
ambiguous entries before loading these documents.

Valid native representations keep their existing dispatch, schema, decoding
and number handling. Dictionary-based constructors are unchanged. Syntax
errors without duplicate keys remain `JSONDecodeError`; missing files keep
their normal filesystem errors. Documents containing both syntax errors and
duplicate keys can fail at whichever issue the decoder encounters first.
This policy does not change YAML readers or unrelated result/ML parsers.
