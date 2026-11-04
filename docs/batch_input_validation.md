# Batch input validation

`fibernet.batch_simulate(configs, output="results.csv")` validates the batch's input shape before generating or simulating a case.

## Accepted inputs

Pass a list of dictionaries. Tuples and other sized iterables are also supported: the input is snapshotted once before processing. Each entry must expose a callable `.get` method, so mapping subclasses and compatible dictionary-like objects remain usable. Unsized iterators, including generators, are rejected before consumption; materialize them as a list first.

Without `report_path`, a malformed entry produces an indexed `ValueError`, such as `configs[1] must provide a callable .get method`. All entry shapes are checked before any generation or simulation starts, so a malformed later entry cannot discard earlier case work. Default calls do not require JSON-safe configuration values.

## Optional reports

With `report_path`, the existing stricter report contract is unchanged: original inputs must be a list or tuple of plain dictionaries containing finite, acyclic JSON-safe values with string keys. These checks happen before normalization and retain their report-specific validation errors rather than indexed entry errors. Report destinations must meet the existing path requirements.

## Scope

This preflight checks input shape, not the meaning of every field or the behavior of custom `.get` implementations. Genuine generation and simulation failures retain their existing per-case handling. An empty batch still raises `RuntimeError`; when requested, its diagnostic report records an empty, all-failed batch. This validation does not introduce a physical solver check.
