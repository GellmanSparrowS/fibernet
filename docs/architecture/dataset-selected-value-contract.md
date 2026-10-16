# Selected values in FiberNetDataset

`fibernet.ml.dataset.FiberNetDataset` validates its current selected data when
`features`, `all_labels`, `labels`, `save`, or `to_csv` consumes it. Missing
selected keys no longer become numerical zero. Selected values must be finite
real numeric scalars; `None`, NaN, infinity, complex values, numerical strings,
and other nonnumeric values are rejected. Python and NumPy real numeric scalars
are supported. Genuine numerical zero is valid.
Python and NumPy boolean scalars retain the existing binary convention: false
converts to 0 and true to 1 in numerical matrices.

Selected names must be nonempty strings and unique within each feature or target
selection, supplied as an ordered sequence. Selected sample fields must be
mappings. Errors identify the sample ID, row, and selected field when the error
belongs to a sample. Invalid/duplicate name errors identify the selection/name.
Validation runs on every access because samples and name lists remain mutable.
Constructing a dataset does not itself consume or validate its values.

The feature matrix validates the declared feature selection; the label matrix
validates the declared target selection. `labels(name)` retains its existing
delegation to `all_labels`, so every declared target must be valid. Declare only
the desired target subset when other results are unavailable. Name auto-detection
remains based on the first sample and does not infer a union across samples.

Both selected matrices are checked before `save` creates its directory or writes
files, and before `to_csv` opens or truncates its destination. Invalid selected
data therefore leaves existing destination files unchanged. This is a selected
data preflight, not a guarantee of multi-file transaction atomicity or validation
of arbitrary metadata serialization.

Empty datasets retain exact two-dimensional shapes `(0, n_features)` and
`(0, n_targets)`; selecting a declared target produces an empty `(0,)` vector.
Explicit empty selections on populated datasets produce `(n_samples, 0)`.
Empty splits and saved/loaded empty datasets follow the same shape contract.

Unavailable information may remain in unselected fields or metadata, including
`{"status": "unavailable", "value": null}`. Metadata is not recursively subjected
to selected-value validation. The reference journey's contact/fracture metadata
continues to round-trip. This guard introduces no new availability schema, units
schema, or checkpoint format, and does not infer physical validity from finiteness.

This intentionally changes the previous handling of incomplete or invalid
selected data. No demonstrated caller requires automatic zero imputation, so
there is no silent legacy-fill switch. Complete finite datasets retain their
numerical values and ordered columns. Old CSV/NPY artifacts may already contain
zeros synthesized from missing keys; those zeros are indistinguishable from real
measurements. Loading them cannot recover or verify their original availability.
Upstream extractors that already substituted explicit fallback zeros also remain
outside this boundary's ability to infer provenance.

The separate `DatasetStream` / `PlanarPhysicalDataset` implementation is unchanged.
Its explicit generate-mode NaN labels, finite physics/surrogate label rules,
atomic checkpoint updates, configuration/schema/source bindings, and existing
resume behavior remain their own contracts.

Regression coverage: `tests/test_dataset_selected_values.py` and the existing
`tests/test_dataset_stream_integrity.py` / `test_physical_dataset_cross_end.py`.
