# Loading FiberNetDataset safely

`FiberNetDataset.load(directory)` reads the existing three-file format:
`features.npy`, `labels.npy`, and `metadata.json`. No migration, manifest or
new dependency is required. The public method signature and return type are
unchanged.

```python
from fibernet.ml.dataset import FiberNetDataset

loaded = FiberNetDataset.load("datasets/current")
X = loaded.features
Y = loaded.all_labels
```

## What loading checks

Before returning a dataset, the loader checks:

- Metadata is an object with ordered feature/target name arrays, `n_samples`,
  and a sample-record array
- Every selected name is a nonblank string and is unique within its own axis.
  Names are not trimmed, sorted or renamed; the same name may occur once on
  each axis
- The declared count is a nonnegative integer, excluding booleans, and exactly
  matches the number of sample records
- Each sample record is an object containing `id`, `unit`, and an array-valued
  `grid`. Grid elements are retained and converted to a tuple
- Each numerical file is one NPY array with exact row and column counts.
  Extra rows/columns, rank changes and disguised NPZ archives are rejected
- Stored arrays have boolean, signed/unsigned integer or real floating-point
  dtype. Strings, complex values (including zero imaginary parts), object,
  structured, datetime and timedelta storage are rejected before conversion
- Every selected value remains finite after the existing conversion to Python
  float. This also catches a finite extended-precision value that overflows
  the float representation

NumPy loading explicitly disables pickle. A disguised NPZ is closed before
rejection. Invalid input is never repaired, silently truncated, filled with
zeros, or rewritten. Loading success and failure both leave source bytes alone.

The mutable in-memory dataset still validates each selected matrix when it is
used, as described in [the selected-value contract](dataset-selected-value-contract.md).
Construction itself remains permissive. Changes to samples or names after a
successful load can therefore make a later matrix access fail.

## Compatibility

Complete finite old saves retain their values, ordered columns, records and
metadata. Boolean storage retains the binary convention, false → 0 and true → 1.
Ordinary real numeric precision/layout/byte-order differences are supported.
Conversion still uses Python float, so it is not a lossless precision guarantee
for every large integer or very small real value.

Modern empty arrays use `(0, number_of_names)`. The historical writer instead
stored `(0,)`, even with declared names. That exact old representation is accepted
when both the declared count and number of sample records are zero. The loaded
matrices expose the modern two-dimensional shapes. This exception applies
independently to each numerical file; it does not allow other empty shapes or
arbitrary zero-size arrays. With N records and no selected names, `(N,0)` is
required.

Sample IDs remain positional and may repeat. The loader does not add ID/unit
content rules or grid dimension/positivity rules. Optional `params` and `metadata`
retain their contents, including nested nulls, unavailable-value descriptors and
legacy metadata NaN/Infinity. Missing optional fields default to `{}`.

Present non-object `params`/`metadata` values retain their historical load
behavior too. A later `save()` may reject such a value because that writer expects
mapping-style `.items()`. Successful loading does not promise every historically
accepted opaque metadata value is re-saveable. Unknown top-level/record keys
remain ignored and are not promised to round-trip.

Historic imputed zeros cannot be distinguished from genuine numerical zeros.
Both remain zero; loading cannot recover the original missingness.

## Errors

Structural and selected-value violations raise `ValueError`. Messages identify
the metadata field or numerical filename and expected versus actual shape/type.
A bad numerical value identifies its sample ID, row, feature/target name and
failure to remain finite after float conversion. Duplicate names fail before
dictionary reconstruction can overwrite a column.

Normal filesystem and parser errors remain informative: missing files raise
`FileNotFoundError`, malformed JSON raises `JSONDecodeError`, and corrupt NPY
files retain NumPy's safe-reader errors. No unsafe/pickle fallback is attempted.
No catch-all wrapper hides memory failures or process interruptions.

The loader validates the metadata envelope first, then both array envelopes,
then scalar values during sample reconstruction. For an artifact with several
faults, the first reported fault follows that order and may differ from the old
loader's incidental indexing error.

## What success cannot prove

This is a structural and selected-value check, not a provenance or physical
validation system. Same-shaped arrays from different saves, reordered metadata
rows, wrong same-length names, or swapped same-shaped files can pass. The format
has no hashes or generation identifiers binding the three files together.

Do not use a successful load to dismiss a recovery-required save exception.
Follow [the persistence recovery procedure](dataset-persistence-usage.md) and
coordinate exclusive access. Loading adds no crash-atomicity, concurrent-reader
snapshot, unit validation, physical-validity or historic-missingness guarantees.

The loader remains eager: arrays, metadata and reconstructed sample dictionaries
are held in memory. Validation reuses the reconstruction loop rather than making
extra dense float64 matrices, but it is not a streaming or arbitrary-size-file
interface.

Regression coverage: `tests/test_dataset_load_validation.py`, including captured
actual historical-writer bytes, empty datasets, both numerical axes, explicit
no-pickle loading, NPZ closure, representability, permissive metadata, unchanged
source bytes, and the documented limits on detecting generation mixes.
