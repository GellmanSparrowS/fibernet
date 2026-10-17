# Explicit source-group dataset splitting

Existing calls retain their historical row-random behavior:

```python
train, validation, test = dataset.split((0.7, 0.15, 0.15), 42)
```

That default is not source-group-safe. Related material placements, scales,
tiles, or other variants can land in different partitions.

Use the additive, explicit group mode when rows share parent sources:

```python
# These metadata keys are a caller-owned convention, not an inferred schema.
groups = [sample.metadata["parent_source"] for sample in dataset.samples]
train, validation, test = dataset.split(
    ratios=(0.7, 0.15, 0.15),
    random_state=42,
    method="group",
    groups=groups,
)
```

## Contract

- groups is a list or tuple of nonempty strings, one per dataset row. Related
  variants must share the same source label. Labels are explicit caller
  assertions; this function cannot verify genealogy or repair false labels.
  Unique sample IDs are not evidence of independent sources. Labels retain
  exact Python string equality: no trimming, Unicode normalization, or NumPy
  string conversion is applied. For example, "source" and "source\0" are
  distinct labels. Sorted distinct labels are encoded as integer group IDs,
  preserving deterministic assignment when input rows are reordered.
- All three requested ratios must be positive finite real values, sum to one
  within absolute tolerance 1e-12, and permit three nonempty group partitions.
  At least three distinct groups are required. Group seed is a non-boolean
  integer from 0 through 2**32 - 1.
- Fractions refer to **groups, not rows**. With G groups, first reserve
  ceil(G * test_fraction) test groups. Of the remaining D groups, reserve
  ceil(D * validation_fraction / (train_fraction + validation_fraction))
  validation groups. The rest train. Reject an allocation that empties any
  partition. To avoid intermediate floating-point roundoff adding a group,
  compute these quotas exactly from each validated float's shortest decimal
  spelling (using rational arithmetic), then take the integer ceiling. For
  example, ten groups with ratios (0.7, 0.2, 0.1) allocate 7/2/1 groups.
  No near-integer snapping is performed: a caller-provided decimal value
  strictly above a boundary remains above it, even when close. The accepted
  sum tolerance does not renormalize the test fraction. This rounds like two
  sequential group holdouts and does not guarantee exact requested row ratios.
  Unequal-size groups can produce substantially different row counts.
- It uses the existing scikit-learn GroupShuffleSplit backend twice, with a
  local random generator seeded by random_state. It does not fit a model.
  Same inputs, seed and tested backend give the same assignment; it does not
  consume NumPy's global RNG. Rows within each output follow original input
  order.
- Invalid inputs and infeasible counts reject before importing the optional
  splitter or constructing result datasets. The final checks require complete,
  nonduplicated row coverage, pairwise-disjoint exact source-group sets, and
  the allocated group counts before constructing any result datasets.
- Source data is not changed. Outputs are new dataset objects with new sample
  lists, retaining the original sample objects and metadata as the legacy
  split does. They are not deep copies of samples or metadata.
- Row mode does not require scikit-learn. Group mode reports an explicit
  ImportError if it is unavailable; it never silently falls back to row mode.
  Providing groups while leaving method="row" is rejected.
- The return remains the existing three-dataset tuple. There is no new Graph,
  dataset class, trainer, experiment registry or implicit metadata field.

## Evaluation exposure is unchanged

Splitting does not erase prior use. Previously evaluated parent groups remain
consumed, even if this call places them in the return value named test.
Metadata such as prior campaign/exposure records is preserved, not inferred,
rewritten, or certified. A fresh final test requires parent groups that have
never informed model selection. Use a separately reviewed exposure/split
manifest for that purpose. This function is a partition operation, not an
untouched-test validator.

## Generation-checkpoint compatibility

The existing generation contract hashes ml/dataset.py as part of its
implementation binding. This additive edit therefore changes that binding.
Resume an old generation campaign using its original source snapshot; do not
weaken the source guard or claim cross-source resume compatibility. No
generation/checkpoint code, format, or guard is changed by this patch.

## Verification boundary

Focused fixtures cover repeated parents/material variants, uneven group sizes,
deterministic seeds, preserved ordering/metadata, consumed-test records,
invalid labels/ratios/seeds, absent optional dependencies, and defensive
partition checks. This is not a training result, physics validation, full
package import test, or full regression-suite pass.

