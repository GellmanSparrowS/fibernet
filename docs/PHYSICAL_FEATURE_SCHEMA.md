# Declaring physical-regressor feature identity

`Regressor.train(X, Y, *, feature_schema=None)` accepts any supported feature
width. Width alone does not identify what the columns mean.

- Omit `feature_schema` for arrays of unknown provenance. The returned history
  has `schema=None`, including for 14-column arrays.
- For columns produced by the canonical `light_features` extractor, pass
  `feature_schema=FEATURE_SCHEMA` from `fibernet.ml.physical_learning`.
  This declaration requires exactly 14 columns.
- A custom declaration must be a nonempty printable string of at most 128
  characters, without leading or trailing whitespace. It is preserved exactly;
  invalid values are rejected rather than converted or trimmed. Custom names
  are opaque caller declarations, not verified column names, units or geometry.
- History always reports `feature_count=X.shape[1]`. Feature declarations do not
  change fitting, normalization, validation rows or prediction width checks.

```python
from fibernet.ml.physical_learning import Regressor, FEATURE_SCHEMA

# X must actually come from the canonical light_features extractor.
history = Regressor("ridge").train(X, Y, feature_schema=FEATURE_SCHEMA)
```

Existing positional training arguments remain valid; `feature_schema` is
keyword-only. Code consuming history should now handle `schema=None` and use
`feature_count` for width. Older histories that always contained the canonical
schema are not evidence of feature provenance.

The workflow example leaves synthetic and imported NPZ arrays undeclared and
writes the returned schema and count to its summary. FiberScope declares the
known generation pipeline, but does not infer identity from cached filenames
or unchecked cached arrays. Existing dataset formats and cache identifiers are
unchanged. Invalid declarations are rejected before backend construction or
callbacks and leave a previous completed fit intact.
