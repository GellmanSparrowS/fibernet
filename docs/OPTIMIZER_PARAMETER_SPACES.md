# Optimizer parameter spaces

`MultiObjectiveOptimizer` uses tuples for numeric intervals and lists for
categorical choices:

```python
param_space = {
    "grid_x": (2, 6),              # Any integer from 2 through 6
    "damping": (0.1, 0.9),         # A continuous numeric interval
    "unit": ["square", "hex"],     # Exactly these two categories
    "segments": [2, 4],            # Exactly 2 or 4, never 3
}
```

Previously, a list containing exactly two items was incorrectly interpreted as
a numeric interval. String choices could fail to convert to a number, and
numeric choices could produce intermediate values outside the requested list.
Two-item lists now follow the same categorical rule as other lists.

## Updating existing configurations

- If a two-item list was intended as an interval, replace it with a tuple.
  For example, change `[2, 6]` to `(2, 6)`.
- If the configuration is stored in JSON, use an explicit dictionary because
  JSON does not preserve Python tuples:

  ```json
  {"grid_x": {"type": "int", "low": 2, "high": 6}}
  ```

  Continuous intervals use `"type": "float"`. Explicit categories can use
  `{"type": "categorical", "choices": [2, 4]}`.
- When resuming a stored Optuna study, preserve the existing parameter's
  distribution type. Use a tuple or explicit interval dictionary to keep a
  previous interval. If changing that parameter to categorical choices,
  use a new study name rather than reusing the incompatible stored study.

Tuple intervals, explicit parameter dictionaries, and list categories with
other lengths retain their existing behavior.
