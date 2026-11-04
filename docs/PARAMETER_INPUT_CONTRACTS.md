# Parameter input contracts

These entry points serve different data models. Their scalar policies are not interchangeable.

| Entry point | Count/type contract | Zero and compatibility |
| --- | --- | --- |
| `fibernet.gen.ordered.square_lattice_2d` | Exactly two nonnegative Python/NumPy integer counts; no bool or float counts. Spacing and radius are finite positive real scalars. | Existing zero-axis strip/empty-grid behavior is retained. |
| `fibernet.gen.disordered.random_straight_2d` | Nonnegative Python/NumPy integer fiber count; no bool or float counts. Fiber length, radius and effective box extents are finite positive real scalars. | Zero fibers produce an empty network when the other inputs are valid. Empty and one-item box sequences retain their existing default/broadcast behavior; only the first two sequence extents are used. |
| `tile_2d`, `tile_3d`, `tile_with_transforms`, `pattern_2d`, `pattern_3d` | A nonnegative Python/NumPy integer scalar broadcasts to all axes, or an iterable supplies exactly two/three integer counts. Python and NumPy bools, floats and negative counts are rejected. Counts normalize to a tuple of Python integers. | A zero axis retains empty tiling behavior; it does not bypass unit construction or other existing validation. Invalid grids are rejected before unit-factory or tiling work. |
| `fibernet.utils.validate_range` and `validate_positive` | The existing built-in `int`/`float` type gate remains, including its bool and NumPy-scalar behavior. NaN is rejected with `ValidationError`. | Existing bound inclusivity, infinity comparisons, arbitrary-size integer handling and successful `None` returns remain. These helpers are not finite-positive-real validators. |

`validate_type` reports tuple and nested-tuple type alternatives in its `ValidationError` message. `ParameterValidator` continues collecting these validation errors and returns `None` on successful final validation.

## Usage

```python
import numpy as np
import fibernet as fn
from fibernet.gen.ordered import square_lattice_2d
from fibernet.utils import validate_range

network = square_lattice_2d(grid_size=(np.int64(2), 3), spacing=1.0, radius=0.05)
graph = fn.pattern_2d(unit="square", grid=np.int64(2), box=(1.0, 1.0), radius=0.05)
validate_range(0.5, "ratio", min_val=0.0, max_val=1.0)
```

Invalid count inputs now raise `ValueError` rather than relying on loop/unpacking failures or accidentally accepting booleans and negative grids. Grid admission may therefore precede errors for other invalid arguments. Strings, bytes and dictionaries are not grid specifications. Normalization preserves ordinary integer values, not custom integer-subclass arithmetic or object identity.

The ordinary generators above return `FiberNetwork`; the tiling/pattern interfaces return `StructureGraph`. Converting between those representations remains approximate: these parameter checks do not establish lossless preservation of arbitrary centerlines, crosslinks, metadata or IDs, nor do they validate physical-model suitability.
