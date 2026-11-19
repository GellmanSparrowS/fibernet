# Crosslinks when combining networks

`fibernet.core.transform.merge` copies the fibers and recorded crosslinks of
each input. With `merge_crosslinks=True` and a proximity threshold, newly
detected welds connect different input entries. It preserves each input's
existing weld records, including parallel records and their attributes.

```python
import numpy as np
from fibernet.core.transform import merge
from fibernet.gen import ZigZagGenerator

unit = ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network()
combined = merge(
    [unit, unit],
    offsets=[np.zeros(3), np.array([10000., 0., 0.])],
    merge_crosslinks=True,
    proximity_threshold=0.5,
)
assert combined.num_fibers == 2 * unit.num_fibers
assert combined.num_crosslinks == 2 * unit.num_crosslinks
```

The separated copies retain their original connectivity. Nearby fibers within
one input are not automatically welded by combining it with another network;
use that input's explicit crosslink-building operations when such joins are
intended. Repeating the same object as two input entries creates two independent
copies, and genuine cross-input contacts are still detected.

`tile`, `duplicate_and_transform` and `create_pattern` use this same merge
behavior. Their existing placement, proximity thresholds and periodic flags
are preserved.
