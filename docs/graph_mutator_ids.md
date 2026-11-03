# Node and edge IDs when editing a StructureGraph

Node movement and node/edge removal accept signed Python and NumPy integer IDs,
including negative stable IDs already present in a live graph. These are lookup
keys, not array-row indices. Booleans, floating-point values and other non-integer
IDs raise `ValueError` before any graph edit. This prevents `True` or `1.0` from
accidentally changing node or edge 1.

Unknown valid IDs retain existing behavior: removal does nothing, while movement
raises `KeyError`. Batch position changes validate every supplied entry before
applying any update. Parallel edges, their remaining deduplication representative,
curve records and stable ID counters retain their existing behavior.

This preserves live-graph editing behavior; it does not broaden the existing
nonnegative-ID rules of the serialized graph reader or downstream adapters.

Mappings follow ordinary Python dictionary semantics: keys already combined by
Python before the method receives a dictionary cannot be recovered or distinguished.
