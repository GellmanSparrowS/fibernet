"""
Graph-based operations for fiber networks.

Provides:
- I/O: JSON/NetworkX conversion for weld graph format
- Weld: Edge intersection detection and welding
"""

from .io import (
    to_networkx,
    from_networkx,
    load_graph_json,
    save_graph_json,
)

from .weld import (
    weld_graph,
    find_intersections,
    merge_coincident_nodes,
)

__all__ = [
    "to_networkx",
    "from_networkx",
    "load_graph_json",
    "save_graph_json",
    "weld_graph",
    "find_intersections",
    "merge_coincident_nodes",
]
