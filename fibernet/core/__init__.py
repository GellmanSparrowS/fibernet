"""
Core data structures for FiberNet.

The central class is StructureGraph — the unified representation for
all fiber networks, lattices, and metamaterials.

Quick Start:
    >>> from fibernet.core import StructureGraph
    >>> g = StructureGraph(dimension=2, box_size=[10, 10])
    >>> n0 = g.add_node([0, 0])
    >>> n1 = g.add_node([10, 0])
    >>> g.add_edge(n0, n1, radius=0.5, n_internal=4)
"""

from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.core.material import Material
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink

__all__ = [
    "StructureGraph", "SNode", "SEdge",
    "Material",
    "Fiber",
    "FiberNetwork", "Crosslink",
]
