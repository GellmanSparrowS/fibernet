"""Public experimental prescribed-temperature planar frame API.

Size admission is independent of physical applicability and process isolation.
The owner uses the existing SciPy sparse CPU path; there is no new backend,
thread setting, solver fallback or GPU implementation here.
"""
from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class ThermoelasticSizeLimits:
    """Explicit, caller-chosen size ceilings for network snapshots only.

    Both integer ceilings are required. ``max_assembly_entries`` is a read-only
    derived property equal to 36 * max_edges, not another constructor argument.
    It bounds conservative 6-DOF element contributions before COO coalescing
    and therefore CSR nnz. Parallel edges still count separately.
    This is NOT a bound on factor fill, solver workspace, RSS, time or output.
    Use external task/process budgets for untrusted or resource-sensitive jobs.
    Omit ``size_limits`` at solve time to retain the 192-node/256-edge default.
    """
    max_nodes: int
    max_edges: int

    def __post_init__(self):
        for name, low, high in (('max_nodes',2,sys.maxsize//3),
                               ('max_edges',1,sys.maxsize//36)):
            value=getattr(self,name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}], not bool')

    @property
    def max_assembly_entries(self):
        """Derived read-only contribution ceiling; not a factorization budget."""
        return 36 * self.max_edges

    def _admit(self, graph):
        # Revalidate at use; frozen dataclasses are not a security boundary.
        self.__post_init__()
        from fibernet.core.structure_graph import StructureGraph
        if not isinstance(graph,StructureGraph):
            raise TypeError('bounded thermoelastic input must be StructureGraph')
        nodes,edges=len(graph.nodes),len(graph.edges)
        if not 2 <= nodes <= self.max_nodes:
            raise ValueError('thermoelastic node count exceeds explicit size limits')
        if not 1 <= edges <= self.max_edges:
            raise ValueError('thermoelastic physical-edge count exceeds explicit size limits')
        upper=36*edges
        if upper>self.max_assembly_entries:
            raise ValueError('thermoelastic assembly-entry upper bound exceeds explicit size limits')
        return {'contract':'thermoelastic_network_size_limits/2',
                'limits':{'max_nodes':self.max_nodes,'max_edges':self.max_edges},
                'derived_limits':{'max_assembly_entries':self.max_assembly_entries,
                                  'assembly_entries_per_physical_edge':36},
                'node_count':nodes,'physical_edge_count':edges,'dof_count':3*nodes,
                'assembly_entries_upper_bound':upper,
                'factorization_memory_or_time_guaranteed':False}


_OWNER_EXPORTS=('ThermoelasticSnapshot','solve_thermoelastic_snapshot',
                'solve_thermoelastic_network_snapshot')
__all__=['ThermoelasticSizeLimits',*_OWNER_EXPORTS]


def __getattr__(name):
    if name in _OWNER_EXPORTS:
        from . import _thermoelastic
        return getattr(_thermoelastic,name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals())|set(__all__))
