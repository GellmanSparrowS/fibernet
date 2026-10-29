"""Small deterministic graphs and explicit requests for preparation tests."""

import math

from fibernet import Material, StructureGraph
from fibernet.sim._exudyn_contract import (
    ChordContactSpec,
    ExudynContactRequest,
    ExudynContractError,
    SphereSpec,
    StepPhase,
)


def make_chain(n=32):
    graph = StructureGraph(dimension=2)
    material = Material(
        name='explicit synthetic test material',
        youngs_modulus=1000000.0,
        poissons_ratio=0.3,
        density=1200.0,
    )
    for i in range(n + 1):
        graph.add_node([-0.001 + 0.002 * i / n, 0, 0], merge=False)
    for i in range(n):
        graph.add_edge(i, i + 1, radius=5e-05, material=material, segments=1)
    return graph


def make_request(graph):
    ends = sorted(graph.nodes, key=lambda n: graph.nodes[n].position[0])
    return ExudynContactRequest(
        profile='straight-shared-chain/1',
        length_unit='m',
        clamped_node_ids=(ends[0], ends[-1]),
        shear_correction=(1.0, 1.0),
        sphere=SphereSpec(
            0.00035,
            4 * math.pi * 0.00035 ** 3 * 1000 / 3,
            (0.0, 0.00062, 0.0),
            (0.0, -0.1, 0.0),
        ),
        contact=ChordContactSpec(100.0, 2e-06, 0.0002),
        phases=(StepPhase(0.0, 0.000195, 39), StepPhase(0.000195, 0.000203, 80)),
        geometry_energy_budget_J=8.979719001510826e-12,
        gravity_m_s2=(0.0, 0.0, 0.0),
        external_forces_N=(),
        initial_fiber_state='reference_rest',
    )


def remap_ids(graph):
    old_nodes = list(graph.nodes.items())
    node_map = {old: 200 - 17 * i for i, (old, _) in enumerate(old_nodes)}
    graph._nodes = {node_map[old]: node for old, node in old_nodes}
    for nid, node in graph.nodes.items():
        node.node_id = nid
    old_edges = list(graph.edges.items())
    edge_map = {old: 500 - 23 * i for i, (old, _) in enumerate(old_edges)}
    graph._edges = {edge_map[old]: edge for old, edge in old_edges}
    for eid, edge in graph.edges.items():
        edge.edge_id = eid
        edge.node_i = node_map[edge.node_i]
        edge.node_j = node_map[edge.node_j]
    graph._next_node_id = max(0, max(graph.nodes) + 1)
    graph._next_edge_id = max(0, max(graph.edges) + 1)
    return graph


class ContractAssertions:
    """Assert stable domain error codes without depending on message wording."""

    def assert_contract_error(self, code, operation):
        with self.assertRaises(ExudynContractError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
