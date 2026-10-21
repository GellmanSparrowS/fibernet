"""Synthetic axial dynamic example; importing this file does not run a solver.

Synthetic demonstration, not empirical polymer parameters.
The three-member example does not infer a safe timestep; see the separately
retained 105-node three-grid example for its specific validated settings.
"""
import math
import numpy as np
from fibernet import StructureGraph, LinearViscoelasticMaterial, simulate


def make_network():
    """One shared junction and three noncollinear members in the authoritative graph."""
    graph = StructureGraph(dimension=3)
    anchors = [graph.add_node(p) for p in ([0.,0.,0.], [.01,0.,0.], [0.,.01,0.])]
    center = graph.add_node([.003,.003,.004])
    for node in anchors:
        graph.add_edge(node, center, radius=5e-5)
    # Noncontiguous stable IDs, without creating a second graph representation.
    record = graph.to_dict()
    node_ids = {row['id']: 17+7*row['id'] for row in record['nodes']}
    for row in record['nodes']:
        row['id'] = node_ids[row['id']]
    for row in record['edges']:
        row['id'] = 91+11*row['id']
        row['node_i'], row['node_j'] = node_ids[row['node_i']], node_ids[row['node_j']]
    record['next_node_id'] = 1+max(row['id'] for row in record['nodes'])
    record['next_edge_id'] = 1+max(row['id'] for row in record['edges'])
    return StructureGraph.from_dict(record), [node_ids[n] for n in anchors], node_ids[center]


def run(*, dt, num_steps):
    """Explicit timestep required: no automatic stability or convergence claim."""
    graph, fixed_ids, moving_id = make_network()
    rows = {node: row for row, node in enumerate(graph.nodes)}
    rest = {eid: math.dist(graph.nodes[e.node_i].position, graph.nodes[e.node_j].position)
            for eid, e in graph.edges.items()}
    # These are explicit caller-selected A0 values, not inferred from graph radius.
    areas = {eid: 7.853981633974483e-9 for eid in graph.edges}
    material = LinearViscoelasticMaterial(1e6, ((2e6,.01), (1e6,.1)))
    return simulate(graph, backend='taichi', mode='dynamics', arch='cpu', num_threads=1,
        material=material, reference_areas_by_id=areas, max_abs_strain=.02,
        rest_lengths_by_id=rest, nodal_masses_by_id={nid:1e-5 for nid in graph.nodes},
        fixed_nodes=[rows[nid] for nid in fixed_ids], external_force=np.zeros((len(rows),3)),
        initial_velocities_by_id={moving_id:[0.,0.,1e-3]},
        state_units=dict(length='m',time='s',mass='kg',frame='world_xyz'),
        dashpot=0., drag=0., dt=dt, num_steps=num_steps,
        save_interval=max(1,num_steps//20), record_state=True)
