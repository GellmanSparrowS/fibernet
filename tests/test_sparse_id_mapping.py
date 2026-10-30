"""Verify sparse stable-ID mapping across graph input, FEM solve, and results.

A three-node axial case checks physical boundaries, asymmetric nodal loads,
parallel-edge forces, and residual identity against analytical equilibrium.
"""
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def test_sparse_id_boundary_load_result_mapping():
    a, r, E, F = .01, .0002, 1e9, .001
    graph = StructureGraph.from_dict({
        'version': '3.0', 'dimension': 3,
        'nodes': [{'id': 7, 'position': [a, 0., 0.]},
                  {'id': 40, 'position': [0., 0., 0.]},
                  {'id': 19, 'position': [2*a, 0., 0.]}],
        'edges': [{'id': 90, 'node_i': 40, 'node_j': 7, 'radius': r},
                  {'id': 3, 'node_i': 7, 'node_j': 40, 'radius': r},
                  {'id': 44, 'node_i': 7, 'node_j': 19, 'radius': r}],
    })
    before = graph.to_dict()
    fem = BeamFrameFEM(E=E, nu=.3)
    inp = fem.graph_to_fem_input(graph, dim=3, pct=.34)
    assert list(graph.nodes) == [7, 40, 19]
    assert list(graph.edges) == [90, 3, 44]
    np.testing.assert_array_equal(inp['node_pos'], [[a,0.,0.],[0.,0.,0.],[2*a,0.,0.]])
    np.testing.assert_array_equal(inp['edge_index'], [[1,0,0],[0,1,2]])
    assert inp['boundaries']['left'] == [1]
    assert inp['boundaries']['right'] == [2]
    # Loads are specified physically first; conversion to input rows is explicit.
    load_by_id = {7: [F,0.,0.], 40: [0.,0.,0.], 19: [2*F,0.,0.]}
    loads = np.array([load_by_id[nid] for nid in graph.nodes])
    raw = fem.solve_3d(inp['edge_index'], inp['node_pos'], inp['radii'],
                       forces=loads, fixed_nodes=inp['boundaries']['left'], damping=0.)
    # Independent series/parallel axial equilibrium: first segment carries 3F
    # split across two EA/a members; final segment carries 2F in one EA/a.
    EA = E*np.pi*r*r
    u_by_id = {40: 0., 7: 3*F*a/(2*EA), 19: 7*F*a/(2*EA)}
    expected = np.zeros((3,6))
    expected[:,0] = [u_by_id[7], u_by_id[40], u_by_id[19]]
    np.testing.assert_allclose(raw['u'], expected, rtol=1e-8, atol=2e-13)
    expected_reactions = np.zeros((3,6)); expected_reactions[1,0] = -3*F
    np.testing.assert_allclose(raw['reactions'], expected_reactions, rtol=1e-8, atol=1e-10)
    # Exercise legacy graph_to_fem_input insertion-order contract, no explicit map.
    result = fem.to_sim_result(raw, graph)
    assert result.metadata['node_ids'] == [7,40,19]
    assert result.metadata['edge_ids'] == [90,3,44]
    for row, nid in enumerate(result.metadata['node_ids']):
        np.testing.assert_allclose(result.displacements[row], [u_by_id[nid],0.,0.], rtol=1e-8, atol=2e-13)
        np.testing.assert_allclose(result.deformed_positions[row], graph.nodes[nid].position + result.displacements[row], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.edge_forces, [1.5*F,1.5*F,2*F], rtol=1e-8, atol=1e-10)
    residual = raw['numerical_diagnostics']['physical_free_residual']
    mapped = result.metadata['numerical_diagnostics']['physical_free_residual']
    assert residual['index_kind'] == 'input_node_row'
    assert residual['node_rows'] == [0]*6 + [2]*6
    assert mapped['index_kind'] == 'stable_node_id'
    assert mapped['node_ids'] == [7]*6 + [19]*6
    assert 'node_rows' not in mapped and 'node_ids' not in residual
    assert mapped['dof_components'] == ['ux','uy','uz','rx','ry','rz']*2
    np.testing.assert_array_equal(mapped['values'], residual['values'])
    np.testing.assert_allclose(mapped['values'], 0., atol=1e-10)
    assert graph.to_dict() == before
