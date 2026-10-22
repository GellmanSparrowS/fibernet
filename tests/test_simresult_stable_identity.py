"""Contract-only synthetic repro; no engine initialization or simulation."""
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.accelerated import SimResult


def make_graph():
    return StructureGraph.from_dict({
        'dimension': 3,
        'nodes': [{'id': 30, 'position': [0., 0., 0.]},
                  {'id': 10, 'position': [1., 0., 0.]},
                  {'id': 20, 'position': [3., 0., 0.]}],
        'edges': [{'id': 70, 'node_i': 30, 'node_j': 10},
                  {'id': 50, 'node_i': 10, 'node_j': 20}]
    })


def result_for(graph):
    return SimResult(
        deformed_positions=np.array([graph.nodes[n].position for n in graph.nodes]),
        displacements=np.zeros((3, 3)),
        metadata={'node_ids': list(graph.nodes), 'edge_ids': list(graph.edges)})


def test_unchanged_graph_serialization_does_not_create_deformation():
    graph = make_graph()
    result = result_for(graph)
    result.compute_detailed(graph, stiffness=1.)
    np.testing.assert_allclose(result.edge_stretches, [1., 1.])
    restored = StructureGraph.from_dict(graph.to_dict())
    assert graph.fingerprint() == restored.fingerprint()
    assert list(graph.nodes) == [30, 10, 20]
    assert list(restored.nodes) == [10, 20, 30]
    result.compute_detailed(restored, stiffness=1.)
    # Same stable IDs, identical geometry, and zero displacement must be unchanged.
    np.testing.assert_allclose(result.edge_stretches, [1., 1.])
    assert result.energy == 0.


def test_edge_rows_remain_in_advertised_id_order():
    graph = make_graph()
    record = graph.to_dict()
    # Keep node order but reverse edge order to isolate edge-row mapping.
    record['nodes'] = [next(n for n in record['nodes'] if n['id'] == nid)
                       for nid in graph.nodes]
    restored = StructureGraph.from_dict(record)
    result = result_for(graph)
    result.deformed_positions[2, 0] = 5.
    result.compute_detailed(graph, stiffness=1.)
    expected = result.edge_stretches.copy()
    assert expected.tolist() == [1., 2.]
    result.compute_detailed(restored, stiffness=1.)
    assert result.metadata['edge_ids'] == [70, 50]
    np.testing.assert_allclose(result.edge_stretches, expected)


def test_invalid_identity_maps_rejected_without_result_mutation():
    graph = make_graph()
    variants = [
        {'node_ids': [30, 10, 20]}, {'edge_ids': [70, 50]},
        {'node_ids': [30, 10, 10], 'edge_ids': [70, 50]},
        {'node_ids': [30, 10, 99], 'edge_ids': [70, 50]},
        {'node_ids': [30, 10, 20], 'edge_ids': [70, 70]},
        {'node_ids': [30, 10, 20], 'edge_ids': [70, 99]},
        {'node_ids': [30, True, 20], 'edge_ids': [70, 50]},
    ]
    for metadata in variants:
        result = result_for(graph)
        result.metadata = metadata
        try:
            result.compute_detailed(graph, stiffness=1.)
        except ValueError:
            assert result.edge_stretches is None
            assert result.n_nodes == 0
        else:
            raise AssertionError('invalid map accepted: ' + repr(metadata))


def test_invalid_graph_identity_and_result_shape_rejected():
    for kind in ('node', 'edge', 'shape'):
        graph = make_graph()
        result = result_for(graph)
        if kind == 'node':
            graph.nodes[30].node_id = 999
        elif kind == 'edge':
            graph.edges[70].edge_id = 999
        else:
            result.deformed_positions = result.deformed_positions[:2]
        try:
            result.compute_detailed(graph, stiffness=1.)
        except ValueError:
            assert result.edge_stretches is None
        else:
            raise AssertionError(kind + ' inconsistency accepted')


def test_multiedges_materials_and_per_edge_stiffness_are_preserved():
    from fibernet.core.material import Material
    graph = make_graph()
    eid = graph.add_edge(30, 10, allow_parallel=True,
                         material=Material(name='distinct material', youngs_modulus=123.))
    original = graph.to_dict()
    result = result_for(graph)
    result.deformed_positions[1, 0] = 2.
    result.compute_detailed(graph, stiffness=np.array([2., 4., 8.]))
    expected_stretches = result.edge_stretches.copy()
    expected_forces = result.edge_forces.copy()
    expected_energy = result.energy
    restored = StructureGraph.from_dict(original)
    result.compute_detailed(restored, stiffness=np.array([2., 4., 8.]))
    np.testing.assert_allclose(result.edge_stretches, expected_stretches)
    np.testing.assert_allclose(result.edge_forces, expected_forces)
    assert result.energy == expected_energy
    assert len(result.edge_stretches) == 3
    assert result.metadata['edge_ids'] == [70, 50, eid]
    assert restored.to_dict() == original == graph.to_dict()


def test_idless_legacy_result_keeps_graph_insertion_order():
    graph = make_graph()
    result = result_for(graph)
    result.metadata = {}
    result.compute_detailed(graph, stiffness=1.)
    np.testing.assert_allclose(result.edge_stretches, [1., 1.])
    assert result.energy == 0.


def test_lammps_compile_preserves_ids_materials_and_rejects_unsupported_geometry():
    import json
    from fibernet.integrations.lammps_backend import (
        prepare_harmonic_beads, HarmonicBond, RunSpec, LAMMPSContractError)
    graph = make_graph()
    def prepare(g):
        return prepare_harmonic_beads(g, length_unit='m', maximum_bond_length_m=5.,
            masses_kg={nid: 1. for nid in g.nodes},
            bonds={eid: HarmonicBond(1., 1.) for eid in g.edges},
            run=RunSpec(0.01, 0, 1, ((-10., 10.),) * 3))
    prepared = prepare(graph)
    assert json.loads(prepared.graph_json) == graph.to_dict()
    assert prepared.manifest()['atom_to_node_id'] == {'1': 10, '2': 20, '3': 30}
    assert prepared.manifest()['bond_to_edge_id'] == {'1': 50, '2': 70}
    for kind in ('parallel', 'curve'):
        g = graph.copy()
        if kind == 'parallel':
            g.add_edge(30, 10, allow_parallel=True)
        else:
            g.edges[70].internal_points = np.array([[0.5, 1., 0.]])
        try:
            prepare(g)
        except LAMMPSContractError as error:
            assert ('parallel' if kind == 'parallel' else 'curved') in str(error)
        else:
            raise AssertionError(kind + ' geometry silently accepted')


def test_invalid_geometry_and_stiffness_fail_before_mutation():
    for kind in ('zero_length', 'nonfinite', 'stiffness_shape', 'displacement_shape'):
        graph = make_graph()
        result = result_for(graph)
        stiffness = 1.
        if kind == 'zero_length':
            graph.nodes[10].position[:] = graph.nodes[30].position
        elif kind == 'nonfinite':
            result.deformed_positions[0, 0] = np.nan
        elif kind == 'stiffness_shape':
            stiffness = np.ones(3)
        else:
            result.displacements = np.zeros((2, 3))
        try:
            result.compute_detailed(graph, stiffness=stiffness)
        except ValueError:
            assert result.n_nodes == 0
            assert result.edge_stretches is None
        else:
            raise AssertionError(kind + ' accepted')


def test_finite_input_overflow_rejected_atomically():
    import copy
    for kind in ('coordinate', 'stiffness', 'displacement'):
        graph = make_graph()
        result = result_for(graph)
        result.compute_detailed(graph, stiffness=1.)
        stiffness = 1.
        if kind == 'coordinate':
            result.deformed_positions[1, 0] = 1e308
        elif kind == 'stiffness':
            result.deformed_positions[1, 0] = 3.
            stiffness = 1e308
        else:
            result.displacements[1, 0] = 1e308
        snapshot = copy.deepcopy(vars(result))
        try:
            result.compute_detailed(graph, stiffness=stiffness)
        except ValueError as error:
            assert 'nonfinite derived metrics' in str(error)
        else:
            raise AssertionError(kind + ' overflow accepted')
        assert set(vars(result)) == set(snapshot)
        for key, value in snapshot.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(result, key), value)
            else:
                assert getattr(result, key) == value, key
