"""Reference checks for the public finite-network deletion example."""

import numpy as np
import pytest

from examples import percolation_transport as example
from fibernet import StructureGraph


def test_parallel_members_and_floating_gauges_after_deletion():
    graph = StructureGraph(dimension=2)
    high, mid, low, extra_a, extra_b, isolate = [
        graph.add_node(point)
        for point in ([0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 2, 0], [1, 2, 0], [1, 3, 0])
    ]
    first = graph.add_edge(high, mid)
    parallel = graph.add_edge(mid, high, allow_parallel=True)
    exit_edge = graph.add_edge(mid, low)
    floating_edge = graph.add_edge(extra_a, extra_b)
    conductance = {first: 2., parallel: 3., exit_edge: 5., floating_edge: 7.}
    electrodes = {'high_V': [high], 'low_V': [low]}
    snapshot = graph.to_dict()
    active = graph.copy()

    original, sample = example._measure(graph, active, conductance, electrodes)
    assert sample['gauge_node_ids'] == [extra_a, isolate]
    assert sample['effective_conductance_S'] == pytest.approx(2.5)
    np.testing.assert_allclose(original.current_A, [1., -1.5, 2.5, 0.])

    active.remove_edge(first)
    conductance[first] = 0.
    after, sample = example._measure(graph, active, conductance, electrodes)
    np.testing.assert_array_equal(after.edge_ids, original.edge_ids)
    np.testing.assert_allclose(after.current_A, [0., -1.875, 1.875, 0.])
    assert sample['effective_conductance_S'] == pytest.approx(1.875)
    assert sample['spanning_component_ids'] == [high]

    active.remove_edge(parallel)
    conductance[parallel] = 0.
    stopped, sample = example._measure(graph, active, conductance, electrodes)
    assert sample['spanning_component_ids'] == []
    np.testing.assert_array_equal(stopped.current_A, np.zeros(4))
    np.testing.assert_array_equal(stopped.node_injection_A, np.zeros(6))
    assert graph.to_dict() == snapshot


@pytest.mark.parametrize('strategy', ['low_flow', 'random'])
def test_deletion_order_replays_a_topological_loss_inside_batch(strategy):
    graph = StructureGraph(dimension=2)
    chain = [graph.add_node([x, 0, 0]) for x in range(4)]
    a, b = (graph.add_node([x, 1, 0]) for x in (1, 2))
    driven = [graph.add_edge(u, v) for u, v in zip(chain, chain[1:])]
    floating = graph.add_edge(a, b)
    fixed = {'high_V': chain[:1], 'low_V': chain[-1:]}
    original_g = dict.fromkeys(graph.edges, 1.)
    volumes = {eid: 4. if eid == floating else 1. for eid in graph.edges}
    fields = {key: [] for key in example.FIELD_NAMES}

    states = example._trajectory(graph, original_g, volumes, fixed, fields, batch_size=3,
                                 strategy=strategy, seed=11)
    assert len(states) == 2
    assert states[0]['effective_conductance_S'] == pytest.approx(1/3)
    assert states[1]['effective_conductance_S'] == 0.
    assert states[1]['removed_edge_fraction'] == 0.75
    batch = states[1]['deleted_edge_ids']
    if strategy == 'low_flow':
        assert batch[0] == floating
        assert states[1]['removed_member_volume_fraction'] == 6/7
    else:
        np.testing.assert_array_equal(batch, np.random.default_rng(11).permutation(4)[:3])
    loss = example._loss_summary(graph, states, fixed, volumes)
    exact_step, cut_edge = next((i, eid) for i, eid in enumerate(batch, 1) if eid in driven)
    assert exact_step < 3
    assert loss['first_topological_loss']['removed_edges'] == exact_step
    assert loss['first_topological_loss']['edge_id'] == cut_edge
    assert loss['last_spanning_sample']['removed_edges'] == 0
    assert loss['first_nonspanning_sample']['removed_edges'] == 3
    assert fields['current_A'][1][cut_edge] == 0.


def test_example_refuses_existing_output_before_generation(tmp_path, monkeypatch):
    def unexpected_generation(*args, **kwargs):
        pytest.fail('generation started for an existing directory')

    monkeypatch.setattr(example, 'pattern_2d', unexpected_generation)
    marker = tmp_path / 'kept'
    marker.write_text('existing')
    with pytest.raises(FileExistsError):
        example.run(tmp_path)
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == 'existing'
