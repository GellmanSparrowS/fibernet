import numpy as np
import pytest
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.accelerated import (
    SimResult,
    TaichiEngine,
    _prepare_displacement_schedule,
    _normalize_spring_stiffness,
)


def _chain(dimension=2):
    graph = StructureGraph(dimension=dimension)
    direction = np.array([1.0, 0.0, 0.0])
    if dimension == 3:
        direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    nodes = [
        graph.add_node(distance * direction)
        for distance in (0.0, 1.0, 3.0)
    ]
    graph.add_edge(nodes[0], nodes[1])
    graph.add_edge(nodes[1], nodes[2])
    return graph, direction


def _engine():
    pytest.importorskip("taichi")
    return TaichiEngine(arch="cpu")


def test_schedule_normalization():
    vector = np.array([1.0, 0.0, 0.0])
    result = _prepare_displacement_schedule(
        {0: [], 1: [(8, vector), (0, vector * 0), (2, vector)]}, 3, 3
    )
    assert 0 not in result
    assert [row[0] for row in result[1]] == [0, 2, 8]
    vector[0] = 99
    assert result[1][-1][1][0] == 1


@pytest.mark.parametrize(
    "schedule",
    [
        {1: [(0, [0, 0, 0]), (0, [1, 0, 0])]},
        {1: [(-1, [0, 0, 0])]},
        {1: [(np.nan, [0, 0, 0])]},
        {1: [(0, [np.inf, 0, 0])]},
        {1: [(0, [0, 0])]},
        {-1: [(0, [0, 0, 0])]},
        {3: [(0, [0, 0, 0])]},
        {True: [(0, [0, 0, 0])]},
        {1.0: [(0, [0, 0, 0])]},
        {1: [(0, [0, 0, 0], "extra")]},
    ],
)
def test_invalid_schedule_rejected(schedule):
    with pytest.raises(ValueError):
        _prepare_displacement_schedule(schedule, 3, 3)


@pytest.mark.parametrize("dimension", [2, 3])
def test_piecewise_schedule_and_cache_reset(dimension):
    engine = _engine()
    graph, direction = _chain(dimension)
    original = np.array([node.position for node in graph.nodes.values()])
    schedule = [
        (0, np.zeros(3)),
        (2, direction),
        (6, direction),
        (8, np.zeros(3)),
    ]
    first = engine.dynamics(
        graph, fixed_nodes=[0, 1], displacement_schedule={2: schedule},
        stiffness=1.0, num_steps=8, save_interval=1,
    )
    expected = [0, 0.5, 1, 1, 1, 1, 1, 0.5, 0]
    assert len(first.positions_trajectory) == len(expected)
    for frame, displacement in zip(first.positions_trajectory, expected):
        np.testing.assert_allclose(
            frame[2], original[2] + displacement * direction, atol=1e-10
        )
    second = engine.dynamics(
        graph, fixed_nodes=[0, 1, 2], displacement_schedule={2: []},
        stiffness=1.0, num_steps=8, save_interval=1,
    )
    for frame in second.positions_trajectory:
        np.testing.assert_allclose(frame, original, atol=1e-10)


@pytest.mark.parametrize("ramp_fraction", [0.0, 0.25, 1.0])
def test_stretch_ramp_and_hold(ramp_fraction):
    engine = _engine()
    graph, _ = _chain()
    original = np.array([node.position for node in graph.nodes.values()])
    result = engine.stretch_test(
        graph, target_stretch=1.1, stiffness=1.0, num_steps=8,
        auto_steps=False, save_interval=1, ramp_fraction=ramp_fraction,
    )
    ramp_steps = int(8 * ramp_fraction)
    for step, frame in enumerate(result.positions_trajectory[1:], start=1):
        fraction = 1.0 if ramp_steps == 0 else min(step / ramp_steps, 1.0)
        np.testing.assert_allclose(
            frame[2, 0] - original[2, 0], 0.3 * fraction, atol=1e-10
        )


@pytest.mark.parametrize("dimension", [2, 3])
def test_per_edge_stiffness_metrics_without_taichi(dimension):
    graph, _ = _chain(dimension)
    original = np.array([node.position for node in graph.nodes.values()])
    final = original * 1.1
    result = SimResult(displacements=final - original, deformed_positions=final)
    result.compute_detailed(graph, stiffness=np.array([10.0, 20.0]))
    np.testing.assert_allclose(result.edge_forces, [1.0, 2.0])
    assert result.edge_forces.shape == (2,)
    assert result.energy == pytest.approx(0.25)


@pytest.mark.parametrize("dimension", [2, 3])
def test_scalar_stiffness_metrics_remain_compatible(dimension):
    graph, _ = _chain(dimension)
    original = np.array([node.position for node in graph.nodes.values()])
    final = original * 1.1
    result = SimResult(displacements=final - original, deformed_positions=final)
    result.compute_detailed(graph, stiffness=10.0)
    np.testing.assert_allclose(result.edge_forces, [1.0, 1.0])
    assert result.edge_forces.shape == (2,)
    assert result.energy == pytest.approx(0.15)


def test_dynamics_uses_effective_stiffness_for_metrics():
    engine = _engine()
    graph, _ = _chain()
    original = np.array([node.position for node in graph.nodes.values()])
    schedule = {
        index: [(0, np.zeros(3)), (2, 0.1 * position)]
        for index, position in enumerate(original)
    }
    result = engine.dynamics(
        graph, displacement_schedule=schedule,
        spring_k=np.array([10.0, 20.0]), num_steps=2, save_interval=1,
    )
    np.testing.assert_allclose(result.edge_forces, [1.0, 2.0])
    assert result.energy == pytest.approx(0.25)


def test_fractional_schedule_interval():
    schedule = [
        (0.5, np.zeros(3)),
        (1.25, np.array([3.0, 0.0, 0.0])),
    ]
    np.testing.assert_allclose(
        TaichiEngine._interpolate_schedule(schedule, 1.0, 2),
        [2.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        TaichiEngine._interpolate_schedule(schedule, 0.5, 2),
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        TaichiEngine._interpolate_schedule(schedule, 1.25, 2),
        [3.0, 0.0, 0.0],
    )


@pytest.mark.parametrize("value", [np.array([[10.0], [20.0]]), [10.0], [10.0, 20.0, 30.0], [np.nan, 20.0], [10.0, np.inf], [-1.0, 20.0]])
def test_invalid_spring_stiffness_rejected(value):
    with pytest.raises(ValueError):
        _normalize_spring_stiffness(value, 2)


def test_scalar_spring_stiffness_is_broadcast_per_edge():
    np.testing.assert_array_equal(_normalize_spring_stiffness(10.0, 2), [10.0, 10.0])


def test_single_waypoint_is_constant_and_future_target_is_partial():
    singleton = [(0, np.array([2.0, 0.0, 0.0]))]
    np.testing.assert_allclose(TaichiEngine._interpolate_schedule(singleton, 1, 8), [2.0, 0.0, 0.0])
    future = [(0, np.zeros(3)), (16, np.array([2.0, 0.0, 0.0]))]
    np.testing.assert_allclose(TaichiEngine._interpolate_schedule(future, 8, 8), [1.0, 0.0, 0.0])


@pytest.mark.parametrize("parameter,value", [("num_steps", 0), ("num_steps", -1), ("num_steps", 1.5), ("num_steps", True), ("save_interval", 0), ("save_interval", -1), ("save_interval", 1.5)])
def test_invalid_step_counts_rejected(parameter, value):
    # Early validation must not require a Taichi runtime or field allocation.
    engine = object.__new__(TaichiEngine)
    graph, _ = _chain()
    with pytest.raises(ValueError):
        engine.dynamics(graph, **{parameter: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_ramp_fraction_rejected(value):
    engine = object.__new__(TaichiEngine)
    graph, _ = _chain()
    with pytest.raises(ValueError):
        engine.stretch_test(graph, ramp_fraction=value)


@pytest.mark.parametrize("ramp_fraction", [0.0, 0.25, 1.0])
def test_stretch_schedule_construction_without_taichi(monkeypatch, ramp_fraction):
    """Inspect the public wrapper's schedule without replacing any kernel."""
    engine = object.__new__(TaichiEngine)
    graph, _ = _chain()
    monkeypatch.setattr(engine, "dynamics", lambda graph, **kwargs: kwargs)
    arguments = engine.stretch_test(
        graph, target_stretch=1.1, num_steps=8, auto_steps=False,
        ramp_fraction=ramp_fraction,
    )
    schedules = _prepare_displacement_schedule(
        arguments["displacement_schedule"], graph.num_nodes, 3
    )
    ramp_steps = int(8 * ramp_fraction)
    for step in range(1, 9):
        fraction = min(step / ramp_steps, 1.0) if ramp_steps else 1.0
        np.testing.assert_allclose(
            engine._interpolate_schedule(schedules[2], step, 8),
            [0.3 * fraction, 0.0, 0.0],
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_stretch_invalid_step_counts_rejected_without_taichi(value):
    engine = object.__new__(TaichiEngine)
    graph, _ = _chain()
    with pytest.raises(ValueError, match="num_steps"):
        engine.stretch_test(graph, num_steps=value)
