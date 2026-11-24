"""Integration checks for a nonuniform four-terminal 3D transport network."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import fibernet as fn


def load_example():
    path = Path(__file__).resolve().parents[1] / "examples/four_terminal_joule_heating.py"
    spec = importlib.util.spec_from_file_location("_four_terminal_example", path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(fn, "solve_linear_transport", side_effect=AssertionError("solve on import")):
        with patch.object(fn, "run_joule_heating", side_effect=AssertionError("solve on import")):
            spec.loader.exec_module(module)
    return module


def test_four_terminal_response_and_thermal_energy_ledger():
    graph, ports, response_S, electrical, insulated, sink = load_example().run_example()
    data = graph.to_dict()
    nodes = list(ports) + sorted(n for n in graph.nodes if n not in ports)
    row = {node: i for i, node in enumerate(nodes)}
    xyz = {n["id"]: np.array(n["position"]) for n in data["nodes"]}
    assert graph.dimension == 3
    assert len(nodes) == 10 and len(data["edges"]) == 14 and len(ports) == 4
    centered = np.stack(list(xyz.values()))
    assert np.linalg.matrix_rank(centered - centered.mean(axis=0)) == 3
    assert set(electrical.edge_ids) == {edge["id"] for edge in data["edges"]}
    dense = np.zeros((len(nodes), len(nodes)))
    rods = []
    for edge in data["edges"]:
        a, b = edge["node_i"], edge["node_j"]
        sigma = edge["material"]["electrical_conductivity"]
        g = sigma * np.pi * edge["radius"] ** 2 / np.linalg.norm(xyz[a] - xyz[b])
        d = np.eye(len(nodes))[row[a]] - np.eye(len(nodes))[row[b]]
        dense += g * np.outer(d, d)
        rods.append((row[a], row[b], g))
    exterior = dense[:4, :4] - dense[:4, 4:] @ np.linalg.solve(dense[4:, 4:], dense[4:, :4])
    np.testing.assert_allclose(response_S, exterior, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(response_S, response_S.T, rtol=0, atol=1e-12)
    assert min(np.linalg.eigvalsh(response_S[1:, 1:])) > 0
    drive = np.array([0.0, 0.3, -0.4, 0.85])
    v = np.r_[drive, np.linalg.solve(dense[4:, 4:], -dense[4:, :4] @ drive)]
    sources_W = np.zeros(len(nodes))
    for a, b, g in rods:
        power = g * (v[a] - v[b]) ** 2
        sources_W[a] += 0.25 * power
        sources_W[b] += 0.75 * power
    total_W = sources_W.sum()
    assert total_W == pytest.approx(6.99820731662)
    assert electrical.boundary_power_W == pytest.approx(total_W, abs=1e-11)
    for run in (insulated, sink):
        t = run.thermal
        assert t.completed and t.times_s[-1] == 0.5
        np.testing.assert_allclose(
            [run.heat_load.joule_source_W[n] for n in nodes], sources_W, rtol=0, atol=1e-12
        )
        heat_in_J = np.r_[0.0, np.cumsum(np.diff(t.times_s) * t.reservoir_injection_W[1:].sum(axis=1))]
        stored_J = t.node_stored_heat_change_J.sum(axis=1)
        np.testing.assert_allclose(stored_J, total_W * t.times_s + heat_in_J, rtol=0, atol=3e-10)
        if run is insulated:
            assert len(t.fixed_node_ids) == 0
            np.testing.assert_array_equal(heat_in_J, np.zeros_like(heat_in_J))
        else:
            assert list(t.fixed_node_ids) == [ports[0]]
            fixed_col = list(t.node_ids).index(ports[0])
            np.testing.assert_array_equal(t.temperature_K[:, fixed_col], 300.0)
            assert heat_in_J[-1] == pytest.approx(-0.18698847938, abs=1e-9)
