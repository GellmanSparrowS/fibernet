"""Two-way DC heating on a small cubic network. Print a JSON report to stdout."""
import json

import numpy as np
from scipy.optimize import root

from fibernet import pattern_3d
from fibernet.sim.electrothermal import TemperatureConductanceLaw, solve_electrothermal


def _reference(step_s, steps):
    """Reduce the symmetric network to one electrical path and two free temperatures."""
    z = np.zeros(2)  # middle and right temperature rise above 300 K
    for _ in range(steps):
        previous = z

        def balance(rise):
            x, y = rise
            R1 = 1 + 0.04 * x / 2
            R2 = (1 + 0.015 * (x + y) / 2) / 2
            current = 2 / (R1 + R2)
            p1, p2 = current**2 * R1, current**2 * R2
            return [
                2 * (x - previous[0]) / step_s + 0.25 * x + 0.75 * (x - y) - (p1 + p2) / 2,
                3 * (y - previous[1]) / step_s + 0.75 * (y - x) - p2 / 2,
            ]

        solved = root(balance, previous, tol=1e-11)
        np.testing.assert_allclose(balance(solved.x), 0, atol=2e-12)
        z = solved.x
    x, y = z
    return (300., 300. + x, 300. + y), 16 / (1 + 0.04 * x / 2 + (1 + 0.015 * (x + y) / 2) / 2)


def run_example():
    graph = pattern_3d(unit="cubic", box=(0.01,) * 3, grid=(2, 1, 1), radius=0.0002)
    # Each x plane has 4 nodes. Each x-directed load path traverses one
    # A and one B edge; transverse ties use B and see no gradient by symmetry.
    plane = {nid: round(node.position[0] / 0.01) for nid, node in graph.nodes.items()}
    if sorted(plane.values()) != [0] * 4 + [1] * 4 + [2] * 4 or len(graph.edges) != 20:
        raise RuntimeError("Unexpected geometry for the cubic reference")
    classes, weights = {}, {}
    for eid, edge in graph.edges.items():
        ends = (edge.node_i, edge.node_j)
        classes[eid] = "A" if {plane[n] for n in ends} == {0, 1} else "B"
        weights[eid] = dict.fromkeys(ends, 0.5)
    # G0 [S], alpha [1/K], thermal conductance h [W/K]: explicit synthetic inputs.
    properties = {"A": (1., 0.04, 0.25), "B": (2., 0.015, 0.75)}

    def property_by_edge(column):
        return {eid: properties[name][column] for eid, name in classes.items()}

    initial = dict.fromkeys(graph.nodes, 300.)
    law = TemperatureConductanceLaw(property_by_edge(0), property_by_edge(1),
                                    dict.fromkeys(graph.edges, 300.), weights, (299., 310.))
    dt, steps = 0.125, 8
    result = solve_electrothermal(
        graph, law,
        prescribed_voltage_V={nid: 2. if x == 0 else 0. for nid, x in plane.items() if x != 1},
        capacity_J_per_K={nid: 3. if x == 2 else 2. for nid, x in plane.items()},
        conductance_W_per_K=property_by_edge(2), initial_temperature_K=initial,
        times_s=[i * dt for i in range(steps + 1)],
        endpoint_heat_weights=weights, length_unit="m",
        fixed_temperature_K={nid: 300. for nid, x in plane.items() if x == 0},
    )
    if not result.completed or len(result.frames) != steps + 1:
        raise RuntimeError("Incomplete electrothermal trajectory")
    start, final = result.frames[0], result.frames[-1]
    expected, expected_power = _reference(dt, steps)
    np.testing.assert_allclose(final.temperature_K, [expected[plane[n]] for n in result.node_ids],
                               rtol=0, atol=8e-11)
    p0 = json.loads(start.electrical_json)["dissipated_power_W"]
    p1 = json.loads(final.electrical_json)["dissipated_power_W"]
    np.testing.assert_allclose(p0, 4 * 2**2 / 1.5, rtol=1e-12)
    np.testing.assert_allclose(p1, expected_power, rtol=1e-11)
    source = sum(final.cumulative_source_heat_J)
    reservoir = sum(final.cumulative_reservoir_heat_J)
    stored = sum(final.node_stored_heat_change_J)
    np.testing.assert_allclose(stored, source + reservoir, rtol=0, atol=8e-11)
    if not (0 < p1 < p0 and reservoir < 0 < stored):
        raise RuntimeError("Expected heating, reduced DC power and net heat removal")
    by_plane = [[T for n, T in zip(result.node_ids, final.temperature_K) if plane[n] == j]
                for j in range(3)]
    return {
        "nodes": len(result.node_ids), "edges": len(result.edge_ids),
        "time_s": final.time_s, "step_s": dt, "frames": len(result.frames),
        "final_plane_mean_temperature_K": [sum(values)/len(values) for values in by_plane],
        "reduced_reference_temperature_K": expected,
        "initial_joule_power_W": p0, "final_joule_power_W": p1,
        "cumulative_joule_heat_J": source, "cumulative_reservoir_heat_J": reservoir,
        "stored_heat_J": stored, "energy_closure_J": stored - (source + reservoir),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, allow_nan=False))
