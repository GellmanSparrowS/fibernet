"""Four electrodes on a synthetic 3D rod network, with one-way Joule heating.

Run with: python -m examples.four_terminal_joule_heating
The output describes a lumped graph, not a solid temperature field.
"""

import json
from math import pi

import fibernet as fn
import numpy as np


MIXED_V = (0.0, 0.3, -0.4, 0.85)


def run_example():
    """Return the graph, electrode IDs, response, and electrical/thermal results."""
    nodes = (710, 49, 605, 13, 281, 117, 942, 369, 55, 804)
    ports = nodes[:4]
    points_m = 0.001 * np.array([
        [0, 0, 0], [0, 6, 1], [10, 3, 0], [6, 1, 9], [2, 1, 2],
        [3, 4, 1], [5, 3, 3], [7, 2, 2], [5, 2, 6], [7, 5, 4],
    ])
    links = ((0, 4), (1, 5), (2, 7), (3, 8), (4, 5), (4, 6), (5, 6),
             (5, 8), (6, 7), (6, 8), (7, 8), (8, 4), (6, 4), (7, 9))
    records, electrical_S, thermal_W_per_K, fractions = [], {}, {}, {}
    for k, (i, j) in enumerate(links):
        edge_id, kind = 3003 + 21 * k, k % 2
        radius_m, sigma_S_per_m = (0.00008, 0.00012)[kind], (8e5, 2e6)[kind]
        length_m = np.linalg.norm(points_m[j] - points_m[i])
        electrical_S[edge_id] = sigma_S_per_m * pi * radius_m**2 / length_m
        thermal_W_per_K[edge_id] = (0.25, 0.6)[kind]
        fractions[edge_id] = {nodes[i]: 0.25, nodes[j]: 0.75}
        records.append(dict(
            id=edge_id, node_i=nodes[i], node_j=nodes[j], radius=radius_m,
            material=fn.Material(electrical_conductivity=sigma_S_per_m).to_dict(),
        ))
    graph = fn.StructureGraph.from_dict(dict(
        dimension=3,
        nodes=[dict(id=n, position=p.tolist()) for n, p in reversed(list(zip(nodes, points_m)))],
        edges=list(reversed(records)),
    ))

    def solve(voltages):
        return fn.solve_linear_transport(
            graph, electrical_S, dict(zip(ports, voltages)), length_unit="m"
        )

    response_S = np.zeros((4, 4))
    for k in range(1, 4):
        result = solve(np.eye(4)[k].tolist())
        injection = dict(zip(result.node_ids, result.node_injection_A))
        response_S[:, k] = [injection[n] for n in ports]
    # A common voltage shift generates no current. The grounded reference
    # column is therefore minus the sum of the three directly solved columns.
    response_S[:, 0] = -response_S[:, 1:].sum(axis=1)

    electrical = solve(MIXED_V)
    capacity_order = ports + tuple(sorted(set(nodes) - set(ports)))
    capacity_J_per_K = {n: 0.30 + 0.05 * i for i, n in enumerate(capacity_order)}
    times_s = [k / 32 for k in range(17)]

    def heat(fixed_temperature_K):
        return fn.run_joule_heating(
            graph, electrical, capacity_J_per_K, thermal_W_per_K,
            dict.fromkeys(nodes, 300.0), times_s, endpoint_weights=fractions,
            length_unit="m", fixed_temperature_K=fixed_temperature_K,
        )

    return graph, ports, response_S, electrical, heat({}), heat({ports[0]: 300.0})


def main():
    _, ports, response_S, electrical, insulated, sink = run_example()
    injections = dict(zip(electrical.node_ids, electrical.node_injection_A))
    port_currents_A = [float(injections[n]) for n in ports]
    cases = {}
    for name, run in (("insulated", insulated), ("clamped_300_K", sink)):
        t = run.thermal
        cases[name] = {
            "final_temperature_K_by_id": dict(zip(map(int, t.node_ids), t.temperature_K[-1].tolist())),
            "stored_J": float(t.node_stored_heat_change_J[-1].sum()),
            "reservoir_influx_J": float(np.diff(t.times_s) @ t.reservoir_injection_W[1:].sum(axis=1)),
        }
    print(json.dumps({
        "port_ids": ports,
        "terminal_response_S": response_S.tolist(),
        "mixed_port_voltage_V": MIXED_V,
        "mixed_port_injection_A": port_currents_A,
        "port_power_W": (np.array(MIXED_V) * port_currents_A).tolist(),
        "rod_power_W_by_id": dict(zip(map(int, electrical.edge_ids), electrical.edge_power_W.tolist())),
        "joule_source_W_by_node": dict(insulated.heat_load.joule_source_W),
        "total_joule_power_W": electrical.dissipated_power_W,
        "duration_s": float(insulated.thermal.times_s[-1]),
        "thermal": cases,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
