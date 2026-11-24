"""Compare damping on one frame at fixed material volume and orientation."""

import argparse
import json
import math

import fibernet as fn
import numpy as np

if __package__:
    from .viscoelastic_network import make_network
else:
    from viscoelastic_network import make_network


# Ranked within direction by the strain energy of the uniform reference
# under the same proportional grip stroke. Values are stable graph edge IDs.
LAYOUTS = {
    "higher_reference_energy": (127, 162, 112, 77, 62, 147, 157, 72, 107, 122, 172),
    "lower_reference_energy": (177, 17, 37, 42, 97, 57, 27, 87, 137, 32, 92),
}


def _inventory(graph, selected):
    counts = [0, 0]
    volumes = []
    for edge_id in selected:
        edge = graph.edges[edge_id]
        a = np.asarray(graph.nodes[edge.node_i].position[:2], dtype=float)
        b = np.asarray(graph.nodes[edge.node_j].position[:2], dtype=float)
        d = b - a
        length_mm = float(np.linalg.norm(d))
        if not math.isclose(length_mm, 4.0, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Expected 4 mm members in each material subset")
        if abs(d[1]) < 1e-10:
            counts[0] += 1
        elif abs(d[0]) < 1e-10:
            counts[1] += 1
        else:
            raise ValueError("Expected axis-aligned members in each material subset")
        volumes.append(math.pi * float(edge.radius)**2 * length_mm * 1e-9)
    if counts != [6, 5]:
        raise ValueError("Expected six horizontal and five vertical long-time members")
    return {"horizontal": counts[0], "vertical": counts[1], "volume_m3": math.fsum(volumes)}


def _run(graph, laws, angular_frequency, left, right, top, samples):
    amplitude = 3e-6
    y_ratio = 0.2
    period = 2 * math.pi / angular_frequency
    stroke = amplitude * np.sin(np.linspace(0.0, 2 * math.pi, samples + 1))
    stroke[[0, -1]] = 0.0
    controls = {i: [[0.0, 0.0, 0.0]] * len(stroke) for i in left}
    controls.update({i: [[float(x), float(y_ratio*x) if i == top else None, None]
                         for x in stroke] for i in right})
    state = None
    end_time = 0.0
    losses = []
    for cycle in range(8):
        times = np.linspace(end_time, (cycle + 1)*period, samples + 1)
        result = fn.simulate(graph, backend="fem", mode="history", material=laws,
                             times=times, constraints=controls, length_unit="mm",
                             initial_state=state)
        end_time = float(times[-1])
        state = result.final_state
        energy = result.energies
        losses.append(float(energy["physical_dissipation"][-1]
                            - energy["physical_dissipation"][0]))
    loss = losses[-1]
    if not (math.isfinite(loss) and loss > 0 and abs(loss-losses[-2])/loss < 5e-4):
        raise RuntimeError("The physical loss has not settled between the last two cycles")
    row = {node_id: j for j, node_id in enumerate(result.node_ids)}
    # Include both prescribed translations in the force conjugate to the stroke.
    force = result.reactions[:, [row[i] for i in right], 0].sum(axis=1)
    force += y_ratio * result.reactions[:, row[top], 1]
    loop_work = float(np.dot((force[:-1] + force[1:])/2, np.diff(stroke)))
    return {
        "physical_loss_J": loss,
        "previous_physical_loss_J": losses[-2],
        "grip_loop_work_J": loop_work,
        "numerical_loss_J": float(energy["numerical_dissipation"][-1]
                                  - energy["numerical_dissipation"][0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-cycle", choices=[64, 128, 256, 512],
                        default=256, type=int)
    args = parser.parse_args()
    graph = make_network()
    if graph.num_nodes != 16 or graph.num_edges != 33:
        raise ValueError("The material layouts require the 16-node, 33-edge fixture")
    inventory = {key: _inventory(graph, ids) for key, ids in LAYOUTS.items()}
    if not math.isclose(*(value["volume_m3"] for value in inventory.values()),
                        rel_tol=1e-13, abs_tol=0):
        raise ValueError("The long-time material volume differs between layouts")
    positions = {i: node.position for i, node in graph.nodes.items()}
    left = [i for i, p in positions.items() if abs(p[0]) < 1e-9]
    xmax = max(p[0] for p in positions.values())
    right = [i for i, p in positions.items() if abs(p[0] - xmax) < 1e-9]
    top = max(right, key=lambda i: positions[i][1])
    material = fn.LinearViscoelasticMaterial.standard_linear_solid
    long_time, short_time = material(1e6, 2e6, 0.20), material(1e6, 2e6, 0.035)
    runs = []
    for omega in (5.0, 1.0 / 0.035):
        for layout, selected in LAYOUTS.items():
            laws = {eid: long_time if eid in selected else short_time for eid in graph.edges}
            runs.append({"angular_frequency_rad_s": omega, "frequency_Hz": omega/(2*math.pi),
                         "layout": layout, **_run(graph, laws, omega, left, right, top,
                                                  args.samples_per_cycle)})
    print(json.dumps({"nodes": graph.num_nodes, "edges": graph.num_edges,
                      "samples_per_cycle": args.samples_per_cycle, "cycles": 8,
                      "long_time_inventory": inventory, "long_time_edge_ids": LAYOUTS,
                      "cases": runs}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
