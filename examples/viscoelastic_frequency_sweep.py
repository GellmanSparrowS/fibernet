"""Compare three drive frequencies on one synthetic 2-D welded SLS network."""

import argparse
import json
import math

import fibernet as fn
import numpy as np

from viscoelastic_network import make_network


def _last_cycle(graph, times, constraints, grip, law, right, top, y_ratio, samples):
    result = fn.simulate(graph, backend='fem', mode='history', material=law,
                         times=times, constraints=constraints, length_unit='mm')
    row = {node_id: j for j, node_id in enumerate(result.node_ids)}
    reactions = result.reactions[-samples-1:]
    # The force conjugate to the x grip drive includes the top y reaction.
    force = reactions[:, [row[i] for i in right], 0].sum(axis=1)
    force += y_ratio * reactions[:, row[top], 1]
    loop = np.dot((force[:-1] + force[1:]) / 2, np.diff(grip[-samples-1:]))
    ledger = result.energies

    def change(key, offset=0):
        values = ledger[key]
        end = len(values) - 1 - offset * samples
        return values[end] - values[end - samples]

    physical_loss = change('physical_dissipation')
    previous_loss = change('physical_dissipation', offset=1)
    if not (0 < physical_loss < math.inf and 0 < previous_loss < math.inf):
        raise FloatingPointError('expected positive finite physical loss in both final cycles')
    return {
        'physical_loss_J': physical_loss,
        'previous_physical_loss_J': previous_loss,
        'numerical_loss_J': change('numerical_dissipation'),
        'stored_energy_change_J': change('stored'),
        'endpoint_work_J': change('work'),
        'trapezoid_loop_work_J': float(loop),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amplitude-um', type=float, default=3.0,
                        help='x grip amplitude in µm, from 0.001 to 10 (default: 3)')
    parser.add_argument('--relaxation-time-s', type=float, default=0.2,
                        help='branch relaxation time in seconds, from 1e-6 to 1e6 (default: 0.2)')
    parser.add_argument('--quick', action='store_true',
                        help='32 instead of 128 samples per cycle')
    args = parser.parse_args()
    tau = args.relaxation_time_s
    if not 0.001 <= args.amplitude_um <= 10:
        parser.error('--amplitude-um must be between 0.001 and 10')
    if not 1e-6 <= tau <= 1e6:
        parser.error('--relaxation-time-s must be between 1e-6 and 1e6')
    amplitude_m = args.amplitude_um * 1e-6

    graph = make_network()  # Positions/radii in mm; controls and outputs in SI.
    x_min = min(node.position[0] for node in graph.nodes.values())
    x_max = max(node.position[0] for node in graph.nodes.values())
    left = [i for i, node in graph.nodes.items() if abs(node.position[0] - x_min) < 1e-9]
    right = [i for i, node in graph.nodes.items() if abs(node.position[0] - x_max) < 1e-9]
    top = max(right, key=lambda i: graph.nodes[i].position[1])
    y_ratio = 0.2
    law = fn.LinearViscoelasticMaterial.standard_linear_solid(1e6, 2e6, tau)
    samples, cycles = (32 if args.quick else 128), 10
    phase = np.arange(cycles * samples + 1) * (2 * math.pi / samples)
    grip = amplitude_m * np.sin(phase)
    grip[::samples] = 0.0  # Zero start and exact cycle boundaries.
    constraints = {i: [[0.0, 0.0, 0.0]] * len(grip) for i in left}
    constraints.update({
        i: [[float(s), float(y_ratio * s) if i == top else None, None] for s in grip]
        for i in right
    })

    cases = []
    for omega_tau in (0.2, 1.0, 5.0):
        times = phase * (tau / omega_tau)
        cases.append({
            'omega_tau': omega_tau,
            'frequency_Hz': omega_tau / (2 * math.pi * tau),
            **_last_cycle(graph, times, constraints, grip, law, right, top, y_ratio, samples),
        })

    print(json.dumps({
        'nodes': len(graph.nodes), 'edges': len(graph.edges), 'cycles': cycles,
        'samples_per_cycle': samples, 'amplitude_m': amplitude_m,
        'relaxation_time_s': tau, 'cases': cases,
    }, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
