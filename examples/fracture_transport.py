"""Measure reference-geometry conductance along a quasistatic fracture history.

Run: python -m examples.fracture_transport --output-dir NEW_DIRECTORY
"""
import argparse
import csv
from hashlib import sha256
import math
from pathlib import Path

from fibernet.sim.fracture import FractureConfig, FractureSolver

from .fracture_scan import FULL_UM, POISSON, STRENGTH_PA, YOUNG_PA, make_network
from .percolation_transport import _electrodes, _measure, _write_json


CONDUCTIVITY_S_PER_M = 1.0


def _electrical_history(graph, fracture, *, mask_edge_ids, electrodes):
    """Apply recorded active masks using the original geometry and fixed electrodes."""
    edge_ids = list(graph.edges)
    node_ids = sorted(graph.nodes)
    if len(mask_edge_ids) != len(edge_ids) or set(mask_edge_ids) != set(edge_ids):
        raise ValueError('fracture mask edge IDs must exactly cover the source graph')
    lengths = dict(zip(edge_ids, graph.edge_lengths()))
    reference = {eid: CONDUCTIVITY_S_PER_M * math.pi * graph.edges[eid].radius**2 / lengths[eid]
                 for eid in edge_ids}
    active_graph = graph.copy()
    remaining = set(edge_ids)
    rows = []
    for index, observation in enumerate(fracture.observations):
        mask = observation['mask']
        if len(mask) != len(mask_edge_ids):
            raise ValueError('fracture observation has the wrong mask width')
        active = {eid for eid, enabled in zip(mask_edge_ids, mask) if enabled}
        if not active <= remaining or not set(observation['pending_failed_ids']) <= active:
            raise ValueError('fracture history must retain only active pending edges and irreversible deletions')
        removed = sorted(remaining - active)
        for eid in removed:
            active_graph.remove_edge(eid)
        remaining = active
        conductance = {eid: reference[eid] if eid in active else 0.0 for eid in edge_ids}
        solved, status = _measure(graph, active_graph, conductance, electrodes)
        if solved.edge_ids.tolist() != edge_ids or solved.node_ids.tolist() != node_ids:
            raise AssertionError('transport rows do not match recorded stable IDs')
        g = status['effective_conductance_S']
        rows.append(dict(
            observation=index, phase=observation['phase'], epoch=observation['epoch'],
            cascade=observation['cascade'], displacement_m=observation['displacement_m'],
            loaded_reaction_force_N=observation['loaded_reaction_force_N'],
            strain_energy_J=observation['strain_energy_J'],
            mechanical_grip_path_present=observation['path_present'],
            pending_failed_edge_ids=observation['pending_failed_ids'],
            removed_edge_ids=removed if index else [],
            active_edges=len(active), status='conducting' if g > 0 else 'open_circuit',
            resistance_ohm=1.0/g if g > 0 else None,
            voltage_V=solved.voltage_V.tolist(), current_A=solved.current_A.tolist(),
            node_injection_A=solved.node_injection_A.tolist(), conductance_S=solved.conductance_S.tolist(),
            **status,
        ))
    if remaining != set(fracture.data['state']['intact_segment_ids']):
        raise ValueError('fracture final active IDs disagree with the observation masks')
    return dict(
        length_unit='m', model='reference_chord_fixed_conductivity_and_area',
        conductivity_S_per_m=CONDUCTIVITY_S_PER_M, delta_voltage_V=1.0,
        electrodes=electrodes, node_ids=node_ids, edge_ids=edge_ids,
        fracture_mask_edge_ids=list(mask_edge_ids), fracture_binding=fracture.data['binding'],
        reference_conductance_S=[reference[eid] for eid in edge_ids],
        initially_inactive_edge_ids=[eid for eid, enabled in zip(mask_edge_ids, fracture.observations[0]['mask'])
                                     if not enabled],
        observations=rows,
    )


def run(output_dir):
    """Solve the fixed notched seed 11 case and measure every saved observation."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    graph, fixed, loaded, notch = make_network(11)
    electrodes = _electrodes(graph, axis=1)
    if electrodes != {'high_V': fixed, 'low_V': loaded}:
        raise AssertionError('original electrode locations differ from the fixed mechanical grips')
    config = FractureConfig(YOUNG_PA, POISSON, STRENGTH_PA, STRENGTH_PA, axis=1,
                            maximum_displacement_m=FULL_UM*1e-6, reference_displacement_m=1e-6)
    solver = FractureSolver(graph, config, fixed_nodes=fixed, loaded_nodes=loaded,
                            length_unit='m', initially_broken_edges=notch)
    fracture = solver.run()
    if not fracture.data['complete'] or fracture.data['binding'] != solver.binding:
        raise RuntimeError('incomplete or mismatched fracture response')
    mask_edge_ids = [eid for eid, _, _ in solver.geometry['original_edges']]
    report = _electrical_history(graph, fracture, mask_edge_ids=mask_edge_ids, electrodes=electrodes)
    rows = report['observations']
    if any(bool(row['spanning_component_ids']) != row['mechanical_grip_path_present'] for row in rows):
        raise AssertionError('fixed-electrode and mechanical grip paths disagree')
    report['seed'] = 11
    graph_path, fracture_path = output/'graph.json', output/'fracture-result.json'
    graph.save_json(graph_path)
    fracture.save(fracture_path)
    report['files_sha256'] = {path.name: sha256(path.read_bytes()).hexdigest()
                              for path in (graph_path, fracture_path)}
    _write_json(output/'electrical-history.json', report)
    fields = ('observation', 'phase', 'epoch', 'cascade', 'displacement_m', 'loaded_reaction_force_N',
              'strain_energy_J', 'active_edges', 'status', 'effective_conductance_S', 'resistance_ohm',
              'high_electrode_current_A', 'dissipated_power_W')
    with (output/'electrical-history.csv').open('x', newline='', encoding='utf-8') as stream:
        table = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        table.writeheader()
        table.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='new directory for saved results')
    args = parser.parse_args()
    rows = run(args.output_dir)['observations']
    print(f"{len(rows)} observations, initial G={rows[0]['effective_conductance_S']:.8g} S, "
          f"final G={rows[-1]['effective_conductance_S']:.8g} S ({rows[-1]['status']})")


if __name__ == '__main__':
    main()
