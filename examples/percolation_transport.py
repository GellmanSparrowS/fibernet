"""Compare batch edge removal in a finite welded resistor network.

Run: python examples/percolation_transport.py --output-dir NEW_DIRECTORY
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from fibernet import StructureGraph
from fibernet.analysis import analyze_topology
from fibernet.gen.pattern import pattern_2d
from fibernet.sim.linear_transport import solve_linear_transport


BATCH_SIZE = 8
RANDOM_SEEDS = (11, 23, 37, 53, 71)
TIE_SEED = 197
CONDUCTIVITY_S_PER_M = 1.0
FIELD_NAMES = ('voltage_V', 'current_A', 'node_injection_A')


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write('\n')
    if json.loads(path.read_text(encoding='utf-8')) != value:
        raise AssertionError(f'JSON readback mismatch: {path.name}')


def _electrodes(graph, axis):
    lower, upper = graph.bounding_box()
    return {
        'high_V': sorted(n for n, node in graph.nodes.items()
                         if abs(node.position[axis] - lower[axis]) <= 1e-12),
        'low_V': sorted(n for n, node in graph.nodes.items()
                        if abs(node.position[axis] - upper[axis]) <= 1e-12),
    }


def _components(active, electrodes):
    topology = analyze_topology(active)
    component = dict(zip(topology.node_ids, topology.component_by_node))
    high = {component[n] for n in electrodes['high_V']}
    low = {component[n] for n in electrodes['low_V']}
    gauges = [c for c, _ in topology.component_sizes if c not in high | low]
    return topology, sorted(high & low), gauges


def _measure(graph, active, conductance, electrodes):
    topology, spanning, gauges = _components(active, electrodes)
    prescribed = dict.fromkeys(electrodes['high_V'], 1.0)
    prescribed.update(dict.fromkeys(electrodes['low_V'] + gauges, 0.0))
    solved = solve_linear_transport(graph, conductance, prescribed, length_unit='m')
    injections = dict(zip(solved.node_ids, solved.node_injection_A))
    current_high = math.fsum(injections[n] for n in electrodes['high_V'])
    current_low = math.fsum(injections[n] for n in electrodes['low_V'])
    if any(injections[n] != 0.0 for n in gauges):
        raise AssertionError('a floating-component gauge supplied current')
    if not math.isclose(current_high, -current_low, rel_tol=1e-9, abs_tol=1e-15):
        raise AssertionError('electrode currents do not balance')
    if not math.isclose(current_high, solved.dissipated_power_W, rel_tol=1e-9, abs_tol=1e-15):
        raise AssertionError('electrode work does not match dissipation at 1 V')
    if (spanning and current_high <= 0) or (not spanning and current_high != 0.0):
        raise AssertionError('transport disagrees with active-edge spanning')
    return solved, {
        'spanning_component_ids': spanning,
        'components': topology.n_components,
        'isolated_nodes': topology.n_isolates,
        'gauge_node_ids': gauges,
        'high_electrode_current_A': current_high,
        'low_electrode_current_A': current_low,
        'effective_conductance_S': current_high / 1.0,
        'dissipated_power_W': solved.dissipated_power_W,
        'free_node_residual_A': solved.free_node_residual_A,
    }


def _trajectory(graph, initial_g, volumes, electrodes, fields, *, batch_size, strategy, seed):
    active = graph.copy()
    edge_ids = list(graph.edges)
    node_ids = sorted(graph.nodes)
    conductance = dict(initial_g)
    volume_total = math.fsum(volumes.values())
    removed = []
    batch = []
    rng = np.random.Generator(np.random.PCG64(seed))
    order = [int(eid) for eid in rng.permutation(sorted(edge_ids))]
    rank = {eid: index for index, eid in enumerate(order)}
    stages = []
    while True:
        solved, status = _measure(graph, active, conductance, electrodes)
        if solved.edge_ids.tolist() != edge_ids or solved.node_ids.tolist() != node_ids:
            raise AssertionError('solver result IDs changed across deletion states')
        if stages and status['effective_conductance_S'] > (
            stages[-1]['effective_conductance_S'] * (1 + 1e-9) + 1e-15
        ):
            raise AssertionError('deleting conductance increased effective conductance')
        field_row = len(fields['current_A'])
        for name in FIELD_NAMES:
            fields[name].append(getattr(solved, name))
        stages.append({
            'step': len(stages), 'field_row': field_row, 'deleted_edge_ids': batch,
            'removed_edges': len(removed), 'removed_edge_fraction': len(removed) / len(edge_ids),
            'removed_member_volume_fraction': math.fsum(volumes[e] for e in removed) / volume_total,
            **status,
        })
        if not status['spanning_component_ids']:
            if not removed:
                raise ValueError('the original graph must span the fixed electrodes')
            return stages

        if strategy == 'low_flow':
            abs_current = np.abs(solved.current_A)
            # Bin normalized currents before ranking so near-zero solver noise
            # does not act as a spatial tie-breaker.
            score = dict(zip(edge_ids, np.round(abs_current / abs_current.max(), 12)))
            candidates = sorted(active.edges, key=lambda eid: (score[eid], rank[eid]))
        else:
            candidates = [eid for eid in order if eid in active.edges]
        batch = candidates[:batch_size]
        for eid in batch:
            active.remove_edge(eid)
            conductance[eid] = 0.0
        removed.extend(batch)


def _loss_summary(graph, stages, electrodes, volumes):
    last, first = stages[-2:]
    bracket_keys = ('step', 'removed_edges', 'removed_edge_fraction', 'removed_member_volume_fraction')
    # The saved deletion order also permits exact topology-only replay inside
    # the terminal batch. Electrical fields are sampled at batch boundaries.
    active = graph.copy()
    removed = []
    for stage in stages[:-1]:
        for eid in stage['deleted_edge_ids']:
            active.remove_edge(eid)
            removed.append(eid)
    for eid in first['deleted_edge_ids']:
        active.remove_edge(eid)
        removed.append(eid)
        _, spans, _ = _components(active, electrodes)
        if not spans:
            return {
                'last_spanning_sample': {key: last[key] for key in bracket_keys},
                'first_nonspanning_sample': {key: first[key] for key in bracket_keys},
                'first_topological_loss': {
                    'edge_id': eid, 'removed_edges': len(removed),
                    'removed_edge_fraction': len(removed) / len(graph.edges),
                    'removed_member_volume_fraction': (
                        math.fsum(volumes[e] for e in removed) / math.fsum(volumes.values())
                    ),
                },
            }
    raise AssertionError('could not replay the final spanning loss')


def run(output_dir):
    """Solve and save the fixed two-direction example in a new directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    graph = pattern_2d(unit='triangle', box=(0.01, 0.01), grid=(16, 16), radius=0.0005,
                       n_internal=0, n_pts_per_side=0, perturbation=0.0, boundary_mode='none')
    if (len(graph.nodes), len(graph.edges)) != (289, 800):
        raise AssertionError('unexpected generated graph size')
    graph_path = output / 'graph.json'
    graph.save_json(graph_path)
    restored = StructureGraph.load_json(graph_path)
    if restored.to_dict() != graph.to_dict():
        raise AssertionError('initial graph readback mismatch')
    graph = restored

    lengths = dict(zip(graph.edges, graph.edge_lengths()))
    areas = {eid: math.pi * edge.radius**2 for eid, edge in graph.edges.items()}
    volumes = {eid: areas[eid] * lengths[eid] for eid in graph.edges}
    initial_g = {eid: CONDUCTIVITY_S_PER_M * areas[eid] / lengths[eid] for eid in graph.edges}
    electrodes = {axis: _electrodes(graph, dim) for dim, axis in enumerate(('x', 'y'))}
    _write_json(output / 'case.json', {
        'length_unit': 'm', 'conductivity_S_per_m': CONDUCTIVITY_S_PER_M,
        'delta_voltage_V': 1.0, 'batch_size': BATCH_SIZE, 'electrodes': electrodes,
        'tie_seed': TIE_SEED, 'random_seeds': list(RANDOM_SEEDS), 'bit_generator': 'PCG64',
        'low_flow_score': 'abs(current_A) / max(abs(current_A)), rounded to 12 decimals',
        'edges': [{'id': eid, 'conductance_S': initial_g[eid], 'member_volume_m3': volumes[eid]}
                  for eid in graph.edges],
    })

    fields = {name: [] for name in FIELD_NAMES}
    histories = []
    for axis, fixed in electrodes.items():
        for strategy, seed in [('low_flow', TIE_SEED), *[('random', s) for s in RANDOM_SEEDS]]:
            stages = _trajectory(graph, initial_g, volumes, fixed, fields, batch_size=BATCH_SIZE,
                                 strategy=strategy, seed=seed)
            histories.append({
                'axis': axis, 'strategy': strategy, 'seed': seed, 'states': stages,
                **_loss_summary(graph, stages, fixed, volumes),
            })
    _write_json(output / 'history.json', histories)

    csv_fields = ('axis', 'strategy', 'seed', 'step', 'field_row', 'removed_edges',
                  'removed_edge_fraction', 'removed_member_volume_fraction', 'spanning',
                  'effective_conductance_S', 'dissipated_power_W', 'components', 'isolated_nodes')
    rows = []
    for history in histories:
        for state in history['states']:
            row = {**state, **{key: history[key] for key in ('axis', 'strategy', 'seed')},
                   'spanning': bool(state['spanning_component_ids'])}
            rows.append([str(row[key]) for key in csv_fields])
    csv_path = output / 'history.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_fields)
        writer.writerows(rows)
    with csv_path.open(newline='', encoding='utf-8') as stream:
        if list(csv.reader(stream)) != [list(csv_fields), *rows]:
            raise AssertionError('CSV readback mismatch')

    field_path = output / 'fields.npz'
    arrays = {'node_ids': sorted(graph.nodes), 'edge_ids': list(graph.edges), **fields}
    np.savez_compressed(field_path, **arrays)
    with np.load(field_path, allow_pickle=False) as saved:
        if any(not np.array_equal(saved[name], values) for name, values in arrays.items()):
            raise AssertionError('NPZ field readback mismatch')
    return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='new directory for saved results')
    args = parser.parse_args()
    histories = run(args.output_dir)
    for history in histories:
        last = history['last_spanning_sample']['removed_edges']
        first = history['first_nonspanning_sample']['removed_edges']
        print(f"{history['axis']} {history['strategy']} seed={history['seed']}: "
              f"last spanning={last}, first nonspanning={first}")


if __name__ == '__main__':
    main()
