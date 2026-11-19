"""Scan displacement bounds for brittle-member removal in a synthetic notched mesh.

Run: python examples/fracture_scan.py --output-dir NEW_DIRECTORY
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from fibernet import Material, StructureGraph
from fibernet.sim.fracture import FractureConfig, FractureSolver


SEEDS = (11, 23, 37)
COLUMNS, ROWS = 9, 7
SPACING_M = 0.01
JITTER_FRACTION = 0.05
RADIUS_M = 0.0005
YOUNG_PA, POISSON = 1e9, 0.3
STRENGTH_PA = 1e6
NOTCH_Y_M, NOTCH_END_X_M = 0.035, 0.026
SHORT_UM, FULL_UM = 10.0, 400.0
ENERGY_RTOL = 1e-8


def _write_json(path, data):
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def make_network(seed):
    """Return the undeformed network, bottom/top grips, and initially absent edges."""
    rng = np.random.Generator(np.random.PCG64(seed))
    graph = StructureGraph(dimension=3)
    material = Material(youngs_modulus=YOUNG_PA, poissons_ratio=POISSON)
    grid = []
    for row in range(ROWS):
        nodes = []
        for col in range(COLUMNS):
            xy = SPACING_M * np.array([col, row], dtype=float)
            if 0 < col < COLUMNS - 1 and 0 < row < ROWS - 1:
                xy += rng.uniform(-JITTER_FRACTION, JITTER_FRACTION, 2) * SPACING_M
            nodes.append(graph.add_node([*xy, 0.0], merge=False))
        grid.append(nodes)
    notch = []
    for row, nodes in enumerate(grid):
        for col, node in enumerate(nodes):
            neighbors = []
            if col + 1 < COLUMNS:
                neighbors.append(nodes[col + 1])
            if row + 1 < ROWS:
                neighbors.append(grid[row + 1][col])
                if col + 1 < COLUMNS:
                    neighbors.append(grid[row + 1][col + 1])
            for neighbor in neighbors:
                edge = graph.add_edge(node, neighbor, radius=RADIUS_M, material=material)
                x0, y0, _ = graph.nodes[node].position
                x1, y1, _ = graph.nodes[neighbor].position
                if y0 < NOTCH_Y_M < y1:
                    cut_x = x0 + (x1 - x0) * (NOTCH_Y_M - y0) / (y1 - y0)
                    if cut_x <= NOTCH_END_X_M:
                        notch.append(edge)
    return graph, grid[0], grid[-1], notch


def summarize_result(result):
    """Measure ordered physical observations including constant-displacement avalanches."""
    data = result.data
    state = data['state']
    observations = result.observations
    load_steps, drops = [], []
    loss_displacement = None
    for prev, obs in zip(observations, observations[1:]):
        step = obs['displacement_m'] - prev['displacement_m']
        changed = prev['mask'] != obs['mask']
        if step < 0 or (changed and step != 0):
            raise RuntimeError('Fracture observations cannot be integrated as a loading trace')
        load_steps.append(step * (prev['loaded_reaction_force_N'] + obs['loaded_reaction_force_N']) / 2)
        if changed:
            drops.append(prev['strain_energy_J'] - obs['strain_energy_J'])
        if loss_displacement is None and prev['path_present'] and not obs['path_present']:
            loss_displacement = obs['displacement_m']
    final_strain = state['strain_energy_J']
    work = math.fsum(load_steps)
    release = work - (final_strain - observations[0]['strain_energy_J'])
    release_at_events = math.fsum(drops)
    tolerance = ENERGY_RTOL * max(abs(work), abs(final_strain), abs(release_at_events), 1e-30)
    residual = release - release_at_events
    if (abs(residual) > tolerance or release < -tolerance
            or any(drop < -tolerance for drop in drops)):
        raise RuntimeError('Fracture work and elastic energy-drop ledger disagree')

    path_loss_observed = loss_displacement is not None
    if path_loss_observed == state['topology']['path_present']:
        raise RuntimeError('Fracture observations and final path status disagree')
    reason = result.termination_reason
    if path_loss_observed:
        status = 'path_loss_observed'
    elif not data['complete']:
        status = 'incomplete'
    elif math.isclose(state['displacement_m'], data['config']['maximum_displacement_m'],
                      rel_tol=4 * math.ulp(1.0)):
        status = 'right_censored_loading_bound'
    else:
        status = 'stopped_without_path_loss'
    return dict(
        status=status, complete=data['complete'], termination_reason=reason,
        path_loss_observed=path_loss_observed,
        path_loss_displacement_m=loss_displacement,
        final_displacement_m=state['displacement_m'],
        peak_reaction_N=max(obs['loaded_reaction_force_N'] for obs in observations),
        failed_edge_count=len(result.events),
        first_failure_displacement_m=result.events[0]['displacement_m'] if result.events else None,
        external_work_J=work, final_strain_energy_J=final_strain,
        released_elastic_energy_J=release, event_elastic_energy_drop_J=release_at_events,
        elastic_energy_ledger_residual_J=residual,
    )


def _run_case(out, seed, horizon, config, graph, fixed, loaded, notch):
    name = f'seed-{seed}-horizon-{horizon}'
    case = out / name
    case.mkdir()
    settings = dict(seed=seed, rng='PCG64', horizon=horizon, length_unit='m',
                    graph=graph.to_dict(), config=asdict(config), fixed_nodes=fixed,
                    loaded_nodes=loaded, initially_broken_edges=notch)
    row = dict(
        case=name, seed=seed, horizon=horizon, displacement_bound_m=config.maximum_displacement_m,
        input_file=f'{name}/input.json', input_sha256=_write_json(case / 'input.json', settings),
    )
    try:
        solver = FractureSolver(graph, config, fixed_nodes=fixed, loaded_nodes=loaded,
                                length_unit='m', initially_broken_edges=notch)
        result = solver.run()
    except (ImportError, MemoryError, RuntimeError, TimeoutError, ValueError) as exc:
        return dict(row, status='error', complete=False, error=f'{type(exc).__name__}: {exc}')
    result_file = case / 'result.json'
    result.save(result_file)
    row.update(result_file=f'{name}/result.json',
               result_sha256=hashlib.sha256(result_file.read_bytes()).hexdigest(),
               result_binding=result.data['binding'])
    try:
        row.update(summarize_result(result))
    except RuntimeError as exc:
        row.update(status='error', complete=False, error=f'{type(exc).__name__}: {exc}')
    return row


def run_scan(out, *, seeds=SEEDS, short_um=SHORT_UM, full_um=FULL_UM):
    """Run two matched displacement horizons per seed, into a new directory."""
    seeds = tuple(seeds)
    if (not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('Provide one to eight distinct seeds in [0, 2**32)')
    if not 0 < short_um < full_um <= 1000:
        raise ValueError('Require 0 < short_um < full_um <= 1000')
    # Keep the reference displacement identical between horizons.
    reference_m = min(1e-6, short_um * 1e-6)
    out = Path(out)
    out.mkdir()
    rows = []
    for seed in seeds:
        graph, fixed, loaded, notch = make_network(seed)
        for horizon, bound_um in (('short', short_um), ('full', full_um)):
            config = FractureConfig(
                YOUNG_PA, POISSON, STRENGTH_PA, STRENGTH_PA, axis=1,
                maximum_displacement_m=bound_um * 1e-6, reference_displacement_m=reference_m,
            )
            rows.append(_run_case(out, seed, horizon, config, graph, fixed, loaded, notch))
    fields = (
        'case', 'seed', 'horizon', 'displacement_bound_m', 'status', 'complete',
        'termination_reason', 'path_loss_observed', 'path_loss_displacement_m',
        'final_displacement_m', 'peak_reaction_N', 'failed_edge_count',
        'first_failure_displacement_m',
        'external_work_J', 'final_strain_energy_J', 'released_elastic_energy_J',
        'event_elastic_energy_drop_J', 'elastic_energy_ledger_residual_J',
        'input_file', 'input_sha256', 'result_file',
        'result_sha256', 'error',
    )
    with (out / 'scan.csv').open('x', encoding='utf-8', newline='') as stream:
        table = csv.DictWriter(stream, fieldnames=fields)
        table.writeheader()
        table.writerows({key: row.get(key) for key in fields} for row in rows)
    _write_json(out / 'summary.json', dict(
        example_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seeds=list(seeds), rng='PCG64', short_um=short_um, full_um=full_um,
        grid=dict(columns=COLUMNS, rows=ROWS, spacing_m=SPACING_M,
                  interior_jitter_fraction=JITTER_FRACTION,
                  notch_y_m=NOTCH_Y_M, notch_end_x_m=NOTCH_END_X_M),
        energy_scope='Elastic energy release in the member-removal model, '
                     'external work minus the change in stored elastic energy.',
        cases=rows,
    ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--short-um', type=float, default=SHORT_UM)
    parser.add_argument('--full-um', type=float, default=FULL_UM)
    args = parser.parse_args()
    try:
        rows = run_scan(args.output_dir, seeds=args.seeds, short_um=args.short_um, full_um=args.full_um)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for row in rows:
        outcome = row.get('error', row.get('termination_reason', ''))
        print(f"{row['case']} {row['status']} ({outcome})")
    return int(any(row['status'] in ('error', 'incomplete') for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
