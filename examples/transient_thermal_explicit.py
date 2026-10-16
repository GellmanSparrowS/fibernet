"""Small explicit SI thermal network; no CTE, deformation or inferred properties.

Run from the project root: python -m examples.transient_thermal_explicit
Importing this module does not solve a model or write a file.
The 0.02-second duration is a quick entry-point demo, not full relaxation
or a new time-accuracy validation.
"""


def build_inputs():
    from fibernet import StructureGraph
    graph = StructureGraph.from_dict({
        'dimension': 2,
        'nodes': [{'id': 101, 'position': [0, 0, 0]},
                  {'id': 907, 'position': [1, 0, 0]}],
        'edges': [{'id': 11, 'node_i': 101, 'node_j': 907},
                  {'id': 31, 'node_i': 101, 'node_j': 907}],
    })
    return graph, {
        'capacity_J_per_K': {101: 2., 907: 1.},
        'conductance_W_per_K': {11: .2, 31: .3},
        'initial_temperature_K': {101: 300., 907: 310.},
        'times_s': [0., .01, .02],
        'source_W': {101: 0., 907: 0.},
        'fixed_temperature_K': {907: 310.},
        'length_unit': 'm',  # Geometry provenance only; g is explicit W/K.
    }


def main():
    from fibernet.sim.transient_thermal import solve_transient_thermal, ThermalSolveError
    graph, inputs = build_inputs()
    print('Synthetic constant-parameter lumped thermal network; all C/g/T0/Q are explicit SI values.')
    print('No effective conductivity, CTE, deformation or general-network time-accuracy claim.')
    print('Quick entry-point demo: 0.02 s only; not full thermal relaxation or a new accuracy validation.')
    try:
        result = solve_transient_thermal(graph, **inputs)
    except ThermalSolveError as exc:
        if exc.partial_result is None:
            print('Numerical failure: initial derived fields are not representable; no frame accepted.')
        else:
            print(f'Numerical failure: last accepted time = {exc.partial_result.times_s[-1]:g} s.')
        print(f'Attempted step {exc.attempted_step}: {exc}')
        return 1
    print(f'Completed requested discrete steps: {result.completed}; final time {result.times_s[-1]:g} s')
    print('t=0 interval residual is unavailable, not zero.')
    for i, node_id in enumerate(result.node_ids):
        residual = (f'{result.free_node_residual_J[-1, i]:.6g} J'
                    if node_id not in inputs['fixed_temperature_K'] else 'unavailable (fixed node)')
        print(f'node {node_id}: T={result.temperature_K[-1,i]:.9g} K; '
              f'stored heat={result.node_stored_heat_change_J[-1,i]:.9g} J; '
              f'reservoir injection={result.reservoir_injection_W[-1,i]:.9g} W; '
              f'interval residual={residual}')
    for i, edge_id in enumerate(result.edge_ids):
        a, b = result.endpoints_by_id[i]
        print(f'edge {edge_id} ({a}->{b}): heat rate={result.edge_heat_rate_W[-1,i]:.9g} W')
    print('Discrete energy checks passed with local algebraic and temperature-representation allowances.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
