"""One StructureGraph, steady electrical power, then explicit transient heat.

Run: python -m examples.one_way_joule_heating
Importing this example performs no solve. Values are synthetic SI inputs,
not material calibration. No temperature feedback or mechanical coupling.
"""


def run_example():
    import fibernet as fn
    graph = fn.StructureGraph.from_dict({
        'dimension': 2,
        'nodes': [{'id': 101, 'position': [0, 0, 0]},
                  {'id': 907, 'position': [1, 0, 0]}],
        'edges': [{'id': 11, 'node_i': 101, 'node_j': 907},
                  {'id': 31, 'node_i': 907, 'node_j': 101},
                  {'id': 67, 'node_i': 101, 'node_j': 907}],
    })
    electrical = fn.solve_linear_transport(
        graph, {11: 1., 31: 3., 67: 0.}, {101: 2., 907: 0.}, length_unit='m')
    # Explicit endpoint-ID fractions; edge orientation does not choose where
    # heat goes. Equal halves would be a different lumping convention.
    endpoint_weights = {eid: {101: .25, 907: .75} for eid in graph.edges}
    heating = fn.run_joule_heating(
        graph, electrical,
        capacity_J_per_K={101: 2., 907: 3.},
        conductance_W_per_K={11: .2, 31: .3, 67: 0.},
        initial_temperature_K={101: 300., 907: 300.},
        times_s=[k/10 for k in range(11)],
        endpoint_weights=endpoint_weights, length_unit='m')
    return graph, electrical, heating


def main():
    _, electrical, heating = run_example()
    print('Synthetic one-way DC heating; explicit constant SI inputs')
    print('Electrical edge powers [W]:', dict(zip(electrical.edge_ids, electrical.edge_power_W)))
    print('Joule sources by node [W]:', heating.heat_load.joule_source_W)
    print('Final temperatures [K]:', dict(zip(heating.thermal.node_ids, heating.thermal.temperature_K[-1])))
    print('Stored heat at 1 s [J]:', float(heating.thermal.node_stored_heat_change_J[-1].sum()))
    print('Expected electrical power: 16 W; closed-network stored heat: 16 J')
    print('Time integration is backward Euler; no general time-accuracy or material-calibration claim')


if __name__ == '__main__':
    main()
