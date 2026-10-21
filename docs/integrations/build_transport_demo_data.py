"""Build labeled synthetic datasets using the reviewed electrical backend.

Run from any directory with the FiberNet project environment. This saves data
only; it does not render figures or claim experimentally calibrated physics.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fibernet import StructureGraph, pattern_2d
from fibernet.sim.linear_transport import solve_linear_transport


def build():
    source = ROOT / 'fibernet/sim/linear_transport.py'
    accepted_hash = '56af003f63a17de0ef1eb09c871cd76c56c8198810edb17e8b3c57e5611ed0b6'
    if hashlib.sha256(source.read_bytes()).hexdigest() != accepted_hash:
        raise RuntimeError('Backend changed; obtain independent numerical acceptance before rebuilding this reviewed dataset')
    directory = Path(__file__).resolve().parent
    fixture = json.loads((directory / 'transport_pilot_fixture.json').read_text())
    analytic = StructureGraph.from_dict({
        'dimension': 3,
        'nodes': [{'id': nid, 'position': xyz} for nid, xyz in
                  zip(fixture['node_ids'], fixture['coordinates_m'])],
        'edges': [{'id': eid, 'node_i': a, 'node_j': b} for eid, (a, b) in
                  zip(fixture['edge_ids'], fixture['endpoints_by_id'])],
    })
    canonical = pattern_2d(unit='square', grid=(3, 3), box=(3., 3.))
    x = {i: float(n.position[0]) for i, n in canonical.nodes.items()}
    lo, hi = min(x.values()), max(x.values())
    boundary = [(i, 1.) for i, value in x.items() if np.isclose(value, lo)]
    boundary += [(i, 0.) for i, value in x.items() if np.isclose(value, hi)]
    setups = [
        ('analytic-series-parallel', analytic,
         dict(zip(fixture['edge_ids'], fixture['conductance_S'])), [(10, 1.), (120, 0.)],
         {'type': 'explicit fixture', 'fixture': 'transport_pilot_fixture.json'},
         {'voltage_V': fixture['expected_voltage_V'],
          'current_A': fixture['expected_oriented_current_A'], 'power_W': 3.}),
        ('generated-square-network', canonical, {i: 1. for i in canonical.edges}, boundary,
         {'function': 'pattern_2d', 'unit': 'square', 'grid': [3, 3], 'box': [3., 3.]},
         {'voltage_V': [(hi-x[i])/(hi-lo) for i in sorted(canonical.nodes)]}),
    ]
    cases = []
    for name, graph, conductance, bc, generation, expected in setups:
        r = solve_linear_transport(graph, conductance, bc, length_unit='m',
                                   model_metadata={'label': 'synthetic capability demonstration',
                                                   'conductance_source': 'explicit prescribed values, not geometry-derived'})
        np.testing.assert_allclose(r.voltage_V, expected['voltage_V'], rtol=1e-10, atol=1e-12)
        if 'current_A' in expected:
            np.testing.assert_allclose(r.current_A, expected['current_A'], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(r.dissipated_power_W, expected['power_W'], rtol=1e-10)
        cases.append({
            'name': name,
            'label': 'SYNTHETIC: ideal steady linear resistor network; not material calibration',
            'input': {'graph': graph.to_dict(), 'generation': generation,
                      'conductance_by_edge_id_S': {str(k): v for k, v in conductance.items()},
                      'dirichlet_voltage_V': [list(pair) for pair in bc],
                      'geometry_length_unit': 'm', 'rtol': 1e-10,
                      'current_atol_A': 0., 'power_atol_W': 0.},
            'expected_analytic_fields': expected,
            'result': r.to_dict(),
            'plotting': {'node_order': 'result.node_ids', 'edge_order': 'result.edge_ids',
                         'edge_orientation': 'result.endpoints_by_id: first ID to second ID',
                         'coordinates': 'input.graph.nodes, joined by stable ID',
                         'voltage_units': 'V', 'current_units': 'A', 'power_units': 'W'},
        })
    sweep = []
    analytic_g = dict(zip(fixture['edge_ids'], fixture['conductance_S']))
    for applied_V in (.25, .5, 1., 2.):
        r = solve_linear_transport(analytic, analytic_g, [(10, applied_V), (120, 0.)],
                                   length_unit='m', model_metadata={
                                       'label': 'synthetic analytic voltage scaling sweep',
                                       'conductance_source': 'explicit fixture'})
        reference = {'terminal_conductance_S': 3.,
                     'terminal_current_A': 3.*applied_V,
                     'dissipated_power_W': 3.*applied_V**2,
                     'voltage_V': (np.array(fixture['expected_voltage_V'])*applied_V).tolist(),
                     'oriented_current_A': (np.array(fixture['expected_oriented_current_A'])*applied_V).tolist()}
        np.testing.assert_allclose(r.voltage_V, reference['voltage_V'], rtol=1e-10, atol=0)
        np.testing.assert_allclose(r.current_A, reference['oriented_current_A'], rtol=1e-10, atol=0)
        np.testing.assert_allclose(r.dissipated_power_W, reference['dissipated_power_W'], rtol=1e-10)
        sweep.append({'applied_voltage_V': applied_V,
                      'input_case': 'analytic-series-parallel',
                      'dirichlet_voltage_V': [[10, applied_V], [120, 0.]],
                      'analytical_reference': reference, 'actual_solver_output': r.to_dict()})
    source = ROOT / 'fibernet/sim/linear_transport.py'
    return {'schema': 'fibernet-synthetic-transport-demo-1',
            'backend_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope': 'Software/analytic capability demonstration; no nonlinear mechanics, material calibration or experimental validation',
            'numerical_guard_status': 'Independent review accepted this backend source hash before dataset generation',
            'cases': cases, 'analytic_voltage_sweep': sweep}


if __name__ == '__main__':
    target = Path(__file__).with_name('transport_demo_dataset.json')
    target.write_text(json.dumps(build(), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(target)
