"""Run a one-way temperature -> mechanics -> conductance -> geometry workflow.

From a checkout: python examples/thermal_to_transport_export.py --output-dir NEW
Requires installed FiberNet and meshio==5.3.5. Importing this file runs no solver.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def _write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def build_graph(length_unit):
    """A 36 mm square network with sparse stable IDs and reversed records."""
    import fibernet as fn
    if length_unit not in ('m', 'mm'):
        raise ValueError('length_unit must be m or mm')
    scale = 1. if length_unit == 'm' else 1000.
    graph = fn.pattern_2d(unit='square', box=(.012*scale, .012*scale),
                          grid=(3, 3), radius=.00005*scale,
                          n_pts_per_side=0, seed=23)
    data = graph.to_dict()
    ids = {nid: 101+17*i for i, nid in enumerate(graph.nodes)}
    for node in data['nodes']:
        node['id'] = ids[node['id']]
    for i, edge in enumerate(data['edges']):
        edge.update(id=43+19*i, node_i=ids[edge['node_i']], node_j=ids[edge['node_j']])
    data['nodes'].reverse()
    data['edges'].reverse()
    data.pop('next_node_id')
    data.pop('next_edge_id')
    return fn.StructureGraph.from_dict(data)


def run(output_dir, *, length_unit='mm'):
    """Run the existing APIs once and verify the written geometry and fields."""
    import meshio
    import fibernet as fn
    from fibernet.core.fiber_network_profile import FiberNetworkProfile
    from fibernet.io.mesh_export import export_stl
    from fibernet.io.meshio_adapter import write_vtu_bundle
    from fibernet.sim.deformed_transport import prepare_deformed_transport
    from fibernet.sim.saved_temperature import read_saved_thermal_json, prepare_saved_temperature
    from fibernet.sim.transient_thermal import solve_transient_thermal

    graph = build_graph(length_unit)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    graph.save_json(out/'source.json')
    xmax = max(node.position[0] for node in graph.nodes.values())
    electrodes = {nid: 1. if node.position[0] == 0. else 0.
                  for nid, node in graph.nodes.items() if node.position[0] in (0., xmax)}
    fixed_temperature = {nid: 300. if voltage == 1. else 310.
                         for nid, voltage in electrodes.items()}
    thermal = solve_transient_thermal(
        graph, capacity_J_per_K={nid: .01 for nid in graph.nodes},
        conductance_W_per_K={eid: .2 for eid in graph.edges},
        initial_temperature_K={nid: fixed_temperature.get(nid, 300.) for nid in graph.nodes},
        times_s=[0., .01, .02, .03], source_W={nid: 0. for nid in graph.nodes},
        fixed_temperature_K=fixed_temperature, length_unit=length_unit)
    _write_json(out/'thermal.json', thermal.to_dict())

    # JSON reload changes record order; stable IDs bind the saved temperature.
    restored = fn.StructureGraph.load_json(out/'source.json')
    frame = prepare_saved_temperature(restored, read_saved_thermal_json(out/'thermal.json'),
                                      frame_index=3, length_unit=length_unit)
    mechanical_bundle = frame.solve(
        restored, profile='network', E_Pa=1e6,
        expansion_alpha_per_K={eid: 1e-6 for eid in restored.edges},
        stress_free_temperature_K=300.,
        constraints={nid: (0., 0., 0.) for nid, node in restored.nodes.items()
                     if node.position[0] == 0.})
    _write_json(out/'mechanical.json', mechanical_bundle)
    mechanical = mechanical_bundle['mechanical']
    # Planar frame rows are [ux_m, uy_m, rotation_rad]; rotation is not z motion.
    translations = {nid: row[:2] for nid, row in
                    zip(mechanical['node_ids'], mechanical['displacement'])}
    derived, conductance, binding = prepare_deformed_transport(
        restored, translations_m_by_id=translations,
        reference_conductance_S={eid: 1. for eid in restored.edges}, length_unit=length_unit)
    binding_path = out/'transport_binding.json'
    _write_json(binding_path, binding)
    preparation = {
        'schema': binding['schema'],
        'input_sha256': binding['input_sha256'],
        'report_file_sha256': hashlib.sha256(binding_path.read_bytes()).hexdigest(),
    }
    electrical = fn.solve_linear_transport(
        derived, conductance, list(electrodes.items()), length_unit=length_unit,
        model_metadata={'deformed_transport_preparation': preparation})
    _write_json(out/'electrical.json', electrical.to_dict())
    derived.save_json(out/'derived.json')

    vectors = {nid: [*translations[nid], 0.] for nid in derived.nodes}
    write_vtu_bundle(out/'vtu', derived, length_unit=length_unit,
        point_data_by_id={'translation_m': vectors},
        field_metadata={'point': {'translation_m': {
            'units': 'm', 'meaning': 'Thermoelastic translational displacement'}}})
    mesh = meshio.read(out/'vtu/network.vtu')
    if set(map(int, mesh.point_data['node_id'])) != set(derived.nodes):
        raise RuntimeError('VTU node identities changed')
    if set(map(int, mesh.cell_data['edge_id'][0])) != set(derived.edges):
        raise RuntimeError('VTU edge identities changed')
    for row, nid in enumerate(mesh.point_data['node_id']):
        np.testing.assert_array_equal(mesh.points[row], derived.nodes[int(nid)].position)
        np.testing.assert_array_equal(mesh.point_data['translation_m'][row], vectors[int(nid)])
    for eid, rows in zip(mesh.cell_data['edge_id'][0], mesh.cells[0].data):
        edge = derived.edges[int(eid)]
        if [int(mesh.point_data['node_id'][r]) for r in rows] != [edge.node_i, edge.node_j]:
            raise RuntimeError('VTU oriented incidence changed')

    fibers, projection = derived.to_fiber_network(profile=FiberNetworkProfile(
        length_unit=length_unit, coordinate_frame='world_xy'))
    _write_json(out/'fiber_projection.json', projection)
    for record in projection['edges']:
        edge = derived.edges[record['edge_id']]
        fiber = fibers.fibers[record['fiber_row']]
        np.testing.assert_array_equal(fiber.start_point, derived.nodes[edge.node_i].position)
        np.testing.assert_array_equal(fiber.end_point, derived.nodes[edge.node_j].position)
        if fiber.radius != edge.radius:
            raise RuntimeError('fiber radius changed')
    export_stl(fibers, str(out/'network.stl'), n_sides=12, binary=True)
    raw = (out/'network.stl').read_bytes()
    triangles = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84+50*triangles:
        raise RuntimeError('binary STL length differs from its triangle count')
    _write_json(out/'stl_geometry.json', {
        'coordinate_unit': length_unit, 'unit_source': 'external declaration',
        'geometry': 'separate capped fiber cylinders', 'triangles': triangles,
        'identity_mapping_file': 'fiber_projection.json',
        'stl_embeds_units_or_stable_ids': False})
    summary = {
        'workflow': 'sequential_one_way', 'feedback': False,
        'node_count': len(graph.nodes), 'edge_count': len(graph.edges),
        'source_length_unit': length_unit, 'mechanical_translation_unit': 'm',
        'max_translation_m': float(np.max(np.linalg.norm(
            np.asarray(mechanical['displacement'])[:, :2], axis=1))),
        'electrical_power_W': electrical.to_dict()['dissipated_power_W'],
        'stl_triangles': triangles,
        'files_sha256': {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(out.rglob('*')) if p.is_file()}}
    _write_json(out/'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--length-unit', choices=('m', 'mm'), default='mm')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir, length_unit=args.length_unit), indent=2))


if __name__ == '__main__':
    main()
