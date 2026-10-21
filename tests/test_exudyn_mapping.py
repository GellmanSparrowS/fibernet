"""Pure straight-chain mapping, provenance, and graph-domain regressions."""

import json
import math
from pathlib import Path
import unittest
from dataclasses import replace

import numpy as np

from fibernet import Material, StructureGraph
from fibernet.sim._exudyn_contract import (
    ExudynContractError,
    prepare_exudyn_contact,
    validate_prepared,
)
from fibernet.sim._exudyn_mapping import compile_straight_chain

from _exudyn_test_support import (
    ContractAssertions,
    make_chain,
    make_request,
    remap_ids,
)


class TestExudynMapping(ContractAssertions, unittest.TestCase):

    def test_n32_shared_layout_and_source_not_mutated(self):
        g = make_chain()
        before = json.dumps(g.to_dict(), sort_keys=True)
        p = prepare_exudyn_contact(g, make_request(g))
        m = p.mapping()
        self.assertTrue(validate_prepared(p))
        self.assertEqual(json.dumps(g.to_dict(), sort_keys=True), before)
        self.assertEqual(
            m['counts'],
            {
                'physical_nodes': 33, 'physical_edges': 32, 'beam_nodes': 33,
                'sphere_nodes': 1, 'ODE2': 201, 'AE': 12, 'welds': 0,
            },
        )
        self.assertEqual(m['sphere']['translation_dofs'], [198, 199, 200])
        self.assertEqual(m['beams'][0]['contact_dofs'], list(range(12)) + [198, 199, 200])
        self.assertEqual(m['beams'][-1]['contact_dofs'], list(range(186, 198)) + [198, 199, 200])
        self.assertEqual(
            m['constraints']['zero_coordinate_dofs'],
            list(range(6)) + list(range(192, 198)),
        )
        self.assertEqual(m['forecast']['accepted_states'], 119)
        self.assertEqual(m['forecast']['phase_start_records'], 2)
        self.assertEqual(
            m['forecast']['q_qt_qdd_AE_float64_payload_bytes'],
            121 * 8 * (3 * 201 + 12),
        )
        self.assertEqual(m['loads'], [])
        self.assertTrue(all((b['native_object_index'] is None for b in m['beams'])))
        self.assertTrue(all((b['contact_mass_matrix'] == 'zero_15_by_15' for b in m['beams'])))

    def test_independent_sections_mass_and_exclusion_radius(self):
        g = make_chain(1)
        r = make_request(g)
        m = prepare_exudyn_contact(g, r).mapping()
        s = m['sections'][0]
        area = math.pi * 5e-05 ** 2
        I = math.pi * 5e-05 ** 4 / 4
        G = 1000000.0 / 2.6
        np.testing.assert_allclose(
            s['stiffness_diagonal_SI'],
            [1000000.0 * area, G * area, G * area, G * 2 * I, 1000000.0 * I, 1000000.0 * I],
            rtol=1e-15,
        )
        self.assertAlmostEqual(m['reference_fiber_mass_kg'], 1200 * area * 0.002, places=20)
        self.assertEqual(
            m['total_reference_mass_kg'],
            m['reference_fiber_mass_kg'] + r.sphere.mass_kg,
        )
        self.assertEqual(m['sphere']['radius_m'], 0.00035)
        self.assertEqual(m['beams'][0]['contact_exclusion_radius_m'], 0.00035 + 5e-05)
        self.assertEqual(s['material'], g.edges[0].material.to_dict())

    def test_distinct_edge_materials_and_radii_preserve_sections_and_mass(self):
        graph = make_chain(2)
        graph.edges[1].radius = 80e-6
        graph.edges[1].material = Material(
            name='second synthetic test material',
            youngs_modulus=2.3e6,
            poissons_ratio=0.24,
            density=890.0,
        )
        request = make_request(graph)
        mapping = prepare_exudyn_contact(graph, request).mapping()
        expected_mass = 0.0
        properties = (
            (0, 50e-6, 1e6, 0.3, 1200.0),
            (1, 80e-6, 2.3e6, 0.24, 890.0),
        )
        for edge_id, radius, young, poisson, density in properties:
            with self.subTest(edge_id=edge_id):
                section = mapping['sections'][edge_id]
                area = math.pi * radius ** 2
                inertia = math.pi * radius ** 4 / 4
                shear = young / (2 * (1 + poisson))
                self.assertEqual(section['source_edge_id'], edge_id)
                self.assertEqual(section['radius_m'], radius)
                self.assertEqual(
                    section['material'], graph.edges[edge_id].material.to_dict()
                )
                np.testing.assert_allclose(
                    section['stiffness_diagonal_SI'],
                    [young * area, shear * area, shear * area,
                     shear * 2 * inertia, young * inertia, young * inertia],
                    rtol=1e-15,
                )
                self.assertAlmostEqual(
                    section['mass_per_length_kg_m'], density * area, places=20
                )
                expected_mass += density * area * 0.001
        self.assertNotEqual(
            mapping['sections'][0]['stiffness_diagonal_SI'],
            mapping['sections'][1]['stiffness_diagonal_SI'],
        )
        self.assertAlmostEqual(
            mapping['reference_fiber_mass_kg'], expected_mass, places=20
        )
        self.assertAlmostEqual(
            mapping['total_reference_mass_kg'],
            expected_mass + request.sphere.mass_kg,
            places=20,
        )

    def test_canonical_identity_ignores_dictionary_order(self):
        g = remap_ids(make_chain())
        p = prepare_exudyn_contact(g, make_request(g))
        g._nodes = dict(reversed(list(g.nodes.items())))
        g._edges = dict(reversed(list(g.edges.items())))
        self.assertEqual(p.identity, prepare_exudyn_contact(g, make_request(g)).identity)

    def test_signed_ids_and_separate_graph_json_boundary(self):
        g = remap_ids(make_chain())
        p = prepare_exudyn_contact(g, make_request(g))
        m = p.mapping()
        self.assertTrue(any((n < 0 for n in m['node_ids'])))
        self.assertTrue(any((e < 0 for e in m['edge_ids'])))
        self.assertEqual(m['node_ids'], sorted(g.nodes))
        self.assertEqual(m['edge_ids'], sorted(g.edges))
        self.assertFalse(p.manifest()['source_graph_json_roundtrip_supported'])
        with self.assertRaises(ValueError):
            StructureGraph.from_dict(p.source_record())
        self.assertEqual(len(m['chain_order']), 33)
        self.assertNotEqual(m['chain_order'], m['node_ids'])

    def test_nonnegative_graph_roundtrip_preserves_identity(self):
        g = make_chain(2)
        p = prepare_exudyn_contact(g, make_request(g))
        restored = StructureGraph.from_dict(p.source_record())
        self.assertEqual(
            p.identity,
            prepare_exudyn_contact(restored, make_request(restored)).identity,
        )
        self.assertTrue(p.manifest()['source_graph_json_roundtrip_supported'])

    def test_reversed_edge_orientation_preserves_source(self):
        g = make_chain(2)
        g.edges[0].node_i, g.edges[0].node_j = (1, 0)
        p = prepare_exudyn_contact(g, make_request(g))
        b = p.mapping()['beams'][0]
        self.assertEqual(b['source_orientation'], -1)
        self.assertEqual(b['source_coordinate_at_native_endpoints'], [1.0, 0.0])
        self.assertEqual(b['source_node_ids_native_order'], [0, 1])
        self.assertEqual(p.source_record()['edges'][0]['node_i'], 1)

    def test_boundary_flags_do_not_create_clamps(self):
        g = make_chain(2)
        g.nodes[1].boundary = (True,) * 6
        m = prepare_exudyn_contact(g, make_request(g)).mapping()
        self.assertEqual(m['constraints']['clamped_node_ids'], [0, 2])
        bad = replace(make_request(g), clamped_node_ids=(0, 1))
        self.assert_contract_error('invalid_boundary', lambda: prepare_exudyn_contact(g, bad))

    def test_invalid_ids_and_record_disagreement(self):
        g = make_chain(1)
        g.nodes[0].node_id = -99
        self.assert_contract_error(
            'invalid_id',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        g = make_chain(1)
        g.edges[0].edge_id = 123
        self.assert_contract_error(
            'invalid_id',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        g = make_chain(1)
        g._nodes[False] = g._nodes.pop(0)
        self.assert_contract_error(
            'invalid_id',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        self.assert_contract_error(
            'invalid_id',
            lambda: replace(make_request(make_chain(1)), clamped_node_ids=(True, 2)),
        )

    def test_unknown_endpoints_and_self_edge(self):
        for target in (999, 0):
            g = make_chain(1)
            g.edges[0].node_j = target
            self.assert_contract_error(
                'invalid_graph',
                lambda: prepare_exudyn_contact(g, make_request(g)),
            )

    def test_curves_and_refinement_explicitly_rejected(self):
        for segments in (4, 32, True, 0, -1):
            g = make_chain(1)
            g.edges[0].segments = segments
            self.assert_contract_error(
                'unsupported_refinement',
                lambda: prepare_exudyn_contact(g, make_request(g)),
            )
        for points in (
            np.empty((0, 3)),
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 1e-05, 0.0]]),
        ):
            g = make_chain(1)
            g.edges[0].internal_points = points
            self.assert_contract_error(
                'unsupported_geometry',
                lambda: prepare_exudyn_contact(g, make_request(g)),
            )

    def test_multiedges_cycles_and_disconnection_rejected(self):
        g = make_chain(2)
        g.add_edge(0, 1, allow_parallel=True, segments=1)
        self.assert_contract_error(
            'unsupported_geometry',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        g = make_chain(2)
        g.edges[1].node_i = 0
        g.edges[1].node_j = 1
        self.assert_contract_error(
            'unsupported_geometry',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        g = make_chain(3)
        g.edges[1].node_i = 0
        g.edges[1].node_j = 2
        self.assert_contract_error(
            'unsupported_geometry',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        g = make_chain(2)
        g.remove_edge(1)
        self.assert_contract_error(
            'unsupported_geometry',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )

    def test_off_axis_coincident_and_scale_rejected(self):
        for pos in (
            [0.001, 1e-05, 0], [0.001, 0, 1e-05], [-0.001, 0, 0],
            [2.0, 0, 0], [0.02, 0, 0],
        ):
            g = make_chain(1)
            g.nodes[1].position = np.array(pos)
            self.assert_contract_error(
                'unsupported_geometry',
                lambda: prepare_exudyn_contact(g, make_request(g)),
            )
        g = make_chain(1)
        g.nodes[1].position = np.array([-0.001 + 1e-08, 0, 0])
        self.assert_contract_error(
            'unsupported_geometry',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )

    def test_nonfinite_and_material_failures(self):
        for field, value in (
            ('youngs_modulus', 0.0), ('shear_modulus', 0.0),
            ('density', -1.0), ('youngs_modulus', float('nan')),
        ):
            g = make_chain(1)
            setattr(g.edges[0].material, field, value)
            with self.assertRaises(ExudynContractError):
                prepare_exudyn_contact(g, make_request(g))
        g = make_chain(1)
        g.edges[0].material.shear_modulus *= 2
        self.assert_contract_error(
            'unsupported_material',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        g = make_chain(1)
        g.edges[0].material.stiffness_tensor = np.eye(6)
        self.assert_contract_error(
            'unsupported_material',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        for radius in (0.0, -1.0, True, float('inf'), 1e+308, 1e-300):
            g = make_chain(1)
            g.edges[0].radius = radius
            with self.assertRaises(ExudynContractError):
                prepare_exudyn_contact(g, make_request(g))
        g = make_chain(1)
        g.nodes[0].position[0] = np.nan
        with self.assertRaises(ExudynContractError):
            prepare_exudyn_contact(g, make_request(g))

    def test_metadata_and_optional_material_data_preserved(self):
        g = make_chain(1)
        g.metadata['experiment'] = {'tags': ['synthetic', 1]}
        g.nodes[0].metadata['owner'] = 'a'
        g.edges[0].metadata['rest_length'] = 99.0
        g.edges[0].material.extra = {'source': 'explicit'}
        p = prepare_exudyn_contact(g, make_request(g))
        self.assertEqual(p.source_record(), g.to_dict())
        self.assertEqual(p.mapping()['beams'][0]['reference_length_m'], 0.002)
        g.metadata['bad'] = {1: 'ambiguous JSON key'}
        with self.assertRaises(ExudynContractError):
            prepare_exudyn_contact(g, make_request(g))

    def test_public_compiler_returns_detached_records(self):
        g = make_chain(1)
        m = compile_straight_chain(g, make_request(g))
        m['sections'][0]['material']['youngs_modulus'] = 1.0
        self.assertEqual(g.edges[0].material.youngs_modulus, 1000000.0)

    def test_malformed_source_allocator_tolerance_and_box(self):
        g = make_chain(1)
        g._next_node_id = 0
        self.assert_contract_error(
            'invalid_graph',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )
        g = make_chain(1)
        g._tolerance = 'bad'
        with self.assertRaises(ExudynContractError):
            prepare_exudyn_contact(g, make_request(g))
        g = make_chain(1)
        g._box_size = np.array([1.0])
        self.assert_contract_error(
            'invalid_graph',
            lambda: prepare_exudyn_contact(g, make_request(g)),
        )

    def test_durable_n32_fixture(self):
        path = Path(__file__).parent / 'fixtures' / 'exudyn_contact' / 'n32_graph.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        g = StructureGraph.from_dict(data)
        self.assertEqual(
            prepare_exudyn_contact(g, make_request(g)).identity,
            prepare_exudyn_contact(make_chain(), make_request(make_chain())).identity,
        )
        self.assertEqual(len(data['nodes']), 33)
        self.assertEqual(len(data['edges']), 32)

    def test_numpy_real_physics_scalars_and_string_types(self):
        g = make_chain(1)
        g.edges[0].radius = np.float64(5e-05)
        g._tolerance = np.float64(1e-06)
        self.assertEqual(
            prepare_exudyn_contact(g, make_request(g)).mapping()['sections'][0]['radius_m'],
            5e-05,
        )
        with self.assertRaises(ExudynContractError):
            replace(make_request(g), profile=np.array(['x', 'y']))
        with self.assertRaises(ExudynContractError):
            replace(make_request(g), length_unit=np.array(['m', 'm']))


if __name__ == '__main__':
    unittest.main()
