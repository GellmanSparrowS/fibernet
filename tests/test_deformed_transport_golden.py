"""Retained arithmetic/identity replay: never imports or calls a physics owner."""
import hashlib
import json
from pathlib import Path
import unittest

from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.deformed_transport import prepare_deformed_transport

FIXTURE = Path(__file__).parent / 'fixtures/deformed_transport_preparation_v1/planar_golden.json'
FIXTURE_SHA256 = 'f8a776a572d0975b097b555d31c932a5267467d5934e12fb7c32afe40b883dc3'


class SavedPreparationGolden(unittest.TestCase):
    def test_actual_public_preparation_matches_retained_case(self):
        raw = FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), FIXTURE_SHA256)
        retained = json.loads(raw.decode('utf-8'))
        graph = StructureGraph.from_dict(retained['source_graph_insertion_record'])
        original = graph.to_dict()
        translations = {int(k): v for k, v in retained['translations_m_by_id'].items()}
        reference_g = {int(k): v for k, v in retained['reference_conductance_S'].items()}
        derived, conductance, binding = prepare_deformed_transport(
            graph, translations_m_by_id=translations,
            reference_conductance_S=reference_g, length_unit='m')
        expected = retained['expected']
        self.assertEqual((len(derived.nodes), len(derived.edges)), (25, 56))
        for field in ('source_full_graph_sha256', 'derived_full_graph_sha256',
                      'source_graph_order_envelope_sha256', 'derived_graph_order_envelope_sha256',
                      'source_node_insertion_order', 'source_edge_insertion_order',
                      'reference_box_size_m', 'derived_node_AABB_m'):
            self.assertEqual(binding[field], expected[field], field)
        self.assertEqual(conductance, {int(k): v for k, v in expected['conductance_S'].items()})
        for field in ('reference_lengths_m', 'derived_lengths_m'):
            self.assertEqual(dict(zip(binding['edge_ids'], binding[field])),
                             {int(k): v for k, v in expected[field].items()}, field)
        # Reconstruct the existing electrical INPUT digest only. The existing
        # solver body is neither imported nor executed, and no response is made.
        ids, eids = sorted(derived.nodes), list(derived.edges)
        boundary = sorted((int(k), v) for k, v in retained['prescribed_voltage_V_for_identity_only'].items())
        identity = dict(node_ids=ids, edge_ids=eids,
                        endpoints=[(derived.edges[e].node_i, derived.edges[e].node_j) for e in eids],
                        conductance_S=[conductance[e] for e in eids],
                        prescribed_voltage_V=boundary,
                        coordinates=[derived.nodes[n].position.tolist() for n in ids], length_unit='m')
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.assertEqual(digest, expected['electrical_input_sha256'])
        self.assertEqual(graph.to_dict(), original)
        self.assertEqual(binding['input_kind'], 'caller_prescribed_translations')
        self.assertFalse(binding['mechanical_result_admitted'])
        self.assertFalse(binding['solver_executed'])
        self.assertFalse(binding['feedback'])
        self.assertNotIn('mechanical_input_sha256', binding)
        self.assertIsNone(binding['coordinate_frame'])
        self.assertEqual(binding['translation_frame'], 'source_graph_xy_axes')


if __name__ == '__main__':
    unittest.main()
