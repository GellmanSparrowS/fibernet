"""Sealed historical identity plus exact same-host preparation-law parity.

No solver is imported. The independent accuracy envelope is derived before any
host norm is evaluated. Historical and current-host electrical digests are
separate: the latter is not claimed to be the former physical input.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import unittest

from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.deformed_transport import prepare_deformed_transport

DIRECTORY = Path(__file__).parent / 'fixtures/deformed_transport_preparation_v1'
FIXTURE = DIRECTORY / 'planar_golden.json'
FIXTURE_SHA256 = 'f8a776a572d0975b097b555d31c932a5267467d5934e12fb7c32afe40b883dc3'
REFERENCE = DIRECTORY / 'arithmetic_reference.py.txt'
REFERENCE_SHA256 = '3f7bea3272972147c1d4cee986bb2ca954a82f396edd184c79c39f547a927827'
NUMERIC_FIELDS = ('reference_lengths_m', 'derived_lengths_m',
                  'frozen_sigma_A_S_m', 'conductance_S')


def sealed_inputs():
    raw, reference_raw = FIXTURE.read_bytes(), REFERENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    assert hashlib.sha256(reference_raw).hexdigest() == REFERENCE_SHA256
    namespace = {'__name__': 'sealed_preparation_arithmetic_reference'}
    exec(compile(reference_raw.decode('utf-8'), str(REFERENCE), 'exec'), namespace)
    retained = json.loads(raw.decode('utf-8'))
    envelopes = namespace['accuracy_envelopes'](retained)
    historical_digest = namespace['check_historical_integrity'](retained, envelopes)
    reference = namespace['same_host_reference'](retained)
    namespace['check_accuracy'](reference, envelopes)
    return retained, namespace, envelopes, reference, historical_digest


def actual_preparation(retained):
    graph = StructureGraph.from_dict(retained['source_graph_insertion_record'])
    original = graph.to_dict()
    derived, conductance, binding = prepare_deformed_transport(
        graph, translations_m_by_id={int(k): v for k, v in retained['translations_m_by_id'].items()},
        reference_conductance_S={int(k): v for k, v in retained['reference_conductance_S'].items()},
        length_unit='m')
    ids, eids = sorted(derived.nodes), list(derived.edges)
    boundary = sorted((int(k), v) for k, v in retained['prescribed_voltage_V_for_identity_only'].items())
    identity = dict(node_ids=ids, edge_ids=eids,
                    endpoints=[(derived.edges[e].node_i, derived.edges[e].node_j) for e in eids],
                    conductance_S=[conductance[e] for e in eids], prescribed_voltage_V=boundary,
                    coordinates=[derived.nodes[n].position.tolist() for n in ids], length_unit='m')
    observed = dict(conductance=conductance, binding=binding, identity=identity,
                    graph_node_order=list(derived.nodes), graph_edge_order=eids,
                    current_host_electrical_input_sha256=identity_digest(identity))
    return graph, original, derived, observed


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def exact_host_comparator(observed, reference, namespace):
    """Actual comparator used by both positive and candidate-only controls."""
    reasons = []
    binding, conductance = observed['binding'], observed['conductance']
    for actual, expected, name in (
        (list(conductance), reference['edge_order'], 'G insertion order/IDs'),
        (observed['graph_node_order'], reference['node_order'], 'graph node order'),
        (observed['graph_edge_order'], reference['edge_order'], 'graph edge order'),
        (binding['node_ids'], reference['node_ids'], 'report sorted node IDs'),
        (binding['edge_ids'], reference['edge_ids'], 'report sorted edge IDs'),
        (binding['source_node_insertion_order'], reference['node_order'], 'source node order'),
        (binding['source_edge_insertion_order'], reference['edge_order'], 'source edge order'),
        (binding['endpoints_by_id'], reference['endpoints'], 'oriented endpoints'),
        (observed['identity']['node_ids'], reference['node_ids'], 'input node IDs'),
        (observed['identity']['edge_ids'], reference['edge_order'], 'input edge order'),
        (observed['identity']['coordinates'], [reference['positions'][n] for n in reference['node_ids']], 'coordinates'),
    ):
        if actual != expected:
            reasons.append(name)
    rows = []
    arrays = {name: dict(zip(binding['edge_ids'], binding[name])) for name in NUMERIC_FIELDS}
    for name in NUMERIC_FIELDS:
        if len(binding[name]) != 56 or set(arrays[name]) != set(reference['edge_ids']):
            reasons.append(name + ' report shape/IDs')
    if binding['reference_conductance_S'] != [reference['reference_conductance_S'][e] for e in reference['edge_ids']]:
        reasons.append('reference G report')
    for eid in reference['edge_ids']:
        row = {'kind': 'edge', 'edge_id': eid}
        for name in ('returned_G',) + NUMERIC_FIELDS:
            actual = conductance.get(eid) if name == 'returned_G' else arrays[name].get(eid)
            expected = reference['conductance_S' if name == 'returned_G' else name][eid]
            valid = type(actual) is float and math.isfinite(actual) and actual > 0
            row[name] = dict(actual_hex=actual.hex() if valid else repr(actual)[:80],
                             expected_hex=expected.hex(),
                             ulps=namespace['ulps'](actual, expected) if valid else None)
            if not valid or actual.hex() != expected.hex():
                reasons.append('%s edge %d' % (name, eid))
        rows.append(row)
    expected_digest = reference['current_host_electrical_input_sha256']
    actual_digest = identity_digest(observed['identity'])
    if actual_digest != observed['current_host_electrical_input_sha256'] or actual_digest != expected_digest:
        reasons.append('current-host electrical input digest')
    if reasons:
        # At most the sealed 56 edge rows, never unittest's truncated dict diff.
        diagnostic = dict(kind='context', reasons=reasons,
                          implementation=platform.python_implementation(),
                          python_version=platform.python_version(), platform=platform.platform(),
                          actual_current_host_digest=actual_digest,
                          expected_current_host_digest=expected_digest,
                          historical_digest_not_substituted=True)
        lines = [json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False)
                 for item in [diagnostic] + rows]
        assert len(lines) == 57 and all(len(line.encode('utf-8')) < 16384 for line in lines)
        raise AssertionError('\n'.join(lines))


class SavedPreparationGolden(unittest.TestCase):
    def test_actual_public_preparation_matches_retained_case(self):
        retained, namespace, envelopes, reference, historical_digest = sealed_inputs()
        graph, original, derived, observed = actual_preparation(retained)
        binding, expected = observed['binding'], retained['expected']
        self.assertEqual((len(derived.nodes), len(derived.edges)), (25, 56))
        for field in ('source_full_graph_sha256', 'derived_full_graph_sha256',
                      'source_graph_order_envelope_sha256', 'derived_graph_order_envelope_sha256',
                      'source_node_insertion_order', 'source_edge_insertion_order',
                      'reference_box_size_m', 'derived_node_AABB_m'):
            self.assertEqual(binding[field], expected[field], field)
        exact_host_comparator(observed, reference, namespace)
        candidate_fields = {name: dict(zip(binding['edge_ids'], binding[name])) for name in NUMERIC_FIELDS}
        namespace['check_accuracy'](candidate_fields, envelopes)
        self.assertEqual(historical_digest, expected['electrical_input_sha256'])
        # These are two separately verified identities, not a cross-host assertion.
        self.assertEqual(observed['current_host_electrical_input_sha256'],
                         reference['current_host_electrical_input_sha256'])
        self.assertEqual(graph.to_dict(), original)
        self.assertEqual(binding['input_kind'], 'caller_prescribed_translations')
        self.assertFalse(binding['mechanical_result_admitted'])
        self.assertFalse(binding['solver_executed'])
        self.assertFalse(binding['feedback'])
        self.assertNotIn('mechanical_input_sha256', binding)
        self.assertIsNone(binding['coordinate_frame'])
        self.assertEqual(binding['translation_frame'], 'source_graph_xy_axes')

    def test_exact_comparator_rejects_candidate_only_mutations(self):
        retained, namespace, envelopes, reference, _ = sealed_inputs()
        _, _, _, observed = actual_preparation(retained)
        exact_host_comparator(observed, reference, namespace)
        eid = reference['edge_ids'][0]
        row = observed['binding']['edge_ids'].index(eid)
        input_row = observed['identity']['edge_ids'].index(eid)
        mutated = copy.deepcopy(observed)
        altered = math.nextafter(mutated['conductance'][eid], math.inf)
        self.assertTrue(namespace['within'](altered, envelopes[eid]['conductance_S']))
        # Coordinate G, report G, actual input vector and its digest. Internal
        # consistency and the accuracy envelope pass; independent parity must not.
        mutated['conductance'][eid] = altered
        mutated['binding']['conductance_S'][row] = altered
        mutated['identity']['conductance_S'][input_row] = altered
        mutated['current_host_electrical_input_sha256'] = identity_digest(mutated['identity'])
        candidate_fields = {name: dict(zip(mutated['binding']['edge_ids'], mutated['binding'][name]))
                            for name in NUMERIC_FIELDS}
        namespace['check_accuracy'](candidate_fields, envelopes)
        with self.assertRaises(AssertionError) as caught:
            exact_host_comparator(mutated, reference, namespace)
        diagnostics = [json.loads(line) for line in str(caught.exception).splitlines()]
        self.assertEqual(len(diagnostics), 57)
        self.assertEqual([entry['edge_id'] for entry in diagnostics[1:]], reference['edge_ids'])
        self.assertTrue(all(entry['kind'] == 'edge' for entry in diagnostics[1:]))
        for field in ('implementation', 'python_version', 'platform',
                      'actual_current_host_digest', 'expected_current_host_digest'):
            self.assertTrue(diagnostics[0][field])
        self.assertTrue(diagnostics[0]['historical_digest_not_substituted'])
        self.assertTrue(all(len(line.encode('utf-8')) < 16384 for line in str(caught.exception).splitlines()))
        for name in NUMERIC_FIELDS:
            with self.subTest(field=name):
                mutated = copy.deepcopy(observed)
                mutated['binding'][name][row] = math.nextafter(mutated['binding'][name][row], math.inf)
                with self.assertRaises(AssertionError):
                    exact_host_comparator(mutated, reference, namespace)
        def reject(label, change):
            with self.subTest(control=label):
                mutated = copy.deepcopy(observed)
                change(mutated)
                with self.assertRaises(AssertionError):
                    exact_host_comparator(mutated, reference, namespace)
        reject('returned G only', lambda x: x['conductance'].__setitem__(eid, altered))
        reject('missing G ID', lambda x: x['conductance'].pop(eid))
        reject('extra G ID', lambda x: x['conductance'].__setitem__(-999, 1.0))
        reject('G insertion order', lambda x: x.__setitem__('conductance', dict(reversed(list(x['conductance'].items())))))
        reject('report edge order', lambda x: x['binding']['edge_ids'].reverse())
        reject('report node order', lambda x: x['binding']['node_ids'].reverse())
        reject('oriented endpoint', lambda x: x['binding']['endpoints_by_id'][row].reverse())
        reject('digest only', lambda x: x.__setitem__('current_host_electrical_input_sha256', '0' * 64))
        reject('coordinate', lambda x: x['identity']['coordinates'][0].__setitem__(0, 123.0))
        # The historical integrity and formula-bound gates also have real negatives.
        changed_history = copy.deepcopy(retained)
        changed_history['expected']['conductance_S'][str(eid)] = math.nextafter(
            changed_history['expected']['conductance_S'][str(eid)], math.inf)
        with self.assertRaises(AssertionError):
            namespace['check_historical_integrity'](changed_history, envelopes)
        outside = copy.deepcopy(reference)
        outside['derived_lengths_m'][eid] = math.nextafter(float(envelopes[eid]['derived_lengths_m'][1]), math.inf)
        self.assertFalse(namespace['within'](outside['derived_lengths_m'][eid], envelopes[eid]['derived_lengths_m']))
        with self.assertRaises(AssertionError) as caught:
            namespace['check_accuracy'](outside, envelopes)
        accuracy_diagnostic = json.loads(str(caught.exception))
        self.assertEqual(accuracy_diagnostic['edge_id'], eid)
        self.assertEqual(accuracy_diagnostic['actual_hex'], outside['derived_lengths_m'][eid].hex())
        self.assertIn('formula_low', accuracy_diagnostic)
        self.assertIn('formula_high', accuracy_diagnostic)


if __name__ == '__main__':
    unittest.main()
