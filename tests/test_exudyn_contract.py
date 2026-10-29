"""Immutable, provider-independent Exudyn preparation contract regressions."""

import math
import unittest
from dataclasses import FrozenInstanceError, replace

from fibernet.sim._exudyn_contract import (
    ChordContactSpec,
    ExudynContractError,
    SphereSpec,
    StepPhase,
    prepare_exudyn_contact,
    source_capabilities,
    validate_prepared,
)

from _exudyn_test_support import ContractAssertions, make_chain, make_request


class TestExudynContract(ContractAssertions, unittest.TestCase):

    def test_all_prepared_payloads_are_immutable_bytes(self):
        graph = make_chain(1)
        prepared = prepare_exudyn_contact(graph, make_request(graph))
        for field in ('source_record_bytes', 'request_bytes', 'mapping_bytes'):
            with self.subTest(field=field):
                payload = getattr(prepared, field)
                self.assertIs(type(payload), bytes)
                self.assert_contract_error(
                    'prepared_integrity',
                    lambda: replace(prepared, **{field: bytearray(payload)}),
                )

    def test_request_revalidated_after_frozen_object_bypass(self):
        cases = (
            ('request', 'length_unit', 'mm', 'invalid_units'),
            ('request', 'profile', 'unsupported/1', 'unsupported_profile'),
            ('sphere', 'radius_m', -1.0, 'invalid_request'),
            ('contact', 'range_m', 0.0, 'invalid_request'),
            ('phase', 'steps', 0, 'invalid_schedule'),
            ('phase', 'end_s', -1.0, 'invalid_schedule'),
        )
        for record, field, value, code in cases:
            with self.subTest(record=record, field=field):
                graph = make_chain(1)
                request = make_request(graph)
                target = {
                    'request': request,
                    'sphere': request.sphere,
                    'contact': request.contact,
                    'phase': request.phases[0],
                }[record]
                object.__setattr__(target, field, value)
                self.assert_contract_error(
                    code, lambda: prepare_exudyn_contact(graph, request)
                )

    def test_prepared_revalidated_after_frozen_object_bypass(self):
        cases = (
            ('identity', '0' * 64),
            ('schema', 'unsupported/1'),
            ('compiler', 'unsupported/1'),
            ('mapping_bytes', bytearray(b'{}')),
        )
        for field, value in cases:
            with self.subTest(field=field):
                graph = make_chain(1)
                prepared = prepare_exudyn_contact(graph, make_request(graph))
                object.__setattr__(prepared, field, value)
                self.assert_contract_error(
                    'prepared_integrity', lambda: validate_prepared(prepared)
                )
                self.assert_contract_error(
                    'prepared_integrity', prepared.manifest
                )

    def test_rehashed_frozen_object_bypass_still_checks_semantics(self):
        import fibernet.sim._exudyn_contract as contract

        graph = make_chain(1)
        prepared = prepare_exudyn_contact(graph, make_request(graph))
        mapping = prepared.mapping()
        mapping['counts']['ODE2'] += 1
        raw = contract._canonical(mapping)
        object.__setattr__(prepared, 'mapping_bytes', raw)
        object.__setattr__(prepared, 'mapping_sha256', contract._sha(raw))
        object.__setattr__(
            prepared,
            'identity',
            contract._identity(
                prepared.source_record_bytes, prepared.request_bytes, raw
            ),
        )
        self.assert_contract_error(
            'prepared_integrity', lambda: validate_prepared(prepared)
        )

    def test_all_typed_records_reject_ordinary_mutation(self):
        graph = make_chain(1)
        request = make_request(graph)
        prepared = prepare_exudyn_contact(graph, request)
        cases = (
            (request, 'length_unit', 'mm'),
            (request.sphere, 'radius_m', 1.0),
            (request.contact, 'range_m', 1.0),
            (request.phases[0], 'steps', 1),
            (prepared, 'identity', 'changed'),
        )
        for record, field, value in cases:
            with self.subTest(record=type(record).__name__, field=field):
                with self.assertRaises(FrozenInstanceError):
                    setattr(record, field, value)

    def test_immutability_and_detached_accessors(self):
        g = make_chain(1)
        r = make_request(g)
        p = prepare_exudyn_contact(g, r)
        identity = p.identity
        g.nodes[0].position[0] -= 0.1
        g.edges[0].material.youngs_modulus = 2000000.0
        g.metadata['late'] = 1
        data = p.mapping()
        data['sections'][0]['material']['youngs_modulus'] = 9.0
        source = p.source_record()
        source['nodes'][0]['position'][0] = 9.0
        req = p.request()
        req['sphere']['radius_m'] = 9.0
        self.assertEqual(p.identity, identity)
        self.assertEqual(p.mapping()['sections'][0]['material']['youngs_modulus'], 1000000.0)
        self.assertTrue(validate_prepared(p))
        with self.assertRaises(FrozenInstanceError):
            p.identity = 'bad'
        with self.assertRaises(FrozenInstanceError):
            r.length_unit = 'mm'
        self.assertIs(type(p.mapping_bytes), bytes)

    def test_input_sequences_detached(self):
        center = [0.0, 0.00062, 0.0]
        velocity = [0.0, -0.1, 0.0]
        s = SphereSpec(0.00035, 1e-07, center, velocity)
        center[0] = 9.0
        velocity[0] = 9.0
        self.assertEqual(s.center_m, (0.0, 0.00062, 0.0))
        self.assertEqual(s.velocity_m_s, (0.0, -0.1, 0.0))

    def test_identity_binds_request_and_material(self):
        g = make_chain(1)
        r = make_request(g)
        p = prepare_exudyn_contact(g, r)
        r2 = replace(r, contact=replace(r.contact, w_N=3e-06))
        self.assertNotEqual(p.identity, prepare_exudyn_contact(g, r2).identity)
        r3 = replace(r, phases=(StepPhase(0.0, 0.000203, 120),))
        self.assertNotEqual(p.identity, prepare_exudyn_contact(g, r3).identity)
        g.edges[0].material.name = 'different provenance'
        self.assertNotEqual(p.identity, prepare_exudyn_contact(g, r).identity)

    def test_prepared_checksum_tamper_rejected(self):
        g = make_chain(1)
        p = prepare_exudyn_contact(g, make_request(g))
        self.assert_contract_error('prepared_integrity', lambda: replace(p, identity='0' * 64))
        self.assert_contract_error(
            'prepared_integrity',
            lambda: replace(p, mapping_bytes=p.mapping_bytes + b' '),
        )
        self.assert_contract_error(
            'prepared_integrity',
            lambda: replace(p, mapping_bytes=bytearray(p.mapping_bytes)),
        )

    def test_consistent_rehashed_but_wrong_mapping_rejected(self):
        import fibernet.sim._exudyn_contract as c
        g = make_chain(1)
        p = prepare_exudyn_contact(g, make_request(g))
        mapping = p.mapping()
        mapping['counts']['ODE2'] = 16
        raw = c._canonical(mapping)
        self.assert_contract_error(
            'prepared_integrity',
            lambda: replace(
                p,
                mapping_bytes=raw,
                mapping_sha256=c._sha(raw),
                identity=c._identity(p.source_record_bytes, p.request_bytes, raw),
            ),
        )

    def test_duplicate_or_unknown_clamps(self):
        g = make_chain(1)
        r = make_request(g)
        self.assert_contract_error('invalid_boundary', lambda: replace(r, clamped_node_ids=(0, 0)))
        self.assert_contract_error(
            'invalid_boundary',
            lambda: prepare_exudyn_contact(g, replace(r, clamped_node_ids=(0, 100))),
        )

    def test_units_loads_profiles_rejected(self):
        r = make_request(make_chain(1))
        for unit in ('SI', 'mm', 'um', 'unknown', None):
            self.assert_contract_error('invalid_units', lambda: replace(r, length_unit=unit))
        self.assert_contract_error(
            'unsupported_profile',
            lambda: replace(r, profile='expanded-welded-web/1'),
        )
        self.assert_contract_error(
            'unsupported_load',
            lambda: replace(r, gravity_m_s2=(0.0, -9.81, 0.0)),
        )
        self.assert_contract_error(
            'unsupported_load',
            lambda: replace(r, external_forces_N=((0, (0.0, 0.0, 0.0)),)),
        )
        self.assert_contract_error(
            'unsupported_profile',
            lambda: replace(r, initial_fiber_state='resume'),
        )

    def test_schedule_order_domain_and_counts(self):
        r = make_request(make_chain(1))
        for phases in (
            (),
            (StepPhase(1.0, 2.0, 1),),
            (StepPhase(0.0, 1.0, 1), StepPhase(1.0000000001, 2.0, 1)),
        ):
            self.assert_contract_error('invalid_schedule', lambda: replace(r, phases=phases))
        for args in [(0.0, 0.0, 1), (0.0, 1.0, 0), (0.0, 1.0, -1), (0.0, 1.0, 1000001)]:
            self.assert_contract_error('invalid_schedule', lambda: StepPhase(*args))
        with self.assertRaises(ExudynContractError):
            StepPhase(0.0, 1.0, True)
        with self.assertRaises(ExudynContractError):
            StepPhase(0.0, math.inf, 1)
        self.assertEqual(r.phases[0].dt_s, 0.000195 / 39)

    def test_scalar_bool_overflow_and_wrong_request_types(self):
        for args in (
            (True, 1e-07, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.0001, -1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ):
            with self.assertRaises(ExudynContractError):
                SphereSpec(*args)
        with self.assertRaises(ExudynContractError):
            ChordContactSpec(100.0, 1.0, 1e-300)
        with self.assertRaises(TypeError):
            prepare_exudyn_contact(make_chain(1), {})
        with self.assertRaises(TypeError):
            prepare_exudyn_contact({}, make_request(make_chain(1)))
        with self.assertRaises(ExudynContractError):
            replace(make_request(make_chain(1)), sphere={})

    def test_capabilities_truthful_and_no_execution_api(self):
        c = source_capabilities()
        self.assertTrue(c['preparation_available'])
        self.assertFalse(c['execution_available'])
        self.assertFalse(c['restart_supported'])
        self.assertEqual(c['runtime_readiness'], 'not_probed')
        self.assertEqual(c['profiles'], ['straight-shared-chain/1'])
        p = prepare_exudyn_contact(make_chain(1), make_request(make_chain(1)))
        self.assertFalse(hasattr(p, 'start'))
        self.assertFalse(hasattr(p, 'resume'))
        c['profiles'].append('bad')
        self.assertEqual(source_capabilities()['profiles'], ['straight-shared-chain/1'])

    def test_no_unrequested_api_expansion(self):
        g = make_chain(1)
        r = make_request(g)
        with self.assertRaises(TypeError):
            prepare_exudyn_contact(g, r, resume=True)
        with self.assertRaises(TypeError):
            prepare_exudyn_contact(g, r, backend='exudyn')
        r = replace(r, phases=(StepPhase(0.0, 1.0, 1),))
        self.assertEqual(
            prepare_exudyn_contact(g, r).mapping()['forecast']['raw_state_records'],
            2,
        )


if __name__ == '__main__':
    unittest.main()
