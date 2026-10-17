"""Ledger JSON bit preservation and atomic-save tests; no physics solve."""
import copy
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import random
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fibernet.sim import _energy_accounting as accounting
from fibernet.sim.accelerated import SimResult


def payload(value):
    return struct.unpack('>Q', struct.pack('>d', value))[0]


def floating(bits):
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def exact_magnitude(bits):
    """Independent test oracle: integer decoding, no production helper calls."""
    exponent, fraction = bits >> 52, bits & ((1 << 52)-1)
    if exponent:
        fraction |= 1 << 52
        power = exponent-1075
    else:
        power = -1074
    return (Fraction(fraction << power) if power >= 0
            else Fraction(fraction, 1 << -power))


def patterns():
    magnitudes = [0, 1, 2, 3, 4, 0x123456789abcd, (1 << 51),
                  (1 << 52)-2, (1 << 52)-1, 1 << 52, (1 << 52)+1,
                  0x3fefffffffffffff, 0x3ff0000000000000, 0x3ff0000000000001,
                  0x7feffffffffffffe, 0x7fefffffffffffff]
    randomizer = random.Random(60217)
    magnitudes += [randomizer.randrange(0x7ff0000000000000) for _ in range(128)]
    return [n | sign for n in magnitudes for sign in (0, 1 << 63)]


def dump(value, **kwargs):
    stream = io.StringIO()
    accounting.dump_binary64_json(value, stream, **kwargs)
    return stream.getvalue()


def zero_result():
    endpoint = dict(kinetic_energy=0., potential_energy=0., total_energy=0.)
    final = dict(step=0, time=0., **endpoint, near_cutoff=False,
                 positions=[[0., 0., 0.]], velocities=[[0., 0., 0.]])
    state = dict(schema='native_axial_state/1', representation='sampled',
                 availability='available', units=None, dt=.125, final=final,
                 work_ledger=dict(status='unavailable', value=None, reason='not_collected'),
                 cutoff_check='sampled_only', node_ids=[17], masses=[1.],
                 steps=[0], times=[0.], positions=[final['positions']],
                 velocities=[final['velocities']], kinetic_energy=[0.],
                 potential_energy=[0.], total_energy=[0.], near_cutoff=[False])
    ledger = accounting.ExactLedger(endpoint, [[0., 0., 0.]])
    row = ledger.sample(0, 0., endpoint)
    result = SimResult(mode='dynamics', n_nodes=1, dynamics_state=state,
                       metadata={'node_ids': [17]})
    result.algorithmic_energy_ledger = ledger.result([row], state, 0., 0., 'unit')
    return result


class TestBinary64LedgerJSON(unittest.TestCase):
    def test_numeric_tokens_roundtrip_bits_and_independent_cells(self):
        for bits in patterns():
            token = dump(floating(bits))
            decimal = json.loads(token, parse_float=Decimal, parse_int=Decimal)
            self.assertIsInstance(decimal, Decimal)
            self.assertEqual(decimal.is_signed(), bool(bits >> 63))
            self.assertEqual(payload(json.loads(token, parse_float=accounting.parse_binary64)), bits)
            magnitude = bits & ((1 << 63)-1)
            if not magnitude:
                self.assertTrue(decimal.is_zero())
                self.assertEqual(token, '-0.0' if bits >> 63 else '0.0')
                continue
            represented = abs(Fraction(decimal))
            center = exact_magnitude(magnitude)
            lower = (exact_magnitude(magnitude-1)+center)/2
            # The overflow midpoint above max-finite is finite and exact too.
            upper_neighbor = (Fraction(1 << 1024) if magnitude == 0x7fefffffffffffff
                              else exact_magnitude(magnitude+1))
            upper = (center+upper_neighbor)/2
            self.assertTrue(lower <= represented <= upper)
            if represented in (lower, upper):
                self.assertEqual(magnitude & 1, 0, 'ties must select an even payload')

    def test_nested_types_escaping_and_default_conversion(self):
        value = {'text': '\n"\\\u4e2d\ud800', 'flags': [True, False, None],
                 'integers': [0, -2, 10**90], 'tuple': (1., -0.),
                 'array': np.frombuffer(struct.pack('>QQ', 1, (1 << 63)|1), dtype='>f8')}
        text = dump(value, default=lambda x: x.tolist())
        decoded = json.loads(text, parse_float=accounting.parse_binary64)
        self.assertEqual(decoded['text'], value['text'])
        self.assertEqual(decoded['flags'], value['flags'])
        self.assertEqual(decoded['integers'], value['integers'])
        self.assertEqual([payload(x) for x in decoded['array']], [1, (1 << 63)|1])
        basic = {None: 'null', False: 'false', 7: 'integer', 1.25: 'float'}
        self.assertEqual(json.loads(dump(basic)), json.loads(json.dumps(basic)))

    def test_decimal_context_is_not_used_or_changed(self):
        value = floating(1)
        expected = dump(value)
        with localcontext() as ctx:
            ctx.prec = 2; ctx.Emin = -2; ctx.Emax = 2; ctx.capitals = 0
            before = (ctx.prec, ctx.rounding, ctx.Emin, ctx.Emax, ctx.capitals,
                      dict(ctx.flags), dict(ctx.traps))
            self.assertEqual(dump(value), expected)
            self.assertEqual(before, (ctx.prec, ctx.rounding, ctx.Emin, ctx.Emax, ctx.capitals,
                                      dict(ctx.flags), dict(ctx.traps)))

    def test_nonfinite_cycles_and_unsupported_values_reject(self):
        for bits in (0x7ff0000000000000, 0xfff0000000000000,
                     0x7ff8000000000000, 0x7ff0000000000001):
            with self.assertRaises(ValueError): dump({'v': floating(bits)})
            with self.assertRaises(ValueError): dump({floating(bits): 'value'})
        cyclic = []; cyclic.append(cyclic)
        with self.assertRaises(ValueError): dump(cyclic)
        with self.assertRaises(TypeError): dump(object())
        with self.assertRaises(TypeError): dump({object(): 1})
        value = object()
        with self.assertRaises(ValueError): dump(value, default=lambda obj: obj)

    def test_zero_closure_and_exact_tamper_refusal(self):
        channels = dict.fromkeys(accounting.CHANNELS, 0.)
        closure = accounting.closure(channels, 0., 0.)
        self.assertEqual(payload(closure['bound']), 4)
        restored = json.loads(dump(closure), parse_float=accounting.parse_binary64)
        self.assertEqual(payload(restored['bound']), 4)
        accounting.validate_closure(restored, channels, 0., 0.)
        for bits in (0, 3, 5):
            tampered = copy.deepcopy(restored); tampered['bound'] = floating(bits)
            with self.assertRaises(ValueError): accounting.validate_closure(tampered, channels, 0., 0.)

    def test_production_save_load_summary_and_detailed(self):
        result = zero_result()
        result.metadata['bit_samples'] = [floating(bits) for bits in patterns()]
        with tempfile.TemporaryDirectory() as area:
            for detailed in (False, True):
                path = Path(area)/('detail.json' if detailed else 'summary.json')
                result.save(path, detailed=detailed)
                if os.name == 'posix':
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                loaded = SimResult.load(path)
                self.assertEqual(payload(loaded.algorithmic_energy_ledger['final']['closure']['bound']), 4)
                self.assertEqual([payload(x) for x in loaded.metadata['bit_samples']], patterns())
                data = json.loads(path.read_text(encoding='utf-8'), parse_float=Decimal)
                self.assertIsInstance(data['algorithmic_energy_ledger']['final']['closure']['bound'], Decimal)
                self.assertEqual(list(Path(area).glob('.*.result-*')), [])

    def test_atomic_failures_retain_old_bytes_and_permissions(self):
        with tempfile.TemporaryDirectory() as area:
            path = Path(area)/'result.json'; path.write_text('original', encoding='utf-8'); path.chmod(0o640)
            original_mode = stat.S_IMODE(path.stat().st_mode)
            for invalid in (floating(0x7ff0000000000000), object()):
                result = zero_result(); result.metadata['invalid'] = invalid
                with self.assertRaises((ValueError, TypeError)): result.save(path)
                self.assertEqual(path.read_text(encoding='utf-8'), 'original')
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), original_mode)
                self.assertEqual(list(Path(area).glob('.*.result-*')), [])
            result = zero_result()
            with patch.object(Path, 'replace', side_effect=OSError('replace sentinel')):
                with self.assertRaisesRegex(OSError, 'replace sentinel'): result.save(path)
            self.assertEqual(path.read_text(encoding='utf-8'), 'original')
            self.assertEqual(list(Path(area).glob('.*.result-*')), [])
            result.save(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), original_mode)

    def test_nonledger_keeps_existing_json_writer(self):
        with tempfile.TemporaryDirectory() as area:
            result = SimResult()
            with patch.object(accounting, 'dump_binary64_json', side_effect=AssertionError('ledger writer called')):
                result.save(Path(area)/'plain.json')


if __name__ == '__main__':
    unittest.main()
