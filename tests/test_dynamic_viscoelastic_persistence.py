"""Saved endpoint consistency only: no constitutive or native advancement.

The numerical helper cases are algebraic binary64 fixtures, not simulated
histories or a claim about the separate stored-energy validation domain.
"""
import copy
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

SAVED = Path(__file__).with_name('fixtures') / 'dynamic_viscoelastic_generalized_summary.json'
SAVED_SHA256 = '359027655bd4c14212a8b83caaafcb2c1a7757dcf54667af2bf08de4f2785448'


def _subnormal(units=1):
    # Construct stored binary64 bits, not an arithmetic result: math.ulp,
    # ldexp, or multiplication can flush these fixtures to zero on FTZ hosts.
    return struct.unpack('>d', struct.pack('>Q', units))[0]


@pytest.mark.parametrize('units', [1, 16, 128])
def test_subnormal_fixtures_preserve_binary64_bits(units):
    assert struct.pack('>d', _subnormal(units)) == struct.pack('>Q', units)


@pytest.fixture
def api(monkeypatch):
    class NoProvider(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.', 1)[0] in {'taichi', 'lammps', 'exudyn'}:
                raise ModuleNotFoundError('native provider unavailable: ' + fullname, name=fullname)

    monkeypatch.setattr(sys, 'meta_path', [NoProvider(), *sys.meta_path])
    from fibernet.core.viscoelastic import LinearViscoelasticMaterial
    from fibernet.sim import accelerated
    from fibernet.sim import _taichi_viscoelastic as material

    def forbidden(*args, **kwargs):
        raise AssertionError('persistence must not advance material/native state')

    monkeypatch.setattr(LinearViscoelasticMaterial, 'trial', forbidden)
    for name in ('__init__', 'prepare', 'commit'):
        monkeypatch.setattr(material.DynamicAxialMaterial, name, forbidden)
    for name in ('_ensure_taichi', '_taichi_runtime_state'):
        monkeypatch.setattr(accelerated, name, forbidden)
    monkeypatch.setattr(accelerated.TaichiEngine, 'dynamics', forbidden)
    return accelerated.SimResult, material


def _saved():
    raw = SAVED.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SAVED_SHA256
    return raw, json.loads(raw)


def test_saved_endpoint_summary_and_detailed_roundtrips(api, tmp_path):
    SimResult, _ = api
    raw, payload = _saved()
    original = copy.deepcopy(payload)
    result = SimResult.load(str(SAVED))
    expected = payload['native_axial_viscoelastic']
    assert result.metadata['native_axial_viscoelastic'] == expected
    before = result.to_dict(detailed=True)
    detached = result.to_dict(detailed=True)
    detached['native_axial_viscoelastic']['final']['increments']['physical_dissipation_J'][0] += 1
    assert result.to_dict(detailed=True) == before
    for detailed in (False, True):
        path = tmp_path / f'roundtrip-{detailed}.json'
        result.save(str(path), detailed=detailed)
        reloaded = SimResult.load(str(path))
        assert reloaded.metadata['native_axial_viscoelastic'] == expected
        assert reloaded.energy == result.energy
        np.testing.assert_array_equal(reloaded.edge_forces, result.edge_forces)
        assert result.to_dict(detailed=True) == before
    assert SAVED.read_bytes() == raw
    assert payload == original


@pytest.mark.parametrize('case,digest,message', [
    (6, '8e7963c6b79ea1044320edfa994454d8175a6bb621d04500a7901b164bb9364d',
     'physical dissipation inconsistent with committed endpoint'),
    (7, 'a188f504a74b68b8c923229bbfb9aaadc4db019e4c3bf40c0f0ffa3907b2575f',
     'physical dissipation inconsistent with committed endpoint'),
    (8, 'bb4f553f2e42af3cbafe49a5c085b0dd9c8d64799f7141b272711bc962e85a78',
     'unsupported material schema or units'),
])
def test_saved_endpoint_corruptions_refused(api, tmp_path, case, digest, message):
    SimResult, _ = api
    original_raw, payload = _saved()
    record = payload['native_axial_viscoelastic']
    if case == 6:
        state = next(s for s in record['samples'] if s['step'] == 32)
        state['increments']['physical_dissipation_J'][0] += 1.0
    elif case == 7:
        assert record['material']['branches'][0][1] == .25
        record['material']['branches'][0][1] = .5
    else:
        record['material']['units']['time'] = 'ms'
    record['final'] = copy.deepcopy(record['samples'][-1])
    raw = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
    # Match the independently retained original false-acceptance witnesses.
    assert hashlib.sha256(raw).hexdigest() == digest
    before = copy.deepcopy(payload)
    path = tmp_path / f'case-{case}.json'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        SimResult.load(str(path))
    assert path.read_bytes() == raw
    assert payload == before
    assert SAVED.read_bytes() == original_raw


def _check(api, q, *, modulus=1., tau=1., dt=1., volume=1., increment=0.):
    _, material = api
    q = np.asarray(q, dtype=np.float64).reshape((-1, 1))
    moduli = np.full(q.shape, modulus, dtype=np.float64)
    times = np.full(q.shape, tau, dtype=np.float64)
    volumes = np.array([volume], dtype=np.float64)
    increments = np.array([increment], dtype=np.float64)
    originals = [a.copy() for a in (q, moduli, times, volumes, increments)]
    try:
        return material._validate_endpoint_physical_dissipation(
            q, moduli, times, dt, volumes, increments)
    finally:
        for value, original in zip((q, moduli, times, volumes, increments), originals):
            np.testing.assert_array_equal(value, original)


@pytest.fixture
def tiny_domain(api):
    _, material = api
    smallest = np.array([1], dtype=np.uint64).view(np.float64)
    try:
        material._require_material_tiny_arithmetic(smallest)
    except ValueError as error:
        assert 'arithmetic unavailable: gradual underflow required' in str(error)
        return False
    return True


def _tiny_check(api, tiny_domain, *args, **kwargs):
    if tiny_domain:
        _check(api, *args, **kwargs)
    else:
        with pytest.raises(ValueError, match='arithmetic unavailable: gradual underflow required'):
            _check(api, *args, **kwargs)


@pytest.mark.parametrize('q,needs_tiny', [([], False), ([0.], False), ([math.ldexp(1., -600)], True)])
def test_zero_rate_requires_exact_zero(api, tiny_domain, q, needs_tiny):
    # Last case deliberately preserves producer q**2 underflow, rather than
    # reassociating it into E*(epsilon-z)**2 or recovering its real-valued loss.
    if needs_tiny:
        _tiny_check(api, tiny_domain, q, increment=0.)
    else:
        _check(api, q, increment=0.)
    message = 'inconsistent with committed endpoint' if tiny_domain else 'arithmetic unavailable'
    with pytest.raises(ValueError, match=message):
        _check(api, q, increment=_subnormal())


def test_normal_ulp_allowance_has_a_finite_boundary(api):
    expected = .5
    _check(api, [1.], dt=.25, volume=2., increment=expected)
    _check(api, [1.], dt=.25, volume=2., increment=math.nextafter(expected, math.inf))
    with pytest.raises(ValueError, match='inconsistent with committed endpoint'):
        _check(api, [1.], dt=.25, volume=2., increment=expected + 256*math.ulp(expected))


def test_positive_sum_reduction_allowance(api):
    q = [1.] + [math.ldexp(1., -27)]*255
    # Correctly rounded exact sum of these power-of-two squared stresses;
    # np.sum may choose a different positive-reduction order.
    exact_sum = 1. + math.ldexp(255., -54)
    _check(api, q, increment=exact_sum)
    with pytest.raises(ValueError, match='inconsistent with committed endpoint'):
        _check(api, q, increment=exact_sum + math.ldexp(1., -30))


def test_subnormal_final_loss_and_underflow_zero(api, tiny_domain):
    q = [math.ldexp(1., -500)]
    _tiny_check(api, tiny_domain, q, dt=math.ldexp(1., -70), increment=_subnormal(16))
    _tiny_check(api, tiny_domain, q, dt=math.ldexp(1., -100), increment=0.)
    _tiny_check(api, tiny_domain, [1.], volume=_subnormal(), increment=_subnormal())
    message = 'inconsistent with committed endpoint' if tiny_domain else 'arithmetic unavailable'
    with pytest.raises(ValueError, match=message):
        _check(api, q, dt=math.ldexp(1., -100), increment=_subnormal(128))


def test_subnormal_density_allowance_propagates_through_large_volume(api, tiny_domain):
    q = [math.ldexp(1., -500)]
    arguments = dict(dt=math.ldexp(1., -70), volume=math.ldexp(1., 1000))
    expected = math.ldexp(1., -70)
    _tiny_check(api, tiny_domain, q, **arguments, increment=expected)
    # One ULP at the density stage is much larger than one final-J ULP.
    _tiny_check(api, tiny_domain, q, **arguments, increment=expected + math.ldexp(1., -74))
    message = 'inconsistent with committed endpoint' if tiny_domain else 'arithmetic unavailable'
    with pytest.raises(ValueError, match=message):
        _check(api, q, **arguments, increment=2*expected)


def test_allowance_products_do_not_overflow_prematurely(api):
    # dt*V is outside binary64, but producer ordering and the exact budget
    # remain finite. This does not exercise a constitutive history update.
    _check(api, [math.ldexp(1., -500)], dt=math.ldexp(1., 800),
           volume=math.ldexp(1., 800), increment=math.ldexp(1., 600))


@pytest.mark.parametrize('arguments', [
    dict(q=[math.inf]), dict(q=[math.nan]), dict(q=[1.], modulus=0.),
    dict(q=[1.], tau=0.), dict(q=[1.], dt=0.), dict(q=[1.], dt=math.inf),
    dict(q=[1.], volume=0.), dict(q=[1.], volume=-1.),
    dict(q=[1.], volume=math.inf), dict(q=[1.], increment=-1.),
    dict(q=[1.], increment=math.inf),
])
def test_endpoint_domain_refused(api, arguments):
    with pytest.raises(ValueError, match='invalid finite domain'):
        _check(api, **arguments)


@pytest.mark.parametrize('arguments', [
    dict(q=[math.ldexp(1., 600)]),
    dict(q=[1.], modulus=math.ldexp(1., -1000), tau=math.ldexp(1., -1000)),
    dict(q=[2.], dt=1e308),
    dict(q=[1.], dt=1e308, volume=2.),
])
def test_nonfinite_producer_intermediates_refused(api, arguments):
    with pytest.raises(ValueError, match='evaluation is nonfinite'):
        _check(api, **arguments)


def test_endpoint_comparison_is_per_edge(api):
    _, material = api
    arguments = (np.array([[1., 2.], [2., 1.]]), np.array([[1.], [2.]]),
                 np.array([[1.], [4.]]), .25, np.array([2., 4.]))
    material._validate_endpoint_physical_dissipation(*arguments, np.array([.75, 4.125]))
    with pytest.raises(ValueError, match='inconsistent with committed endpoint'):
        material._validate_endpoint_physical_dissipation(*arguments, np.array([.75, 5.125]))


def test_endpoint_requires_gradual_underflow(api, monkeypatch):
    from fibernet.sim import _roundoff

    def unavailable():
        raise _roundoff._SubnormalRoundoffError('simulated unavailable arithmetic')

    # Test the error boundary without changing any process floating-point mode.
    monkeypatch.setattr(_roundoff, '_require_gradual_underflow', unavailable)
    # Ordinary values do not require an environment-wide gradual mode.
    _check(api, [1.], increment=1.)
    # All physical stages are normal, even though ulp(S) is subnormal.
    # Its exact bit-decoded budget must not need the tiny-arithmetic guard.
    _check(api, [math.ldexp(1., -500)], dt=math.ldexp(1., 800),
           volume=math.ldexp(1., 800), increment=math.ldexp(1., 600))
    with pytest.raises(ValueError, match='arithmetic unavailable: gradual underflow required'):
        _check(api, [math.ldexp(1., -600)], increment=0.)


@pytest.mark.parametrize('area,rest,needs_tiny', [
    (_subnormal(), math.ldexp(1., -39), True),
    (math.ldexp(1., 1023), 2., False),
])
def test_record_reference_volume_product_must_be_finite_positive(api, tiny_domain, area, rest, needs_tiny):
    _, material = api
    raw, payload = _saved()
    record = payload['native_axial_viscoelastic']
    record['reference_areas_m2'][0] = area
    record['rest_lengths_m'][0] = rest
    before = copy.deepcopy(record)
    message = ('arithmetic unavailable' if needs_tiny and not tiny_domain else
               'reference volumes must remain finite and positive')
    with pytest.raises(ValueError, match=message):
        material.validate_record(record)
    assert record == before
    assert SAVED.read_bytes() == raw
