"""Random generator keyword admission without generation or solver execution."""
import pytest

from fibernet import api
from fibernet.gen import disordered


ROUTES = [
    ('random_2d', 'random_straight_2d'),
    ('random_3d', 'random_straight_3d'),
    ('random_walk', 'random_walk_fibers'),
]


@pytest.mark.parametrize('route,owner', ROUTES)
@pytest.mark.parametrize('via_api', [False, True])
@pytest.mark.parametrize('options', [
    {'fiber_lenght': 3}, {'num_step': 2}, {'z_option': 1, 'a_option': 2},
])
def test_unknown_options_reject_before_validation_rng_or_allocation(
        route, owner, via_api, options, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('unknown options reached validation, RNG, or allocation')

    for name in ('nonnegative_count', 'positive_real', 'Material', 'FiberNetwork',
                 '_compute_percolation_box_2d', '_compute_percolation_box_3d'):
        monkeypatch.setattr(disordered, name, forbidden)
    monkeypatch.setattr(disordered.np.random, 'RandomState', forbidden)
    with pytest.raises(TypeError) as caught:
        if via_api:
            api.create(route, num_fibers=-1, **options)
        else:
            getattr(disordered, owner)(num_fibers=-1, **options)
    assert str(caught.value) == (
        owner + ' does not support keyword options: ' + ', '.join(sorted(options)))


@pytest.mark.parametrize('route,owner', ROUTES)
def test_declared_options_still_reach_owner_rng_without_generation(
        route, owner, monkeypatch):
    class ReachedRNG(Exception):
        pass

    seen = []

    def stop_at_rng(seed):
        seen.append(seed)
        raise ReachedRNG

    monkeypatch.setattr(disordered.np.random, 'RandomState', stop_at_rng)
    options = dict(num_fibers=0, box_size=2., radius=.1, material=object(),
                   seed=17, ensure_connected=False)
    if route == 'random_walk':
        options.update(num_steps=2, step_size=.5, threshold_factor=2.)
    else:
        options['fiber_length'] = 1.
        if route == 'random_3d':
            options['threshold_factor'] = 2.
    with pytest.raises(ReachedRNG):
        api.create(route, **options)
    assert seen == [17]


@pytest.mark.parametrize('route,owner', ROUTES)
def test_facade_forwards_named_options_unchanged(route, owner, monkeypatch):
    expected = dict(num_fibers=0, box_size=2., radius=.1, material=object(),
                    seed=17, ensure_connected=False)
    if route == 'random_walk':
        expected.update(num_steps=2, step_size=.5, threshold_factor=2.)
    else:
        expected['fiber_length'] = 1.
        if route == 'random_3d':
            expected['threshold_factor'] = 2.
    result = object()
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setitem(api._registry.generators, route, capture)
    assert api.create(route, **expected) is result
    assert calls == [expected]


def test_custom_registered_factory_retains_arbitrary_options(monkeypatch):
    marker = object()
    calls = []

    def custom(**kwargs):
        calls.append(kwargs)
        return marker

    monkeypatch.setitem(api._registry.generators, '_keyword_contract_custom', custom)
    assert api.create('_keyword_contract_custom', custom_metadata=marker) is marker
    assert calls == [{'custom_metadata': marker}]
