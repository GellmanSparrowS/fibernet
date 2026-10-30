"""Real plate-factory dispatch with signature-bound, geometry-free sentinels."""

import inspect
from types import SimpleNamespace

import pytest

from fibernet import api, gen
from fibernet.gen import _fixes, metamaterials, unified


PLATE_SIGNATURE = inspect.signature(metamaterials.plate_lattice_3d)
GYROID_SIGNATURE = inspect.signature(metamaterials.gyroid_lattice_3d)


@pytest.fixture(autouse=True)
def forbid_geometry_and_capture_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dispatch reached geometry, random draws, or conversion")

    seeds = []

    def capture_rng(seed=None):
        seeds.append(seed)
        return SimpleNamespace(uniform=forbidden)

    monkeypatch.setattr(
        _fixes, "np",
        SimpleNamespace(random=SimpleNamespace(RandomState=capture_rng),
                        array=forbidden),
    )
    monkeypatch.setattr(_fixes, "_edges_to_network_safe", forbidden)
    for name in ("_NodeGraph", "_graph_to_network", "plate_lattice_3d",
                 "gyroid_lattice_3d"):
        monkeypatch.setattr(metamaterials, name, forbidden)
    return seeds


@pytest.fixture(params=[False, True], ids=["direct", "api"])
def create_lattice(request):
    # Never replace the registered factory or the public wrapper under test.
    assert api._registry.generators["lattice_3d"] is unified.lattice_3d
    assert gen.lattice_3d is unified.lattice_3d
    if request.param:
        return lambda **kwargs: api.create("lattice_3d", **kwargs)
    return unified.lattice_3d


@pytest.fixture
def capture_plate(monkeypatch):
    result = object()
    calls = []

    def capture(*args, **kwargs):
        bound = PLATE_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        return result

    monkeypatch.setattr(metamaterials, "plate_lattice_3d", capture)
    return result, calls


@pytest.mark.parametrize("options,spacing", [({}, 10.0), ({"cell_size": 2.5}, 2.5)])
def test_plate_default_and_custom_cell_size(create_lattice, capture_plate, options, spacing):
    result, calls = capture_plate
    assert create_lattice(topology="plate", **options) is result
    assert calls == [{"spacing": spacing, "grid_size": (3, 3, 3),
                      "plate_thickness": 0.5, "material": None, "radius": 0.1}]


def test_plate_shared_and_backend_options_are_forwarded(
    create_lattice, capture_plate, forbid_geometry_and_capture_rng
):
    result, calls = capture_plate
    material = object()
    options = {"plate_thickness": 0.8}
    assert create_lattice(
        topology="PLATE", cell_size=4.0, grid_size=(1, 2, 4), radius=0.25,
        material=material, seed=17, **options,
    ) is result
    assert calls == [{"spacing": 4.0, "grid_size": (1, 2, 4),
                      "plate_thickness": 0.8, "material": material, "radius": 0.25}]
    assert options == {"plate_thickness": 0.8}
    assert forbid_geometry_and_capture_rng == [17]


@pytest.mark.parametrize(
    "options",
    [{"spacing": 2.5}, {"cell_size": 10.0, "spacing": 2.5},
     {"cell_size": 2.5, "spacing": 2.5}, {"cell_size": 4.0, "spacing": 2.5}],
    ids=["implicit-default", "explicit-default", "equal-values", "conflicting-values"],
)
def test_plate_spacing_is_not_a_second_factory_spelling(
    create_lattice, capture_plate, options
):
    _, calls = capture_plate
    with pytest.raises(TypeError) as caught:
        create_lattice(topology="plate", **options)
    assert str(caught.value) == "lattice_3d(topology='plate') uses cell_size, not spacing"
    assert calls == []


def test_plate_unknown_backend_option_still_rejects(create_lattice, capture_plate):
    _, calls = capture_plate
    with pytest.raises(TypeError, match="plate_thikness"):
        create_lattice(topology="plate", plate_thikness=0.8)
    assert calls == []


def test_gyroid_dispatch_keeps_its_existing_keyword_contract(create_lattice, monkeypatch):
    result = object()
    calls = []

    def capture(*args, **kwargs):
        GYROID_SIGNATURE.bind(*args, **kwargs)
        calls.append(kwargs)
        return result

    monkeypatch.setattr(metamaterials, "gyroid_lattice_3d", capture)
    assert create_lattice(topology="gyroid", cell_size=2.5,
                          resolution=6, threshold=0.2) is result
    assert calls == [{"cell_size": 2.5, "resolution": 6, "threshold": 0.2}]


def test_public_factory_signature_is_unchanged():
    expected = [("topology", "octet"), ("cell_size", 10.0),
                ("grid_size", (3, 3, 3)), ("perturbation", 0.0),
                ("radius", 0.1), ("material", None), ("seed", None)]
    for factory in (unified.lattice_3d, _fixes.lattice_3d_fixed):
        parameters = list(inspect.signature(factory).parameters.values())
        assert [(p.name, p.default) for p in parameters[:-1]] == expected
        assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                   for p in parameters[:-1])
        assert parameters[-1].name == "kwargs"
        assert parameters[-1].kind == inspect.Parameter.VAR_KEYWORD


def test_custom_generator_kwargs_and_existing_registry_are_preserved(monkeypatch):
    registry = api._registry.generators
    before = dict(registry)
    name = "_plate_dispatch_contract_sentinel"
    assert name not in registry
    marker, result = object(), object()
    calls = []
    with monkeypatch.context() as context:
        context.setattr(api._registry, "generators", dict(registry))

        @api.register_generator(name)
        def custom(**kwargs):
            calls.append(kwargs)
            return result

        assert api.create(name, cell_size=2.5, spacing=4.0, custom_option=marker) is result
        assert calls == [{"cell_size": 2.5, "spacing": 4.0, "custom_option": marker}]
        assert all(api._registry.generators[key] is value for key, value in before.items())
    assert api._registry.generators is registry
    assert registry == before
