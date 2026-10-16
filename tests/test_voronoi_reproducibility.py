"""Public Voronoi defaults are reproducible without changing explicit seeds."""
import copy

import pytest

from fibernet import Material, StructureGraph
from fibernet.gen.pattern import _unit_voronoi, pattern_2d


def _pattern(**kwargs):
    options = dict(unit="voronoi", box=(4., 3.), grid=(2, 1),
                   n_internal=1, n_pts_per_side=1,
                   unit_kwargs={"n_seeds": 6})
    options.update(kwargs)
    return pattern_2d(**options)


@pytest.mark.parametrize("seed_kwargs", [{}, {"seed": None}])
def test_public_default_is_repeatable_and_matches_seed_42(seed_kwargs):
    first = _pattern(**seed_kwargs).to_dict()
    assert first == _pattern(**seed_kwargs).to_dict()
    assert first == _pattern(seed=42).to_dict()


@pytest.mark.parametrize("seed", [0, 1, 42, 987])
def test_explicit_integer_seed_is_repeatable(seed):
    assert _pattern(seed=seed).to_dict() == _pattern(seed=seed).to_dict()


def test_different_seeds_change_geometry():
    assert _pattern(seed=0).fingerprint() != _pattern(seed=1).fingerprint()


@pytest.mark.parametrize("override", [None, 0, 7])
def test_unit_kwargs_seed_wins_without_mutating_options(override):
    options = {"n_seeds": 6, "seed": override}
    before = copy.deepcopy(options)
    result = _pattern(seed=99, unit_kwargs=options)
    expected = _pattern(seed=42 if override is None else override)
    assert result.to_dict() == expected.to_dict()
    assert options == before


def test_2d_roundtrip_preserves_geometry_material_and_source(tmp_path):
    material = Material(name="voronoi-test", youngs_modulus=1234.,
                        poissons_ratio=.25, density=900.,
                        extra={"source": {"labels": ["original"]}})
    material_before = copy.deepcopy(material.to_dict())
    graph = _pattern(material=material, radius=.07)
    before = copy.deepcopy(graph.to_dict())
    assert graph.dimension == 2
    assert len(graph.nodes) > 0 and len(graph.edges) > 0
    for node in graph.nodes.values():
        assert node.position[2] == 0.
    for edge in graph.edges.values():
        assert edge.radius == .07
        assert edge.material.to_dict() == material_before
    path = tmp_path / "voronoi.json"
    graph.save_json(path)
    restored = StructureGraph.load_json(path)
    assert restored.dimension == 2
    assert restored.to_dict() == before
    assert graph.to_dict() == before
    assert material.to_dict() == material_before
    next(iter(restored.nodes.values())).position[0] += 1.
    next(iter(restored.edges.values())).material.extra["source"]["labels"].append("changed")
    assert graph.to_dict() == before
    assert material.to_dict() == material_before


def test_private_factory_omitted_default_remains_42():
    options = dict(box=(4., 3.), n_seeds=6, n_pts_per_side=1,
                   material=Material())
    assert _unit_voronoi(**options).to_dict() == _unit_voronoi(seed=42, **options).to_dict()
