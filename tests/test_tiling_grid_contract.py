"""Grid admission precedes construction while zero and valid tilings remain usable."""
import copy
import numpy as np
import pytest
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.material import Material
from fibernet.core._grid_contract import normalize_grid
from fibernet.core import tiling
from fibernet.gen import pattern

TILERS = (tiling.tile_2d, tiling.tile_3d, tiling.tile_with_transforms)

@pytest.mark.parametrize('dimension', (2, 3))
@pytest.mark.parametrize('bad', (-1, True, False, np.bool_(True), 1.0, np.float64(2), None, '2'))
def test_invalid_scalar(dimension, bad):
    with pytest.raises(ValueError, match='grid'):
        normalize_grid(bad, dimension)

@pytest.mark.parametrize('dimension', (2, 3))
def test_sequences_and_numpy_counts(dimension):
    for scalar in (0, 1, np.int64(0), np.int32(2), np.uint64(2)):
        assert normalize_grid(scalar, dimension) == (int(scalar),) * dimension
    expected = (0,) + (2,) * (dimension - 1)
    for counts in (expected, list(expected), np.array(expected), iter(expected)):
        actual = normalize_grid(counts, dimension)
        assert actual == expected and all(type(n) is int for n in actual)
    for counts in ((), (1,) * (dimension + 1), (1,) * (dimension - 1),
                   (-1,) + (1,) * (dimension - 1), (True,) + (1,) * (dimension - 1),
                   (np.bool_(False),) + (1,) * (dimension - 1),
                   (1.0,) + (1,) * (dimension - 1)):
        with pytest.raises(ValueError, match='grid'):
            normalize_grid(counts, dimension)

@pytest.mark.parametrize('tiler', TILERS)
def test_core_rejects_before_accessing_cell(tiler):
    with pytest.raises(ValueError, match='grid'):
        tiler(object(), grid=-1)

@pytest.mark.parametrize('name,dimension', (('pattern_2d', 2), ('pattern_3d', 3)))
def test_pattern_rejects_before_factory_and_rng(monkeypatch, name, dimension):
    def forbidden(**kwargs):
        raise AssertionError('factory/RNG must not run for invalid grid')
    registry = pattern._UNIT_FACTORIES if dimension == 2 else pattern._UNIT_FACTORIES_3D
    monkeypatch.setitem(registry, 'grid_contract_sentinel', forbidden)
    monkeypatch.setattr(pattern.np.random, 'default_rng', forbidden)
    for grid in (-1, True, 1.0, (1,) * (dimension + 1)):
        with pytest.raises(ValueError, match='grid'):
            getattr(pattern, name)(unit='grid_contract_sentinel', grid=grid)

@pytest.mark.parametrize('tiler', TILERS)
def test_zero_and_valid_numpy_grid_preserve_source(tiler):
    dim = 3 if tiler is tiling.tile_3d else 2
    g = StructureGraph(dimension=dim, box_size=[1.] * dim)
    a = g.add_node([0., 0., 0.]); b = g.add_node([1., 1., 1. if dim == 3 else 0.])
    mat = Material(youngs_modulus=2e8)
    g.add_edge(a, b, radius=.12, material=mat, internal_points=np.array([[.5, .3, 0.]]))
    g.add_edge(b, a, radius=.17, material=mat, allow_parallel=True)
    g.metadata['sample'] = {'values': [1, 2]}
    before = copy.deepcopy(g.to_dict())
    expected = tiler(g, grid=(1,) * dim).to_dict()
    for grid in (1, np.int64(1), [np.int32(1)] * dim, np.ones(dim, dtype=np.int64)):
        assert tiler(g, grid=grid).to_dict() == expected
    for grid in (0, (0,) + (1,) * (dim - 1)):
        empty = tiler(g, grid=grid)
        assert empty.num_nodes == empty.num_edges == 0
    assert g.to_dict() == before and g.edges[0].material is mat

@pytest.mark.parametrize('name,dimension', (('pattern_2d', 2), ('pattern_3d', 3)))
def test_public_numpy_grid_rotation_and_zero(name, dimension):
    kwargs = dict(unit='square' if dimension == 2 else 'cubic', seed=17)
    if dimension == 2:
        kwargs['rotation'] = 15.
    generate = getattr(pattern, name)
    expected = generate(grid=(1,) * dimension, **kwargs).to_dict()
    assert generate(grid=np.int64(1), **kwargs).to_dict() == expected
    for grid in (0, (0,) + (1,) * (dimension - 1)):
        graph = generate(grid=grid, **kwargs)
        assert graph.num_nodes == graph.num_edges == 0
