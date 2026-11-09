"""Reject invalid native 3-D cell extents before factories or meshing run."""

import copy

import numpy as np
import pytest

from fibernet import StructureGraph, pattern_3d
from fibernet.gen import pattern

ERROR = "box must contain three finite positive real dimensions"
TPMS_UNITS = ("gyroid", "schwarz_p", "schwarz_d", "iwp", "neovius", "lidinoid")
ORDINARY_UNITS = (
    "cubic",
    "octet",
    "diamond_3d",
    "bcc",
    "fcc",
    "hcp",
    "chiral_3d",
    "reentrant_3d",
)


@pytest.mark.parametrize(
    "box",
    [
        (0.0, 10.0, 10.0),
        (-10.0, 10.0, 10.0),
        (True, 10.0, 10.0),
        np.array([10.0 + 2.0j, 10.0, 10.0]),
    ],
)
def test_cubic_rejects_collapsed_reflected_or_nonreal_cells(box):
    with pytest.raises(ValueError, match=ERROR):
        pattern_3d(unit="cubic", box=box, grid=(2, 1, 1))


@pytest.mark.parametrize("unit", ("cubic", "gyroid", "custom-box-contract"))
@pytest.mark.parametrize(
    "box",
    [
        None,
        10.0,
        "123",
        b"123",
        {1: 1.0, 2: 2.0, 3: 3.0},
        {1.0, 2.0, 3.0},
        (),
        (10.0,),
        (10.0, 10.0),
        (10.0, 10.0, 10.0, 10.0),
        np.array(10.0),
        np.array([[10.0], [10.0], [10.0]]),
        np.zeros((3, 0)),
        (0.0, 10.0, 10.0),
        (10.0, -1.0, 10.0),
        (10.0, 10.0, np.nan),
        (np.inf, 10.0, 10.0),
        (10.0, -np.inf, 10.0),
        (True, 10.0, 10.0),
        (10.0, np.bool_(True), 10.0),
        ("10", 10.0, 10.0),
        (10.0, None, 10.0),
        (10.0 + 0.0j, 10.0, 10.0),
        np.array([10.0 + 2.0j, 10.0, 10.0]),
        np.array([10.0 + 2.0j, 10.0, 10.0], dtype=object),
        np.ma.array([10.0, 10.0, 10.0], mask=[False, True, False]),
        (10**400, 10.0, 10.0),
    ],
)
def test_invalid_box_precedes_factory_tiling_repair_and_rng(monkeypatch, unit, box):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid box reached generation work")

    monkeypatch.setitem(pattern._UNIT_FACTORIES_3D, unit, forbidden)
    monkeypatch.setattr(pattern, "tile_3d", forbidden)
    monkeypatch.setattr(pattern, "_post_tile_connectivity_repair", forbidden)
    monkeypatch.setattr(pattern.np.random, "default_rng", forbidden)
    with pytest.raises(ValueError, match=ERROR):
        pattern_3d(unit=unit, box=box, grid=(1, 1, 1))


@pytest.mark.parametrize("grid", ((0, 1, 1), (1, 0, 1), (1, 1, 0)))
def test_invalid_box_does_not_bypass_admission_with_zero_grid(grid):
    with pytest.raises(ValueError, match=ERROR):
        pattern_3d(unit="cubic", box=(0.0, 10.0, 10.0), grid=grid)


@pytest.mark.parametrize("box", ((0.0, 8.0, 10.0), None, (6.0, 8.0)))
def test_overridden_unit_kwargs_cannot_rescue_invalid_top_level_box(box):
    with pytest.raises(ValueError, match=ERROR):
        pattern_3d(unit="cubic", box=box, grid=1, unit_kwargs={"box": (6.0, 8.0, 10.0)})


@pytest.mark.parametrize("unused", (None, (0.0, -1.0, np.nan), "unused"))
def test_valid_top_level_box_still_overrides_unit_kwargs(unused):
    options = {"box": unused}
    graph = pattern_3d(unit="cubic", box=(6.0, 8.0, 10.0), grid=1, unit_kwargs=options)
    expected = pattern_3d(unit="cubic", box=(6.0, 8.0, 10.0), grid=1)
    assert graph.to_dict() == expected.to_dict()
    assert options["box"] is unused


@pytest.mark.parametrize(
    "box",
    [
        (6.0, 8.0, 10.0),
        [6, 8, 10],
        np.array([6.0, 8.0, 10.0], dtype=np.float32),
        np.array([6, 8, 10], dtype=np.int64),
        np.array([6.0, 8.0, 10.0], dtype=object),
        np.ma.array([6.0, 8.0, 10.0], mask=False),
    ],
)
def test_valid_box_objects_reach_custom_factory_unchanged_and_detach_from_output(
    monkeypatch, box
):
    cubic = pattern._UNIT_FACTORIES_3D["cubic"]
    calls = []

    def factory(**kwargs):
        calls.append(kwargs["box"])
        return cubic(**kwargs)

    monkeypatch.setitem(pattern._UNIT_FACTORIES_3D, "custom-box-contract", factory)
    before = copy.deepcopy(box)
    graph = pattern_3d(unit="custom-box-contract", box=box, grid=(2, 1, 1))
    assert calls == [box]
    assert calls[0] is box
    np.testing.assert_array_equal(box, before)
    assert (graph.num_nodes, graph.num_edges) == (12, 20)
    np.testing.assert_array_equal(graph.box_size, [12.0, 8.0, 10.0])
    graph.box_size = [99.0, 99.0, 99.0]
    np.testing.assert_array_equal(box, before)


@pytest.mark.parametrize("unit", ORDINARY_UNITS)
def test_valid_rectangular_builtin_geometry_and_native_roundtrip(unit):
    options = dict(unit=unit, grid=(1, 1, 1), n_internal=2, seed=37)
    graph = pattern_3d(box=np.array([6.0, 8.0, 10.0]), **options)
    expected = pattern_3d(box=(6.0, 8.0, 10.0), **options)
    assert graph.to_dict() == expected.to_dict()
    assert graph.num_nodes > 0 and graph.num_edges > 0
    np.testing.assert_array_equal(graph.box_size, [6.0, 8.0, 10.0])
    assert StructureGraph.from_dict(graph.to_dict()).to_dict() == graph.to_dict()


@pytest.mark.parametrize("unit", TPMS_UNITS)
@pytest.mark.parametrize("as_array", (False, True))
def test_real_tpms_rectangular_box_preserves_provider_geometry_and_native_roundtrip(
    unit, as_array
):
    pytest.importorskip("skimage.measure")
    options = dict(
        unit=unit, grid=(1, 1, 1), unit_kwargs={"resolution": 8, "voxel_factor": 2.0}
    )
    box = np.array([6.0, 8.0, 10.0]) if as_array else (6.0, 8.0, 10.0)
    # Preserve each scalar container's existing provider arithmetic. NumPy and
    # Python scalar promotion need not produce bit-identical float32 vertices.
    graph = pattern_3d(box=box, **options)
    expected = pattern_3d(box=copy.deepcopy(box), **options)
    assert graph.to_dict() == expected.to_dict()
    assert graph.num_nodes > 0 and graph.num_edges > 0
    assert np.isfinite(graph.node_positions()).all()
    np.testing.assert_array_equal(graph.box_size, [6.0, 8.0, 10.0])
    assert StructureGraph.from_dict(graph.to_dict()).to_dict() == graph.to_dict()


@pytest.mark.parametrize("unit", ("cubic", "hcp", "gyroid"))
def test_valid_box_keeps_empty_tiling_contract(unit):
    if unit == "gyroid":
        pytest.importorskip("skimage.measure")
    graph = pattern_3d(unit=unit, box=(6.0, 8.0, 10.0), grid=(0, 1, 1))
    assert (graph.num_nodes, graph.num_edges) == (0, 0)
    np.testing.assert_array_equal(graph.box_size, [0.0, 8.0, 10.0])
