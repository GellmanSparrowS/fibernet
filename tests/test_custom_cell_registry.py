"""Custom cells share APP JSON and compile through the public library API."""
from pathlib import Path
import sys

import numpy as np
import pytest

from fibernet.gen import (CustomCell, CustomCellRegistry,
                          PlanarManufacturingConfig, compile_planar)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FiberScope"))
from fslab.structure import _cell_factory, valid_cell_spec


SPEC = {"nodes": [[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 0],
                  [0, 4], [1, 4], [2, 4], [3, 4]],
        "zh": "自定义", "en": "Custom",
        "settings": {"expansion_rule": "translate"}}


def test_app_custom_cell_json_roundtrip_and_route(tmp_path):
    assert valid_cell_spec(SPEC)
    cell = CustomCell.from_mapping(SPEC)
    graph = cell.to_graph()
    app_graph = _cell_factory(SPEC)(box=(10, 10), n_pts_per_side=0)
    np.testing.assert_array_equal(graph.node_positions(), app_graph.node_positions())
    np.testing.assert_array_equal(graph.edge_array(), app_graph.edge_array())
    registry = CustomCellRegistry(tmp_path / "custom_units.json")
    key = registry.save("My-Cell", cell, reserved={"square"})
    assert key == "my_cell"
    loaded = CustomCellRegistry(registry.path).load()
    assert loaded[key].to_mapping() == SPEC
    network = compile_planar(PlanarManufacturingConfig(
        unit=key, grid_x=2, grid_y=2, n_pts_per_side=1,
        base_graph=loaded[key].to_graph()))
    assert len(network.route_edges) == len(network.edges)
    assert network.route_nodes[0] == network.route_nodes[-1]
    assert registry.delete(key)
    assert CustomCellRegistry(registry.path).load() == {}


def test_custom_cell_rejects_invalid_or_oversized_input(tmp_path):
    registry = CustomCellRegistry(tmp_path / "custom_units.json", max_bytes=1024)
    with pytest.raises(ValueError, match="duplicate"):
        CustomCell.from_mapping({**SPEC, "edges": SPEC["edges"] + [[1, 0]]})
    with pytest.raises(ValueError, match="integers"):
        CustomCell.from_mapping({**SPEC, "edges": [[0.5, 1]]})
    with pytest.raises(ValueError, match="built-in"):
        registry.save("square", SPEC, reserved={"square"})
    registry.path.write_text("x" * 1025, encoding="utf-8")
    with pytest.raises(MemoryError, match="file budget"):
        registry.load()


def test_registry_save_preserves_existing_app_entries(tmp_path):
    path = tmp_path / "custom_units.json"
    path.write_text('{"first": ' + __import__("json").dumps(SPEC) + '}', encoding="utf-8")
    second = CustomCellRegistry(path)
    second.save("second", SPEC)
    assert sorted(CustomCellRegistry(path).load()) == ["first", "second"]
    assert CustomCellRegistry(path).delete("first")
    assert sorted(CustomCellRegistry(path).load()) == ["second"]
    CustomCellRegistry(path).replace_all({"third": SPEC})
    assert sorted(CustomCellRegistry(path).load()) == ["third"]
