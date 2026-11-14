"""Real native generation, linear solve, persisted analysis and VTK ordering."""

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import fibernet as fn


def test_generated_radius_workflow(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "examples/heterogeneous_radius_workflow.py"
    )
    spec = importlib.util.spec_from_file_location("radius_workflow", source)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    output = tmp_path / "workflow"
    report = example.run(output)
    graph = fn.StructureGraph.load_json(output / "graph.json")
    result = fn.SimResult.load(output / "result.json")
    saved = json.loads((output / "analysis.json").read_text())
    assert report == saved
    assert (graph.num_nodes, graph.num_edges) == (35, 118)
    assert report["node_ids"] == result.metadata["node_ids"]
    assert report["edge_ids"] == result.metadata["edge_ids"]
    assert report["max_displacement_m"] > 0
    assert report["max_axial_force_N"] > 0
    assert 0 < report["tensile_recruited_count"] <= graph.num_edges

    # This tiled octet has 54 axis-aligned members and 64 body diagonals.
    assert sum(e.radius == 0.0005 for e in graph.edges.values()) == 54
    assert sum(e.radius == 0.00035 for e in graph.edges.values()) == 64
    axis_length = 18 * (0.01 + 0.015 + 0.02)
    diagonal_length = 64 * math.sqrt(0.01**2 + 0.015**2 + 0.02**2) / 2
    volume = math.pi * (0.0005**2 * axis_length + 0.00035**2 * diagonal_length)
    area = 2 * math.pi * (0.0005 * axis_length + 0.00035 * diagonal_length)
    assert report["summed_cylinder_volume_m3"] == pytest.approx(volume)
    assert report["summed_lateral_area_m2"] == pytest.approx(area)
    assert report["node_bbox_volume_m3"] == pytest.approx(0.02 * 0.03 * 0.04)
    with (output / "cylinder_features.csv").open(newline="") as stream:
        csv_row = next(csv.DictReader(stream))
    for name, value in report["cylinder_features"].items():
        assert float(csv_row[name]) == value

    # Parse the actual VTK float fields; check edge-row identity and radii.
    tokens = (output / "member_fields.vtk").read_text().split()
    cell_count = int(tokens[tokens.index("CELL_DATA") + 1])
    parsed = {}
    for name in ("source_edge_id", "radius_m", "signed_axial_force_N"):
        index = tokens.index(name)
        assert tokens[index - 1 : index + 5] == [
            "SCALARS",
            name,
            "float",
            "1",
            "LOOKUP_TABLE",
            "default",
        ]
        parsed[name] = np.array(
            tokens[index + 5 : index + 5 + cell_count], dtype=np.float32
        )
    expected_ids, expected_radii, expected_forces = [], [], []
    edge_rows = {eid: i for i, eid in enumerate(result.metadata["edge_ids"])}
    for fiber, (eid, edge) in zip(graph.to_fiber_network().fibers, graph.edges.items()):
        count = len(fiber.centerline) - 1
        expected_ids.extend([eid] * count)
        expected_radii.extend([edge.radius] * count)
        expected_forces.extend([result.edge_forces[edge_rows[eid]]] * count)
    for name, expected in zip(parsed, (expected_ids, expected_radii, expected_forces)):
        serialized = np.array([f"{v:.6e}" for v in expected], dtype=np.float32)
        np.testing.assert_array_equal(parsed[name], serialized)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        example.run(output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
