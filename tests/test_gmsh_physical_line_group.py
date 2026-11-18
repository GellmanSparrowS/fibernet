"""The named physical group of a legacy MSH export contains all line elements."""

from collections import defaultdict

import numpy as np
import pytest

from fibernet import api, gen
from fibernet.io import to_gmsh


def _section(text, name):
    body = text.split(f'${name}\n', 1)[1].split(f'$End{name}\n', 1)[0].splitlines()
    count, *rows = body
    assert int(count) == len(rows)
    return rows


def _check_mesh(path, source, expected_nodes, expected_lines):
    mesh = path.read_text()
    assert mesh.startswith('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
    names = _section(mesh, 'PhysicalNames')
    assert names == ['1 1 "fibers"']

    nodes = {}
    for row in _section(mesh, 'Nodes'):
        tag, *xyz = row.split()
        assert len(xyz) == 3
        nodes[int(tag)] = [float(value) for value in xyz]
    assert len(nodes) == expected_nodes

    by_fiber = defaultdict(list)
    elements = _section(mesh, 'Elements')
    assert len(elements) == expected_lines
    for expected_id, row in enumerate(elements, start=1):
        element_id, element_type, num_tags, physical_tag, curve_tag, start, end = map(int, row.split())
        assert (element_id, element_type, num_tags, physical_tag) == (expected_id, 1, 2, 1)
        assert start in nodes and end in nodes
        by_fiber[curve_tag].append((start, end))

    assert list(by_fiber) == list(range(1, source.num_fibers + 1))
    for fiber, elements in zip(source.fibers, by_fiber.values()):
        assert len(elements) == 5
        assert [end for _, end in elements[:-1]] == [start for start, _ in elements[1:]]
        np.testing.assert_allclose(nodes[elements[0][0]], fiber.start_point, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(nodes[elements[-1][1]], fiber.end_point, rtol=1e-10, atol=1e-9)
    return mesh


def test_default_export_names_every_zigzag_beam(tmp_path):
    pytest.importorskip('networkx')
    source = gen.ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network()
    assert (source.num_fibers, source.num_crosslinks, source.dimension) == (45, 90, 2)
    before = source.to_dict()

    inferred = tmp_path / 'zigzag.msh'
    assert api.export(source, inferred) == inferred
    default = _check_mesh(inferred, source, expected_nodes=216, expected_lines=225)

    explicit = tmp_path / 'zigzag-explicit.msh'
    assert to_gmsh(source, explicit, 5, 2) == explicit
    assert _check_mesh(explicit, source, expected_nodes=216, expected_lines=225) == default
    assert source.to_dict() == before


def test_explicit_3d_export_names_curves_without_changing_xyz(tmp_path):
    source = gen.entangled_3d(num_fibers=60, seed=23)
    assert (source.num_fibers, source.dimension) == (60, 3)
    assert source.crosslinks
    before = source.to_dict()

    spatial = tmp_path / 'spatial.msh'
    assert api.export(source, spatial, dimension=3) == spatial
    explicit = _check_mesh(spatial, source, expected_nodes=360, expected_lines=300)

    default = tmp_path / 'spatial-default.msh'
    assert api.export(source, default) == default
    assert default.read_text() == explicit
    assert source.to_dict() == before
