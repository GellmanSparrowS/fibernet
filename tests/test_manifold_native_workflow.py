"""Small real-provider manufacturing journeys, not general mesh qualification."""

import csv
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest


@pytest.fixture
def provider():
    return pytest.importorskip("manifold3d")


def _assert_owned_closed_mesh(vertices, faces):
    assert vertices.dtype == np.float64 and vertices.flags.owndata
    assert vertices.flags.writeable and vertices.flags.c_contiguous
    assert faces.dtype == np.int32 and faces.flags.owndata
    assert faces.flags.writeable and faces.flags.c_contiguous
    assert vertices.shape[1] == faces.shape[1] == 3
    assert np.isfinite(vertices).all()
    assert len(faces) > 0 and faces.min() >= 0 and faces.max() < len(vertices)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)
    forward = directed[np.lexsort(directed.T[::-1])]
    reversed_edges = directed[:, ::-1]
    backward = reversed_edges[np.lexsort(reversed_edges.T[::-1])]
    np.testing.assert_array_equal(forward, backward)
    triangles = vertices[faces]
    assert np.all(
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        > 0
    )


def test_real_capsule_binary64_geometry_and_input_ownership(provider):
    from fibernet.gen.tube_solid import tube_mesh

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = np.array([[0, 1]])
    original_points, original_edges = points.copy(), edges.copy()
    radius = 0.25
    vertices, faces = tube_mesh(points, edges, radius)
    _assert_owned_closed_mesh(vertices, faces)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(edges, original_edges)
    assert not np.shares_memory(vertices, points)
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [2.5, 0.5, 0.5], atol=1e-12)
    assert len(vertices) - 3 * len(faces) // 2 + len(faces) == 2
    triangles = vertices[faces]
    volume = (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )
    exact_capsule = np.pi * radius**2 * 2 + 4 * np.pi * radius**3 / 3
    assert 0.95 * exact_capsule < volume <= exact_capsule


@pytest.mark.parametrize("curved", [False, True])
def test_real_public_solid_export_and_readback(provider, tmp_path, curved):
    import fibernet as fn

    config = fn.PlanarManufacturingConfig(
        unit="square", grid_x=1, grid_y=1, n_pts_per_side=0
    )
    if curved:
        network = fn.compile_surface(
            np.array(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.2], [2.0, 2.0, 0.4], [0.0, 2.0, 0.1]]
            ),
            [[0, 1, 2, 3]],
            config,
        )
    else:
        network = fn.compile_planar(config)
    before_points, before_edges = network.positions.copy(), network.edges.copy()
    settings = fn.PrintSettings(
        width=20.0, depth=20.0, height=1.0, diameter=1.0, curved=curved
    )
    progress = []
    solid = fn.build_solid(network, settings, progress=progress.append)
    _assert_owned_closed_mesh(solid.vertices, solid.faces)
    assert solid.validate() > 0 and progress[-1] == 100
    np.testing.assert_array_equal(network.positions, before_points)
    np.testing.assert_array_equal(network.edges, before_edges)
    for suffix in ("stl", "3mf"):
        fn.export_solid(tmp_path / ("network." + suffix), solid)
    stl = (tmp_path / "network.stl").read_bytes()
    assert struct.unpack("<I", stl[80:84])[0] == len(solid.faces)
    assert len(stl) == 84 + 50 * len(solid.faces)
    records = np.frombuffer(
        stl[84:],
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )
    np.testing.assert_array_equal(
        records["vertices"], solid.vertices[solid.faces].astype("<f4")
    )
    with zipfile.ZipFile(tmp_path / "network.3mf") as archive:
        assert archive.testzip() is None
        model = ET.fromstring(archive.read("3D/3dmodel.model"))
    assert model.attrib["unit"] == "millimeter"
    positions = np.array(
        [
            [float(vertex.get(axis)) for axis in ("x", "y", "z")]
            for vertex in model.findall(".//{*}vertex")
        ]
    )
    faces = np.array(
        [
            [int(face.get(key)) for key in ("v1", "v2", "v3")]
            for face in model.findall(".//{*}triangle")
        ]
    )
    np.testing.assert_array_equal(
        positions.view(np.uint64), solid.vertices.view(np.uint64)
    )
    np.testing.assert_array_equal(faces, solid.faces)
    fn.export_route(tmp_path / "route.csv", network, solid.centers)
    with (tmp_path / "route.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(network.edges) + 1
    assert {int(row["incoming_edge_id"]) for row in rows[1:]} == set(
        range(len(network.edges))
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "network.3mf",
        "network.stl",
        "route.csv",
    ]


@pytest.mark.parametrize("cancel_at_union", [False, True])
def test_real_cancellation_then_retry_preserves_inputs(provider, cancel_at_union):
    from fibernet.gen.tube_solid import tube_mesh

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = np.array([[0, 1]])
    original_points, original_edges = points.copy(), edges.copy()
    progress = []

    def stop():
        return not cancel_at_union or (progress and progress[-1] >= 85)

    with pytest.raises(InterruptedError, match="solid export cancelled"):
        tube_mesh(points, edges, 0.25, progress=progress.append, stop_cb=stop)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(edges, original_edges)
    vertices, faces = tube_mesh(points, edges, 0.25)
    _assert_owned_closed_mesh(vertices, faces)
