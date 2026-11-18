"""Public OBJ import should match the APP and reject invalid geometry."""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fibernet.gen import (
    MappingConfig, PlanarManufacturingConfig, compile_surface, load_obj, map_cells,
)
from examples.obj_surface_workflow import ObjSurfaceWorkflow
from examples.surface_mapping_workflow import SurfaceMappingWorkflow


def test_import_bundled_surface_without_app_import():
    path = (Path(__file__).resolve().parents[1] / "FiberScope" /
            "assets" / "obj" / "demo_pyramid.obj")
    vertices, faces, info = load_obj(path, return_info=True)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert len(faces) == info["quad_faces"] > 0
    assert all(len(face) == 4 for face in faces)
    assert np.isfinite(vertices).all()
    assert min(min(face) for face in faces) >= 0
    assert max(max(face) for face in faces) < len(vertices)


@pytest.mark.parametrize("content", [
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n",
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 2\n",
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 2 3\n",
])
def test_invalid_obj_indices(tmp_path, content):
    path = tmp_path / "bad.obj"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid OBJ face indices"):
        load_obj(path)


def test_obj_to_closed_route_workflow(tmp_path):
    path = (Path(__file__).resolve().parents[1] / "FiberScope" /
            "assets" / "obj" / "demo_pyramid.obj")
    summary = ObjSurfaceWorkflow(path, tmp_path).run()
    assert summary["route_closed"]
    assert summary["edges"] > 100
    with np.load(tmp_path / "curved_route.npz", allow_pickle=False) as saved:
        assert len(saved["route_edges"]) == len(saved["edges"])


_NOTCHED_PANEL = np.array([
    [0, 0, 0], [3, 0, 0], [3, 3, 0], [2, 3, 0],
    [2, 1, 0], [1, 1, 0], [1, 3, 0], [0, 3, 0],
], dtype=float)


def _write_obj(path, vertices, faces):
    coordinates = [f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in vertices]
    polygons = ["f " + " ".join(str(i + 1) for i in face) for face in faces]
    path.write_text("\n".join(coordinates + polygons) + "\n", encoding="utf-8")
    return path


def _signed_areas(vertices, faces):
    x, y = vertices[faces, :2].transpose(2, 0, 1)
    return np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1) / 2


def _crosses_notch(segment, margin):
    """Intersect the segment parameter intervals for the open notch rectangle."""
    first, last = 0.0, 1.0
    for axis, upper in enumerate((2, 3)):
        start, end = segment[:, axis]
        low, high = 1 + margin, upper - margin
        if abs(end - start) < 1e-13:
            if not low < start < high:
                return False
            continue
        entry, exit = sorted(((low - start) / (end - start), (high - start) / (end - start)))
        first, last = max(first, entry), min(last, exit)
    return first < last


@pytest.mark.parametrize("target_faces", [None, 1500])
@pytest.mark.parametrize("winding", [1, -1])
@pytest.mark.parametrize(("scale", "offset"), [
    (1e-8, 49e-8), (1.0, 49.0), (1e8, 49e8), (1.0, 1e6), (1e-6, 100.0),
])
def test_concave_obj_preserves_area_winding_and_notch(tmp_path, scale, offset, winding, target_faces):
    rotation = Rotation.from_euler("xyz", [19, 31, 13], degrees=True).as_matrix()
    shift = offset * np.array([1.0, 1.5, -0.85])
    source = (_NOTCHED_PANEL @ rotation) * scale + shift
    path = _write_obj(tmp_path / "panel.obj", source, [list(range(len(source)))[::winding]])
    vertices, faces, info = load_obj(path, return_info=True, target_faces=target_faces)
    assert info == dict(source_vertices=8, source_faces=1, triangles=0,
                        reduced=False, quad_faces=18, converted=True)
    local = ((vertices - shift) / scale) @ rotation.T
    margin = max(1e-10, 32 * np.finfo(float).eps * offset / scale)
    np.testing.assert_allclose(local[:, 2], 0, atol=margin)
    quads = np.asarray(faces)
    areas = winding * _signed_areas(local, quads)
    assert (areas > 0).all()
    assert areas.sum() == pytest.approx(7.0)
    segments = np.stack((quads, np.roll(quads, -1, axis=1)), axis=-1).reshape(-1, 2)
    assert not any(_crosses_notch(segment, margin) for segment in local[segments])
    # A quad covering the whole notch would have no edges crossing it.
    corners = local[quads, :2]
    edges = np.roll(corners, -1, axis=1) - corners
    to_center = np.array([1.5, 2.0]) - corners
    side = winding * (edges[..., 0] * to_center[..., 1] - edges[..., 1] * to_center[..., 0])
    assert not (side >= -margin).all(axis=1).any()


@pytest.mark.parametrize("close_vertex", [1e-12, 1.0 + 1e-12])
def test_concave_obj_retains_short_boundary_edges(tmp_path, close_vertex):
    additions = [[close_vertex, 0.0, 0.0]]
    if close_vertex > 1.0:
        additions.insert(0, [1.0, 0.0, 0.0])
    source = np.concatenate((_NOTCHED_PANEL[:1], additions, _NOTCHED_PANEL[1:]))
    path = _write_obj(tmp_path / "short_edge.obj", source, [list(range(len(source)))])
    vertices, faces = load_obj(path)
    assert np.bincount(np.ravel(faces))[:len(source)].min() > 0
    areas = _signed_areas(vertices, np.asarray(faces))
    assert (areas > 0).all()
    assert areas.sum() == pytest.approx(7.0)


def test_skinny_concave_obj_is_still_quadrangulated(tmp_path):
    scale = np.array([1.0, 1e-12, 1.0])
    path = _write_obj(tmp_path / "skinny.obj", _NOTCHED_PANEL * scale, [list(range(8))])
    vertices, faces = load_obj(path)
    assert len(faces) == 18
    areas = _signed_areas(vertices / scale, np.asarray(faces))
    assert (areas > 0).all()
    assert areas.sum() == pytest.approx(7.0)


def test_concave_quad_is_repaired_in_all_quad_obj(tmp_path):
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0.6, 0.6, 0], [0, 2, 0]])
    path = _write_obj(tmp_path / "concave_quad.obj", vertices, [[0, 1, 2, 3]])
    vertices, faces = load_obj(path)
    areas = _signed_areas(vertices, np.asarray(faces))
    assert len(faces) == 6
    assert (areas > 0).all()
    assert areas.sum() == pytest.approx(1.2)


def test_concave_face_keeps_collinear_shared_edges_with_convex_quads(tmp_path):
    source = np.insert(_NOTCHED_PANEL, 2, [3.0, 1.5, 0.0], axis=0)
    source = np.vstack((source, [[4, 0, 0], [4, 1.5, 0], [4, 3, 0]]))
    path = _write_obj(tmp_path / "joined.obj", source,
                      [list(range(9)), [1, 9, 10, 2], [2, 10, 11, 3]])
    vertices, faces = load_obj(path)
    np.testing.assert_array_equal(vertices[:len(source)], source)
    counts = Counter(tuple(sorted((a, b))) for face in faces
                     for a, b in zip(face, face[1:] + face[:1]))
    for a, b in ((1, 2), (2, 3), (2, 10)):
        midpoint = (source[a] + source[b]) / 2
        matches = np.flatnonzero(np.all(vertices == midpoint, axis=1))
        assert len(matches) == 1
        index = int(matches[0])
        assert counts[tuple(sorted((a, index)))] == counts[tuple(sorted((b, index)))] == 2
    areas = _signed_areas(vertices, np.asarray(faces))
    assert (areas > 0).all()
    assert areas.sum() == pytest.approx(10.0)


def test_concave_obj_workflows_leave_the_notch_unmapped(tmp_path):
    path = _write_obj(tmp_path / "panel.obj", _NOTCHED_PANEL, [list(range(8))])
    vertices, faces = load_obj(path)
    route = compile_surface(vertices, faces, PlanarManufacturingConfig(
        unit="square", grid_x=1, grid_y=1, n_pts_per_side=0), overscale=1.0)
    points, segments = map_cells(vertices, faces, [], config=MappingConfig(overscale=1.0))
    assert route.route_nodes[0] == route.route_nodes[-1]
    assert not any(_crosses_notch(segment, 1e-10) for segment in route.positions[route.edges])
    assert not any(_crosses_notch(segment, 1e-10) for segment in points[segments])

    assert ObjSurfaceWorkflow(path, tmp_path).run()["route_closed"]
    assert SurfaceMappingWorkflow(path, tmp_path).run()["mapped_faces"] == len(faces)
    for filename, positions, edges in (
        ("curved_route.npz", "positions", "edges"),
        ("mapped_cells.npz", "points", "segments"),
    ):
        with np.load(tmp_path / filename, allow_pickle=False) as saved:
            segments = saved[positions][saved[edges]]
            # Both examples deliberately overscale patches by 6%.
            assert not any(_crosses_notch(segment, 0.04) for segment in segments)
