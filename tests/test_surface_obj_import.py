"""Public OBJ import should match the APP and reject invalid geometry."""
from pathlib import Path

import numpy as np
import pytest

from fibernet.gen import load_obj
from examples.obj_surface_workflow import ObjSurfaceWorkflow


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
