"""Inspect a trusted FiberNet Tet4 VTU in a local browser."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from uuid import uuid4

_COLOR_CYCLE = (
    (57, 174, 235), (246, 168, 68), (181, 135, 230), (64, 205, 160),
    (237, 106, 134), (218, 210, 106), (132, 156, 225), (174, 189, 198),
)
_IDS = ("source_cell_id_hi", "source_cell_id_lo")
_TAG = "source_material_tag"
_RGB = "material_rgb"
_LIMIT_CELLS = 250_000
_LIMIT_FILE = 64 * 1024 * 1024


@dataclass(frozen=True)
class _Scene:
    mesh: object
    locator: object
    diagonal_m: float


def _scenes(path, section_z_mm, scale):
    import numpy as np
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkPlane, vtkStaticCellLocator, VTK_TETRA
    from vtkmodules.vtkFiltersCore import vtkCutter
    from vtkmodules.vtkFiltersGeneral import vtkWarpVector
    from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    path = Path(path)
    if path.suffix.lower() != ".vtu" or not path.is_file():
        raise ValueError("Choose a local FiberNet Tet4 .vtu file")
    if not 0 < path.stat().st_size <= _LIMIT_FILE:
        raise ValueError("The local VTU must be non-empty and no larger than 64 MiB")
    if not math.isfinite(scale) or not 0 <= scale <= 100:
        raise ValueError("--scale must be between 0 and 100")
    if not math.isfinite(section_z_mm):
        raise ValueError("--section-z-mm must be finite")
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    errors = []
    reader.AddObserver("ErrorEvent", lambda *_: errors.append(True))
    reader.Update()
    grid = reader.GetOutput()
    nc, npnt = grid.GetNumberOfCells(), grid.GetNumberOfPoints()
    if errors or not 0 < nc <= _LIMIT_CELLS or not 0 < npnt <= 4 * _LIMIT_CELLS:
        raise ValueError("Invalid or oversized Tet4 VTU; limit is 250,000 cells")
    if not np.all(vtk_to_numpy(grid.GetCellTypesArray()) == VTK_TETRA):
        raise ValueError("All volume cells must be linear tetrahedra")

    def field(attrs, name, shape):
        vtk_array = attrs.GetArray(name)
        if vtk_array is None:
            raise ValueError(f"Missing VTU field {name}")
        arr = vtk_to_numpy(vtk_array)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"Invalid VTU field {name}")
        return arr

    cells, points = grid.GetCellData(), grid.GetPointData()
    ids = field(cells, "cell_id", (nc,))
    tags = field(cells, "material_id", (nc,))
    xyz = vtk_to_numpy(grid.GetPoints().GetData())
    u = field(points, "displacement_m", (npnt, 3))
    if (not np.isfinite(xyz).all() or ids.dtype.kind not in "iu" or (ids < 0).any()
            or len(np.unique(ids)) != nc or tags.dtype.kind not in "iu" or
            (tags < 0).any() or int(tags.max()) >= 2**32):
        raise ValueError("Coordinates and material tags must be valid; cell_id must be unique, nonnegative integer")
    unique_tags = np.unique(tags)
    if len(unique_tags) > len(_COLOR_CYCLE):
        raise ValueError("This viewer supports at most eight material tags")
    colors = {int(tag): _COLOR_CYCLE[i] for i, tag in enumerate(unique_tags)}
    palette = np.array(tuple(colors.values()), dtype=np.uint8)
    z = section_z_mm / 1000
    if not float(xyz[:, 2].min()) < z < float(xyz[:, 2].max()):
        raise ValueError("--section-z-mm must lie strictly inside the undeformed z bounds")
    relative_slack = 8 * np.finfo(xyz.dtype).eps * max(abs(z), float(np.ptp(xyz[:, 2])))
    if np.min(np.abs(xyz[:, 2] - z)) <= relative_slack:
        raise ValueError("Section crosses a mesh vertex; move --section-z-mm slightly")
    xyz_warped = xyz + u * scale
    diagonal = float(np.linalg.norm(np.ptp(xyz_warped, axis=0)))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Invalid deformed coordinates")
    # Use only fields understood by the viewer. Integer categorical data is attached
    # to cells, never passed through a point-field interpolator.
    core_u = np.array(u, copy=True)
    ids = ids.astype(np.uint64)
    tags = tags.astype(np.uint32)
    points.Initialize()
    cells.Initialize()

    def attach(attrs, name, arr):
        values = numpy_to_vtk(arr, deep=True)
        values.SetName(name)
        attrs.AddArray(values)

    attach(points, "displacement_m", core_u)
    attach(cells, _IDS[0], (ids >> np.uint64(32)).astype(np.uint32))
    attach(cells, _IDS[1], (ids & np.uint64(0xffffffff)).astype(np.uint32))
    attach(cells, _TAG, tags)
    surface = vtkDataSetSurfaceFilter()
    surface.SetInputData(grid)
    plane = vtkPlane()
    plane.SetOrigin(0., 0., z)
    plane.SetNormal(0., 0., 1.)
    section = vtkCutter()
    section.SetCutFunction(plane)
    section.SetInputData(grid)
    section.GenerateTrianglesOn()
    result = {}
    for mode, extract in (("surface", surface), ("slice", section)):
        warp = vtkWarpVector()
        warp.SetInputConnection(extract.GetOutputPort())
        warp.SetInputArrayToProcess(0, 0, 0, 0, "displacement_m")
        warp.SetScaleFactor(scale)
        warp.Update()
        mesh = warp.GetOutput()
        if not mesh.GetNumberOfCells():
            raise ValueError(f"No cells in {mode}; choose a different section height")
        attrs = mesh.GetCellData()
        material = vtk_to_numpy(attrs.GetArray(_TAG))
        attach(attrs, _RGB, palette[np.searchsorted(unique_tags, material)])
        locator = vtkStaticCellLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        result[mode] = _Scene(mesh, locator, diagonal)
    return result, colors


def _ray_hit(scene, payload):
    from vtkmodules.vtkCommonCore import reference
    from vtkmodules.vtkCommonDataModel import vtkGenericCell

    ray, est = payload.get("ray"), payload.get("worldPosition")

    def valid_vector(row):
        return isinstance(row, (list, tuple)) and len(row) == 3 and all(
            type(x) in (float, int) and abs(x) < 1e9 and math.isfinite(x) for x in row
        )

    if not (isinstance(ray, (list, tuple)) and len(ray) == 2 and
            all(map(valid_vector, ray)) and valid_vector(est)):
        return None
    hit, coords = [0.] * 3, [0.] * 3
    t, sub, idx = reference(0.), reference(0), reference(0)
    if not scene.locator.IntersectWithLine(
        *ray, scene.diagonal_m * 1e-9, t, hit, coords, sub, idx, vtkGenericCell()
    ):
        return None
    if (math.dist(hit, est) > scene.diagonal_m * 1.5e-5 or
            min(coords[:2] + [1 - coords[0] - coords[1]]) < 1e-3):
        return None
    attrs = scene.mesh.GetCellData()
    hi, lo = (int(attrs.GetArray(name).GetValue(int(idx))) for name in _IDS)
    tag = int(attrs.GetArray(_TAG).GetValue(int(idx)))
    return str((hi << 32) | lo), str(tag), ", ".join(f"{v*1000:.4f}" for v in hit) + " mm"


def _app(scenes, colors, z_mm, scale):
    from trame.app import get_server
    from trame.ui.html import DivLayout
    from trame.widgets import html, vtk

    server = get_server("fibernet-" + uuid4().hex, client_type="vue3")
    state = server.state
    state.update(dict(mode="surface", pickData=None, details="", material="", world_point=""))

    @state.change("mode")
    def reset_panel(**_):
        state.details = state.material = state.world_point = ""
        state.pickData = None

    @state.change("pickData")
    def select_cell(pickData, **_):
        state.details = state.material = state.world_point = ""
        if not isinstance(pickData, dict) or pickData.get("representationId") != state.mode:
            return
        scene = scenes.get(state.mode) if isinstance(state.mode, str) else None
        if scene is None:
            return
        hit = _ray_hit(scene, pickData)
        if hit is None:
            state.details = "Pick inside a visible mesh face"
            return
        state.details, state.material, state.world_point = hit

    with DivLayout(server):
        with html.Div(style="font-family:system-ui;color:#fff;background:#111e2c;position:fixed;inset:0;display:flex;flex-direction:column;"):
            with html.Div(style="height:52px;padding:12px"):
                html.Span("FiberNet · Tet4 results", style="margin-right:12px")
                html.Button("Exterior", id="select-surface", click="mode = 'surface'")
                html.Button("Material section", id="select-slice", click="mode = 'slice'")
                html.Span("{{ mode }}", id="visible-mode", style="margin-left:8px")
                html.Span(f"z = {z_mm:g} mm in reference geometry · displacement ×{scale:g}",
                          style="margin-left:12px;color:#a9bac9")
            with html.Div(style="position:relative;flex:1;min-height:0"):
                for mode, scene in scenes.items():
                    with vtk.VtkView(
                        v_show=f"mode === '{mode}'", picking_modes=("pickModes", ["click"]),
                        background=("viewBg", [.06, .09, .13]),
                        style="position:absolute;inset:0;width:100%;height:100%",
                        click="pickData = $event.worldPosition ? $event : pickData",
                    ):
                        with vtk.VtkGeometryRepresentation(
                            id=mode, mapper=(mode+"Mapper", {"scalarVisibility": True,
                            "scalarMode": 4, "colorMode": 2, "colorByArrayName": _RGB}),
                        ):
                            vtk.VtkMesh(mode+"_data", dataset=scene.mesh, cell_arrays=[*_IDS, _TAG, _RGB])
            with html.Div(style="position:absolute;right:12px;top:75px;padding:10px;background:#132434;pointer-events:none;min-width:236px"):
                html.Div("Original Tet4", style="font-weight:600")
                html.Div("{{ details }}", id="picked-id")
                html.Div("Material tag", style="color:#a9bac9;margin-top:4px")
                html.Div("{{ material }}", id="picked-material")
                html.Div("{{ world_point }}", id="picked-point", style="color:#a9bac9;margin-top:4px")
                for tag, rgb in colors.items():
                    html.Span(str(tag), style=f"display:inline-block;margin:6px 5px 0 0;padding:3px 7px;color:#07131e;background:rgb{rgb};border-radius:4px")
    return server


def _restrict_to_local_origin(server, port):
    from aiohttp import web

    host = f"127.0.0.1:{port}"
    origin = "http://" + host

    @web.middleware
    async def local_only(request, handler):
        if (request.headers.get("Host") != host
                or request.headers.get("Origin") not in {None, origin}
                or request.headers.get("Sec-Fetch-Site") not in {None, "none", "same-origin"}
                or (request.headers.get("Upgrade", "").lower() == "websocket"
                    and request.headers.get("Origin") != origin)):
            raise web.HTTPForbidden()
        response = await handler(request)
        if not response.prepared:
            response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response

    @server.controller.add("on_server_bind")
    def guard(wslink_server):
        wslink_server.app.middlewares.append(local_only)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtu", type=Path, help="your trusted local FiberNet Tet4 VTU")
    parser.add_argument("--section-z-mm", type=float, required=True,
                        help="z height of the section before deformation, in mm")
    parser.add_argument("--scale", type=float, default=1., help="display-only displacement multiplier")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    try:
        scenes, colors = _scenes(args.vtu, args.section_z_mm, args.scale)
        server = _app(scenes, colors, args.section_z_mm, args.scale)
        _restrict_to_local_origin(server, args.port)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in {"vtkmodules", "trame", "trame_vtk", "aiohttp"}:
            raise
        parser.error("Set up Python 3.12 and install FiberNet with its [browser-viz] extra")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    server.start(port=args.port, host="127.0.0.1", open_browser=False)


if __name__ == "__main__":
    main()
