"""Open the installed local Tet4 viewer and verify provenance with a native browser."""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ready(proc, url):
    for _ in range(150):
        if proc.poll() is not None:
            raise RuntimeError("The installed viewer stopped during startup")
        try:
            with urlopen(url, timeout=1) as res:
                if res.status == 200:
                    return
        except OSError:
            time.sleep(.2)
    raise RuntimeError("The installed viewer did not answer")


def _check_local_boundary(url):
    for headers in ({"Origin": "https://example.invalid"}, {"Host": "example.invalid"},
                    {"Sec-Fetch-Site": "cross-site"}, {"Upgrade": "websocket"}):
        try:
            with urlopen(Request(url, headers=headers), timeout=3):
                raise AssertionError(f"Non-local request succeeded: {tuple(headers)}")
        except HTTPError as exc:
            assert exc.code == 403


def _native_reference(path):
    r = vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    grid = r.GetOutput()
    assert grid.GetNumberOfCells() == 28512
    arrays = grid.GetCellData()
    cell_ids = vtk_to_numpy(arrays.GetArray("cell_id")).tolist()
    tags = vtk_to_numpy(arrays.GetArray("material_id")).tolist()
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(float)
    u = vtk_to_numpy(grid.GetPointData().GetArray("displacement_m")).astype(float)
    return {str(n): (i, str(tag)) for i, (n, tag) in enumerate(zip(cell_ids, tags))}, grid, points + u, u


def _check_physical_pick(page, reference, cell_id, tag):
    rows, grid, drawn_points, u = reference
    row, original_tag = rows[cell_id]
    assert original_tag == tag and int(cell_id) > 2**53
    cell = grid.GetCell(row)
    ids = [cell.GetPointId(i) for i in range(4)]
    nodes_mm = drawn_points[ids] * 1e3
    vectors_um = u[ids] * 1e6
    base = nodes_mm[0]
    gradient = np.linalg.solve(nodes_mm[1:] - base, vectors_um[1:] - vectors_um[0])
    location_text = page.locator("#picked-point").inner_text()
    field_text = page.locator("#picked-u").inner_text()
    assert location_text.endswith(" mm") and field_text.endswith(" µm")
    position = np.array([float(x) for x in location_text.removesuffix(" mm").split(",")])
    vector = np.array([float(x) for x in field_text.removesuffix(" µm").split(",")])
    expected = vectors_um[0] + (position - base) @ gradient
    # Coordinates shown to 4 decimal places in mm; propagate their worst-case
    # rounding through the original tetrahedron's linear displacement field.
    bound = 5e-5 * np.abs(gradient).sum(axis=0) + 6e-7
    assert np.all(np.abs(expected - vector) <= bound), (cell_id, expected, vector, bound)


def _render_field(page, material_image, u, picks):
    crop = (180, 210, 790, 590)
    original = material_image.crop(crop)
    page.locator("#field-ux").click()
    label = page.locator("#ux-scale")
    label.wait_for(state="visible")
    cap = float(np.max(np.abs(u[:,0])) * 1e6)
    assert math.isfinite(cap) and cap > 0
    assert label.inner_text() == f"x displacement: {-cap:.4g} to {cap:.4g} µm"
    for _ in range(30):
        physics_image = Image.open(io.BytesIO(page.screenshot())).convert("RGB").crop(crop)
        change = np.asarray(ImageChops.difference(original, physics_image), dtype=np.uint8)
        if np.count_nonzero(np.max(change, axis=2) > 12) >= 1500:
            break
        page.wait_for_timeout(150)
    else:
        raise AssertionError("Displacement color rendering did not change the mesh")
    # Re-pick a known position after the scalar mapper changes. Verify that
    # provenance and physical values are still accessible in the field view.
    for (x, y), cell_id, material, physical_u in picks:
        page.mouse.click(x, y)
        page.wait_for_function(
            "(id) => document.querySelector('#picked-id').textContent.trim() === id", arg=cell_id
        )
        assert page.locator("#picked-material").inner_text() == material
        assert page.locator("#picked-u").inner_text() == physical_u
    page.locator("#field-material").click()
    label.wait_for(state="hidden")


def _render(url, reference, executable, channel):
    with sync_playwright() as pw:
        args = []
        if sys.platform.startswith("linux"):
            args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        browser = pw.chromium.launch(executable_path=executable, channel=channel,
                                   chromium_sandbox=True, args=args)
        try:
            page = browser.new_page(viewport={"width": 980, "height": 680}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("document.querySelectorAll('canvas').length >= 2")
            pixels = {"7": (57, 174, 235), "19": (246, 168, 68)}
            seen = {}
            for mode in ("surface", "slice"):
                if mode == "slice":
                    page.locator("#select-slice").click()
                    page.wait_for_function("document.querySelector('#visible-mode').textContent.trim() === 'slice'")
                    page.wait_for_function("document.querySelector('#picked-id').textContent.trim() === ''")
                for _ in range(30):
                    shot = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
                    # Ignore the controls and legend. Choose stable colors from the actual material view.
                    candidates = {m: [] for m in pixels}
                    for y in range(210, 590, 5):
                        for x in range(180, 790, 5):
                            row = shot.getpixel((x, y))
                            for material, rgb in pixels.items():
                                if max(abs(a-b) for a,b in zip(row,rgb)) <= 4:
                                    candidates[material].append((x, y))
                    if all(len(v) >= 10 for v in candidates.values()):
                        break
                    page.wait_for_timeout(200)
                else:
                    raise AssertionError(f"No rendered material colors in {mode}")
                picks = []
                for material, points in candidates.items():
                    # Spread probes to avoid repeatedly selecting one shared-edge neighbourhood.
                    points = points[::max(1,len(points)//25)]
                    for x,y in points:
                        page.mouse.click(x,y)
                        page.wait_for_timeout(120)
                        src = page.locator("#picked-id").inner_text()
                        if src.isdigit() and page.locator("#picked-material").inner_text() == material:
                            _check_physical_pick(page, reference, src, material)
                            seen[mode,material] = src
                            picks.append(((x, y), src, material, page.locator("#picked-u").inner_text()))
                            break
                    else:
                        raise AssertionError(f"Unable to pick {mode} material {material}")
                _render_field(page, shot, reference[-1], picks)
            assert not errors, errors
            return seen
        finally:
            browser.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group()
    choice.add_argument("--browser-executable", default=None)
    choice.add_argument("--browser-channel", choices=["chrome"], default=None)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        subprocess.run([sys.executable, "-m", "fibernet.examples.tet4_tagged_grip_iterative",
                        "--output", str(root/"result"), "--write-vtu"], cwd=root, check=True,
                       capture_output=True, text=True, timeout=120)
        vtu = root/"result"/"deformation.vtu"
        reference = _native_reference(vtu)
        port = _port()
        url = f"http://127.0.0.1:{port}"
        with (root/"viewer.log").open("w",encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable,"-m","fibernet.viz.tet4_browser",str(vtu),
                                     "--section-z-mm","0.9265","--port",str(port)],
                                    cwd=root,stdout=log,stderr=subprocess.STDOUT)
            try:
                _ready(proc,url)
                _check_local_boundary(url)
                result = _render(url,reference,args.browser_executable,args.browser_channel)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                if sys.exc_info()[0] is not None:
                    print((root/"viewer.log").read_text(encoding="utf-8",errors="replace"),file=sys.stderr)
        print(json.dumps({"/".join(key):value for key,value in result.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
