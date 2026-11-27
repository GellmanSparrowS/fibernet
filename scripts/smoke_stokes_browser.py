"""Exercise installed native pressure solve, VTU writer and local browser."""

import argparse
from io import BytesIO
from itertools import product
import json
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

from fibernet.examples.tet4_periodic_pore import _voxel_bundle
from fibernet.sim.tet4_open_stokes import solve_open_pressure_stokes, write_open_stokes_velocity_vtu
from fibernet.viz.stokes_browser import _scenes


def _check_browser(url, args):
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path=args.browser_executable,
            channel=args.browser_channel,
            chromium_sandbox=True,
            args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"],
        )
        try:
            page=b.new_page(viewport={"width":1200,"height":740}, device_scale_factor=1)
            errors=[]
            page.on("pageerror", lambda exc:errors.append(str(exc)))
            page.goto(url,wait_until="load")
            page.wait_for_function("document.querySelector('canvas') !== null")
            page.locator("#set-view").click()
            page.wait_for_timeout(700)
            assert "49 streamlines · 1,152 tetra10 cells" in page.locator("#details").text_content()
            def image():
                return Image.open(BytesIO(page.screenshot())).convert("RGB").crop((80,180,880,640))
            a=image()
            colored=sum((max(px)-min(px)>30 and min(px)<120) for px in a.get_flattened_data())
            assert colored>1000
            page.locator("#wall-toggle").click()
            page.wait_for_timeout(500)
            assert page.locator("#wall-toggle").get_attribute("aria-pressed")=="true"
            delta=sum(max(px)>10 for px in ImageChops.difference(a,image()).get_flattened_data())
            assert not errors and delta>2000,(errors,delta)
            page.set_viewport_size({"width":390,"height":730})
            bb=page.locator("#details").bounding_box()
            assert bb["x"]>=0 and bb["x"]+bb["width"]<=390
            return delta,colored
        finally:
            b.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    p=parser.add_mutually_exclusive_group()
    p.add_argument("--browser-executable")
    p.add_argument("--browser-channel")
    args=parser.parse_args()
    with TemporaryDirectory() as directory:
        run=Path(directory)
        brief=_voxel_bundle(product(range(12), range(4), range(4)))
        bc=dict(viscosity_Pa_s=.001,solver="direct",include_quadratic_velocity=True)
        positive=solve_open_pressure_stokes(brief,pressure_drop_Pa=.0035,**bc)
        reverse=solve_open_pressure_stokes(brief,pressure_drop_Pa=-.0035,**bc)
        assert positive.axial_flow_m3_s>0. and reverse.axial_flow_m3_s<0.
        pos=run/"source-flow.vtu"; rev=run/"reverse-flow.vtu"
        write_open_stokes_velocity_vtu(pos,positive.quadratic_velocity)
        write_open_stokes_velocity_vtu(rev,reverse.quadratic_velocity)
        _,_,meta,_=_scenes(rev,True)
        assert meta["traces"]==49 and meta["cells"]==1152
        bad=run/"unsupported.vtu"; bad.write_text("<VTKFile type='ImageData'/>",encoding="utf-8")
        try: _scenes(bad)
        except ValueError as err: assert "Expected FiberNet's" in str(err)
        else: raise AssertionError("Unsupported VTK input was admitted")
        nonsource=run/"not-a-file.vtu"
        nonsource.mkdir()
        try: _scenes(nonsource)
        except ValueError as err: assert "regular" in str(err)
        else: raise AssertionError("Admitted a nonregular VTU input")
        with socket.socket() as s:
            s.bind(("127.0.0.1",0));port=s.getsockname()[1]
        url=f"http://127.0.0.1:{port}"
        with (run/"server.log").open("w",encoding="utf-8") as log:
            proc=subprocess.Popen([sys.executable,"-I","-m","fibernet.viz.stokes_browser",str(pos),"--port",str(port)],
                                  stdout=log,stderr=subprocess.STDOUT,cwd=run)
            try:
                for _ in range(100):
                    try:
                        with urlopen(url,timeout=.4) as response: assert response.status==200
                        break
                    except OSError:
                        assert proc.poll() is None,(run/"server.log").read_text(encoding="utf-8")
                        time.sleep(.2)
                else: raise RuntimeError("Local Stokes browser did not start")
                delta,colored=_check_browser(url,args)
            finally:
                proc.terminate()
                proc.wait(timeout=5)
        print(json.dumps(dict(cells=len(brief.tetrahedra),render_changed_pixels=delta,
            colored_pixels=colored,saved_file_replay=True,reverse_cap=True)))


if __name__=="__main__":
    main()
