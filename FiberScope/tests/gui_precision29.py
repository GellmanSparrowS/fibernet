"""Visible controls, regular perturbation and real high-resolution GIF validation."""
import os
import sys
import time
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication, QMessageBox
from PIL import Image
from studio.main import MainWindow
from fslab.structure import StructureFactory
from fslab.manufacturing import compile_planar


def main():
    app=QApplication.instance() or QApplication([])
    win=MainWindow(lang='zh',mode='light');win.show();app.processEvents()
    assert win.tab_sim.chk_weld.isChecked() and not win.tab_sim.chk_contact.isChecked()
    assert len({x.height() for x in (win.ver_chip,win.ai_btn,win.lang_combo,win.theme_btn)})==1
    with patch.object(QMessageBox,'information') as about:
        win.ver_chip.click()
        assert '复旦大学高分子系杨云浩' in about.call_args.args[2]
    assert win.tabs.indexOf(win.tab_manufacturing)==win.tabs.indexOf(win.tab_surface)+1
    win.tabs.setCurrentWidget(win.tab_surface);app.processEvents()
    for button in (win.tab_surface.import_btn,win.tab_surface.reset_btn,win.tab_surface.mesh_btn):
        assert button.width()>=button.fontMetrics().horizontalAdvance(button.text())+24
    factory=StructureFactory(unit='ring',grid_x=2,grid_y=2,n_pts_per_side=5,
        line_displacements=[[.01,.04]]*5,perturbation=.5)
    actual=compile_planar(factory)
    explicit=compile_planar(replace(factory,perturbation=0.,line_displacements=factory.effective_spectrum().tolist()))
    assert np.array_equal(actual.positions,explicit.positions)
    assert np.array_equal(actual.route_nodes,explicit.route_nodes)
    zero=replace(factory,line_displacements=[[0.,0.]]*5)
    assert np.array_equal(compile_planar(zero).positions,compile_planar(replace(zero,perturbation=0.)).positions)
    win.tab_surface.apply_structure(factory)
    assert np.array_equal(win.tab_surface._follow_spec,factory.effective_spectrum())
    tab=win.tab_sim
    tab.set_spec(StructureFactory(grid_x=2,grid_y=2,n_pts_per_side=1))
    tab.quant.setValue(3);tab.run_sync();tab._play_timer.stop()
    assert tab.run is not None
    tab.chk_contact.setChecked(True)
    with patch.object(tab, "run_sync"):
        win.ai_panel.registry["run_stretch"]()
    assert not tab.chk_contact.isChecked()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'simulation.gif'
        tab._export_gif(str(path))
        deadline=time.monotonic()+90
        while not path.exists():
            app.processEvents();time.sleep(.005)
            assert time.monotonic()<deadline,tab.status.text()
        with Image.open(path) as animation:
            assert animation.size==(1600,1000)
            assert animation.n_frames==tab.run.n_frames
            duration=0
            for frame in range(animation.n_frames):
                animation.seek(frame);duration+=animation.info['duration']
            assert abs(duration-animation.n_frames*1000/48)<=5.1
        assert not path.with_name(path.name+'.tmp').exists()
    win.close();app.processEvents()
    print('[precision29] parameter equivalence, fixed zeros, surface following, unclipped buttons, author click, defaults, 1600px GIF and 48FPS average PASS')


if __name__=='__main__': main()
