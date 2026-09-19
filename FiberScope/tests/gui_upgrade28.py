"""UI type follow, trained unit lock, dark radar and recording command routing."""
import os
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication,QInputDialog
from studio.main import MainWindow
from fslab.structure import StructureFactory
from studio.theme import colors


class Model:
    def predict(self,x): return np.ones((len(np.atleast_2d(x)),3))


def main():
    app=QApplication.instance() or QApplication([])
    win=MainWindow(lang='zh',mode='light'); win.show(); app.processEvents()
    design=win.tab_design
    assert design.follow_type.isChecked() and not design.model_btn.isChecked()
    assert design.resume.isHidden() and not design.unit_combo.isEnabled()
    assert win.tab_surface.chk_follow.isChecked()
    win.tab_struct.load_spec(StructureFactory(unit='ring',grid_x=2,grid_y=2))
    app.processEvents()
    assert design.unit_combo.currentData()=='ring'
    assert win.tab_surface._follow_unit=='ring'
    win.tab_ml.trained_model=Model(); win.tab_ml.trained_unit='hexagon'
    win.tab_ml.unit_combo.setCurrentIndex(win.tab_ml.unit_combo.findData('square'))
    with patch.object(QInputDialog,'getItem',return_value=('max_peak',True)):
        design.model_btn.click()
    assert design.selected_model is not None and design.unit_combo.currentData()=='hexagon'
    win.tab_struct.load_spec(StructureFactory(unit='triangle'))
    assert design.unit_combo.currentData()=='hexagon'
    design.model_btn.click()
    assert design.unit_combo.currentData()=='triangle'
    design.follow_type.setChecked(False); assert design.unit_combo.isEnabled()
    win.tab_sim.chk_weld.setChecked(True)
    assert win.tab_sim.collect_cfg().weld_intersections
    win.tab_features.set_mode('dark'); app.processEvents()
    radar=win.tab_features.fingerprint
    pixel=radar.grab().toImage().pixelColor(1,1).name()
    assert radar.mode=='dark' and pixel==colors('dark')['card2'],pixel
    ai=win.ai_panel
    with patch.object(win.tab_ml,'start_gen') as generate:
        ai._generate_learning(n=300,unit='hexagon')
        assert generate.called and win.tab_ml.n_spin.value()==300
    ai._tool_surface('金字塔',None,None,1.)
    surface=win.tab_surface
    assert surface.chk_follow.isChecked() and surface._follow_unit=='triangle'
    path=surface._cur_path
    surface._mesh_settings[path]=(100,1)
    surface._mesh_cache.pop(path,None); surface._cur_path=None
    surface._compute_mapping()
    assert len(surface._mesh_cache[path][1])<=400
    with patch.object(ai.workflow,'start',return_value={}) as start:
        ai.input.setText('请执行决赛录像流程：生成结构、拉伸2、300样本训练、J逆设计、金字塔、STL拓竹。')
        ai.send()
        options=start.call_args.kwargs
        assert options['samples']==300 and options['target']=='J' and options['stretch']==2
        assert options['surface_model']=='demo_pyramid.obj' and options['create_demo']
    win.close(); app.processEvents()
    print('[gui-upgrade28] defaults, trained-type locking, dark radar, welding and full recording command PASS')


if __name__=='__main__': main()
