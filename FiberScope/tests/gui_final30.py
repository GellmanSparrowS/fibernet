"""Final interaction contract: visible parameters, automatic fabrication and traced navigation."""
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
from PySide6.QtWidgets import QApplication
from studio.main import MainWindow
from studio.recording_command import recording_options
from studio import workflow,ml_tab
from fslab.structure import StructureFactory,CELL
from fslab.manufacturing import compile_planar


def main():
    app=QApplication.instance() or QApplication([])
    win=MainWindow(lang='zh',mode='light');win.show();win.ai_panel.show();app.processEvents()
    tab=win.tab_struct
    tab.pert.setValue(40);tab.push_spec()
    actual=np.array(tab.editor.disp)
    assert actual.shape==(5,2) and np.any(actual)
    assert np.allclose(actual,np.array(tab.disp_row.values()),rtol=0,atol=1e-12)
    assert np.array_equal(actual,tab.collect_factory().effective_spectrum())
    tab.pert.setValue(80)
    assert np.allclose(tab.editor.disp,2*actual,atol=.0011)
    tab.pert.setValue(40)
    assert np.array_equal(tab.editor.disp,actual)
    seed=tab.seed.value();tab.seed.setValue(seed+1)
    assert not np.array_equal(tab.editor.disp,actual)
    tab.seed.setValue(seed)
    assert np.array_equal(tab.editor.disp,actual)
    tab.disp_row.spins[0].setValue(3.5)
    tab.pert.setValue(80)
    assert abs(tab.editor.disp[0][0]-.07)<1e-12
    tab.pert.setValue(40)
    tab.disp_row.spins[0].setValue(100)
    tab.pert.setValue(80)
    assert abs(tab.editor.disp[0][0]-1.)<1e-12
    assert np.allclose(tab.editor.disp,tab.disp_row.values(),atol=1e-12)
    tab.pts.setValue(7)
    assert len(tab.disp_row.spins)==14 and len(tab.editor.disp)==7
    tab.pts.setValue(5)
    factory=tab.collect_factory();tab.load_spec(factory)
    assert np.array_equal(factory.build().node_positions(),tab.collect_factory().build().node_positions())
    tab.pert.setValue(0);assert np.count_nonzero(tab.editor.disp)==0
    tab._random_seed();assert np.count_nonzero(tab.editor.disp)>0
    base=compile_planar(StructureFactory(unit='ring',grid_x=1,grid_y=1,n_pts_per_side=5))
    points=np.unique(np.round(base.positions,6),axis=0)
    radii=np.linalg.norm(points-[CELL/2,CELL/2,0],axis=1)
    assert len(points)==48 and np.isclose(radii,CELL*.36,atol=2e-6).sum()==24
    assert base.health['odd']==0 and len(base.route_edges)==len(base.edges)
    text='请执行决赛录像流程：生成中等复杂的方形网络，拉伸比2。生成300个带真实物理标签的样本，以J型曲线为目标进行200次预算。映射到金字塔，纤维直径2毫米，适配250毫米打印空间。'
    options=recording_options(text)
    assert (options['demo_unit'],options['iterations'],options['samples'])==('square',200,300)
    assert recording_options(text.replace('方形','六边形').replace('200次','60次'))['iterations']==60
    with patch.object(win.ai_panel.workflow,'start',return_value={}) as start:
        win.ai_panel.input.setText(text);win.ai_panel.send()
        assert start.call_args.kwargs==options
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory);(folder/'ml').mkdir()
        with patch.object(workflow,'writable_data_dir',lambda _:str(folder/'workflows')),patch.object(ml_tab,'_cache_dir',lambda:str(folder/'ml')):
            runner=win.ai_panel.workflow
            runner.start(samples=10,iterations=2,curved=True,open_slicer=False,
                         demo_unit='square',surface_model='demo_pyramid.obj',create_demo=True,target='J')
            # Skip only presentation pauses in this short regression.
            seen=set();until=time.monotonic()+150
            while runner.running:
                app.processEvents();time.sleep(.005)
                seen.add(win.tabs.currentWidget())
                if getattr(runner,"_preview_until",None) is not None: runner._preview_until=0
                assert time.monotonic()<until,runner.status()
            assert runner.status()['state']=='complete',runner.status()
            assert all(page in seen for page in (win.tab_struct,win.tab_sim,win.tab_features,
                win.tab_ml,win.tab_design,win.tab_surface,win.tab_manufacturing))
            assert win.tabs.currentWidget() is win.tab_manufacturing
            dialog=win.tab_manufacturing.dialog
            assert dialog.canvas.three_d and dialog.view2d.isHidden() and dialog.view3d.isHidden()
            buttons=(dialog.stl,dialog.threemf,dialog.route,dialog.bambu,dialog.cancel)
            assert len({button.geometry().y() for button in buttons})==1
            assert all(button.width()>=button.fontMetrics().horizontalAdvance(button.text())+14 for button in buttons)
            assert len({b.geometry().y() for b in (win.tab_manufacturing.planar,win.tab_manufacturing.curved,win.tab_manufacturing.play)})==1
            tools=[entry for entry in win.ai_panel._log if entry[0]=='tool' and entry[1].startswith('workflow.')]
            assert len(tools)==11 and all(entry[4] for entry in tools)
            assert all(stage['page']=='制造' for stage in runner.record['stages'][-3:])
            assert dialog.source_network is runner.network
            win.tab_manufacturing.open_source(False)
            assert not win.tab_manufacturing.dialog.canvas.three_d
            win.close()
            until=time.monotonic()+15
            while win.isVisible():
                app.processEvents();time.sleep(.01)
                assert time.monotonic()<until
    print('[final30] exact visible random parameters, four-arc ring density, parsed square/200, seven actual pages, eleven tool results, automatic views and single-row controls PASS')


if __name__=='__main__': main()
