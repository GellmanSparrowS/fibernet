"""Final interaction contract: visible parameters, automatic fabrication and traced navigation."""
import os
import sys
import time
import tempfile
import json
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
            result=runner.status()
            assert result['state']=='failed',result
            assert result['message']=='ambiguous zero-thickness solid contact',result
            assert result['printer_state']=='not_submitted' and not result['busy']
            assert [stage['name'] for stage in result['stages']]==list(runner.STAGES[:9])
            assert all(stage['state']=='complete' for stage in result['stages'][:-1])
            assert result['stages'][-1]['state']=='failed'
            assert result['stages'][-1]['name']=='solid'
            assert runner.solid is None and runner.network is None
            assert all(win.tabs.widget(i).isEnabled() for i in range(win.tabs.count()))
            assert not any((runner.folder/name).exists() for name in
                           ('FiberScope.stl','FiberScope.3mf','path.csv'))
            saved=json.loads((runner.folder/'progress.json').read_text(encoding='utf-8'))
            assert saved['state']=='failed' and saved['message']==result['message']
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
            assert len(tools)==9 and all(entry[4] for entry in tools[:-1])
            assert not tools[-1][4] and tools[-1][1]=='workflow.solid'
            assert runner.record['stages'][-1]['page']=='制造'
            assert dialog.solid is None and dialog.canvas.solid is None
            assert 'ambiguous zero-thickness solid contact' in dialog.status.text()
            assert all(not button.isEnabled() for button in
                       (dialog.stl,dialog.threemf,dialog.route,dialog.bambu,dialog.play))
            until=time.monotonic()+15
            while dialog.worker.isRunning():
                app.processEvents();time.sleep(.005)
                assert time.monotonic()<until
            assert dialog.refresh.isEnabled()
            curved_factory=StructureFactory(unit='square',grid_x=3,grid_y=3,
                                             n_pts_per_side=2,perturbation=.35,seed=7)
            win.tab_struct.load_spec(curved_factory)
            surface=win.tab_surface
            index=[Path(path).name for path in surface._obj_files].index('3_Lung_quad_500.obj')
            asset=surface._obj_files[index]
            surface._mesh_settings[asset]=(48,0)
            surface._mesh_cache.pop(asset,None)
            surface.obj_combo.setCurrentIndex(index)
            surface.chk_follow.setChecked(True)
            surface.overscale.setValue(106)
            surface.apply_structure(curved_factory)
            surface._debounce.stop()
            surface._recompute()
            assert surface._mapping_error is None,surface._mapping_error
            assert len(surface._manufacturing_source[1])==48
            win.tab_manufacturing.open_source(True)
            dialog=win.tab_manufacturing.dialog
            dialog.controls['width'].setValue(250.)
            dialog.controls['depth'].setValue(250.)
            dialog.controls['diameter'].setValue(2.)
            until=time.monotonic()+90
            while dialog.solid is None:
                app.processEvents();time.sleep(.005)
                assert time.monotonic()<until,dialog.status.text()
                if dialog.worker is not None and not dialog.worker.isRunning():
                    app.processEvents()
                    assert dialog.solid is not None,dialog.status.text()
            assert dialog.canvas.three_d and dialog.solid.settings.up_axis=='y'
            assert dialog.source_network is surface.manufacturing_network
            assert dialog.solid.validate()>0
            exported=folder/'curved-mesh';exported.mkdir()
            for kind in ('stl','3mf','csv'):
                path=exported/('lung.'+kind)
                assert dialog.save(kind,str(path))==str(path),dialog.status.text()
                assert path.stat().st_size>100
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            faces=dialog.solid.faces;edges=faces[:,[0,1,1,2,2,0]].reshape(-1,2)
            adjacency=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),
                                 shape=(len(dialog.solid.vertices),)*2).tocsr()
            components,_=connected_components(adjacency,directed=False)
            print('[curved-mesh]',components,'mesh boundary components; formats exported',flush=True)
            win.tab_manufacturing.open_source(False)
            assert not win.tab_manufacturing.dialog.canvas.three_d
            win.close()
            until=time.monotonic()+15
            while win.isVisible():
                app.processEvents();time.sleep(.01)
                assert time.monotonic()<until
    print('[final30] exact visible random parameters, four-arc ring density, parsed square/200, seven actual pages, exact unsafe-solid refusal, unlocked UI, curved mesh export and single-row controls PASS')


if __name__=='__main__': main()
