"""Curved GUI identity, print orientation and hidden workflow controls."""
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from studio.main import MainWindow
from studio.manufacturing_dialog import ManufacturingWorker, ManufacturingCanvas
from fslab.print_export import PrintSettings


def main():
    app=QApplication.instance() or QApplication([])
    win=MainWindow(lang='zh',mode='light')
    win.show(); app.processEvents()
    ai=win.ai_panel
    assert ai.workflow_btn.isHidden() and ai.workflow_stop.isHidden()
    with patch.object(ai.workflow,'cancel') as cancel:
        ai.workflow.running=True
        ai.input.setText('停止流程'); ai.send()
        assert cancel.called
        ai.workflow.running=False
    tab=win.tab_surface
    win.tabs.setCurrentWidget(tab)
    tab.obj_combo.setCurrentIndex(tab.obj_combo.count()-1)
    tab._compute_mapping()
    app.processEvents()
    assert tab.manufacturing_network.up_axis=='z'
    expected_network=tab.manufacturing_network
    got=[]; failed=[]
    worker=ManufacturingWorker(tab._manufacturing_source[2],tab._manufacturing_source[:2]+(1.06,),
        PrintSettings(width=200,depth=200,curved=True),network=tab.manufacturing_network)
    worker.completed.connect(lambda n,s:got.append((n,s)))
    worker.failed.connect(failed.append)
    with patch('studio.manufacturing_dialog.compile_surface',side_effect=AssertionError('unexpected recompilation')):
        worker.run()
    assert not failed and got,failed
    canvas=ManufacturingCanvas(); canvas.resize(800,600)
    canvas.network,canvas.solid=got[0]; canvas.three_d=True
    canvas.show(); app.processEvents(); canvas.grab()
    assert canvas.solid is got[0][1]
    assert len(canvas.solid.faces)==len(got[0][1].faces)
    canvas.close()
    assert got[0][0] is expected_network
    assert np.ptp(got[0][1].vertices,axis=0)[2]>100
    tab.obj_combo.setCurrentIndex(0); tab._compute_mapping()
    assert tab.manufacturing_network.up_axis=='y'
    settings=PrintSettings(width=250,depth=250,curved=True,up_axis='y')
    points=settings.coordinates(tab.manufacturing_network)
    old=np.ptp(tab.manufacturing_network.positions,axis=0)
    new=np.ptp(points,axis=0)
    assert np.allclose(new/new.max(),old[[0,2,1]]/old.max())
    win.close(); app.processEvents()
    print('[gui-geometry27] hidden controls, text cancellation, exact surface handoff and Y/Z print orientation PASS')


if __name__=='__main__': main()
