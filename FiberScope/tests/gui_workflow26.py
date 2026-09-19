"""Real asynchronous full workflow, cancellation, UI focus, and atomic outputs."""
import os
import sys
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from studio.main import MainWindow
from fslab.structure import StructureFactory
from studio import workflow, ml_tab


def wait(app,predicate,seconds=240):
    until=time.monotonic()+seconds
    while not predicate():
        app.processEvents()
        if time.monotonic()>until:
            raise AssertionError('workflow timeout')
        time.sleep(.01)
    app.processEvents()


def main():
    app=QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix='workflow26_') as folder:
        with patch.object(workflow,'writable_data_dir',lambda _:folder),patch.object(ml_tab,'_cache_dir',lambda:folder):
            win=MainWindow(lang='zh',mode='light')
            win.show(); win.ai_panel.show(); app.processEvents()
            win.tab_struct.load_spec(StructureFactory(grid_x=1,grid_y=1,n_pts_per_side=1))
            ai=win.ai_panel
            ai.input.setText('')
            ai.send_btn.setFocus(); app.processEvents()
            before=ai.input.grab().toImage().pixelColor(6,20).name()
            QTest.mouseClick(ai.input,Qt.LeftButton); app.processEvents()
            after=ai.input.grab().toImage().pixelColor(6,20).name()
            assert before==after,(before,after)
            runner=ai.workflow
            runner.start(samples=10,iterations=2,open_slicer=False)
            wait(app,lambda:not runner.running)
            result=runner.status()
            assert result['state']=='complete',result
            assert len(result['stages'])==11 and all(s['state']=='complete' for s in result['stages']),result
            assert result['printer_state']=='files_ready'
            assert win.tab_manufacturing.dialog.source_network is runner.network
            for name in ('initial.json','features.csv','optimized.json','optimized_features.csv',
                         'optimized_curve.csv','FiberScope.3mf','FiberScope.stl','path.csv','progress.json'):
                assert (runner.folder/name).stat().st_size>0,name
            saved=json.loads((runner.folder/'progress.json').read_text(encoding='utf-8'))
            assert saved['topology']=='topnet26'
            assert win.tab_ml.model is not None and win.tab_design._result['best_spec']
            print('[workflow26] real simulation, features, physics dataset/training, C inverse, verification, surface, solid and files PASS',flush=True)
            runner.start(samples=10,iterations=2,open_slicer=False)
            runner.cancel()
            assert runner.status()['state']=='cancelled'
            assert all(win.tabs.widget(i).isEnabled() for i in range(win.tabs.count()))
            win.close(); app.processEvents()
            print('[workflow26] focus color, cancellation, per-stage journal and UI unlock PASS',flush=True)


if __name__=='__main__':
    main()
