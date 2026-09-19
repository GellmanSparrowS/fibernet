"""Real asynchronous manufacturing dialog, views, route and exports."""
import os
import sys
import time
import tempfile
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from studio.main import MainWindow


def main():
    app=QApplication.instance() or QApplication([])
    window=MainWindow(lang='zh',mode='light')
    window.show()
    window.tab_struct.manufacturing_btn.click()
    dialog=window.tab_struct.manufacturing_dialog
    deadline=time.monotonic()+90
    while dialog.solid is None:
        app.processEvents()
        time.sleep(.01)
        assert time.monotonic()<deadline,dialog.status.text()
    assert dialog.network.health['odd']==0
    assert dialog.view2d.isChecked() and dialog.stl.isEnabled()
    assert dialog.view2d.isHidden() and dialog.view3d.isHidden()
    assert not dialog.canvas.three_d
    dialog.play.click()
    app.processEvents()
    assert dialog.timer.isActive()
    with tempfile.TemporaryDirectory() as folder:
        assert dialog.save('3mf',str(Path(folder)/'model.3mf'))
        assert dialog.save('csv',str(Path(folder)/'route.csv'))
    dialog.controls['width'].setValue(80.)
    assert not dialog.stl.isEnabled()
    dialog.rebuild()
    window.close()
    assert window.isVisible(), 'window closed while a manufacturing thread was active'
    deadline=time.monotonic()+20
    while window.isVisible():
        app.processEvents()
        time.sleep(.02)
        assert time.monotonic()<deadline
    assert not dialog.worker.isRunning()
    app.processEvents()
    print('[gui-manufacturing] entry, async solid, distinct 2D/3D, route playback, export and stale-state guard PASS')


if __name__=='__main__':
    main()
