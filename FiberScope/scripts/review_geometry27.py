"""Save local-only review views and a pyramid print demo; no printer job is sent."""
import os
import sys
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from studio.main import MainWindow
from studio.manufacturing_dialog import ManufacturingDialog
from fslab.print_export import PrintSettings, build_solid, export_solid


class GeometryReview:
    def run(self):
        root=Path(__file__).resolve().parents[1]
        folder=root/'_tmp'/'review27'
        folder.mkdir(parents=True,exist_ok=True)
        app=QApplication.instance() or QApplication([])
        win=MainWindow(lang='zh',mode='light'); win.show()
        tab=win.tab_surface; win.tabs.setCurrentWidget(tab)
        for index in range(tab.obj_combo.count()):
            tab.obj_combo.setCurrentIndex(index); tab._compute_mapping()
            app.processEvents()
            tab.canvas.grab().save(str(folder/('surface_%d.png'%index)))
        v,f,factory,scale=tab._manufacturing_source
        network=tab.manufacturing_network
        settings=PrintSettings(width=200,depth=200,curved=True,up_axis=network.up_axis)
        solid=build_solid(network,settings)
        dialog=ManufacturingDialog(factory,surface=(v,f,scale),auto_build=False,network=network)
        dialog._done(network,solid); dialog.show(); dialog.view3d.click(); app.processEvents()
        dialog.canvas.grab().save(str(folder/'pyramid_solid.png'))
        export_solid(folder/'pyramid.3mf',solid)
        export_solid(folder/'pyramid.stl',solid)
        dialog.close(); win.close(); app.processEvents()
        print('[review27] local views and pyramid STL/3MF:',folder)


if __name__=='__main__': GeometryReview().run()
