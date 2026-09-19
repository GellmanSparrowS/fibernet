"""Repeatable offscreen layout audit; optional captures are local QA only.

Run: python scripts/audit_ui_layout.py [--capture-dir _tmp/finals_qa]
"""
import argparse
import json
import os
import sys
os.environ['QT_QPA_PLATFORM'] = 'windows' if '--native' in sys.argv else 'offscreen'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


class LayoutAudit:
    def run(self, capture_dir=None, native=False):
        from studio.main import MainWindow
        from studio.cell_editor import CellEditorDialog
        app = QApplication.instance() or QApplication([])
        win = MainWindow(lang='zh', mode='light')
        if native:
            win.setAttribute(Qt.WA_DontShowOnScreen)
        win.resize(1366, 900)
        win.show()
        win.ai_panel.show()
        win.tab_struct.push_spec()
        app.processEvents()
        records = []
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
        for name in ('structure', 'features', 'ml', 'design', 'surface'):
            win.ai_panel._navigate(name)
            app.processEvents()
            records.append(dict(page=name, width=win.width(), height=win.height(),
                                page_width=win.tabs.currentWidget().width()))
            if capture_dir:
                win.grab().save(os.path.join(capture_dir, name+'.png'))
        dlg = CellEditorDialog(mode='light')
        if native:
            dlg.setAttribute(Qt.WA_DontShowOnScreen)
        dlg.show()
        dlg._build_preview()
        app.processEvents()
        records.append(dict(page='workbench', width=dlg.width(), height=dlg.height(),
                            preview_width=dlg.preview.width(), preview_height=dlg.preview.height()))
        assert dlg.save_btn.isVisible() and dlg.canvas.height() >= 300
        if capture_dir:
            dlg.grab().save(os.path.join(capture_dir, 'workbench.png'))
        dlg.close()
        win.close()
        print(json.dumps(records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-dir')
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    LayoutAudit().run(args.capture_dir, args.native)
