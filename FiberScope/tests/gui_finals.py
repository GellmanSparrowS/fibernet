"""Finals UI regression. Run: python tests/gui_finals.py."""
import json
import os
import sys
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtWidgets import QApplication


def main():
    from studio.main import MainWindow
    from studio.ai_assistant import _ToolRequest
    from studio.cell_editor import CellEditorDialog
    from fslab.structure import CELL_UNITS
    app = QApplication([])
    win = MainWindow(lang='zh', mode='light')
    win.show()
    app.processEvents()
    panel = win.ai_panel
    panel.cfg = dict(panel.cfg, api_key='')
    panel.input.setText('打开特征分析')
    panel.send()
    assert win.tabs.currentWidget() is win.tab_features
    assert panel.worker is None
    # Page objects remain authoritative even if the visible tab order changes.
    win.tabs.tabBar().moveTab(0, 2)
    panel.registry['set_tab'](tab='structure')
    assert win.tabs.currentWidget() is win.tab_struct
    req = _ToolRequest('set_expansion', {'rule': 'rotate', 'grid_x': 2, 'grid_y': 3})
    panel._run_tool(req)
    result = json.loads(req.result)
    assert result['ok'] and result['structure']['expansion_rule'] == 'rotate'
    assert result['structure']['grid_y'] == 3
    req = _ToolRequest('edit_line_point', {'index': 0, 'dx': .01, 'dy': .02})
    panel._navigate('features')
    panel._run_tool(req)
    assert win.tabs.currentWidget() is win.tab_struct
    assert 'error' not in json.loads(req.result)
    panel._navigate('replay')
    assert win.tab_design.replay_toggle.isChecked()
    panel._navigate('design')
    assert not win.tab_design.replay_toggle.isChecked()
    win.tabs.tabBar().moveTab(2, 0)
    dlg = CellEditorDialog(mode='light')
    spec = CELL_UNITS['octagon']
    dlg.canvas.set_spec(spec['nodes'], spec['edges'])
    dlg.rule.setCurrentIndex(2)
    dlg.gx.setValue(2)
    dlg.gy.setValue(3)
    dlg._build_preview()
    assert dlg._preview_factory.expansion_rule == 'mirror'
    assert dlg.preview.graph_arrays is not None
    from studio.network_canvas import NetworkCanvas
    from studio.theme import colors
    from PySide6.QtGui import QColor
    import numpy as np
    canvas = NetworkCanvas(mode='light')
    canvas.resize(400, 300)
    canvas.set_static(np.array([[0., 0.], [10., 0.], [5., 8.]]),
                      np.array([[0, 1], [1, 2], [2, 0]]), np.array([0]), np.array([1]))
    image = canvas._render_scene().toImage()
    scale, ox, oy = canvas._transform()
    expected = QColor(colors('light')['warn']).rgb()
    assert image.pixel(round(ox), round(oy)) == expected, 'grip marker not transformed'
    assert image.pixel(0, 0) != expected, 'grip marker leaked to raw model origin'
    import tempfile
    from fslab import structure as structures
    original_path, original_cells = structures.CUSTOM_CELL_FILE, dict(structures.CUSTOM_CELLS)
    with tempfile.TemporaryDirectory(prefix='fs_name_test_') as folder:
        try:
            structures.CUSTOM_CELL_FILE = os.path.join(folder, 'cells.json')
            saved_keys = []
            for _ in range(2):
                editor = CellEditorDialog(mode='light')
                editor.name_zh.setText('Same name')
                editor.name_en.setText('square')
                editor.rule.setCurrentIndex(2)
                editor.amplitude.setValue(1.7)
                editor._save()
                assert editor.saved_key and editor.saved_key != 'square'
                saved_keys.append(editor.saved_key)
                editor.deleteLater()
            assert len(set(saved_keys)) == 2, 'new cell overwrote a same-name cell'
            structures.load_custom_cells()
            for key in saved_keys:
                assert structures.CUSTOM_CELLS[key]['settings']['amplitude'] == 1.7
                assert structures.CUSTOM_CELLS[key]['settings']['expansion_rule'] == 'mirror'
        finally:
            structures.CUSTOM_CELL_FILE = original_path
            structures.CUSTOM_CELLS.clear()
            structures.CUSTOM_CELLS.update(original_cells)
            structures._CELLS_REGISTERED.clear()
    win.lang_combo.setCurrentIndex(1)
    app.processEvents()
    assert panel.target_combo.itemText(0) == 'Automatic'
    dlg.close()
    win.close()
    print('[gui_finals] local navigation, reordered tabs, spectrum tools and actual preview ok')


if __name__ == '__main__':
    main()
