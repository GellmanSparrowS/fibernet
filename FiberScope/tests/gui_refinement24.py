"""Rerunnable interaction/model-search regression: python tests/gui_refinement24.py."""
import os
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from studio.main import MainWindow
from studio.cell_editor import CellCanvas
from fslab import StructureFactory
from fslab import structure
from fslab.learning import Regressor, MODEL_SPECS
from fslab.inverse import run_inverse
from fslab.model_inverse import run_model_inverse, PredictedStructure
from fslab.rl_process import run_external
from studio import ml_tab


def main():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix='fs24_') as folder:
        original_path = structure.CUSTOM_CELL_FILE
        original_cells = dict(structure.CUSTOM_CELLS)
        with patch.object(ml_tab, '_cache_dir', lambda: folder), patch.dict(os.environ, FIBERSCOPE_TRAINING_DIR=folder):
            win = MainWindow(lang='zh', mode='light')
            win.show()
            app.processEvents()
            try:
                ml, design = win.tab_ml, win.tab_design
                assert ml.unit_combo.currentData() == 'hexagon'
                assert ml.data_note.isHidden()
                assert design.target_combo.currentText() == 'C'
                assert not design.model_btn.isChecked()
                design.budget.setValue(1000000)
                assert design.budget.value() == 1000000
                assert win.tab_struct.rule_combo.isHidden() and win.tab_struct.rule_edit.isHidden()
                assert win.tab_surface.amp_slider.maximum() == 500
                from studio.surface_tab import OBJ_NAMES
                assert len(OBJ_NAMES) == 7 and not any('real_' in x for x in OBJ_NAMES)
                with patch.object(QMessageBox, 'about') as about, patch.object(QMessageBox, 'information') as info:
                    win._show_about()
                    assert '复旦大学高分子系杨云浩' in str(about.call_args or info.call_args)
                canvas = CellCanvas()
                canvas.resize(600, 600)
                canvas.tool = 'add'
                canvas.show()
                app.processEvents()
                QTest.mouseClick(canvas, Qt.LeftButton, pos=canvas._px(-.2, .5).toPoint())
                assert canvas.nodes and canvas.nodes[0][0] < 0, canvas.nodes
                nodes, edges = [[-.2,.5],[.8,.5],[1.8,.5]], [[0,1],[1,2]]
                structure.CUSTOM_CELL_FILE = str(Path(folder)/'cells.json')
                key = structure.save_custom_cell('outside24', nodes, edges)
                assert structure.valid_cell_spec(structure.CUSTOM_CELLS[key])
                assert '-0.2' in Path(structure.CUSTOM_CELL_FILE).read_text(encoding='utf-8')
                graph = StructureFactory(unit=key, grid_x=2, grid_y=2).build()
                assert np.isfinite(graph.node_positions()).all()
                canvas.close()
                from studio.rule_dialog import ExpansionDialog
                rule = ExpansionDialog('mirror', [[{'turn': 0, 'flip': False}]])
                assert rule.rule.currentData() == 'mirror'
                with patch('studio.rule_dialog.RuleDialog.exec', return_value=1), patch('studio.rule_dialog.RuleDialog.pattern', return_value=[[{'turn': 1, 'flip': True}]]):
                    rule.edit.click()
                assert rule.rule.currentData() == 'custom' and rule.custom_rule[0][0]['turn'] == 1
                rule.close()
                from fslab.surface_mapping import load_obj, map_cells
                for name in OBJ_NAMES:
                    vertices, faces = load_obj(Path('assets/obj')/name)
                    for unit in ('square', 'triangle', 'hexagon'):
                        positions, segments = map_cells(vertices, faces, [[1.5, -1.5]], unit)
                        assert len(segments) and np.isfinite(positions).all()
                ml.n_spin.setValue(10)
                ml.start_gen()
                start = time.monotonic()
                while ml._busy():
                    app.processEvents()
                    time.sleep(.01)
                    assert time.monotonic()-start < 120
                app.processEvents()
                assert ml.X.shape == (10,14) and np.isfinite(ml.Y).all(), ml.status.text()
                win.tabs.setCurrentWidget(ml)
                ml.apply_sample_btn.click()
                assert win.tabs.currentWidget() is win.tab_struct
                factory = StructureFactory(unit='hexagon', grid_x=2, grid_y=2, n_pts_per_side=2)
                for key in MODEL_SPECS:
                    model = Regressor(key)
                    model.train(ml.X, ml.Y, epochs=5)
                    seen = []
                    with patch('fslab.inverse.Engine2', side_effect=AssertionError('unexpected physics')):
                        result = run_model_inverse(factory, model, 'max_peak', budget=4,
                            callback=lambda rec, preview: seen.append(preview))
                    assert len(seen) == 4 and all(isinstance(p, PredictedStructure) for p in seen)
                    assert result['best_run'] is None and np.isfinite(result['best_prediction']).all()
                    design._evals, design._bests = [], []
                    for rec, preview in zip(result['records'], seen):
                        design._on_progress(rec, preview)
                    design._on_done(result)
                    assert design.canvas.static is not None and design.best_run is None
                    assert '预测' in design.lbl_target_info.text()
                    print('[refinement24] learned CEM + preview:', key)
                ml.model = model
                ml.trained_model = model
                ml.trained_unit = ml.unit_combo.currentData()
                with patch.object(QInputDialog, 'getItem', return_value=('max_peak', True)):
                    design.model_btn.click()
                assert design.selected_model is not None and design.target_combo.currentText() == 'max_peak'
                design.budget.setValue(4)
                design.start_run()
                start = time.monotonic()
                while design.worker.isRunning():
                    app.processEvents()
                    time.sleep(.01)
                    assert time.monotonic()-start < 30
                app.processEvents()
                assert design._result['best_preview'] is not None, design.status.text()
                assert design.model_btn.isEnabled()
                design.model_btn.click()
                assert design.selected_model is None and design.target_combo.currentText() == 'C'
                for algorithm in ('PPO','A2C','DQN','SAC','TD3','DDPG'):
                    seen = []
                    result = run_external(factory, 'max_peak', algorithm, {'n_steps': 4} if algorithm in ('PPO', 'A2C') else {'learning_starts': 1, 'batch_size': 4}, 12, 7, model=model,
                        callback=lambda rec, preview: seen.append(preview))
                    assert len(seen) == 12 and all(isinstance(p, PredictedStructure) for p in seen)
                    assert result['best_preview'] is not None and result['best_run'] is None
                    print('[refinement24] learned RL + preview:', algorithm)
                result = run_inverse(lambda *args: None, 'max_peak', budget=10005, fixed_unit='square',
                    pts=1, evaluator=lambda *args: (1., None))
                assert result['evaluations'] == 10005 and len(result['records']) == 2000
                assert result['records'][-1].eval_id == 10005
                print('[refinement24] defaults, navigation, boundary drawing, hidden author, 10005 evaluations PASS')
            finally:
                win.close()
                app.processEvents()
                structure.CUSTOM_CELL_FILE = original_path
                structure.CUSTOM_CELLS.clear()
                structure.CUSTOM_CELLS.update(original_cells)


if __name__ == '__main__':
    main()
