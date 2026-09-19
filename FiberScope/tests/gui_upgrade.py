"""2.2 end-to-end GUI flows using isolated datasets and asynchronous workers."""
import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from studio.main import MainWindow
from studio import ml_tab
from studio.rule_dialog import RuleDialog
from studio.algorithm_dialog import AlgorithmDialog
from fslab.learning import MODEL_SPECS, ACQUISITIONS


def main():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as folder:
        original = ml_tab._cache_dir
        ml_tab._cache_dir = lambda: folder
        win = MainWindow(lang='zh', mode='light')
        try:
            def wait(predicate, timeout=90):
                start = time.monotonic()
                while predicate():
                    app.processEvents()
                    time.sleep(.01)
                    assert time.monotonic()-start < timeout, 'worker timed out'
                app.processEvents()
            tab = win.tab_ml
            assert tab.mode_combo.currentData() == 'physics'
            assert tab.mode_combo.count() == 1
            assert '方' in tab.unit_combo.itemText(tab.unit_combo.findData('square'))
            tab.n_spin.setValue(10)
            tab.start_gen()
            wait(tab._busy)
            assert tab.X.shape == (10,14) and np.isfinite(tab.Y).all(), tab.status.text()
            assert len(tab.sample_specs) == 10
            tab.mode_combo.setCurrentIndex(tab.mode_combo.findData('physics'))
            tab.start_gen()
            wait(tab._busy)
            assert np.isfinite(tab.Y).all(), tab.status.text()
            tab.algorithm, tab.parameters = 'ridge', {'alpha': 1.}
            tab.start_train()
            wait(tab._busy)
            assert tab.model is not None, tab.status.text()
            assert tab.history['validation_count'] == 2
            assert len(tab.scatter.getData()[0]) == 2
            tab.scatter_plot.setRange(xRange=(0, 1), yRange=(0, 1))
            tab._update_scatter()
            xs, ys = tab.scatter.getData()
            ranges = tab.scatter_plot.viewRange()
            assert min(xs) >= ranges[0][0] and max(xs) <= ranges[0][1]
            assert min(ys) >= ranges[1][0] and max(ys) <= ranges[1][1]
            dialog = AlgorithmDialog(MODEL_SPECS, 'deep_mlp', acquisitions=ACQUISITIONS)
            assert set(dialog.controls) == {'hidden','depth'}
            dialog.combo.setCurrentIndex(dialog.combo.findData('ridge'))
            assert set(dialog.controls) == {'alpha'}
            tab.algorithm, tab.parameters = 'deep_mlp', {'hidden': 24, 'depth': 3}
            tab._sync_model_controls()
            assert tab.hidden_spin.value() == 24 and tab.lr_combo.isEnabled()
            tab.hidden_spin.setValue(20)
            assert tab.parameters['hidden'] == 20
            tab.algorithm, tab.parameters = 'ridge', {'alpha': 1.}
            tab._sync_model_controls()
            assert not tab.lr_combo.isEnabled()
            rule = RuleDialog([[{'turn':1,'flip':True}]])
            rule.rows.setValue(2)
            rule.cols.setValue(3)
            assert rule.pattern()[0][0] == {'turn':1,'flip':True}
            assert len(rule.pattern()[1]) == 3
            rule.close()
            dialog.close()
            print('[gui22] generate/physical/train/predict provenance, model forms and rule resizing PASS')
        finally:
            win.close()
            app.processEvents()
            ml_tab._cache_dir = original


if __name__ == '__main__':
    main()
