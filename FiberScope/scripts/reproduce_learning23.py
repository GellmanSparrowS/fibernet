"""Reproduce actual physical-data training and plot geometry. Run directly."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtWidgets import QApplication
from studio.main import MainWindow
from studio import ml_tab


def main():
    app = QApplication.instance() or QApplication([])
    folder = Path('_cache/qa23')
    folder.mkdir(parents=True, exist_ok=True)
    ml_tab._cache_dir = lambda: str(folder)
    win = MainWindow(lang='zh', mode='light')
    win.resize(1366, 900)
    win.show()
    tab = win.tab_ml
    tab.show()
    for parent in win.findChildren(__import__('PySide6.QtWidgets', fromlist=['QTabWidget']).QTabWidget):
        if parent.indexOf(tab) >= 0:
            parent.setCurrentWidget(tab)

    def wait():
        start = time.monotonic()
        while tab._busy():
            app.processEvents()
            time.sleep(.01)
            if time.monotonic() - start > 300:
                raise TimeoutError(tab.status.text())
        for _ in range(10):
            app.processEvents()
            time.sleep(.01)

    try:
        tab.mode_combo.setCurrentIndex(tab.mode_combo.findData('physics'))
        tab.start_gen()
        wait()
        assert tab.X is not None and np.isfinite(tab.Y).all(), tab.status.text()
        for model in ('mlp', 'random_forest'):
            tab.algorithm, tab.parameters = model, {}
            tab.start_train()
            wait()
            assert tab.model is not None, tab.status.text()
            x, y = tab.scatter.getData()
            print(model, tab.status.text(), 'points', len(x), 'range', tab.scatter_plot.viewRange(),
                  'data', [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))],
                  'geometry', tab.scatter_plot.geometry().getRect(), 'visible', tab.scatter_plot.isVisible(), flush=True)
            win.grab().save(str(folder / (model + '.png')))
    finally:
        win.close()
        app.processEvents()


if __name__ == '__main__':
    main()
