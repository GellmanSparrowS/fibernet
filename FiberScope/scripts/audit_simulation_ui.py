"""Bounded offscreen simulation/playback regression check.

From the repository root, after installing FiberScope/requirements.txt:
    python FiberScope/scripts/audit_simulation_ui.py --capture-dir demo_ui

Captures are optional local QA artifacts, not release screenshots. This checks
Qt rendering and interactions, not native-window or frozen-build portability.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QStyle, QStyleOptionButton,
                               QStyleOptionComboBox)

from fslab.structure import StructureFactory
from studio.main import MainWindow
from studio import sim_tab
from studio.sim_tab import FPS


def control_fit(tab):
    checks = {}
    for label, widget in (("view", tab.view_combo), ("fps", tab.speed_combo)):
        option = QStyleOptionComboBox()
        widget.initStyleOption(option)
        field = widget.style().subControlRect(
            QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, widget)
        checks[label] = dict(text=widget.currentText(),
                            text_width=widget.fontMetrics().horizontalAdvance(widget.currentText()),
                            available_width=field.width())
    widget = tab.loop_btn
    option = QStyleOptionButton()
    widget.initStyleOption(option)
    field = widget.style().subElementRect(QStyle.SE_PushButtonContents, option, widget)
    checks["loop"] = dict(text=widget.text(),
                          text_width=widget.fontMetrics().horizontalAdvance(widget.text()),
                          available_width=field.width())
    assert all(c["text_width"] <= c["available_width"] for c in checks.values()), checks
    assert tab.slider.width() >= 120, tab.slider.width()
    return checks


def check_playback(tab):
    for _ in range(3):
        assert not tab._play_timer.isActive()
        QTest.mouseClick(tab.play_btn, Qt.LeftButton)
        assert tab._play_timer.isActive()
        QTest.mouseClick(tab.play_btn, Qt.LeftButton)
        assert not tab._play_timer.isActive()
    for index, fps in enumerate(FPS):
        tab.speed_combo.setCurrentIndex(index)
        assert tab._play_timer.interval() == int(round(1000 / fps))
    tab.speed_combo.setCurrentIndex(FPS.index(48))
    if tab.loop_btn.isChecked():
        QTest.mouseClick(tab.loop_btn, Qt.LeftButton)
    tab.slider.setValue(tab.slider.maximum())
    tab._play_step()
    assert not tab._play_timer.isActive()
    assert tab.slider.value() == tab.slider.maximum()
    QTest.mouseClick(tab.loop_btn, Qt.LeftButton)
    tab._play_step()
    assert tab.slider.value() == 0
    tab.slider.setValue(tab.slider.maximum())
    tab.retranslate()


def run(capture_dir=None):
    output = Path(capture_dir) if capture_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    records = []
    with tempfile.TemporaryDirectory(prefix="fiberscope-ui-") as cache:
        previous_cache = sim_tab.CACHE_DIR
        sim_tab.CACHE_DIR = cache
        win = None
        try:
            win = MainWindow(lang="en", mode="light")
            win.resize(1366, 900)
            win.show()
            win.tab_struct.load_spec(StructureFactory(
                unit="square", grid_x=2, grid_y=2, n_pts_per_side=2, seed=7))
            win.tabs.setCurrentWidget(win.tab_sim)
            tab = win.tab_sim
            tab.stretch.setValue(1.1)
            tab.run_sync()
            tab._play_timer.stop()
            for language in ("en", "zh"):
                win.lang_combo.setCurrentIndex(1 if language == "en" else 0)
                check_playback(tab)
                for size in ((1366, 900), (1120, 700)):
                    win.resize(*size)
                    for _ in range(5):
                        app.processEvents()
                    assert (win.width(), win.height()) == size
                    record = dict(language=language, size=list(size),
                                  controls=control_fit(tab), slider_width=tab.slider.width(),
                                  frames=tab.run.n_frames, nodes=tab.run.frames_xy.shape[1],
                                  edges=tab.run.n_edges, playback_checks="passed")
                    if output:
                        path = output / ("simulation-%s-%d.png" % (language, size[0]))
                        assert win.grab().save(str(path))
                        record["screenshot"] = path.name
                    records.append(record)
        finally:
            if win is not None:
                win.close()
                app.processEvents()
            sim_tab.CACHE_DIR = previous_cache
    if output:
        (output / "simulation-ui.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", help="Optional directory for PNGs and JSON evidence")
    run(parser.parse_args().capture_dir)
