"""Verify English startup and the explicit Chinese language switch."""
import os
import sys

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QMessageBox


def main():
    from studio.i18n import get_lang
    from studio.main import MainWindow

    app = QApplication([])
    assert get_lang() == 'en'
    window = MainWindow()
    assert get_lang() == 'en'
    assert window.lang_combo.currentText() == 'English'
    assert window.tabs.tabText(0) == 'Structure studio'

    messages = []
    original = QMessageBox.information
    QMessageBox.information = lambda _parent, _title, body: messages.append(body)
    try:
        window._show_about()
        assert 'Department of Macromolecular Science' in messages[-1]
        window.lang_combo.setCurrentIndex(0)
        assert get_lang() == 'zh'
        window._show_about()
        assert '复旦大学' in messages[-1]
    finally:
        QMessageBox.information = original
        window.close()
        app.processEvents()
    print('[gui_default_language] PASS')


if __name__ == '__main__':
    main()
