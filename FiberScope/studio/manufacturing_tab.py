"""Main fabrication workspace with explicit planar and surface sources."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from .i18n import get_lang


class ManufacturingTab(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.dialog = None
        self.layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.planar = QPushButton()
        self.curved = QPushButton()
        self.planar.clicked.connect(lambda:self.open_source(False))
        self.curved.clicked.connect(lambda:self.open_source(True))
        row.addWidget(self.planar)
        row.addWidget(self.curved)
        self.play=QPushButton()
        self.play.setEnabled(False)
        self.play.clicked.connect(self._play)
        row.addWidget(self.play)
        row.addStretch()
        self.layout.addLayout(row)
        self.layout.addStretch()
        self.retranslate()

    def retranslate(self):
        zh = get_lang()=='zh'
        self.planar.setText('使用当前平面结构' if zh else 'Use planar structure')
        self.curved.setText('使用当前曲面结构' if zh else 'Use surface structure')
        self.play.setText('播放一笔画路径' if zh else 'Play continuous route')
        if self.dialog: self.dialog.retranslate()

    def _play(self):
        if self.dialog and self.dialog.play.isEnabled(): self.dialog.play_route()

    def mount(self, dialog):
        if self.dialog is not None and self.dialog is not dialog:
            self.dialog.timer.stop()
            self.layout.removeWidget(self.dialog)
            self.dialog.hide()
            self.dialog.deleteLater()
        self.dialog = dialog
        dialog.readiness_changed.connect(self.play.setEnabled)
        self.play.setEnabled(dialog.solid is not None and dialog.stl.isEnabled())
        dialog.play.hide()
        dialog.setParent(self)
        dialog.setWindowFlags(Qt.Widget)
        self.layout.insertWidget(1,dialog,1)
        dialog.show()
        self.window.tabs.setCurrentWidget(self)

    def open_source(self, curved):
        if self.dialog and self.dialog.worker and self.dialog.worker.isRunning():
            self.window.tabs.setCurrentWidget(self)
            return
        from .manufacturing_dialog import ManufacturingDialog
        source = self.window.tab_surface if curved else self.window.tab_struct
        if curved:
            source._compute_mapping()
            if source._mapping_error: return
            v,f,factory,overscale = source._manufacturing_source
            dialog = ManufacturingDialog(factory,(v,f,overscale),self.window.mode,self,
                                         network=source.manufacturing_network)
        else:
            dialog = ManufacturingDialog(source.collect_factory(),mode=self.window.mode,parent=self)
        # Both aliases refer to the live workspace, avoiding stale deleted dialogs.
        self.window.tab_struct.manufacturing_dialog = dialog
        self.window.tab_surface.manufacturing_dialog = dialog
        self.mount(dialog)

    def set_mode(self, mode):
        if self.dialog:
            self.dialog.mode = self.dialog.canvas.mode = mode
            self.dialog.canvas.update()
