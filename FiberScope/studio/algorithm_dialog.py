"""Schema-driven algorithm settings shared by learning and inverse design."""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox,
                              QSpinBox, QDoubleSpinBox, QDialogButtonBox, QLabel, QWidget)
from .i18n import get_lang


class AlgorithmDialog(QDialog):
    def __init__(self, specs, selected, parameters=None, acquisitions=None,
                 acquisition='random', parent=None):
        super().__init__(parent)
        self.specs, self.values = specs, dict(parameters or {})
        self.controls = {}
        self.lang = 0 if get_lang() == 'zh' else 1
        self.setWindowTitle(('算法与参数', 'Algorithms and parameters')[self.lang])
        self.resize(440, 360)
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        for key, spec in specs.items():
            self.combo.addItem(spec[self.lang], key)
        self.combo.setCurrentIndex(max(0, self.combo.findData(selected)))
        layout.addWidget(self.combo)
        self.form_widget = QWidget()
        self.form = QFormLayout(self.form_widget)
        layout.addWidget(self.form_widget)
        self.acquisition = QComboBox()
        if acquisitions:
            layout.addWidget(QLabel(('主动学习选样策略', 'Active learning acquisition')[self.lang]))
            for key, names in acquisitions.items():
                self.acquisition.addItem(names[self.lang], key)
            self.acquisition.setCurrentIndex(max(0, self.acquisition.findData(acquisition)))
            layout.addWidget(self.acquisition)
        self.combo.currentIndexChanged.connect(self._rebuild)
        self._rebuild()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _rebuild(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.controls = {}
        key = self.combo.currentData()
        for name, (default, low, high, zh, en) in self.specs[key][2].items():
            spin = QSpinBox() if isinstance(default, int) else QDoubleSpinBox()
            if isinstance(spin, QDoubleSpinBox):
                spin.setDecimals(6)
            spin.setRange(low, high)
            spin.setValue(self.values.get(key, {}).get(name, default))
            spin.valueChanged.connect(lambda value, k=key, n=name: self.values.setdefault(k, {}).update({n: value}))
            self.form.addRow((zh, en)[self.lang], spin)
            self.controls[name] = spin

    def selection(self):
        return self.combo.currentData(), {k: v.value() for k, v in self.controls.items()}
