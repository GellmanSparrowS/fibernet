"""Positive integer entry without a product-level iteration ceiling."""
from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import QLineEdit


class CountInput(QLineEdit):
    def __init__(self, value=40, parent=None):
        super().__init__(parent)
        self._last_value = int(value)
        self.setValidator(QRegularExpressionValidator(QRegularExpression('[1-9][0-9]*'), self))
        self.setValue(value)
        self.editingFinished.connect(self._normalize)

    def value(self):
        text = self.text().strip()
        if self.hasAcceptableInput():
            try:
                self._last_value = int(text)
            except ValueError:
                pass
        return self._last_value

    def setValue(self, value):
        value = int(value)
        if value < 1:
            raise ValueError('iteration count must be positive')
        self._last_value = value
        self.setText(str(value))

    def _normalize(self):
        self.setValue(self.value())
