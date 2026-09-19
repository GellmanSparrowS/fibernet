"""Safe visual editor for periodic orientation rules; no executable expressions."""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QSpinBox, QTableWidget, QComboBox, QDialogButtonBox)
from fslab.cell_rules import validate_rule
from .i18n import get_lang, tr


class ExpansionDialog(QDialog):
    """Compact entry to built-in rules and the optional periodic table."""
    def __init__(self, rule='translate', pattern=None, parent=None):
        super().__init__(parent)
        from copy import deepcopy
        from PySide6.QtWidgets import QPushButton
        from fslab.cell_rules import RULES
        self.custom_rule = deepcopy(pattern)
        self.setWindowTitle(tr('expansion_rule'))
        self.resize(360, 160)
        layout = QVBoxLayout(self)
        self.rule = QComboBox()
        for key in RULES:
            self.rule.addItem(tr('rule_' + key), key)
        self.rule.setCurrentIndex(max(0, self.rule.findData(rule)))
        layout.addWidget(self.rule)
        self.edit = QPushButton(tr('rule_edit'))
        self.edit.clicked.connect(self._edit)
        layout.addWidget(self.edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _edit(self):
        dialog = RuleDialog(self.custom_rule, self)
        if dialog.exec():
            self.custom_rule = dialog.pattern()
            self.rule.setCurrentIndex(self.rule.findData('custom'))


class RuleDialog(QDialog):
    def __init__(self, pattern=None, parent=None):
        super().__init__(parent)
        zh = get_lang() == 'zh'
        self.setWindowTitle('自定义周期规律' if zh else 'Custom periodic rule')
        self.resize(540, 360)
        layout = QVBoxLayout(self)
        label = QLabel('设置一个周期内各位置的方向，向右、向上循环重复。镜像后旋转；不改变单元尺寸。'
                       if zh else 'Set orientations in one repeating block. Mirror, then rotate; cell size stays fixed.')
        label.setWordWrap(True)
        layout.addWidget(label)
        table = validate_rule(pattern)
        row = QHBoxLayout()
        self.rows, self.cols = QSpinBox(), QSpinBox()
        for spin, text, value in ((self.rows, '行' if zh else 'Rows', len(table)),
                                  (self.cols, '列' if zh else 'Columns', len(table[0]))):
            spin.setRange(1, 4)
            spin.setValue(value)
            row.addWidget(QLabel(text))
            row.addWidget(spin)
        layout.addLayout(row)
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self._fill(table)
        self.rows.valueChanged.connect(self._resize)
        self.cols.valueChanged.connect(self._resize)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _fill(self, pattern):
        self.table.setRowCount(self.rows.value())
        self.table.setColumnCount(self.cols.value())
        for j in range(self.rows.value()):
            for i in range(self.cols.value()):
                combo = QComboBox()
                for flip in (False, True):
                    for turn in range(4):
                        prefix = ('镜像 + ' if get_lang() == 'zh' else 'Mirror + ') if flip else ''
                        combo.addItem(prefix + str(turn * 90) + '°', {'turn': turn, 'flip': flip})
                if j < len(pattern) and i < len(pattern[j]):
                    cell = pattern[j][i]
                    combo.setCurrentIndex(cell['turn'] + 4 * int(cell['flip']))
                self.table.setCellWidget(j, i, combo)
        self.table.resizeColumnsToContents()

    def pattern(self):
        return [[self.table.cellWidget(j, i).currentData()
                 for i in range(self.table.columnCount())]
                for j in range(self.table.rowCount())]

    def _resize(self):
        self._fill(self.pattern())
