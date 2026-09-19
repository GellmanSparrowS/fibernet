"""Adjust OBJ reduction and conforming quad subdivision without editing sources."""
from PySide6.QtWidgets import QDialog,QFormLayout,QSpinBox,QDialogButtonBox
from .i18n import get_lang


class MeshDialog(QDialog):
    def __init__(self,target=1500,levels=0,parent=None):
        super().__init__(parent)
        zh=get_lang()=='zh'
        self.setWindowTitle('网格设置' if zh else 'Mesh settings')
        layout=QFormLayout(self)
        self.target=QSpinBox(); self.target.setRange(12,10000); self.target.setValue(target)
        self.levels=QSpinBox(); self.levels.setRange(0,3); self.levels.setValue(levels)
        layout.addRow('抽稀目标面数' if zh else 'Reduction face budget',self.target)
        layout.addRow('四边面细分等级' if zh else 'Quad subdivision levels',self.levels)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self): return self.target.value(),self.levels.value()
