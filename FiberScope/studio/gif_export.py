"""Bounded-memory, high-resolution GIF export driven by the Qt event loop."""
import os
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QProgressDialog
from PIL import Image, GifImagePlugin
from .network_canvas import NetworkCanvas
from .i18n import get_lang


class SimulationGif:
    def __init__(self, tab, path, width=1600, fps=48):
        self.tab, self.path, self.fps = tab, Path(path), fps
        self.temporary = self.path.with_name(self.path.name+'.tmp')
        self.stream = self.temporary.open('wb')
        self.index = 0
        self.active = True
        self.canvas = NetworkCanvas(mode=tab.mode)
        self.canvas.resize(width, round(width*.625))
        self.canvas.set_color_mode(tab.canvas.color_mode)
        self.canvas.set_data(tab.run, tab.perc if tab.canvas.color_mode=='percolation' else None)
        self.canvas.set_legend(tab.canvas.legend)
        self.dialog = QProgressDialog('导出 GIF' if get_lang()=='zh' else 'Export GIF',
                                     '取消' if get_lang()=='zh' else 'Cancel', 0, tab.run.n_frames, tab)
        self.dialog.setMinimumDuration(0)
        self.dialog.canceled.connect(self.cancel)
        self.timer = QTimer(tab)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.step)
        self.timer.start()

    def step(self):
        try:
            self.canvas.set_frame(self.index)
            qimage = self.canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
            frame = Image.frombytes('RGBA', (qimage.width(), qimage.height()),
                                    bytes(qimage.constBits()), 'raw', 'RGBA', qimage.bytesPerLine()).convert('RGB')
            palette = frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            if self.index == 0:
                header, _ = GifImagePlugin.getheader(palette, info={'loop':0})
                for block in header: self.stream.write(block)
            delay = 10*(round((self.index+1)*100/self.fps)-round(self.index*100/self.fps))
            for block in GifImagePlugin.getdata(palette, duration=delay, disposal=2, include_color_table=True):
                self.stream.write(block)
            self.index += 1
            self.dialog.setValue(self.index)
            if self.index == self.tab.run.n_frames:
                self.stream.write(b';')
                self.stream.close()
                os.replace(self.temporary, self.path)
                self.finish()
                self.tab.status.setText(str(self.path))
        except Exception as exc:
            self.cancel()
            self.tab._export_fail(exc)

    def finish(self):
        if not self.active: return
        self.active = False
        self.timer.stop()
        self.timer.deleteLater()
        self.canvas.deleteLater()
        self.dialog.close()
        self.dialog.deleteLater()
        self.tab.act_gif.setEnabled(True)

    def cancel(self):
        if not self.active: return
        self.stream.close()
        self.temporary.unlink(missing_ok=True)
        self.finish()
