"""Separate 2D/3D fabrication views backed by one verified multigraph and solid."""
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
from PySide6.QtCore import Qt, QPointF, QLineF, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath, QPixmap, QPolygonF
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QDoubleSpinBox, QFileDialog, QProgressBar, QGridLayout)
from fslab.manufacturing import compile_planar, compile_surface
from fslab.print_export import PrintSettings, build_solid, export_solid, export_route
from .i18n import get_lang
from .theme import colors


class ManufacturingWorker(QThread):
    completed = Signal(object, object)
    failed = Signal(str)
    progress = Signal(int)

    def __init__(self, factory, surface, settings, parent=None, network=None):
        super().__init__(parent)
        self.factory, self.surface, self.settings = factory, surface, settings
        self.source_network = network
        self.stop = False

    def request_stop(self):
        self.stop = True

    def run(self):
        try:
            network = self.source_network
            if network is None:
                network = (compile_planar(self.factory) if self.surface is None else
                    compile_surface(*self.surface[:2], self.factory, self.surface[2]))
            from fslab.solid_process import build_isolated
            solid = build_isolated(network, self.settings, self.progress.emit, lambda: self.stop)
            if self.stop:
                raise InterruptedError('solid export cancelled')
            self.completed.emit(network, solid)
        except Exception as exc:
            self.failed.emit(str(exc))


class ManufacturingCanvas(QWidget):
    def __init__(self, mode='dark', parent=None):
        super().__init__(parent)
        self.mode, self.network, self.solid = mode, None, None
        self.three_d = False
        self.yaw, self.pitch, self.zoom = -20., 50., 1.
        self.fraction = 0.
        self._drag = None
        self._cache_key = None
        self._route_count = 0
        self.setMinimumSize(320, 300)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = colors(self.mode)
        painter.fillRect(self.rect(), QColor(palette['bg']))
        if self.solid is None:
            return
        centers = self.solid.centers
        origin = (centers.min(0)+centers.max(0))/2
        xyz = centers-origin
        rotation = np.eye(3)
        if self.three_d:
            basis = np.array([[1.,0,0],[0,0,1.],[0,-1.,0]]) if self.solid.settings.curved else np.eye(3)
            y, p = np.radians([self.yaw, self.pitch])
            rotation = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1.]])
            rotation = np.array([[1.,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])@basis@rotation
            xyz = xyz@rotation.T
        span = np.ptp(xyz[:,:2],axis=0)+self.solid.settings.diameter
        scale = min((self.width()-60)/max(span[0],1.),(self.height()-60)/max(span[1],1.))*self.zoom
        view_center = (xyz[:,:2].min(0)+xyz[:,:2].max(0))/2
        xy = (xyz[:,:2]-view_center)*[scale,-scale]+[self.width()/2,self.height()/2]
        width = max(.5,self.solid.settings.diameter*scale)
        cache_key = (id(self.solid),self.width(),self.height(),self.three_d,self.yaw,self.pitch,self.zoom,self.mode)
        if self._cache_key != cache_key:
            self._cache_key = cache_key
            self._route_count = 0
            self._base = QPixmap(self.size())
            self._base.fill(Qt.transparent)
            ink = QPainter(self._base)
            ink.setRenderHint(QPainter.Antialiasing)
            self._paint_solid(ink,origin,rotation,view_center,scale,xy,width,palette)
            ink.end()
        painter.drawPixmap(0,0,self._base)
        if self.fraction > 0:
            count = max(2,int(self.fraction*len(self.network.route_nodes)))
            if self._route_count == 0 or count < self._route_count:
                self._route_path = QPainterPath(QPointF(*xy[self.network.route_nodes[0]]))
                self._route_count = 1
            for node in self.network.route_nodes[self._route_count:count]:
                self._route_path.lineTo(QPointF(*xy[node]))
            self._route_count = count
            painter.setPen(QPen(QColor(palette['hot']),max(1.2,width*.3)))
            painter.drawPath(self._route_path)

    def _paint_solid(self, ink, origin, rotation, view_center, scale, xy, width, palette):
        if len(self.solid.faces)>100000:
            # Render every physical centerline at its true projected diameter.
            # This avoids topology-destroying decimation of thin fibers.
            depth=((self.solid.centers-origin)@rotation.T)[:,2]
            edges=np.unique(np.sort(self.network.edges,axis=1),axis=0)
            order=np.argsort(depth[edges].mean(axis=1))
            base=QColor(palette['accent'])
            for edge in edges[order]:
                a,b=xy[edge]
                line=QLineF(QPointF(*a),QPointF(*b))
                for fraction, shade in ((1.,.5),(.72,.8),(.35,1.)):
                    color=QColor(int(base.red()*shade),int(base.green()*shade),int(base.blue()*shade))
                    ink.setPen(QPen(color,max(.5,width*fraction),Qt.SolidLine,Qt.RoundCap))
                    ink.drawLine(line)
            return
        vertices=(self.solid.vertices-origin)@rotation.T
        triangles=vertices[self.solid.faces]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        lengths=np.maximum(np.linalg.norm(normals,axis=1),1e-15)
        normals/=lengths[:,None]
        visible=np.flatnonzero(normals[:,2]>0.)
        visible=visible[np.argsort(triangles[visible,:,2].mean(1))]
        projected=(vertices[:,:2]-view_center)*[scale,-scale]+[self.width()/2,self.height()/2]
        light=np.array([-.35,.45,.82]); light/=np.linalg.norm(light)
        shades=np.clip(.3+.7*(normals@light),.18,1.)
        base=QColor(palette['accent'])
        brushes=[QColor(int(base.red()*k/63),int(base.green()*k/63),int(base.blue()*k/63))
                 for k in range(64)]
        ink.setPen(Qt.NoPen)
        for face in visible:
            ink.setBrush(brushes[int(shades[face]*63)])
            ink.drawPolygon(QPolygonF([QPointF(*projected[n]) for n in self.solid.faces[face]]))

    def mousePressEvent(self,event):
        self._drag = event.position()

    def mouseMoveEvent(self,event):
        if self._drag is not None and self.three_d:
            delta = event.position()-self._drag
            self._drag = event.position()
            self.yaw += delta.x()*.5
            self.pitch = np.clip(self.pitch+delta.y()*.5,-89.,89.)
            self.update()

    def mouseReleaseEvent(self,event):
        self._drag = None

    def wheelEvent(self,event):
        self.zoom = np.clip(self.zoom*(1.1 if event.angleDelta().y()>0 else 1/1.1),.2,5.)
        self.update()

    def mouseDoubleClickEvent(self,event):
        curved = self.solid is not None and self.solid.settings.curved
        self.zoom, self.yaw, self.pitch = 1., -25. if curved else -20., 15. if curved else 50.
        self.update()


class ManufacturingDialog(QDialog):
    readiness_changed = Signal(bool)
    def __init__(self, factory, surface=None, mode='dark', parent=None, auto_build=True, network=None):
        super().__init__(parent)
        self.factory, self.surface, self.mode = replace(factory), surface, mode
        self.source_network = network
        self.worker = None
        self.network = self.solid = None
        self.bambu_executable = None
        zh = get_lang() == 'zh'
        self.setWindowTitle('连续制造 · 毫米模型' if zh else 'Continuous fabrication · millimetres')
        self.resize(1120,760)
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        form = QFormLayout()
        self.form = form
        self.controls = {}
        for key, label, value in [('width','外宽 / Width (mm)',100.),('depth','外长 / Depth (mm)',100.),
                                  ('height','平面厚度 / Height (mm)',2.),('diameter','纤维直径 / Diameter (mm)',2.)]:
            box = QDoubleSpinBox()
            box.setRange(.1,1000.)
            box.setDecimals(2)
            box.setValue(200. if surface is not None and key in ('width','depth') else value)
            self.controls[key] = box
            form.addRow(label.split(' / ')[0]+'（mm）' if zh else label.split(' / ')[-1],box)
        if surface is not None:
            limit = QDoubleSpinBox()
            limit.setRange(.1,250.)
            limit.setValue(250.)
            self.controls['max_height'] = limit
            form.addRow('高度上限（mm）' if zh else 'Height limit (mm)',limit)
            self.controls['width'].setMaximum(250.)
            self.controls['depth'].setMaximum(250.)
        self.controls['height'].setEnabled(surface is None)
        self.refresh = QPushButton('更新制造模型' if zh else 'Update fabrication model')
        self.refresh.clicked.connect(self.rebuild)
        form.addRow(self.refresh)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setMaximumWidth(265)
        form.addRow(self.info)
        explanation = ('曲面保持比例，三轴均不超过250mm；厚度为纤维直径。\n' if surface is not None else '')
        explanation += '重叠纤维保留独立路径；实体在交汇处融合。拓竹切片会重新规划打印路径。'
        self.note = QLabel(explanation if zh else 'Distinct overlapping fibers retain their route identities. Solid overlaps are fused. Bambu Studio replans the print path. Curved models keep their aspect ratio.')
        self.note.setWordWrap(True)
        self.note.setMaximumWidth(265)
        self.note.hide()
        body.addLayout(form)
        preview = QVBoxLayout()
        views = QHBoxLayout()
        self.view2d = QPushButton('二维制造视图' if zh else '2D fabrication')
        self.view3d = QPushButton('三维实体视图' if zh else '3D fabrication')
        for button, dimension in ((self.view2d,False),(self.view3d,True)):
            button.setCheckable(True)

            button.hide()
        self.play = QPushButton('播放一笔画路径' if zh else 'Play continuous route')
        self.play.clicked.connect(self.play_route)
        views.addWidget(self.play)
        preview.addLayout(views)
        self.canvas = ManufacturingCanvas(mode)
        preview.addWidget(self.canvas,1)
        body.addLayout(preview,1)
        root.addLayout(body,1)
        self.progress = QProgressBar()
        root.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        exports = QHBoxLayout()
        exports.setSpacing(6)
        self.stl = QPushButton('导出 STL' if zh else 'Export STL')
        self.threemf = QPushButton('导出 3MF' if zh else 'Export 3MF')
        self.route = QPushButton('导出路径' if zh else 'Route CSV')
        self.bambu = QPushButton('导入拓竹' if zh else 'Bambu Studio')
        self.cancel = QPushButton('停止' if zh else 'Stop')
        self.stl.clicked.connect(lambda:self.save('stl'))
        self.threemf.clicked.connect(lambda:self.save('3mf'))
        self.route.clicked.connect(lambda:self.save('csv'))
        self.bambu.clicked.connect(self.open_bambu)
        self.cancel.clicked.connect(self.stop)
        for index,button in enumerate((self.stl,self.threemf,self.route,self.bambu,self.cancel)):
            button.setStyleSheet("padding:6px 8px;")
            exports.addWidget(button)
        root.addLayout(exports)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        for control in self.controls.values():
            control.valueChanged.connect(self._invalidate)
        self.set_view(surface is not None)
        if auto_build:
            QTimer.singleShot(0,self.rebuild)

    def retranslate(self):
        zh = get_lang()=='zh'
        labels = {'width':('外宽','Width'), 'depth':('外长','Depth'),
                  'height':('平面厚度','Planar thickness'), 'diameter':('纤维直径','Fiber diameter'),
                  'max_height':('高度上限','Height limit')}
        for key,control in self.controls.items():
            self.form.labelForField(control).setText(labels[key][0 if zh else 1]+' (mm)')
        for button,cn,en in ((self.refresh,'更新制造模型','Update fabrication model'),
            (self.view2d,'二维制造视图','2D fabrication'),(self.view3d,'三维实体视图','3D fabrication'),
            (self.play,'播放一笔画路径','Play continuous route'),(self.stl,'导出 STL','Export STL'),
            (self.threemf,'导出 3MF','Export 3MF'),(self.route,'导出路径','Route CSV'),
            (self.bambu,'导入拓竹','Bambu Studio'),(self.cancel,'停止','Stop')):
            button.setText(cn if zh else en)
        self.note.setText('')
        if self.solid is not None:
            self.info.setText('%.2f × %.2f × %.2f mm' % tuple(np.ptp(self.solid.vertices,axis=0)))

    def set_view(self,three_d):
        self.canvas.three_d = three_d
        self.view2d.setChecked(not three_d)
        self.view3d.setChecked(three_d)
        self.canvas.update()

    def _invalidate(self):
        self.readiness_changed.emit(False)
        for button in (self.stl,self.threemf,self.route,self.bambu,self.play):
            button.setEnabled(False)
        self.status.setText('参数已改变，请更新制造模型' if get_lang()=='zh' else 'Settings changed; update the fabrication model')

    def rebuild(self):
        if self.worker is not None and self.worker.isRunning():
            return
        settings = PrintSettings(**{k:v.value() for k,v in self.controls.items()}, curved=self.surface is not None,
                                 up_axis=getattr(self.source_network,'up_axis','y') if self.surface is not None else 'z')
        self.timer.stop()
        self.canvas.fraction = 0.
        self._invalidate()
        self.refresh.setEnabled(False)
        self.cancel.setEnabled(True)
        for control in self.controls.values():
            control.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText('正在校验路径并构建毫米实体…' if get_lang()=='zh' else 'Checking route and building solid…')
        self.worker = ManufacturingWorker(self.factory,self.surface,settings,self,network=self.source_network)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.completed.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _release(self):
        self.refresh.setEnabled(True)
        self.cancel.setEnabled(False)
        for key,control in self.controls.items():
            control.setEnabled(key!='height' or self.surface is None)

    def _done(self,network,solid):
        self.set_view(self.surface is not None)
        self.network,self.solid = network,solid
        self.timer.setInterval(100 if len(network.edges)>20000 else 33)
        self.canvas.network,self.canvas.solid = network,solid
        if solid.settings.curved:
            self.canvas.yaw,self.canvas.pitch = -25.,15.
        self.canvas.update()
        self._release()
        for button in (self.stl,self.threemf,self.route,self.bambu,self.play):
            button.setEnabled(True)
        size = np.ptp(solid.vertices,axis=0)
        self.info.setText(('%.2f × %.2f × %.2f mm\n%d %s · %d %s' %
            (*size,len(network.positions),'节点' if get_lang()=='zh' else 'nodes',
             len(network.edges),'纤维段' if get_lang()=='zh' else 'segments')))
        self.status.setText('已就绪' if get_lang()=='zh' else 'Ready')
        self.readiness_changed.emit(True)

    def _failed(self,message):
        self._release()
        self.status.setToolTip(message)
        if get_lang()=="zh":
            if "budget" in message: message="模型规模过大，请降低网格密度"
            elif "cancel" in message.lower(): message="已停止生成"
            elif "timed out" in message: message="生成超时，请降低网格密度后重试"
        self.status.setText(message)

    def stop(self):
        if self.worker is not None:
            self.worker.stop = True

    def play_route(self):
        if self.timer.isActive():
            self.timer.stop()
            return
        self.started = time.monotonic()
        self.timer.start()

    def _tick(self):
        self.canvas.fraction = min(1.,(time.monotonic()-self.started)/10.)
        self.canvas.update()
        if self.canvas.fraction == 1.:
            self.timer.stop()

    def save(self,kind,path=None):
        if self.solid is None:
            return
        if not path:
            path,_ = QFileDialog.getSaveFileName(self,'Export','FiberScope.'+kind,kind.upper()+' (*.'+kind+')')
        if not path:
            return
        try:
            if kind=='csv':
                export_route(path,self.network,self.solid.centers)
            else:
                export_solid(path,self.solid)
            self.status.setText(str(path))
            return str(path)
        except (OSError,ValueError) as exc:
            self.status.setText(str(exc))

    def open_bambu(self):
        from fslab.bambu import find_bambu,open_in_bambu
        from fslab.storage import writable_data_dir
        executable = self.bambu_executable or find_bambu()
        if not executable:
            executable,_ = QFileDialog.getOpenFileName(self,'选择 Bambu Studio / Select Bambu Studio','', 'Executable (*.exe)')
        if not executable:
            self.status.setText('未找到拓竹，可先导出STL后手动导入。' if get_lang()=='zh' else 'Export STL and import it manually if Bambu Studio is not installed.')
            return
        self.bambu_executable = executable
        folder = Path(writable_data_dir('manufacturing'))
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/('FiberScope_'+str(time.time_ns())+'.stl')
        if self.save('stl',str(path)):
            try:
                open_in_bambu(path,executable)
                self.status.setText(('已发送到拓竹，请选择打印机、材料并切片：' if get_lang()=='zh' else 'Opened in Bambu Studio; select printer/material and slice: ')+str(path))
            except OSError as exc:
                self.status.setText(str(exc))

    def reject(self):
        self.close()

    def closeEvent(self,event):
        self.timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.stop()
            event.ignore()
            QTimer.singleShot(100,self.close)
        else:
            super().closeEvent(event)
