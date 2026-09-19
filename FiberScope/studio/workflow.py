"""Observable asynchronous UI workflow, with atomic per-stage journal."""
from dataclasses import asdict
import json
from pathlib import Path
import time
from PySide6.QtCore import QObject, QTimer, Signal, QEvent
from PySide6.QtWidgets import QApplication, QWidget
from fslab.storage import writable_data_dir
from fslab.exporter import export_json
from fslab.print_export import PrintSettings, export_solid, export_route
from .manufacturing_dialog import ManufacturingWorker, ManufacturingDialog


class PrintWorkflow(QObject):
    changed = Signal(str)
    STAGES = ('structure','simulation','features','dataset','training','inverse',
              'verification','surface','solid','export','slicer')
    LABELS = ('结构生成','物理仿真','特征分析','物理标注数据','模型训练','逆向设计',
              '优化后复核','三维映射','制造实体','打印文件','拓竹交接')

    def __init__(self, panel):
        super().__init__(panel)
        self.panel, self.win = panel, panel.win
        self.running, self.worker = False, None
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.record = {}

    def start(self, samples=10, iterations=20, curved=False, open_slicer=True,
              target='C', stretch=2., surface_model=None, create_demo=False,
              demo_unit='hexagon', diameter=2., print_size=200.):
        if self.running or self._busy(self.worker):
            raise ValueError('workflow already running or stopping')
        if not 10<=int(samples)<=2000 or not 1<=int(iterations):
            raise ValueError('samples must be 10..2000; iterations must be positive')
        from fslab.inverse import TARGETS, SCALARS
        if target not in TARGETS and target not in SCALARS:
            raise ValueError('unknown inverse target')
        if not 1.01<=float(stretch)<=20.:
            raise ValueError('stretch must be 1.01..20')
        if surface_model is not None and surface_model not in [Path(p).name for p in self.win.tab_surface._obj_files]:
            raise ValueError('unknown surface model')
        from fslab.structure import all_unit_keys
        if demo_unit not in all_unit_keys():
            raise ValueError('unknown demo structure type')
        if not .1<=float(diameter)<float(print_size)<=250.:
            raise ValueError('fiber diameter must be smaller than print size, at most 250 mm')
        for owner in (self.win.tab_struct,self.win.tab_surface):
            dialog = getattr(owner,'manufacturing_dialog',None)
            if self._busy(getattr(dialog,'worker',None)):
                raise ValueError('wait for the current manufacturing task to finish')
        for tab in (self.win.tab_sim,self.win.tab_design,self.win.tab_features,self.win.tab_ml):
            for key in ('worker','_worker','gen_worker','train_worker'):
                if self._busy(getattr(tab,key,None)):
                    raise ValueError('wait for the current analysis task to finish')
        self.folder = Path(writable_data_dir('workflows'))/str(time.time_ns())
        self.folder.mkdir(parents=True)
        self.options = dict(samples=int(samples),iterations=int(iterations),curved=bool(curved),open_slicer=bool(open_slicer),
                            target=target,stretch=float(stretch),surface_model=surface_model,create_demo=bool(create_demo),
                            demo_unit=demo_unit,diameter=float(diameter),print_size=float(print_size))
        self.record = dict(state='running',options=self.options,stages=[],initial=asdict(self.win.tab_struct.collect_factory()),
                           topology='topnet26',printer_state='not_submitted')
        self.index, self.entered, self.success, self.error = 0, False, False, None
        self.network = self.solid = None
        self.running = True
        self._locked = [self.win.tabs.widget(i) for i in range(self.win.tabs.count())]
        QApplication.instance().installEventFilter(self)
        self._save()
        self.timer.start()
        return self.status()

    def eventFilter(self, watched, event):
        blocked=(QEvent.MouseButtonPress,QEvent.MouseButtonRelease,QEvent.MouseButtonDblClick,
                 QEvent.KeyPress,QEvent.KeyRelease,QEvent.Wheel,QEvent.Shortcut,QEvent.ContextMenu)
        if self.running and isinstance(watched,QWidget) and self.win.tabs.isAncestorOf(watched):
            if event.type() in blocked: return True
        return super().eventFilter(watched,event)

    def _report_stage(self, state, message=None):
        if getattr(self,'_stage_reported',True): return
        self._stage_reported=True
        stage=self.STAGES[self.index]
        result=self.panel.registry['get_app_state']()
        result.update(stage=stage,state=state,page=result['tab'],options=self.options)
        if stage in ('dataset','training'):
            result['samples']=len(self.win.tab_ml.X) if self.win.tab_ml.X is not None else 0
        if stage in ('solid','export','slicer') and self.solid is not None:
            result.update(triangles=len(self.solid.faces),view='3D' if self.options['curved'] else '2D')
        if message: result['message']=message
        self.record['stages'][-1]['page']=result['page']
        self.panel._workflow_tool_finished(stage,result,state=='complete')

    @staticmethod
    def _busy(worker):
        try:
            return worker is not None and worker.isRunning()
        except RuntimeError:
            return False

    def _save(self):
        temp = self.folder/'progress.json.tmp'
        temp.write_text(json.dumps(self.record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temp.replace(self.folder/'progress.json')

    def status(self):
        return dict(busy=self.running,**self.record,folder=str(getattr(self,'folder','')))

    def _ok(self,*_):
        self.success = True

    def _fail(self,message):
        self.error = str(message)

    def _watch(self,worker,signal):
        if worker is None:
            raise RuntimeError('stage did not start')
        self.worker = worker
        getattr(worker,signal).connect(self._ok)
        worker.failed.connect(self._fail)

    def _tick(self):
        if not self.running:
            return
        try:
            if self.error:
                self._finish('failed',self.error)
                return
            if not self.entered:
                self.entered, self.success, self.worker = True, False, None
                self._preview_until = None
                self._stage_reported = False
                self.record['stages'].append(dict(name=self.STAGES[self.index],state='running'))
                self._save()
                self.changed.emit('%d/11 · %s' % (self.index+1,self.LABELS[self.index]))
                self.panel._workflow_tool_started(self.STAGES[self.index],self.options)
                self._begin(self.STAGES[self.index])
            elif self.success and not self._busy(self.worker):
                self._report_stage('complete')
                if self.options['create_demo']:
                    simulation = self.STAGES[self.index] in ('simulation','verification')
                    if self._preview_until is None:
                        self._preview_until = time.monotonic()+(6. if simulation else (5. if self.STAGES[self.index]=='solid' else 2.5))
                        if simulation: self.win.tab_sim._start_play()
                    if time.monotonic()<self._preview_until:
                        return
                    if simulation: self.win.tab_sim._play_timer.stop()
                self.record['stages'][-1]['state'] = 'complete'
                self.index += 1
                self._save()
                if self.index==len(self.STAGES):
                    self._finish('complete','打印文件已就绪；在拓竹中确认打印机、材料并切片打印。')
                else:
                    self.entered = False
        except Exception as exc:
            self._finish('failed',str(exc))

    def _begin(self,stage):
        w, p = self.win,self.panel
        page = dict(structure='structure',simulation='stretch',features='features',dataset='ml',training='ml',
                    inverse='design',verification='stretch',surface='surface',solid='manufacturing',export='manufacturing',slicer='manufacturing')
        if stage!='solid': p._navigate(page[stage])
        if stage=='structure':
            if self.options['create_demo']:
                from fslab.structure import StructureFactory
                w.tab_struct.load_spec(StructureFactory(unit=self.options['demo_unit'],grid_x=3,grid_y=3,n_pts_per_side=5,
                    seed=7,perturbation=.04,line_displacements=[[0.,v] for v in (.02,.035,.04,.035,.02)]))
                w.tab_sim.chk_weld.setChecked(True)
                w.tab_sim.chk_contact.setChecked(False)
            w.tab_struct.push_spec()
            self.record['initial'] = asdict(w.tab_struct.collect_factory())
            export_json(w.tab_struct.collect_factory(),str(self.folder/'initial.json'))
        elif stage in ('simulation','verification'):
            w.tab_sim.stretch.setValue(self.options['stretch'])
            w.tab_sim.start_run()
            self._watch(w.tab_sim.worker,'finished_ok')
            return
        elif stage=='features':
            w.tab_features.chk_batch.setChecked(False)
            w.tab_features.refresh()
            if w.tab_features._feats is None:
                raise RuntimeError(w.tab_features.status.text())
            w.tab_features._export_feat_csv(silent=str(self.folder/'features.csv'))
        elif stage=='dataset':
            tab = w.tab_ml
            tab._train_after_generation = False
            unit = w.tab_struct.collect_factory().unit
            if tab.unit_combo.findData(unit)<0:
                tab.unit_combo.addItem(unit,unit)
            tab.unit_combo.setCurrentIndex(tab.unit_combo.findData(unit))
            tab.n_spin.setValue(self.options['samples'])
            tab.mode_combo.setCurrentIndex(tab.mode_combo.findData('physics'))
            tab.start_gen()
            self._watch(tab.gen_worker,'done')
            return
        elif stage=='training':
            w.tab_ml.start_train()
            self._watch(w.tab_ml.train_worker,'done')
            return
        elif stage=='inverse':
            tab = w.tab_design
            tab.selected_model = None
            tab.model_btn.setChecked(False)
            tab.algorithm, tab.parameters = 'cem', {}
            tab.follow_type.setChecked(True)
            result = p._run_inverse(self.options['target'],self.options['iterations'],True)
            if isinstance(result,str):
                result = json.loads(result)
            if not result.get('ok'):
                raise RuntimeError(result.get('error','inverse did not start'))
            self._watch(tab.worker,'finished_ok')
            return
        elif stage=='surface':
            w.tab_features.refresh()
            w.tab_features._export_feat_csv(silent=str(self.folder/'optimized_features.csv'))
            w.tab_sim._export_csv(silent=str(self.folder/'optimized_curve.csv'))
            if self.options['surface_model'] is not None:
                index = [Path(p).name for p in w.tab_surface._obj_files].index(self.options['surface_model'])
                w.tab_surface.obj_combo.setCurrentIndex(index)
            w.tab_surface.chk_follow.setChecked(True)
            w.tab_surface.apply_structure(w.tab_struct.collect_factory())
            w.tab_surface._debounce.stop() if hasattr(w.tab_surface,'_debounce') else None
            w.tab_surface._recompute()
            if w.tab_surface._mapping_error:
                raise RuntimeError(w.tab_surface._mapping_error)
        elif stage=='solid':
            surface = None
            if self.options['curved']:
                v,f,factory,scale = w.tab_surface._manufacturing_source
                surface = (v,f,scale)
            else:
                factory = w.tab_struct.collect_factory()
            self.final_factory, self.final_surface = factory,surface
            network=w.tab_surface.manufacturing_network if surface is not None else None
            self.manufacturing_dialog=dialog=ManufacturingDialog(factory,surface,w.mode,w.tab_manufacturing,
                                                                  auto_build=False,network=network)
            for key in ('width','depth'):
                dialog.controls[key].setValue(self.options['print_size'] if surface is not None else 100.)
            dialog.controls['diameter'].setValue(self.options['diameter'])
            w.tab_struct.manufacturing_dialog=w.tab_surface.manufacturing_dialog=dialog
            w.tab_manufacturing.mount(dialog)
            dialog.rebuild()
            worker=dialog.worker
            self._watch(worker,'completed')
            worker.completed.connect(self._solid)
            return
        elif stage=='export':
            export_json(w.tab_struct.collect_factory(),str(self.folder/'optimized.json'))
            export_solid(self.folder/'FiberScope.3mf',self.solid)
            export_solid(self.folder/'FiberScope.stl',self.solid)
            export_route(self.folder/'path.csv',self.network,self.solid.centers)
            self.record['topology_id'] = self.network.topology_id
            self.record['final'] = asdict(self.final_factory)
        elif stage=='slicer':
            from fslab.bambu import find_bambu,open_in_bambu
            executable = find_bambu() if self.options['open_slicer'] else None
            if executable:
                open_in_bambu(self.folder/'FiberScope.stl',executable)
                self.record['printer_state'] = 'opened_in_bambu'
            else:
                self.record['printer_state'] = 'files_ready'
        self.success = True

    def _solid(self,network,solid):
        self.network,self.solid = network,solid
        self.manufacturing_dialog.source_network=network

    def cancel(self):
        if not self.running:
            return
        if self._busy(self.worker) and hasattr(self.worker,'request_stop'):
            self.worker.request_stop()
        self.win.tab_ml._train_after_generation = False
        self.win.tab_design._auto_apply = False
        self._finish('cancelled','流程已停止，已完成的结果保留。')

    def _finish(self,state,message):
        self.timer.stop()
        if state!='complete' and self._busy(self.worker) and hasattr(self.worker,'request_stop'):
            self.worker.request_stop()
        self.running = False
        QApplication.instance().removeEventFilter(self)
        if state!='complete': self._report_stage(state,message)
        self.record.update(state=state,message=message)
        if self.record.get('stages') and self.record['stages'][-1]['state']=='running':
            self.record['stages'][-1]['state']=state
        self._save()
        for tab in self._locked:
            tab.setEnabled(True)
        self.changed.emit(message+'\n'+str(self.folder))
