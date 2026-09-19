"""Machine-learning tab: structure features -> mechanical properties.

Surrogate MLP trained on generated datasets, with live loss curves
(train/val appended per epoch via signals), early stopping and a stop
button.  Dataset generation and training both run in QThreads; the UI
only consumes signals.
"""
import os
import sys as _sys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QSlider,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

from fslab import UNIT_PRESETS
from fslab.structure import all_unit_keys, unit_display
from fslab.mlmodel import MLP, TARGET_NAMES, gen_dataset, r2_score
from fslab.learning import Regressor, MODEL_SPECS, ACQUISITIONS
from fslab.dataset_stream import dataset_config, config_id
from .i18n import tr, get_lang
from .theme import apply_plot_theme, colors


def _cache_dir():
    if getattr(_sys, 'frozen', False):
        from fslab.storage import writable_data_dir
        return writable_data_dir('datasets')
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_cache')


def dataset_path(unit):
    return os.path.join(_cache_dir(), 'ml_dataset_%s.npz' % unit)


class GenWorker(QThread):
    """Generate samples incrementally; emits progress every sample."""
    progress = Signal(int, int)          # i, n
    done = Signal(object, object, str)   # X, Y, path
    failed = Signal(str)

    def __init__(self, unit, n, amp, pert, seed0, path, parent=None, mode='physics', acquisition='random', model=None):
        super().__init__(parent)
        self.unit, self.n, self.amp = unit, int(n), float(amp)
        self.label_mode, self.acquisition, self.model = mode, acquisition, model
        self.pert, self.seed0, self.path = float(pert), int(seed0), path
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            X, Y = gen_dataset(
                self.unit, self.n, self.amp, self.pert, self.seed0,
                self.path,
                progress_cb=lambda i, n: self.progress.emit(i, n),
                stop_cb=lambda: self._stop, mode=self.label_mode,
                acquisition=self.acquisition, model=self.model)
            self.done.emit(X, Y, self.path)
        except Exception as e:
            self.failed.emit('%s: %s' % (e.__class__.__name__, e))


class TrainWorker(QThread):
    """Train the MLP; emits (epoch, train_loss, val_loss) each epoch."""
    epoch = Signal(int, float, float)
    done = Signal(object)                # {'model': MLP, 'history': dict}
    failed = Signal(str)

    def __init__(self, X, Y, hidden, lr, epochs, patience, min_delta,
                 parent=None, algorithm='mlp', parameters=None):
        super().__init__(parent)
        self.X, self.Y = X, Y
        self.algorithm, self.parameters = algorithm, parameters
        self.hidden, self.lr = int(hidden), float(lr)
        self.epochs, self.patience = int(epochs), int(patience)
        self.min_delta = float(min_delta)
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            model = Regressor(self.algorithm, self.parameters or ({'hidden': self.hidden} if self.algorithm in ('mlp', 'deep_mlp') else {}))
            hist = model.train(
                self.X, self.Y, lr=self.lr, epochs=self.epochs,
                patience=self.patience, min_delta=self.min_delta,
                epoch_cb=lambda e, tr, va: self.epoch.emit(e, tr, va),
                stop_cb=lambda: self._stop)
            self.done.emit({'model': model, 'history': hist})
        except Exception as e:
            self.failed.emit('%s: %s' % (e.__class__.__name__, e))


_ML_KEYS = ('gb_data', 'unit', 'count', 'amp', 'pert', 'gb_train', 'lr',
            'epochs', 'hidden', 'patience', 'min_delta', 'gen', 'train',
            'stop', 'loss', 'scatter', 'target', 'targets', 'ready',
            'need_data', 'busy_gen', 'busy_train')


def _T():
    # this tab reads the central i18n table (keys under ml_*)
    return {k: tr('ml_' + k) for k in _ML_KEYS}




class MLTab(QWidget):
    apply_structure = Signal(object)
    def __init__(self, mode='dark', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.algorithm, self.parameters, self.acquisition = 'mlp', {}, 'random'
        self.label_mode = 'physics'
        self.trained_model = None
        self.sample_specs = []
        self._train_after_generation = False
        self.X = None
        self.Y = None
        self.model = None
        self.history = None
        self.gen_worker = None
        self.train_worker = None
        self._build()
        self.retranslate()
        self._load_existing()

    # ---------------- UI ----------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        body = QSplitter(Qt.Horizontal)

        # ---- left: configuration ----
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        self.gb_data = QGroupBox()
        df = QFormLayout(self.gb_data)
        df.setSpacing(8)
        self.lbl_unit = QLabel()
        self.unit_combo = QComboBox()
        for key in all_unit_keys():
            self.unit_combo.addItem(unit_display(key, get_lang()), key)
        self.unit_combo.setCurrentIndex(self.unit_combo.findData('hexagon'))
        self.unit_combo.currentIndexChanged.connect(self._load_existing)
        df.addRow(self.lbl_unit, self.unit_combo)
        self.lbl_n = QLabel()
        self.n_spin = QSpinBox()
        self.n_spin.setRange(10, 2000)
        self.n_spin.setValue(60)
        df.addRow(self.lbl_n, self.n_spin)
        self.lbl_amp = QLabel()
        arow = QHBoxLayout()
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(0, 100)         # 0 .. 1.0
        self.amp_slider.setValue(20)             # default 0.20
        self.amp_val = QLabel('0.20')
        self.amp_val.setObjectName('chip')
        self.amp_slider.valueChanged.connect(
            lambda v: self.amp_val.setText('%.2f' % (v / 100.0)))
        arow.addWidget(self.amp_slider)
        arow.addWidget(self.amp_val)
        df.addRow(self.lbl_amp, arow)
        self.lbl_pert = QLabel()
        prow = QHBoxLayout()
        self.pert_slider = QSlider(Qt.Horizontal)
        self.pert_slider.setRange(0, 100)        # 0 .. 1.0
        self.pert_slider.setValue(10)            # default 0.10
        self.pert_val = QLabel('0.10')
        self.pert_val.setObjectName('chip')
        self.pert_slider.valueChanged.connect(
            lambda v: self.pert_val.setText('%.2f' % (v / 100.0)))
        prow.addWidget(self.pert_slider)
        prow.addWidget(self.pert_val)
        df.addRow(self.lbl_pert, prow)
        self.mode_combo = QComboBox()
        for key in ('physics',):
            self.mode_combo.addItem(key, key)
        self.mode_combo.currentIndexChanged.connect(self._load_existing)
        self.mode_combo.hide()
        self.data_note = QLabel()
        self.data_note.setWordWrap(True)
        self.data_note.hide()
        self.algorithm_btn = QPushButton()
        self.algorithm_btn.clicked.connect(self._configure)
        df.addRow(self.algorithm_btn)
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 1)
        self.apply_sample_btn = QPushButton()
        self.apply_sample_btn.clicked.connect(self._apply_sample)
        sample_row = QHBoxLayout()
        sample_row.addWidget(self.sample_spin)
        sample_row.addWidget(self.apply_sample_btn)
        df.addRow(sample_row)
        lv.addWidget(self.gb_data)

        self.gb_train = QGroupBox()
        tf = QFormLayout(self.gb_train)
        tf.setSpacing(8)
        self.lbl_lr = QLabel()
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(['1e-2', '3e-3', '1e-3'])
        self.lr_combo.setCurrentIndex(1)
        tf.addRow(self.lbl_lr, self.lr_combo)
        self.lbl_epochs = QLabel()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 5000)
        self.epochs_spin.setValue(200)
        tf.addRow(self.lbl_epochs, self.epochs_spin)
        self.lbl_hidden = QLabel()
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(4, 256)
        self.hidden_spin.setValue(32)
        self.hidden_spin.valueChanged.connect(self._hidden_changed)
        tf.addRow(self.lbl_hidden, self.hidden_spin)
        self.lbl_patience = QLabel()
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 500)
        self.patience_spin.setValue(15)
        tf.addRow(self.lbl_patience, self.patience_spin)
        self.lbl_mindelta = QLabel()
        self.mindelta_spin = QDoubleSpinBox()
        self.mindelta_spin.setRange(1e-6, 1e-2)
        self.mindelta_spin.setDecimals(6)
        self.mindelta_spin.setSingleStep(1e-5)
        self.mindelta_spin.setValue(1e-4)
        tf.addRow(self.lbl_mindelta, self.mindelta_spin)
        lv.addWidget(self.gb_train)

        self.gen_btn = QPushButton()
        self.gen_btn.setProperty('primary', True)
        self.gen_btn.clicked.connect(self.start_gen)
        lv.addWidget(self.gen_btn)
        brow = QHBoxLayout()
        self.train_btn = QPushButton()
        self.train_btn.setProperty('primary', True)
        self.train_btn.clicked.connect(self.start_train)
        self.stop_btn = QPushButton()
        self.stop_btn.clicked.connect(self.stop)
        brow.addWidget(self.train_btn)
        brow.addWidget(self.stop_btn)
        lv.addLayout(brow)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(16)
        self.progress.setFormat('%p%')
        lv.addWidget(self.progress)
        self.status = QLabel()
        self.status.setObjectName('subtitle')
        self.status.setWordWrap(True)
        lv.addWidget(self.status)
        lv.addStretch(1)
        left.setMinimumWidth(250)
        left.setMaximumWidth(330)
        body.addWidget(left)

        # ---- right: live plots ----
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)

        self.loss_plot = pg.PlotWidget()
        apply_plot_theme(self.loss_plot, self.mode)
        lpi = self.loss_plot.getPlotItem()
        lpi.setLabel('bottom', 'epoch')
        lpi.setLabel('left', 'loss (MSE)')
        lpi.addLegend()
        c = colors(self.mode)
        self.tr_curve = lpi.plot([], [], pen=pg.mkPen(c['accent'], width=2),
                                 name='train', symbol='o', symbolSize=4)
        self.va_curve = lpi.plot([], [], pen=pg.mkPen(c['hot'], width=2),
                                 name='val', symbol='o', symbolSize=4)
        rv.addWidget(self.loss_plot, 3)

        r2row = QHBoxLayout()
        self.lbl_target = QLabel()
        r2row.addWidget(self.lbl_target)
        self.target_combo = QComboBox()
        self.target_combo.addItems(list(TARGET_NAMES))
        self.target_combo.setFixedWidth(120)
        self.target_combo.currentIndexChanged.connect(self._update_scatter)
        r2row.addWidget(self.target_combo)
        r2row.addStretch(1)
        self.r2_labels = []
        for _ in TARGET_NAMES:
            lab = QLabel('--')
            lab.setObjectName('chip')
            lab.setWordWrap(True)
            self.r2_labels.append(lab)
            r2row.addWidget(lab)
        rv.addLayout(r2row)

        self.scatter_plot = pg.PlotWidget()
        apply_plot_theme(self.scatter_plot, self.mode)
        spi = self.scatter_plot.getPlotItem()
        spi.setLabel('bottom', 'actual')
        spi.setLabel('left', 'predicted')
        self.id_line = spi.plot([], [], pen=pg.mkPen(
            c['sub'], width=1, style=Qt.DashLine))
        self.scatter = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush(c['accent2']))
        spi.addItem(self.scatter)
        self.scatter_plot.setMinimumHeight(180)
        rv.addWidget(self.scatter_plot, 2)

        body.addWidget(right)
        outer.addWidget(body, 1)

    # ---------------- dataset generation ----------------
    def _apply_sample(self):
        from fslab.structure import StructureFactory
        if self.sample_specs:
            self.apply_structure.emit(StructureFactory(**self.sample_specs[self.sample_spin.value()-1]))

    def _read_specs(self, path):
        import json
        self.sample_specs = []
        if os.path.exists(path):
            try:
                with np.load(path, allow_pickle=False) as z:
                    if 'specs' in z:
                        self.sample_specs = json.loads(str(z['specs']))
            except (ValueError, OSError, KeyError):
                self.sample_specs = []
        self.sample_spin.setRange(1, max(1, len(self.sample_specs)))
        self.apply_sample_btn.setEnabled(bool(self.sample_specs))

    def _hidden_changed(self, value):
        if self.algorithm in ('mlp', 'deep_mlp'):
            self.parameters['hidden'] = int(value)
            self.trained_model = None

    def _sync_model_controls(self):
        neural = self.algorithm in ('mlp', 'deep_mlp')
        for widget in (self.lr_combo, self.epochs_spin, self.hidden_spin,
                       self.patience_spin, self.mindelta_spin):
            widget.setEnabled(neural)
        if neural and 'hidden' in self.parameters:
            self.hidden_spin.setValue(int(self.parameters['hidden']))

    def _configure(self):
        from .algorithm_dialog import AlgorithmDialog
        dialog = AlgorithmDialog(MODEL_SPECS, self.algorithm, {self.algorithm: self.parameters},
                                 ACQUISITIONS, self.acquisition, self)
        if dialog.exec():
            self.trained_model = None
            self.algorithm, self.parameters = dialog.selection()
            self._sync_model_controls()
            self.acquisition = dialog.acquisition.currentData()
            self.algorithm_btn.setText(MODEL_SPECS[self.algorithm][0 if get_lang() == 'zh' else 1] + ' · …')
            self._load_existing()

    def _dataset_path(self):
        unit = self.unit_combo.currentData()
        config = dataset_config(unit, self.amp_slider.value()/100., self.pert_slider.value()/100.,
                                0, self.mode_combo.currentData(), self.acquisition)
        return os.path.join(_cache_dir(), 'ml_' + config_id(config) + '.npz')

    def start_gen(self):
        if self._busy():
            return
        unit = self.unit_combo.currentData()
        path = self._dataset_path()
        if self.mode_combo.currentData() == 'surrogate':
            import uuid
            path = path[:-4] + '_' + uuid.uuid4().hex[:8] + '.npz'
        amp = self.amp_slider.value() / 100.0
        pert = self.pert_slider.value() / 100.0
        self.gen_worker = GenWorker(unit, self.n_spin.value(), amp, pert,
                                    0, path, self, mode=self.mode_combo.currentData(),
                                    acquisition=self.acquisition, model=self.trained_model)
        self.gen_worker.progress.connect(self._gen_progress)
        self.gen_worker.done.connect(self._gen_done)
        self.gen_worker.failed.connect(self._worker_failed)
        self._set_busy(True)
        self.progress.setValue(0)
        self.status.setText(_T()['busy_gen'])
        self.gen_worker.start()

    def _gen_progress(self, i, n):
        self.progress.setValue(int(100 * i / max(n, 1)))
        self.status.setText('%s %d/%d'
                            % (_T()['busy_gen'], i, n))

    def _gen_done(self, X, Y, path):
        self.X, self.Y = X, Y
        self.label_mode = self.gen_worker.label_mode
        self._read_specs(path)
        self.model = None
        self.history = None
        self.gen_worker.deleteLater()
        self.gen_worker = None
        self._set_busy(False)
        self.progress.setValue(100)
        self.status.setText('%d samples -> %s'
                            % (X.shape[0], os.path.basename(path)))
        self._refresh_r2()
        if self._train_after_generation:
            self._train_after_generation = False
            QTimer.singleShot(0, self.start_train)

    # ---------------- training ----------------
    def start_train(self):
        if self._busy():
            return
        if self.X is None or self.X.shape[0] < 5 or self.label_mode != 'physics' or not np.isfinite(self.Y).all():
            self.status.setText(_T()['need_data'])
            return
        self._training_unit = self.unit_combo.currentData()
        lr = float(self.lr_combo.currentText())
        self.train_worker = TrainWorker(
            self.X, self.Y, self.hidden_spin.value(), lr,
            self.epochs_spin.value(), self.patience_spin.value(),
            self.mindelta_spin.value(), self, algorithm=self.algorithm,
            parameters=self.parameters)
        self.train_worker.epoch.connect(self._on_epoch)
        self.train_worker.done.connect(self._train_done)
        self.train_worker.failed.connect(self._worker_failed)
        self.tr_curve.setData([], [])
        self.va_curve.setData([], [])
        self._tr_pts, self._tr_loss = [], []
        self._va_pts, self._va_loss = [], []
        self._set_busy(True)
        forest = self.algorithm in ('random_forest', 'extra_trees')
        neural = self.algorithm in ('mlp', 'deep_mlp')
        self._progress_kind = ('树数量' if forest else ('轮次' if neural else '拟合')) if get_lang() == 'zh' else ('trees' if forest else ('epoch' if neural else 'fit'))
        self.loss_plot.setLabel('bottom', self._progress_kind)
        self.progress.setRange(0, 0)          # busy indicator
        self.status.setText(_T()['busy_train'])
        self.train_worker.start()

    def _on_epoch(self, ep, tr, va):
        # live update of the loss curves (setData, pyqtgraph 0.14-safe)
        self._tr_pts.append(ep)
        self._tr_loss.append(tr)
        self._va_pts.append(ep)
        self._va_loss.append(va)
        self.tr_curve.setData(self._tr_pts, self._tr_loss)
        self.va_curve.setData(self._va_pts, self._va_loss)
        self.loss_plot.enableAutoRange()
        self.status.setText('%s %d  train %.4f  val %.4f' % (self._progress_kind, ep, tr, va))

    def _train_done(self, res):
        self.model = res['model']
        self.trained_unit = getattr(self, '_training_unit', self.unit_combo.currentData())
        self.trained_model = self.model
        self.history = res['history']
        self.train_worker.deleteLater()
        self.train_worker = None
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        h = self.history
        tag = 'early-stop' if h['early_stopped'] else 'finished'
        self.status.setText(
            '%s: %d epochs (best %d), final train %.4f / val %.4f'
            % (tag, h['epochs_run'], h['best_epoch'],
               h['train'][-1], h['val'][-1]))
        if h.get('progress_kind') != 'epoch':
            self.status.setText(('拟合完成 · ' if get_lang() == 'zh' else 'Fit complete · ') +
                                self._progress_kind + ' ' + str(h.get('trees_fitted') or 1) +
                                ' · val MSE %.4f' % h['val'][-1])
        self._refresh_r2()

    def stop(self):
        self._train_after_generation = False
        if self.gen_worker is not None:
            self.gen_worker.request_stop()
        if self.train_worker is not None:
            self.train_worker.request_stop()

    def _worker_failed(self, msg):
        self._train_after_generation = False
        self.status.setText(msg)
        self._set_busy(False)
        self.progress.setRange(0, 100)
        if self.gen_worker is not None:
            self.gen_worker.deleteLater()
            self.gen_worker = None
        if self.train_worker is not None:
            self.train_worker.deleteLater()
            self.train_worker = None

    def _busy(self):
        return ((self.gen_worker is not None
                 and self.gen_worker.isRunning())
                or (self.train_worker is not None
                    and self.train_worker.isRunning()))

    def _set_busy(self, busy):
        self.gen_btn.setEnabled(not busy)
        self.train_btn.setEnabled(not busy)
        self.unit_combo.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.algorithm_btn.setEnabled(not busy)
        for widget in (self.amp_slider, self.pert_slider, self.n_spin):
            widget.setEnabled(not busy)

    # ---------------- results ----------------
    def _load_existing(self, *_):
        if not hasattr(self, 'mode_combo') or not hasattr(self, 'status'):
            return
        self.label_mode = self.mode_combo.currentData()
        unit = self.unit_combo.currentData()
        path = self._dataset_path()
        self._read_specs(path)
        if os.path.exists(path):
            try:
                with np.load(path) as z:
                    self.X = np.asarray(z['X'], float)
                    self.Y = np.asarray(z['Y'], float)
                self.status.setText('%s: %d samples' % (unit,
                                                        self.X.shape[0]))
            except Exception:
                self.X = self.Y = None
        else:
            self.X = self.Y = None
        self.model = None
        self.history = None
        self._refresh_r2()

    def _refresh_r2(self):
        t = _T()
        if self.model is None or self.X is None:
            for lab in self.r2_labels:
                lab.setText('--')
            self.scatter.setData(x=[], y=[])
            self.id_line.setData([], [])
            return
        ids = getattr(self.model, 'val_indices', np.arange(len(self.X)))
        pred = self.model.predict(self.X[ids])
        for j, lab in enumerate(self.r2_labels):
            r2 = r2_score(self.Y[ids, j], pred[:, j])
            tag = '验证' if get_lang() == 'zh' else 'validation'
            lab.setText('%s %s R²=%.3f' % (t['targets'][j], tag, r2))
        self._update_scatter()

    def _update_scatter(self, *_):
        if self.model is None or self.X is None:
            return
        j = max(self.target_combo.currentIndex(), 0)
        ids = getattr(self.model, 'val_indices', np.arange(len(self.X)))
        pred = self.model.predict(self.X[ids])[:, j]
        truth = self.Y[ids, j]
        self.scatter.setData(x=truth.tolist(), y=pred.tolist())
        lo = float(min(truth.min(), pred.min()))
        hi = float(max(truth.max(), pred.max()))
        pad = 0.05 * max(hi - lo, 1e-9)
        self.id_line.setData([lo - pad, hi + pad], [lo - pad, hi + pad])
        self.scatter_plot.setRange(xRange=(lo-pad, hi+pad), yRange=(lo-pad, hi+pad), padding=.03)
        self.scatter_plot.setLabel('bottom', '物理真实值' if get_lang() == 'zh' else 'Physical truth')
        self.scatter_plot.setLabel('left', '模型预测值' if get_lang() == 'zh' else 'Prediction')

    # ---------------- theming / i18n ----------------
    def set_mode(self, mode):
        self.mode = mode
        apply_plot_theme(self.loss_plot, mode)
        apply_plot_theme(self.scatter_plot, mode)
        c = colors(mode)
        self.tr_curve.setPen(pg.mkPen(c['accent'], width=2))
        self.va_curve.setPen(pg.mkPen(c['hot'], width=2))
        self.id_line.setPen(pg.mkPen(c['sub'], width=1, style=Qt.DashLine))
        self.scatter.setBrush(pg.mkBrush(c['accent2']))

    def retranslate(self):
        zh = get_lang() == 'zh'
        self.apply_sample_btn.setText('查看结构' if zh else 'View structure')
        self._sync_model_controls()
        self.mode_combo.blockSignals(True)
        self.data_note.setText('生成结构并自动完成物理标注，完成后即可训练。' if zh else 'Generate structures with physical labels, ready for training.')
        for i, names in enumerate((('生成并物理标注', 'Generate and simulate'),)):
            self.mode_combo.setItemText(i, names[0 if zh else 1])
        self.mode_combo.blockSignals(False)
        self.algorithm_btn.setText(MODEL_SPECS[self.algorithm][0 if zh else 1] + ' · …')
        self.unit_combo.blockSignals(True)
        for i in range(self.unit_combo.count()):
            self.unit_combo.setItemText(i, unit_display(self.unit_combo.itemData(i), get_lang()))
        self.unit_combo.blockSignals(False)
        t = _T()
        self.gb_data.setTitle(t['gb_data'])
        self.gb_train.setTitle(t['gb_train'])
        self.lbl_unit.setText(t['unit'])
        self.lbl_n.setText(t['count'])
        self.lbl_amp.setText(t['amp'])
        self.lbl_pert.setText(t['pert'])
        self.lbl_lr.setText(t['lr'])
        self.lbl_epochs.setText(t['epochs'])
        self.lbl_hidden.setText(t['hidden'])
        self.lbl_patience.setText(t['patience'])
        self.lbl_mindelta.setText(t['min_delta'])
        self.gen_btn.setText(t['gen'])
        self.train_btn.setText(t['train'])
        self.stop_btn.setText(t['stop'])
        self.lbl_target.setText(t['target'])
        for j in range(len(TARGET_NAMES)):
            self.target_combo.setItemText(j, t['targets'][j])
        self.loss_plot.getPlotItem().setTitle(t['loss'])
        self.scatter_plot.getPlotItem().setTitle(t['scatter'])
        if not self._busy():
            self.status.setText(t['ready'])
        self._refresh_r2()
