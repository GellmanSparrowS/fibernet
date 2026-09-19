"""Resumable physical/GUI factorial regression. Run --section ml|search|all.

Six regressors x four acquisitions x three topologies; seven search methods
x three topologies x two objectives. Defaults are retained (60 samples,
40 evaluations). Each case is atomic and keyed by relevant source hashes.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from PySide6.QtWidgets import QApplication
from fslab.learning import MODEL_SPECS, ACQUISITIONS
from fslab.search_algorithms import SEARCH_SPECS, atomic_json
from fslab.structure import StructureFactory, all_unit_keys
from studio import ml_tab
from studio.design_tab import DesignTab


class PhysicalMatrix:
    units = ('square', 'triangle', 'reentrant')

    def __init__(self):
        self.app = QApplication.instance() or QApplication([])
        self.root = Path('_cache/validation23').resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.report = Path('docs/validation/matrix23')
        self.report.mkdir(parents=True, exist_ok=True)
        os.environ['FIBERSCOPE_TRAINING_DIR'] = str(self.root / 'training')

    def case(self, section, key, action):
        files = ['fslab/engine2.py', 'fslab/structure.py']
        files += (['fslab/learning.py', 'fslab/mlmodel.py', 'fslab/dataset_stream.py', 'studio/ml_tab.py'] if section == 'ml'
                  else ['fslab/inverse.py', 'fslab/rl_process.py', 'fslab/search_algorithms.py', 'scripts/rl_worker.py', 'studio/design_tab.py'])
        digest = hashlib.sha256(b''.join(Path(f).read_bytes() for f in files)).hexdigest()
        path = self.report / (section + '_' + key + '.json')
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old.get('source_hash') == digest and old.get('passed') and (section != 'ml' or 'parameters' in old):
                return
        start = time.monotonic()
        try:
            data = action()
        except Exception as exc:
            atomic_json(path, dict(passed=False, source_hash=digest, error=str(exc)))
            raise
        atomic_json(path, dict(passed=True, source_hash=digest, seconds=time.monotonic()-start, **data))
        print(section, key, 'PASS', flush=True)

    def wait(self, done, status, timeout=900):
        start = time.monotonic()
        while not done():
            self.app.processEvents()
            time.sleep(.005)
            if time.monotonic()-start > timeout:
                raise TimeoutError(status())
        for _ in range(4):
            self.app.processEvents()

    def learning(self):
        ml_tab._cache_dir = lambda: str(self.root)
        tab = ml_tab.MLTab(mode='light')
        tab.resize(1250, 820)
        tab.show()
        try:
            assert tab.algorithm == 'mlp' and tab.mode_combo.currentData() == 'physics'
            assert tab.mode_combo.count() == 1
            for unit in self.units:
                tab.unit_combo.setCurrentIndex(tab.unit_combo.findData(unit))
                assert tab.unit_combo.currentData() == unit, unit
                for acquisition in ACQUISITIONS:
                    tab.acquisition = acquisition
                    tab._load_existing()
                    tab.start_gen()
                    self.wait(lambda: tab.gen_worker is None, tab.status.text)
                    assert tab.X.shape == (60, 14) and tab.Y.shape == (60, 3), tab.status.text()
                    assert np.isfinite(tab.Y).all() and len(tab.sample_specs) == 60
                    for key in MODEL_SPECS:
                        def check():
                            tab.algorithm = key
                            tab.parameters = {k: v[0] for k, v in MODEL_SPECS[key][2].items()}
                            tab.model = None
                            tab._sync_model_controls()
                            tab.start_train()
                            self.wait(lambda: tab.train_worker is None, tab.status.text)
                            assert tab.model is not None, tab.status.text()
                            assert len(tab.model.val_indices) == 12
                            assert not set(tab.model.val_indices) & set(tab.model.train_indices)
                            for target in range(3):
                                tab.target_combo.setCurrentIndex(target)
                                tab.scatter_plot.setRange(xRange=(-2, -1), yRange=(-2, -1))
                                tab._update_scatter()
                                self.app.processEvents()
                                x, y = tab.scatter.getData()
                                ranges = tab.scatter_plot.viewRange()
                                assert len(x) == 12 and np.isfinite(y).all()
                                assert min(x) >= ranges[0][0] and max(x) <= ranges[0][1]
                                assert min(y) >= ranges[1][0] and max(y) <= ranges[1][1]
                            assert tab.scatter_plot.isVisible() and tab.scatter_plot.height() >= 180
                            assert len(tab.tr_curve.getData()[0]) > 0
                            if key in ('random_forest', 'extra_trees'):
                                assert len(tab.model.estimator.estimators_) == 80
                                assert tab._tr_pts == list(range(10, 81, 10))
                            if key == 'mlp':
                                from fslab.mlmodel import MLP
                                assert isinstance(tab.model.estimator, MLP)
                            return dict(samples=60, validation=12, targets=3, parameters=tab.parameters,
                                        progress_points=len(tab._tr_pts), status=tab.status.text())
                        self.case('ml', '_'.join((unit, acquisition, key)), check)
        finally:
            tab.close()

    def search(self):
        tab = DesignTab(mode='light')
        tab.resize(1250, 820)
        tab.show()
        assert tab.unit_combo.currentData() == 'square'
        assert set(tab.unit_combo.itemData(i) for i in range(tab.unit_combo.count())) == set(all_unit_keys())
        try:
            for unit in self.units:
                tab.unit_combo.setCurrentIndex(tab.unit_combo.findData(unit))
                tab.set_spec(StructureFactory(unit='square'))
                assert tab.factory.unit == unit
                for target in ('J', 'max_peak'):
                    for algorithm in SEARCH_SPECS:
                        def check():
                            tab.algorithm, tab.parameters = algorithm, {}
                            tab.target_combo.setCurrentText(target)
                            frames = []
                            errors = []
                            tab.start_run()
                            worker = tab.worker
                            def capture(rec, run):
                                if run is not None:
                                    frames.append(hashlib.sha256(run.frames_xy[0].tobytes()).hexdigest())
                            worker.progress.connect(capture)
                            worker.failed.connect(errors.append)
                            self.wait(lambda: getattr(tab, '_result', None) is not None or bool(errors), tab.status.text)
                            worker.wait(5000)
                            assert not errors, errors
                            res = tab._result
                            assert len(res['records']) == 40 and len(frames) == 40, (len(res['records']), len(frames))
                            assert len(set(frames)) > 5, 'candidate geometry failed to change'
                            assert all(r.label == unit for r in res['records'])
                            assert res['best_dist'] <= res['records'][0].dist
                            assert tab.best_run is not None and len(tab.curve_best.getData()[0]) == 24
                            assert len(tab.curve_conv.getData()[0]) == 40
                            assert tab.run_btn.isEnabled() and tab.gb_cfg.isEnabled()
                            return dict(evaluations=40, candidate_updates=len(frames), unique_structures=len(set(frames)),
                                        initial=res['records'][0].dist, best=res['best_dist'], amplitude=tab.amplitude.value())
                        self.case('search', '_'.join((unit, target, algorithm)), check)
        finally:
            if tab.worker and tab.worker.isRunning():
                tab.worker.request_stop()
                tab.worker.wait()
            tab.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', choices=('ml', 'search', 'all'), default='all')
    args = parser.parse_args()
    matrix = PhysicalMatrix()
    if args.section in ('ml', 'all'):
        matrix.learning()
    if args.section in ('search', 'all'):
        matrix.search()


if __name__ == '__main__':
    main()
