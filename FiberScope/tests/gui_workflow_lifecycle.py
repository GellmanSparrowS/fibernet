"""Deterministic Qt worker-lifecycle regressions without physics backends.

Run from the repository root:
    python FiberScope/scripts/run_tests.py --only gui_workflow_lifecycle

Exercises the actual workflow and QObject observer with real QThreads and queued
signals. Backend imports are explicit doubles; this is not a full-app GUI or
physics test. All journal files live in a temporary directory.
"""
import dataclasses
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import tempfile
import time
import types

APP_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SOURCE = APP_ROOT / 'studio/workflow.py'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_LOGGING_RULES'] = 'qt.pyside.libpyside.warning=true'


class NoScienceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'fibernet', 'numpy', 'scipy', 'pyqtgraph', 'sklearn', 'matplotlib',
            'manifold3d', 'exudyn', 'mujoco', 'trimesh', 'skimage',
        }:
            raise RuntimeError('Forbidden science/native backend import: ' + fullname)
        return None


sys.meta_path.insert(0, NoScienceImports())
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QThread, QTimer, Signal, qInstallMessageHandler, qVersion
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QTabWidget, QWidget
import PySide6
import shiboken6


APP = QApplication([])
MAIN_IDENT = threading.get_ident()
QT_MESSAGES = []
PYTHON_EXCEPTIONS = []
WORKERS = []
PANELS = []
RUNNERS = []
RESULTS = []
BACKEND_STUBS = []
TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix='fibernet-workflow-lifecycle-')
JOURNALS = Path(TEMP_DIRECTORY.name)


def qt_message(kind, context, message):
    QT_MESSAGES.append({'kind': str(kind), 'message': message})


def python_exception(kind, value, traceback):
    PYTHON_EXCEPTIONS.append(f'{kind.__name__}: {value}')


qInstallMessageHandler(qt_message)
sys.excepthook = python_exception


def gui_thread():
    assert QThread.currentThread() == APP.thread(), 'callback outside QApplication thread'
    assert threading.get_ident() == MAIN_IDENT


def forbidden_backend(*args, **kwargs):
    raise AssertionError('No physics/export/manufacturing backend may execute')


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    BACKEND_STUBS.append(name)
    return module


stub('fslab', __path__=[])
stub('fslab.storage', writable_data_dir=lambda kind: str(JOURNALS))
stub('fslab.exporter', export_json=forbidden_backend)
stub('fslab.print_export', PrintSettings=object, export_solid=forbidden_backend, export_route=forbidden_backend)
stub('fslab.inverse', TARGETS={'C': object()}, SCALARS={})
stub('fslab.structure', all_unit_keys=lambda: ['hexagon'])


def load_workflow(source, package):
    stub(package, __path__=[])
    stub(package + '.manufacturing_dialog', ManufacturingWorker=forbidden_backend, ManufacturingDialog=forbidden_backend)
    name = package + '.workflow'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORKFLOW = load_workflow(WORKFLOW_SOURCE, 'lifecycle_test_studio')


@dataclasses.dataclass
class Factory:
    unit: str = 'hexagon'


class Panel(QWidget):
    def __init__(self):
        super().__init__()
        self.tabs = QTabWidget(self)
        self.tabs.addTab(QWidget(), 'test')
        self.navigation = []
        self.events = []
        self.win = types.SimpleNamespace(
            tabs=self.tabs,
            tab_struct=types.SimpleNamespace(collect_factory=Factory, manufacturing_dialog=None),
            tab_surface=types.SimpleNamespace(_obj_files=[], manufacturing_dialog=None),
            tab_sim=types.SimpleNamespace(worker=None, stretch=QDoubleSpinBox(self)),
            tab_design=types.SimpleNamespace(worker=None, _auto_apply=False),
            tab_features=types.SimpleNamespace(worker=None),
            tab_ml=types.SimpleNamespace(gen_worker=None, train_worker=None, X=None, _train_after_generation=False),
        )
        self.win.tab_sim.stretch.setRange(1, 20)
        self.registry = {'get_app_state': lambda: {'tab': 'test'}}
        PANELS.append(self)

    def _navigate(self, page):
        gui_thread()
        self.navigation.append(page)

    def _workflow_tool_started(self, stage, options):
        gui_thread()
        self.events.append(('started', stage))

    def _workflow_tool_finished(self, stage, result, success):
        gui_thread()
        self.events.append(('finished', stage, success))


class ProbeMixin:
    """Instrumentation delegates directly to every unchanged product method."""
    def _ok(self, *args):
        gui_thread()
        self.probe.append(('ok', self.network, self.solid))
        return super()._ok(*args)

    def _fail(self, message):
        gui_thread()
        self.probe.append(('fail', message))
        return super()._fail(message)

    def _solid(self, network, solid):
        gui_thread()
        self.probe.append(('solid', network, solid))
        return super()._solid(network, solid)

    def _save(self):
        gui_thread()
        return super()._save()


def new_runner(module=WORKFLOW):
    panel = Panel()
    cls = type('InstrumentedWorkflow', (ProbeMixin, module.PrintWorkflow), {})
    runner = cls(panel)
    runner.probe = []
    runner.start(samples=10, iterations=1, open_slicer=False)
    runner.timer.stop()  # Every event-loop entry below is explicit and bounded.
    runner.manufacturing_dialog = types.SimpleNamespace(source_network=None)
    RUNNERS.append(runner)
    return runner


class Worker(QThread):
    succeeded0 = Signal()
    succeeded1 = Signal(object)
    succeeded3 = Signal(object, object, object)
    succeeded_sim = Signal(object, object, bool)
    succeeded_gen = Signal(object, object, str)
    finished_ok = Signal(object, object, bool)
    completed = Signal(object, object)
    failed = Signal(str)

    def __init__(self, events, *, finish_inside_start=False, hold_before=False, hold_after=False):
        super().__init__()
        self.events = events
        self.finish_inside_start = finish_inside_start
        self.hold_before = hold_before
        self.hold_after = hold_after
        self.ready = threading.Event()
        self.emitted = threading.Event()
        self.release = threading.Event()
        self.stops = 0
        self.error = None
        self.emitter_ident = None
        WORKERS.append(self)

    def request_stop(self):
        self.stops += 1  # Keep the thread alive until the harness releases it.

    def start(self):
        assert not any(w is not self and shiboken6.isValid(w) and w.isRunning() for w in WORKERS)
        super().start()
        if self.finish_inside_start:
            assert self.wait(1000), 'worker did not finish inside start()'
            assert self.error is None, self.error

    def run(self):
        try:
            self.emitter_ident = threading.get_ident()
            assert self.emitter_ident != MAIN_IDENT
            self.ready.set()
            if self.hold_before:
                assert self.release.wait(1.0), 'pre-emission worker release timeout'
            for signal, args in self.events:
                getattr(self, signal).emit(*args)
            self.emitted.set()
            if self.hold_after:
                assert self.release.wait(1.0), 'post-emission worker release timeout'
        except BaseException as exc:
            self.error = repr(exc)
            self.ready.set()
            self.emitted.set()


def join(worker):
    assert worker.wait(1000), 'QThread join timeout'
    assert not worker.isRunning()
    assert worker.error is None, worker.error


def pump(milliseconds=10, check_exceptions=True):
    gui_thread()
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    if check_exceptions:
        assert not PYTHON_EXCEPTIONS, PYTHON_EXCEPTIONS


def stop_runner(runner):
    if runner.running:
        runner.cancel()
    pump()
    assert not runner.running and not runner.timer.isActive()


def test_completion_before_start_returns():
    runner = new_runner()
    worker = Worker([('finished_ok', (object(), object(), True))], finish_inside_start=True)

    def start_run(*, autostart=True):
        runner.win.tab_sim.worker = worker
        if autostart:
            worker.start()
        return worker

    runner.win.tab_sim.start_run = start_run
    runner.index = 1
    runner._tick()  # Actual _tick -> _begin -> prepare -> subscribe -> start.
    assert not runner.success, 'GUI slot executed before queued events were processed'
    join(worker)
    pump()
    assert runner.success
    assert worker.emitter_ident != MAIN_IDENT
    stop_runner(runner)
    return {'completion_before_start_returns_captured': True}


def test_arities_and_terminal_order():
    details = []
    for signal, args in [('succeeded0', ()), ('succeeded1', (object(),)), ('succeeded3', (object(), object(), object())),
                         ('succeeded_sim', (object(), object(), True)),
                         ('succeeded_gen', (object(), object(), 'cached'))]:
        runner = new_runner()
        worker = Worker([(signal, args), (signal, args), ('failed', ('late',))], finish_inside_start=True)
        runner._start_worker(worker, signal)
        assert not runner.success and runner.error is None
        observer = runner._observer
        assert observer.thread() == APP.thread() and observer.parent() is runner
        pump()
        assert runner.success and runner.error is None
        assert [x[0] for x in runner.probe] == ['ok']
        stop_runner(runner)
        assert not shiboken6.isValid(observer)
        details.append(signal)
    runner = new_runner()
    worker = Worker([('failed', ('',)), ('succeeded0', ()), ('failed', ('later',))], finish_inside_start=True)
    runner._start_worker(worker, 'succeeded0')
    pump()
    assert not runner.success and runner.error == 'stage failed without an error message'
    assert [x[0] for x in runner.probe] == ['fail']
    stop_runner(runner)
    return {'success_arities': details, 'first_terminal_wins': True, 'empty_failure_preserved': True}


def test_solid_payload_atomicity():
    runner = new_runner()
    network, solid, replacement_network, replacement_solid = object(), object(), object(), object()
    worker = Worker([('completed', (network, solid)), ('completed', (replacement_network, replacement_solid)), ('failed', ('late',))], finish_inside_start=True)
    runner._start_worker(worker, 'completed', solid=True)
    assert not runner.success and runner.network is None and runner.solid is None
    pump()
    assert runner.success and runner.error is None
    assert runner.network is network and runner.solid is solid
    assert runner.manufacturing_dialog.source_network is network
    assert [x[0] for x in runner.probe] == ['solid', 'ok']
    assert runner.probe[1][1] is network and runner.probe[1][2] is solid
    stop_runner(runner)
    return {'payload_before_success': True, 'duplicate_did_not_replace': True}


def test_queued_stale_signals_cancel_restart():
    runner = new_runner()
    old_network, old_solid = object(), object()
    old = Worker([('completed', (old_network, old_solid)), ('failed', ('stale-failure',))], hold_after=True)
    runner._start_worker(old, 'completed', solid=True)
    assert old.emitted.wait(1.0), 'old signals not queued'
    assert old.isRunning() and not runner.success
    retired = runner._observer
    runner.cancel()  # Retire + disconnect while old callback events remain queued.
    assert runner.status()['state'] == 'cancelled' and old.stops >= 1
    try:
        runner.start(samples=10, iterations=1, open_slicer=False)
    except ValueError as exc:
        assert 'running or stopping' in str(exc)
    else:
        raise AssertionError('restart accepted while old QThread still running')
    old.release.set()
    join(old)
    old.deleteLater()  # Also exercise tab-owned worker deletion after completion.
    runner.start(samples=10, iterations=1, open_slicer=False)
    runner.timer.stop()
    # Restart-before-first-tick: do not install the next observer until stale drain.
    pump()
    assert not runner.success and runner.error is None and runner.network is None and runner.solid is None
    assert not runner.probe, runner.probe
    assert not shiboken6.isValid(retired) and not shiboken6.isValid(old)
    fresh = Worker([('succeeded1', (object(),))], finish_inside_start=True)
    runner._start_worker(fresh, 'succeeded1')
    pump()
    assert runner.success and runner.error is None and len(runner.probe) == 1
    stop_runner(runner)
    return {'running_restart_rejected': True, 'stale_success_failure_solid_ignored': True, 'deferred_deletion_safe': True}


def test_replaced_observer_stale_queue():
    runner = new_runner()
    old = Worker([('succeeded3', (1, 2, 3)), ('failed', ('old',))], finish_inside_start=True)
    runner._start_worker(old, 'succeeded3')
    retired = runner._observer
    fresh = Worker([('succeeded0', ())], hold_before=True)
    runner._start_worker(fresh, 'succeeded0')
    assert fresh.ready.wait(1.0)
    pump()
    assert not runner.success and runner.error is None and not runner.probe
    assert not shiboken6.isValid(retired)
    fresh.release.set()
    join(fresh)
    pump()
    assert runner.success and [x[0] for x in runner.probe] == ['ok']
    stop_runner(runner)
    return {'old_queue_cannot_complete_new_stage': True}


def main():
    started = time.monotonic()
    failure = None
    try:
        for test in [test_completion_before_start_returns, test_arities_and_terminal_order,
                     test_solid_payload_atomicity, test_queued_stale_signals_cancel_restart,
                     test_replaced_observer_stale_queue]:
            detail = test()
            RESULTS.append({'test': test.__name__, 'status': 'PASS', 'detail': detail})
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        # All holds are released even after an assertion; never destroy live QThread.
        cleanup_errors = []
        alive = []
        for worker in WORKERS:
            try:
                if shiboken6.isValid(worker):
                    worker.release.set()
            except BaseException as exc:
                cleanup_errors.append(f'release {id(worker)}: {type(exc).__name__}: {exc}')
        cleanup_deadline = time.monotonic() + 2.0
        for worker in WORKERS:
            try:
                if shiboken6.isValid(worker):
                    remaining = max(0, int((cleanup_deadline - time.monotonic()) * 1000))
                    if not worker.wait(remaining):
                        alive.append(id(worker))
                    else:
                        worker.deleteLater()
            except BaseException as exc:
                cleanup_errors.append(f'join {id(worker)}: {type(exc).__name__}: {exc}')
                alive.append(id(worker))  # Unknown lifecycle is not safe teardown.
        if not alive:
            for runner in RUNNERS:
                try:
                    if runner.running:
                        runner.cancel()
                except BaseException as exc:
                    cleanup_errors.append(f'cancel: {type(exc).__name__}: {exc}')
            for panel in PANELS:
                try:
                    panel.deleteLater()
                except BaseException as exc:
                    cleanup_errors.append(f'panel delete: {type(exc).__name__}: {exc}')
            try:
                pump(check_exceptions=False)
            except BaseException as exc:
                cleanup_errors.append(f'deferred cleanup: {type(exc).__name__}: {exc}')
        bad = [x for x in QT_MESSAGES if any(s in x['message'].lower() for s in
               ['different thread', 'cannot queue', 'destroyed while thread', 'timers cannot', 'qobject::setparent'])]
        observer_dynamic = [x for x in QT_MESSAGES if '_WorkflowObserver' in x['message'] and 'Registering dynamic slot' in x['message']]
        if failure is None and (cleanup_errors or PYTHON_EXCEPTIONS or bad or observer_dynamic):
            failure = 'callback/Qt/cleanup diagnostics failed; see receipt lists'
        report = {
            'status': 'PASS' if failure is None and not alive else 'FAIL',
            'failure': failure, 'surviving_workers': alive, 'tests': RESULTS,
            'wall_seconds': time.monotonic() - started, 'qt_version': qVersion(),
            'backend_stub_modules': BACKEND_STUBS,
            'instrumentation': 'GUI-thread assertions wrap/delegate unchanged _ok/_fail/_solid/_save; observer class and slots unchanged',
            'scope': 'Actual workflow/observer with real Qt and controlled worker/panel/backend boundaries; not full-app GUI or physics coverage',
            'python_exceptions': PYTHON_EXCEPTIONS, 'qt_messages': QT_MESSAGES,
            'cleanup_errors': cleanup_errors, 'forbidden_qt_diagnostics': bad,
            'observer_dynamic_slot_warnings': observer_dynamic,
            'production_signal_signatures': {'SimWorker.finished_ok': ['object', 'object', 'bool'], 'GenWorker.done': ['object', 'object', 'str'], 'train_design_done': ['object']},
        }
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        if alive:
            os._exit(2)  # Avoid unsafe Python/C++ destructor teardown of live threads.
        TEMP_DIRECTORY.cleanup()
    return 0 if failure is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
