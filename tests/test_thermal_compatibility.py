"""Compatibility/side-effect checks; all solver calls here are test doubles."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import builtins
import numpy as np
import pytest
# Preload shared package imports before the repository default pytest-forked
# subprocesses; importing threaded optional dependencies after fork can hang.
import fibernet
from fibernet import api
from fibernet.utils.config import create_template_config, run_from_config
from fibernet.sim import transient_thermal as th

ROOT=Path(__file__).parents[1]


def load_example(name):
    spec=importlib.util.spec_from_file_location('thermal_example_test_'+name,ROOT/'examples'/name)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_old_api_immediate_clear_failure(monkeypatch):
    from fibernet import api
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name!='fibernet.sim.thermal'
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    with pytest.raises(NotImplementedError,match='not an equivalent replacement'):
        api.simulate_thermal(object())


def test_thermal_template_refuses_before_writing(tmp_path,monkeypatch):
    from fibernet.utils.config import create_template_config
    def forbidden(*a,**k):raise AssertionError('output write attempted')
    monkeypatch.setattr(Path,'write_text',forbidden)
    with pytest.raises(NotImplementedError,match='no file was written'):
        create_template_config(tmp_path/'no.yaml',template_type='thermal')
    assert not list(tmp_path.iterdir())


def test_thermal_run_refuses_before_generation_or_hash(tmp_path):
    from fibernet.utils.config import run_from_config
    class Sentinel:
        simulation={'type':'thermal'}
        def validate(self):raise AssertionError('validation/generation phase reached')
        def compute_hash(self):raise AssertionError('execution metadata phase reached')
    with pytest.raises(NotImplementedError,match='no network was generated'):
        run_from_config(Sentinel())
    assert not list(tmp_path.iterdir())


def test_thermal_file_config_also_rejects(tmp_path):
    from fibernet.utils.config import run_from_config
    path=tmp_path/'input.json';path.write_text('{"simulation":{"type":"thermal"}}')
    with pytest.raises(NotImplementedError,match='unsupported'):run_from_config(path)
    assert sorted(x.name for x in tmp_path.iterdir())==['input.json']


@pytest.mark.parametrize('name',['03_thermal_analysis.py','transient_thermal_explicit.py'])
def test_example_import_has_no_backend_import_or_output(name,monkeypatch,capsys):
    original=builtins.__import__
    def guard(module,*args,**kwargs):
        assert not module.startswith(('fibernet','numpy'))
        return original(module,*args,**kwargs)
    # Limit the import guard to the example globals, not pytest's own imports.
    namespace={'__name__':'import_safe_example',
               '__builtins__':dict(vars(builtins),__import__=guard)}
    exec(compile((ROOT/'examples'/name).read_text(),str(ROOT/'examples'/name),'exec'),namespace)
    assert capsys.readouterr().out==''


def test_old_notice_nonzero_and_clear(capsys):
    m=load_example('03_thermal_analysis.py');assert m.main()==2
    out=capsys.readouterr().out
    assert 'unsupported' in out and 'DIFFERENT' in out and 'CTE' in out


def test_explicit_example_inputs_and_success_use_double(monkeypatch,capsys):
    from fibernet.sim import transient_thermal as th
    calls=[]
    def fake(graph,**kw):
        calls.append((graph,kw))
        return SimpleNamespace(completed=True,times_s=np.array([0,.01,.02]),node_ids=np.array([101,907]),
           temperature_K=np.array([[300.,310.]]*3),node_stored_heat_change_J=np.zeros((3,2)),reservoir_injection_W=np.zeros((3,2)),free_node_residual_J=np.array([[np.nan,np.nan],[0,np.nan],[0,np.nan]]),edge_ids=np.array([11,31]),endpoints_by_id=np.array([[101,907],[101,907]]),edge_heat_rate_W=np.zeros((3,2)))
    monkeypatch.setattr(th,'solve_transient_thermal',fake)
    assert load_example('transient_thermal_explicit.py').main()==0
    graph,kw=calls[0];assert list(graph.edges)==[11,31]
    assert kw=={'capacity_J_per_K':{101:2.,907:1.},'conductance_W_per_K':{11:.2,31:.3},'initial_temperature_K':{101:300.,907:310.},'times_s':[0.,.01,.02],'source_W':{101:0.,907:0.},'fixed_temperature_K':{907:310.},'length_unit':'m'}
    assert 'unavailable' in capsys.readouterr().out


@pytest.mark.parametrize('partial',[None,SimpleNamespace(times_s=np.array([0,.01]))])
def test_example_failure_prefix_messages(monkeypatch,capsys,partial):
    from fibernet.sim import transient_thermal as th
    def fail(*a,**kw):raise th.ThermalSolveError('injected test failure',partial,0 if partial is None else 2,0 if partial is None else .02)
    monkeypatch.setattr(th,'solve_transient_thermal',fail)
    assert load_example('transient_thermal_explicit.py').main()==1
    out=capsys.readouterr().out
    assert ('no frame accepted' if partial is None else 'last accepted time = 0.01') in out


def test_no_root_thermal_aliases():
    import fibernet
    assert not hasattr(fibernet,'simulate_thermal')
    assert not hasattr(fibernet,'solve_transient_thermal')


def test_import_does_not_call_solver(monkeypatch):
    calls=[]
    def forbidden(*a,**kw):
        calls.append(True)
        raise AssertionError('solver called during import')
    monkeypatch.setattr(th,'solve_transient_thermal',forbidden)
    load_example('transient_thermal_explicit.py')
    load_example('03_thermal_analysis.py')
    assert calls==[]


def test_legacy_script_subprocess_is_notice_only():
    import subprocess,sys
    completed=subprocess.run([sys.executable,str(ROOT/'examples/03_thermal_analysis.py')],capture_output=True,text=True,timeout=3)
    assert completed.returncode==2
    assert 'unsupported' in completed.stdout and 'DIFFERENT' in completed.stdout
    assert 'Traceback' not in completed.stderr
