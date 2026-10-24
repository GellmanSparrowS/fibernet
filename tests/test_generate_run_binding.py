"""Real small graph/features/persistence; engine sentinel, no Taichi solve."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np
from fibernet.ml.dataset import FiberNetDataset
from fibernet.ml import _generate_contract as c
import fibernet.gen.pattern as pattern
import fibernet.sim.accelerated as accelerated

BASE = dict(units=['square'],grid_range=[(1,1)],radius_range=[.05,.2],
            n_internal_range=[0],box=(1,1),seed=42,num_steps=2,verbose=False)

@pytest.fixture
def trace(monkeypatch):
    calls=[]
    actual=pattern.pattern_2d
    def generate(**kw):
        g=actual(**kw)
        calls.append(('graph',kw,g.to_dict()))
        return g
    class Engine:
        def __init__(self): calls.append(('engine',))
        def stretch_test(self,g,**kw):
            calls.append(('solve',kw))
            return SimpleNamespace(max_force=float(sum(e.radius for e in g.edges.values())),
                max_stretch=1.5,mean_stretch=1.,std_stretch=0.,time_seconds=0.)
    monkeypatch.setattr(pattern,'pattern_2d',generate)
    monkeypatch.setattr(accelerated,'TaichiEngine',Engine)
    return calls

def snap(path):
    return {str(f.relative_to(path)):f.read_bytes() for f in path.rglob('*') if f.is_file()}

def run(path=None,**changes):
    return FiberNetDataset.generate(**{**BASE,**changes},save_dir=path)

def rewrite(path,change):
    f=path/'checkpoint.json';j=json.loads(f.read_text());change(j)
    j['samples_digest']=c.digest(j['samples']);f.write_text(json.dumps(j))

def test_real_radius_product_and_complete_resume(tmp_path,trace):
    ds=run(tmp_path)
    graphs=[v for v in trace if v[0]=='graph']
    assert [g[1]['radius'] for g in graphs]==[.05,.2]
    assert [g[1]['seed'] for g in graphs]==[42,42]
    assert graphs[0][2]['nodes']==graphs[1][2]['nodes']
    for item,r in zip(graphs,[.05,.2]):
        assert all(e['radius']==r for e in item[2]['edges'])
    assert [s.features['mean_radius'] for s in ds.samples]==[.05,.2]
    assert len({s.sample_id for s in ds.samples})==2
    assert all(s.sample_id.startswith('v2-') and len(s.sample_id)==67 for s in ds.samples)
    before=snap(tmp_path);trace.clear();again=run(tmp_path,checkpoint_every=1)
    assert not trace and before==snap(tmp_path)
    assert again.samples[0].features==ds.samples[0].features
    assert FiberNetDataset.load(tmp_path).n_samples==2

@pytest.mark.parametrize('change',[{'box':(2,2)},{'stiffness':2e5},{'damping':.7},
 {'target_stretch':1.1},{'num_steps':3},{'radius_range':[.1,.2]},
 {'radius_range':[.2,.05]},{'seed':43},{'n_internal_range':[1]},
 {'grid_range':[(2,1)]},{'max_samples':1},{'units':['triangle']},
 {'geometry_policy':'endpoint_chord'}])
def test_changed_run_rejects_before_callbacks_and_writes(tmp_path,trace,change):
    run(tmp_path);trace.clear();before=snap(tmp_path)
    with pytest.raises(ValueError,match='binding mismatch'):run(tmp_path,**change)
    assert not trace and snap(tmp_path)==before

@pytest.mark.parametrize('kind',['legacy','version','digest','row','foreign','duplicate','count'])
def test_invalid_checkpoint_fails_closed(tmp_path,trace,kind):
    run(tmp_path)
    def change(j):
        if kind=='legacy':j.pop('checkpoint_schema')
        elif kind=='version':j['checkpoint_schema']='unknown'
        elif kind=='digest':j['run_binding_sha256']='bad'
        elif kind=='row':j['samples'][0]['params']['radius']=.8
        elif kind=='foreign':j['samples'][0]['id']='v2-foreign'
        elif kind=='duplicate':j['samples'][1]=copy.deepcopy(j['samples'][0])
        elif kind=='count':j['n_samples']=90
    rewrite(tmp_path,change);before=snap(tmp_path);trace.clear()
    with pytest.raises(ValueError):run(tmp_path)
    assert not trace and snap(tmp_path)==before


def test_partial_resume_only_missing(tmp_path,trace):
    ds=run(tmp_path)
    def change(j):j['samples']=j['samples'][:1];j['n_samples']=1
    rewrite(tmp_path,change);trace.clear();again=run(tmp_path)
    assert len([v for v in trace if v[0]=='solve'])==1
    assert [s.features for s in again.samples]==[s.features for s in ds.samples]


def test_implementation_change_rejects(tmp_path,trace,monkeypatch):
    run(tmp_path);before=snap(tmp_path);trace.clear();original=c.implementation_recipe
    def recipe(*a):
        d=original(*a);d['source_sha256']['sim/accelerated.py']='changed';return d
    monkeypatch.setattr(c,'implementation_recipe',recipe)
    with pytest.raises(ValueError,match='binding mismatch'):run(tmp_path)
    assert not trace and before==snap(tmp_path)


def test_registry_custom_fresh_and_resume_rejected_for_checkpoint(tmp_path,trace,monkeypatch):
    run(tmp_path);trace.clear();before=snap(tmp_path)
    monkeypatch.setitem(pattern._UNIT_FACTORIES,'square',pattern._unit_triangle)
    for path in (tmp_path,tmp_path/'new'):
        with pytest.raises(ValueError,match='known built-in'):run(path)
    assert not trace and snap(tmp_path)==before
    assert run(None).n_samples==2

@pytest.mark.parametrize('change',[{'radius_range':[.1,.1]},{'units':['square','square']},
 {'max_samples':0},{'seed':True},{'num_steps':1.5},{'checkpoint_every':0},
 {'radius_range':[float('nan')]},{'box':(float('inf'),1)},{'radius_range':[-1]}])
def test_invalid_inputs_early(tmp_path,trace,change):
    with pytest.raises(ValueError):run(tmp_path,**change)
    assert not trace and not list(tmp_path.iterdir())


def test_forced_id_collision_early(tmp_path,trace,monkeypatch):
    monkeypatch.setattr(c,'sample_id',lambda p:'v2-collision')
    with pytest.raises(ValueError,match='duplicate or colliding'):run(tmp_path)
    assert not trace and not list(tmp_path.iterdir())


def test_shuffle_and_empty_plan(tmp_path,trace):
    ds=run(tmp_path/'one',max_samples=1)
    expected=[.05,.2];np.random.RandomState(42).shuffle(expected)
    assert ds.samples[0].params['radius']==expected[0]
    trace.clear();assert run(tmp_path/'empty',units=[]).n_samples==0
    assert not trace
    before=snap(tmp_path/'empty');assert run(tmp_path/'empty',units=[]).n_samples==0
    assert not trace and snap(tmp_path/'empty')==before


def test_old_tmp_sentinel_untouched(tmp_path,trace):
    sentinel=tmp_path/'checkpoint.json.tmp';sentinel.write_bytes(b'old sidecar')
    run(tmp_path);assert sentinel.read_bytes()==b'old sidecar'


def test_native_without_checkpoint_rejects(tmp_path,trace):
    (tmp_path/'metadata.json').write_text('{}');before=snap(tmp_path)
    with pytest.raises(ValueError,match='no bound checkpoint'):run(tmp_path)
    assert not trace and before==snap(tmp_path)

@pytest.mark.parametrize('changes',[{'radius_range':[]},{'n_internal_range':[]},{'grid_range':[]},{'units':[]}])
def test_explicit_empty_ranges_are_zero_work(tmp_path,trace,changes):
    assert run(tmp_path,**changes).n_samples==0
    assert not trace and json.loads((tmp_path/'checkpoint.json').read_text())['samples']==[]

@pytest.mark.parametrize('changes',[{'units':[' ']},{'max_samples':False},{'max_samples':-1},
 {'seed':-1},{'seed':2**32},{'radius_range':True},{'grid_range':[(1,)]}])
def test_more_invalid_inputs(tmp_path,trace,changes):
    with pytest.raises(ValueError):run(tmp_path,**changes)
    assert not trace and not list(tmp_path.iterdir())


def test_large_cap_and_seed_upper_bound(tmp_path,trace):
    ds=run(tmp_path,max_samples=100,seed=2**32-1)
    assert ds.n_samples==2
    assert all(s.params['seed']==2**32-1 for s in ds.samples)
    trace.clear()
    with pytest.raises(ValueError,match='binding mismatch'):run(tmp_path,max_samples=101,seed=2**32-1)
    assert not trace

@pytest.mark.parametrize('kind',['none','bool_count','nan_label','metadata'])
def test_bad_envelope_rows(tmp_path,trace,kind):
    run(tmp_path)
    def mutate(j):
        if kind=='none':j['samples']=None
        elif kind=='bool_count':j['n_samples']=True
        elif kind=='metadata':j['samples'][0]['metadata']=[]
        else:j['samples'][0]['labels']['max_force']=float('nan')
    f=tmp_path/'checkpoint.json';j=json.loads(f.read_text());mutate(j)
    if kind!='nan_label':j['samples_digest']=c.digest(j['samples'])
    f.write_text(json.dumps(j));before=snap(tmp_path);trace.clear()
    with pytest.raises(ValueError):run(tmp_path)
    assert not trace and snap(tmp_path)==before


def test_failed_jobs_retry_without_accepted_nan(tmp_path,trace,monkeypatch):
    class Failing:
        def stretch_test(self,g,**kw):
            raise RuntimeError('synthetic failure')
    with monkeypatch.context() as m:
        m.setattr(accelerated,'TaichiEngine',Failing)
        assert run(tmp_path).n_samples==0
    assert json.loads((tmp_path/'checkpoint.json').read_text())['samples']==[]
    trace.clear();assert run(tmp_path).n_samples==2
    assert len([v for v in trace if v[0]=='solve'])==2


def test_replace_failure_preserves_checkpoint_and_sidecar(tmp_path,trace,monkeypatch):
    import fibernet.ml.dataset as d
    run(tmp_path)
    rewrite(tmp_path,lambda j:(j.update(samples=j['samples'][:1],n_samples=1)))
    sentinel=tmp_path/'checkpoint.json.tmp';sentinel.write_bytes(b'untouched')
    before=snap(tmp_path)
    def fail(*a):raise OSError('replace failed')
    monkeypatch.setattr(d.os,'replace',fail)
    with pytest.raises(OSError,match='replace failed'):run(tmp_path)
    assert snap(tmp_path)==before


def test_cleanup_handler_cannot_mask_commit(tmp_path,trace,monkeypatch):
    import fibernet.ml.dataset as d
    original=Path.unlink
    def fail_owned(path,*a,**kw):
        if path.name.startswith('.checkpoint-'):raise OSError('cleanup failed')
        return original(path,*a,**kw)
    class Logger:
        def warning(self,*a,**kw):raise RuntimeError('handler failed')
    with monkeypatch.context() as m:
        m.setattr(Path,'unlink',fail_owned)
        m.setattr(d.logging,'getLogger',lambda *a:Logger())
        assert run(tmp_path).n_samples==2
    assert json.loads((tmp_path/'checkpoint.json').read_text())['n_samples']==2


def test_complete_checkpoint_does_not_repair_missing_native_files(tmp_path,trace):
    run(tmp_path);(tmp_path/'features.npy').unlink();trace.clear();before=snap(tmp_path)
    assert run(tmp_path).n_samples==2
    assert not trace and snap(tmp_path)==before
    with pytest.raises((FileNotFoundError,ValueError)):FiberNetDataset.load(tmp_path)


def test_duplicate_json_keys_rejected(tmp_path,trace):
    run(tmp_path);f=tmp_path/'checkpoint.json'
    f.write_text(f.read_text().replace('{','{"checkpoint_schema":"shadow",',1))
    trace.clear();before=snap(tmp_path)
    with pytest.raises(ValueError,match='duplicate JSON'):run(tmp_path)
    assert not trace and snap(tmp_path)==before


def test_zero_stiffness_retains_backend_supported_scalar(tmp_path,trace):
    assert run(tmp_path,stiffness=0).n_samples==2
    assert all(v[1]['stiffness']==0 for v in trace if v[0]=='solve')

@pytest.mark.parametrize('change',[{'target_stretch':0},{'target_stretch':-1},
                                  {'damping':1.5},{'damping':-.1}])
def test_binding_does_not_invent_solver_physical_ranges(tmp_path,trace,change):
    assert run(tmp_path,**change).n_samples==2
    key,value=next(iter(change.items()))
    assert all(v[1][key]==value for v in trace if v[0]=='solve')
