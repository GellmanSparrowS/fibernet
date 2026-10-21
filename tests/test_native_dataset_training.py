"""Native easy-training names, selection, split intent, and retained assertions."""
import copy
import json
import io
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('sklearn')
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from fibernet.easy import train_model, predict_named
from fibernet.ml.dataset import DatasetSample, FiberNetDataset


@pytest.fixture
def dataset():
    return FiberNetDataset([
        DatasetSample(f'row-{i}', 'fixture', (1, 1), {'unselected': object()},
                      {'length_m': float(i), 'area_m2': float(i % 5) ** 2},
                      {'force_N': float(2*i + 3*(i % 5)**2), 'unused_label': None},
                      {'length_unit': 'm', 'target_unit': 'N',
                       'meaning': {'text': 'source assertion', 'availability': [None, True, 2, 1.5]},
                       'unselected': object()})
        for i in range(24)
    ], ['length_m', 'area_m2'], ['force_N', 'unused_label'])


def native(dataset, **kwargs):
    options=dict(target='force_N', model_type='lr', test_size=.25, split='row')
    options.update(kwargs)
    return train_model(dataset, **options)


@pytest.fixture
def no_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid native request entered estimator.fit')
    for cls in (LinearRegression, RandomForestRegressor, GradientBoostingRegressor):
        monkeypatch.setattr(cls, 'fit', forbidden)


@pytest.fixture
def no_split(no_fit, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid selected input entered splitting')
    monkeypatch.setattr('sklearn.model_selection.train_test_split', forbidden)
    monkeypatch.setattr('sklearn.model_selection.GroupShuffleSplit.split', forbidden)


@pytest.mark.parametrize('kind,cls,settings', [
    ('lr', LinearRegression, {}),
    ('rf', RandomForestRegressor, {'n_estimators': 100, 'random_state': 42}),
    ('gb', GradientBoostingRegressor, {'n_estimators': 100, 'random_state': 42}),
])
def test_native_estimators_raw_names_artifact_and_named_inference(dataset, tmp_path, kind, cls, settings):
    names=['area_m2', 'length_m']
    view=FiberNetDataset(dataset.samples, names, ['force_N'])
    X,y=view.features,view.labels('force_N')
    train,test=train_test_split(np.arange(len(y)), test_size=.25, random_state=42)
    original=cls(**settings).fit(X[train], y[train])
    path=tmp_path/'native.pkl'
    result=native(dataset, features=names, model_type=kind, save_path=path)
    assert result['features']==names and result['features'] is not names
    assert result['target']=='force_N' and result['n_samples']==24
    assert result['r2']==r2_score(y[test],original.predict(X[test]))
    assert result['rmse']==np.sqrt(mean_squared_error(y[test],original.predict(X[test])))
    with path.open('rb') as handle: artifact=pickle.load(handle)
    assert set(artifact)=={'model','features','metrics','artifact_version','target'}
    assert artifact['metrics']==result and artifact['artifact_version']==1
    assert type(artifact['model']) is cls
    np.testing.assert_array_equal(artifact['model'].predict(X),original.predict(X))
    records=[{name:s.features[name] for name in reversed(names)} for s in dataset.samples]
    answer=predict_named(artifact,records)
    np.testing.assert_array_equal(answer['predictions'],original.predict(X))
    assert answer['validation']=='named_columns_only' and answer['target']=='force_N'
    assert dataset.feature_names==['length_m','area_m2']


def test_native_default_feature_order_selected_target_and_duplicate_ids(dataset):
    for sample in dataset.samples: sample.sample_id='duplicate-id'
    result=native(dataset)
    assert result['features']==dataset.feature_names
    provenance=result['training_data']
    assert provenance['source_kind']=='FiberNetDataset'
    assert provenance['metadata_keys']==[]
    assert provenance['rows']==[{'row_index':i,'sample_id':'duplicate-id'} for i in range(24)]
    split=provenance['split']
    train,test=train_test_split(np.arange(24),test_size=.25,random_state=42)
    assert split=={'method':'row','random_state':42,'train_rows':train.tolist(),'test_rows':test.tolist()}
    assert dataset.samples[0].labels['unused_label'] is None
    assert dataset.target_names==['force_N','unused_label']
    json.dumps(result,allow_nan=False)


def test_native_csv_path_keyword_accepts_explicit_object(dataset):
    assert train_model(csv_path=dataset,target='force_N',model_type='lr',split='row',test_size=.25)['target']=='force_N'


def test_raw_label_prefix_is_not_native_namespace(dataset):
    for sample in dataset.samples: sample.features['label_sensor']=sample.features['length_m']
    dataset.feature_names.append('label_sensor')
    assert native(dataset,features=['label_sensor'])['features']==['label_sensor']


def test_metadata_is_selected_deep_copied_and_source_unmodified(dataset):
    names_before=list(dataset.feature_names)
    features_before=[dict(s.features) for s in dataset.samples]
    labels_before=[dict(s.labels) for s in dataset.samples]
    metadata_before=[dict(s.metadata) for s in dataset.samples]
    keys=['length_unit','target_unit','meaning']
    result=native(dataset,metadata_keys=keys)
    provenance=result['training_data']
    assert provenance['metadata_keys']==keys and provenance['metadata_keys'] is not keys
    assert [row['metadata'] for row in provenance['rows']]==[
        {key:s.metadata[key] for key in keys} for s in dataset.samples]
    assert dataset.feature_names==names_before
    assert [s.features for s in dataset.samples]==features_before
    assert [s.labels for s in dataset.samples]==labels_before
    assert [s.metadata for s in dataset.samples]==metadata_before
    dataset.samples[0].metadata['meaning']['availability'].append('later-source-edit')
    assert provenance['rows'][0]['metadata']['meaning']['availability']==[None,True,2,1.5]
    provenance['rows'][1]['metadata']['meaning']['availability'].append('result-edit')
    assert dataset.samples[1].metadata['meaning']['availability']==[None,True,2,1.5]
    json.dumps(provenance,allow_nan=False)


def test_metadata_empty_selection_ignores_all_opaque_metadata(dataset):
    dataset.samples[0].metadata=object()
    assert native(dataset,metadata_keys=[])['training_data']['metadata_keys']==[]


def test_metadata_shared_noncyclic_containers_are_copied(dataset):
    shared=[1,2]
    for sample in dataset.samples: sample.metadata['meaning']={'left':shared,'right':shared}
    rows=native(dataset,metadata_keys=['meaning'])['training_data']['rows']
    rows[0]['metadata']['meaning']['left'].append(3)
    assert rows[0]['metadata']['meaning']['right']==[1,2]
    assert rows[1]['metadata']['meaning']['left']==[1,2] and shared==[1,2]


@pytest.mark.parametrize('bad',[None,np.nan,np.inf,-np.inf,'1',1+0j,np.ma.masked,np.array(1),10**400])
@pytest.mark.parametrize('selected_kind',['features','labels'])
@pytest.mark.parametrize('model_type',['lr','rf','gb'])
def test_invalid_selected_values_before_split(dataset,no_split,bad,selected_kind,model_type):
    key='length_m' if selected_kind=='features' else 'force_N'
    getattr(dataset.samples[23],selected_kind)[key]=bad
    with pytest.raises(ValueError): native(dataset,model_type=model_type)


@pytest.mark.parametrize('options',[
    {'features':[]},{'features':'length_m'},{'features':('length_m',)},
    {'features':['length_m','length_m']},{'features':[None]},
    {'features':['feat_length_m']},{'features':['missing']},
    {'target':None},{'target':''},{'target':'label_force_N'},
    {'model_type':'other'},{'split':None},{'split':'automatic'},
    {'groups':['g']*24},
    {'split':'group','groups':None},{'split':'group','groups':['g']*24},
    {'split':'group','groups':['a','b']},{'split':'group','groups':['']*24},
    {'split':'group','groups':np.arange(24)},
    {'metadata_keys':'meaning'},{'metadata_keys':['meaning','meaning']},
    {'metadata_keys':[None]},{'metadata_keys':['missing']},
])
def test_invalid_options_before_split(dataset,no_split,options):
    with pytest.raises(ValueError): native(dataset,**options)


@pytest.mark.parametrize('test_size',[0,1,-.1,np.inf,np.nan,True,np.bool_(False),'.25',[],10**400])
def test_invalid_test_fraction_before_split(dataset,no_split,test_size):
    with pytest.raises(ValueError,match='test_size'): native(dataset,test_size=test_size)


@pytest.mark.parametrize('bad',[None,1,'',' '])
def test_native_invalid_sample_identity_before_split(dataset,no_split,bad):
    dataset.samples[5].sample_id=bad
    with pytest.raises(ValueError,match='sample_id'): native(dataset)


@pytest.mark.parametrize('attribute,value',[
    ('feature_names',[]),('feature_names',['length_m','length_m']),('feature_names','length_m'),
    ('target_names',[]),('target_names',['force_N','force_N']),('target_names',[None]),
])
def test_invalid_declared_names(dataset,no_split,attribute,value):
    setattr(dataset,attribute,value)
    with pytest.raises(ValueError): native(dataset)


def test_native_rejects_any_declared_target_feature(dataset,no_split):
    dataset.feature_names.append('unused_label')
    for sample in dataset.samples: sample.features['unused_label']=1.
    with pytest.raises(ValueError,match='declared target'): native(dataset,features=['unused_label'])


@pytest.mark.parametrize('bad',[np.nan,np.inf,np.float64(1),np.int64(1),np.bool_(True),np.array([1]),(1,),{1:2},{1},object()])
def test_selected_metadata_json_native_policy(dataset,no_split,bad):
    dataset.samples[7].metadata['meaning']={'nested':bad}
    with pytest.raises(ValueError,match='metadata'): native(dataset,metadata_keys=['meaning'])


@pytest.mark.parametrize('container',['list','dict'])
def test_selected_metadata_cycles_fail_before_fit_or_save(dataset,no_split,tmp_path,container):
    cycle=[] if container=='list' else {}
    if container=='list': cycle.append(cycle)
    else: cycle['again']=cycle
    dataset.samples[0].metadata['meaning']=cycle
    saved=tmp_path/'keep.pkl'; saved.write_bytes(b'original')
    with pytest.raises(ValueError,match='cycles'):
        native(dataset,metadata_keys=['meaning'],save_path=saved)
    assert saved.read_bytes()==b'original'


def test_selected_metadata_requires_each_row_mapping(dataset,no_split):
    dataset.samples[3].metadata=None
    with pytest.raises(ValueError,match='mapping'): native(dataset,metadata_keys=['meaning'])


def test_single_test_row_and_unsplittable_groups_rejected_before_fit(dataset,no_fit):
    with pytest.raises(ValueError,match='two test rows'): native(dataset,test_size=.01)
    with pytest.raises(ValueError,match='usable holdout'):
        native(dataset,split='group',groups=['a']*12+['b']*12,test_size=.99)


def test_group_fraction_is_fraction_of_groups_and_preserves_caller_vector(dataset):
    groups=['g0']*2+['g1']*2+['g2']*2+['g3']*18
    result=native(dataset,split='group',groups=groups,test_size=.25)
    split=result['training_data']['split']
    assert split['method']=='group' and split['groups']==groups
    assert split['groups'] is not groups
    assert len({groups[i] for i in split['test_rows']})==1
    assert len(split['test_rows'])==2  # One of four groups, not six of 24 rows.
    assert len(split['train_rows'])==22
    assert {groups[i] for i in split['train_rows']}.isdisjoint(groups[i] for i in split['test_rows'])
    groups[0]='edited'
    assert split['groups'][0]=='g0'


def test_known_duplicate_network_groups_never_overlap(dataset):
    groups=[]
    for i,sample in enumerate(dataset.samples):
        group=i//4; groups.append(f'origin-{group}')
        sample.features={'length_m':float(group),'area_m2':float(group**2)}
        sample.labels['force_N']=float(group*3)
    split=native(dataset,split='group',groups=tuple(groups))['training_data']['split']
    assert set(groups[i] for i in split['train_rows']).isdisjoint(groups[i] for i in split['test_rows'])
    assert len({s.sample_id for s in dataset.samples})==24


@pytest.mark.parametrize('kind,cls,settings',[
    ('lr',LinearRegression,{}),('rf',RandomForestRegressor,{'n_estimators':100,'random_state':42}),
    ('gb',GradientBoostingRegressor,{'n_estimators':100,'random_state':42}),
])
def test_legacy_csv_keyword_exact_predictions_and_keys(dataset,tmp_path,kind,cls,settings):
    pd=pytest.importorskip('pandas')
    path=tmp_path/'data.csv'
    # Only selected single target is exportable; irrelevant None remains native-only.
    selected=FiberNetDataset(dataset.samples,dataset.feature_names,['force_N'])
    selected.to_csv(path)
    names=['feat_area_m2','feat_length_m']
    frame=pd.read_csv(path); X=frame[names].to_numpy(); y=frame['label_force_N'].to_numpy()
    a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42)
    original=cls(**settings).fit(a,c)
    saved=tmp_path/'csv.pkl'
    result=train_model(csv_path=path,target='label_force_N',features=names,
                       model_type=kind,test_size=.25,save_path=saved)
    assert result=={'r2':r2_score(d,original.predict(b)),
                   'rmse':np.sqrt(mean_squared_error(d,original.predict(b))),
                   'n_samples':24,'features':names}
    with saved.open('rb') as handle: artifact=pickle.load(handle)
    assert set(artifact)=={'model','features','metrics','target','artifact_version'}
    np.testing.assert_array_equal(artifact['model'].predict(X),original.predict(X))


@pytest.mark.parametrize('options',[{'split':'row'},{'groups':['a']},{'metadata_keys':[]}])
def test_csv_rejects_native_options(tmp_path,options):
    with pytest.raises(ValueError,match='native FiberNetDataset'):
        train_model(csv_path=tmp_path/'not_read.csv',**options)


def test_native_without_pandas_or_torch_in_fresh_process(tmp_path):
    script=r'''
import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'pandas','torch'}:
            raise ModuleNotFoundError('blocked '+fullname)
sys.meta_path.insert(0,Block())
from fibernet.easy import train_model,predict_named
from fibernet.ml.dataset import FiberNetDataset,DatasetSample
samples=[DatasetSample(str(i),'x',(1,1),{}, {'x_m':float(i)}, {'force_N':float(i*2)}) for i in range(12)]
ds=FiberNetDataset(samples,['x_m'],['force_N'])
import pickle
for kind in ['lr','rf','gb']:
    result=train_model(ds,target='force_N',model_type=kind,split='group',groups=[str(i//3) for i in range(12)],test_size=.25,save_path='native-'+kind+'.pkl')
    with open('native-'+kind+'.pkl','rb') as handle: artifact=pickle.load(handle)
    assert predict_named(artifact,[s.features for s in samples])['target']=='force_N'
assert 'pandas' not in sys.modules and 'torch' not in sys.modules
assert 'sklearn' in sys.modules
print('NATIVE_OPTIONAL_IMPORTS_OK')
'''
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',TI_DISABLE_VERSION_CHECK='1',
             MPLCONFIGDIR=str(tmp_path/'mpl'),XDG_CACHE_HOME=str(tmp_path/'cache'))
    run=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'NATIVE_OPTIONAL_IMPORTS_OK' in run.stdout


@pytest.mark.parametrize("console_encoding", ["ascii", "cp1252"])
def test_native_training_status_supports_strict_narrow_console(dataset, tmp_path,
                                                              monkeypatch, console_encoding):
    """A status message must not fail an otherwise successful fit or save."""
    buffer = io.BytesIO()
    console = io.TextIOWrapper(buffer, encoding=console_encoding, errors="strict")
    monkeypatch.setattr(sys, "stdout", console)
    path = tmp_path / "model.pkl"
    result = native(dataset, save_path=path)
    console.flush()
    output = buffer.getvalue().decode(console_encoding)
    assert "OK: R^2=" in output
    assert "OK: Model saved to" in output
    assert path.is_file()
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    assert predict_named(artifact, [dataset.samples[0].features])["target"] == "force_N"
    assert result["n_samples"] == dataset.n_samples
