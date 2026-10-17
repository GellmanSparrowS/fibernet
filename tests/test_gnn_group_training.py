"""Group-mode preflight and bounded synthetic training integration."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_gnn_graph_batch import fixture, snapshot


def inputs():
    pytest.importorskip('torch')
    from fibernet.ml.gnn import graphs_from_batch
    groups = ['s' + str(i//2) for i in range(16)]
    gs = graphs_from_batch(snapshot(*[fixture(1+i/20) for i in range(16)]), source_groups=groups)
    return gs, np.arange(16, dtype=np.float32)/10, groups


def model():
    from fibernet.ml.gnn import FiberGNN
    return FiberGNN(4, hidden=8, n_layers=1, dropout=.2)


@pytest.mark.parametrize('change', ['no_split', 'row', 'groups_missing', 'groups_mismatch',
 'one_group', 'mixed', 'unknown', 'duplicate_ids', 'wrong_names', 'unit', 'target_shape',
 'target_masked', 'target_bool', 'target_inf', 'target_tiny', 'target_large', 'float_model',
 'multi_output', 'bad_gat', 'epochs_bool', 'epochs_zero', 'batch_float', 'patience_negative',
 'lr_nan', 'lr_zero', 'fraction_zero', 'fraction_one', 'float_edges', 'bad_adjacency',
 'subclass', 'input_double'])
def test_preflight_before_optimizer(change, monkeypatch):
    torch = pytest.importorskip('torch')
    from fibernet.ml.gnn import FiberGNN, train_gnn
    gs, y, groups = inputs()
    m = model()
    kw = dict(epochs=2, batch_size=4, early_stopping=0, verbose=False, split='group', groups=groups)
    if change == 'no_split': kw.pop('split')
    elif change == 'row': kw['split'] = 'row'
    elif change == 'groups_missing': kw.pop('groups')
    elif change == 'groups_mismatch': kw['groups'] = groups[::-1]
    elif change == 'one_group': kw['groups'] = ['same']*16
    elif change == 'mixed': gs[0].pop('input_schema')
    elif change == 'unknown': gs[0]['input_schema'] = 'unknown'
    elif change == 'duplicate_ids': gs[1]['graph_id'] = gs[0]['graph_id']
    elif change == 'wrong_names': gs[0]['node_feature_names'] = ('z','y','x','dimension')
    elif change == 'unit': gs[0]['length_unit'] = 'm'
    elif change == 'target_shape': y = y[:, None]
    elif change == 'target_masked': y = np.ma.array(y, mask=False)
    elif change == 'target_bool': y = y.astype(bool)
    elif change == 'target_inf': y[0] = np.inf
    elif change == 'target_tiny': y = y.astype(float); y[0] = 1e-200
    elif change == 'target_large': y = y.astype(float); y[0] = 1e200
    elif change == 'float_model': m.double()
    elif change == 'multi_output': m = FiberGNN(4, hidden=8, n_layers=1, n_outputs=2)
    elif change == 'bad_gat': m = FiberGNN(4, hidden=7, n_heads=4, n_layers=2, layer_type='gat')
    elif change == 'epochs_bool': kw['epochs'] = True
    elif change == 'epochs_zero': kw['epochs'] = 0
    elif change == 'batch_float': kw['batch_size'] = 2.5
    elif change == 'patience_negative': kw['early_stopping'] = -1
    elif change == 'lr_nan': kw['lr'] = np.nan
    elif change == 'lr_zero': kw['lr'] = 0
    elif change == 'fraction_zero': kw['val_split'] = 0
    elif change == 'fraction_one': kw['val_split'] = 1
    elif change == 'float_edges': gs[0]['edge_index'] = gs[0]['edge_index'].float()
    elif change == 'bad_adjacency': gs[0]['adjacency'] = None
    elif change == 'subclass':
        class Custom(FiberGNN): pass
        m = Custom(4, hidden=8, n_layers=1)
    elif change == 'input_double': gs[0]['node_features'] = gs[0]['node_features'].double()
    before = {k:v.clone() for k,v in m.state_dict().items()}
    def forbidden(*a, **k): raise AssertionError('optimizer constructed before preflight')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    with pytest.raises(ValueError): train_gnn(m, gs, y, **kw)
    assert all(torch.equal(v, m.state_dict()[k]) for k,v in before.items())


def test_missing_sklearn_and_legacy_groups_error(monkeypatch):
    pytest.importorskip('torch')
    from fibernet.ml.gnn import train_gnn
    gs,y,groups = inputs()
    monkeypatch.setitem(sys.modules, 'sklearn.model_selection', None)
    with pytest.raises(ImportError, match='scikit-learn'):
        train_gnn(model(), gs, y, split='group', groups=groups)
    for gd in gs: gd.pop('input_schema')
    with pytest.raises(ValueError): train_gnn(model(), gs, y, groups=groups)


def cpu_smoke(directory):
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    from fibernet.ml.gnn import FiberGNN, graph_from_structure, graphs_from_batch, train_gnn
    from fibernet.ml.training import save_model_bundle, load_model_bundle
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gs,y,groups = inputs()
    train,val = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
                     .split(np.arange(16),groups=groups))
    # Final-test IDs are excluded entirely from the trainer invocation.
    final_test_ids = {'unseen-final-a', 'unseen-final-b'}
    from fibernet.ml.graph_data import graph_batch
    final_snapshot = graph_batch([fixture(4), fixture(5)], graph_ids=sorted(final_test_ids),
                                  length_unit='mm', coordinate_frame='specimen')
    final_gs = graphs_from_batch(final_snapshot, source_groups=['final-source-a', 'final-source-b'])
    assert not {gd['source_group'] for gd in final_gs} & set(groups)
    def fit(labels):
        torch.manual_seed(42)
        m = model()
        assert m.dropout.p == .2
        before = {k:v.clone() for k,v in m.state_dict().items()}
        calls = []
        hook = m.register_forward_pre_hook(lambda mdl,args: calls.append(
            (mdl.training, torch.is_grad_enabled(), [g['graph_id'] for g in args[0]])))
        from unittest.mock import patch
        original_loss = torch.nn.functional.mse_loss
        def loss_spy(pred, target):
            indices = train if torch.is_grad_enabled() else val
            torch.testing.assert_close(target, torch.tensor(labels[indices]), rtol=0, atol=0)
            return original_loss(pred, target)
        with patch('torch.nn.functional.mse_loss', side_effect=loss_spy):
            result = train_gnn(m,gs,labels,split='group',groups=groups,epochs=2,
                               val_split=.25,batch_size=32,early_stopping=0,verbose=False)
        hook.remove()
        assert calls == [(True,True,[str(i) for i in train]),(False,False,[str(i) for i in val])]*2
        assert not final_test_ids & {gid for _,_,ids in calls for gid in ids}
        assert any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items())
        assert all(torch.isfinite(p).all() for p in m.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        return m,result
    a,result = fit(y)
    changed = y.copy(); changed[val] += 1000
    b,repeated = fit(changed)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert result['history'].val_losses != repeated['history'].val_losses
    assert result['best_val_loss'] == min(result['history'].val_losses)
    record = result['training_data']
    json.dumps(record, allow_nan=False)
    assert not set(record['split']['train_groups']) & set(record['split']['validation_groups'])
    assert record['split']['train_indices'] == train.tolist()
    assert record['split']['validation_indices'] == val.tolist()
    assert record['training']['epochs_run'] == 2 and not record['training']['best_state_restored']
    with torch.no_grad():
        pred = a(final_gs)
        assert pred.shape == (2,1) and torch.isfinite(pred).all()
        assert torch.equal(pred,b(final_gs))
    path = str(Path(directory)/'own-group.pt')
    save_model_bundle(a,path,feature_names=record['feature_names'],target_names=['synthetic'],
                      metadata={'training_data':record,'origin':'synthetic_fixture'})
    loaded,info = load_model_bundle(path,model_class=FiberGNN,node_dim=4,hidden=8,n_layers=1,dropout=.2)
    assert info['metadata']['training_data'] == record
    with torch.no_grad(): assert torch.equal(pred,loaded(final_gs))
    # Legacy remains unmarked and does not acquire group-mode return fields.
    torch.manual_seed(42)
    legacy = FiberGNN(5,hidden=8,n_layers=1,dropout=0)
    legacy_graph = fixture()
    legacy_graph.edges[9].radius = .2  # Legacy converter does not accept missing radius.
    legacy_result = train_gnn(legacy,[graph_from_structure(legacy_graph) for _ in range(8)],
                              np.arange(8,dtype=np.float32),epochs=2,early_stopping=0,verbose=False)
    assert set(legacy_result) == {'history','best_val_loss'}
    import resource
    print(json.dumps(dict(status='passed',epochs=2,train=12,validation=4,final_test=2,
          train_losses=result['history'].train_losses,
          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))


def test_bounded_group_training(tmp_path):
    pytest.importorskip('torch')
    if sys.platform != 'linux': pytest.skip('Linux address-space budget fixture')
    code = '''
import resource,runpy,sys
resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
sys.path.insert(0,sys.argv[1])
runpy.run_path(sys.argv[2])['cpu_smoke'](sys.argv[3])
'''
    env = dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    r = subprocess.run([sys.executable,'-c',code,str(Path(__file__).parent),__file__,str(tmp_path)],
                        env=env,capture_output=True,text=True,timeout=30)
    print(r.stdout)
    assert r.returncode == 0,r.stdout+r.stderr


def test_preflight_records_effective_gat_and_legacy_schema():
    pytest.importorskip('torch')
    from fibernet.ml.gnn import FiberGNN, _gnn_group_preflight
    gs,y,groups = inputs()
    m = FiberGNN(4,hidden=8,n_layers=2,n_heads=4,dropout=0,layer_type='gat')
    _,_,_,record = _gnn_group_preflight(m,gs,y,groups,2,.001,32,.25,3)
    assert record['model']['dropout'] == 0
    assert all(layer['attention_dropout'] == .1 for layer in record['model']['layers'])
    assert record['training']['early_stopping'] == 3
    assert record['split']['validation_role'] == 'validation'
    json.dumps(record,allow_nan=False)
    for gd in gs: gd.pop('input_schema')
    _,_,_,record = _gnn_group_preflight(model(),gs,y,groups,2,.001,32,.25,0)
    assert record['input_schema'] == 'caller-legacy'
    assert record['feature_names'] is None and record['graph_ids'] is None
